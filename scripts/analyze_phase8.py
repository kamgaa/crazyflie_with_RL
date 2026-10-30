"""Phase8 measured data audit and TensorBoard evaluation export. No optimizer."""
import argparse,json,csv,time,math,copy,shutil
from pathlib import Path
import numpy as np,torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ppo_common import ROOT,dump_json,sha256
from phase6_common import preserved,differences,equal_state
from phase8_reward import TERMS
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();start=time.perf_counter()
def read(p):return json.loads(Path(p).read_text())
def csvwrite(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
plan=read(r/'experiment_plan.json');assert read(r/'evaluation_finished.json')['success']
checks={};details={};training={};counter_audit={};history=[];selected=[];conditions=[];tb_compare={}
def check(k,value,detail=None):
    checks[k]=bool(value)
    if detail is not None:details[k]=detail
    if not value:raise AssertionError((k,detail))
def aggregate(xs):
    result=dict(completed_count=sum(x['completed'] for x in xs),mean_duration_s=float(np.mean([x['duration_s'] for x in xs])),condition_count=len(xs))
    metrics=['position_rmse_m','max_position_error_m','max_abs_z_error_m','max_tilt_deg','roll_rms_rad','body_omega_rms_rad_s',
        'z_std_m','z_peak_to_peak_m','action_clipping_fraction','allocation_clipping_fraction',
        'tail_position_mean_m','tail_position_rmse_m','tail_speed_mean_m_s','tail_z_mean_m','tail_z_std_m',
        'tail_z_peak_to_peak_m','tail_body_omega_rms_rad_s','tail_roll_rms_rad','tail_roll_rad_band_10_20_power',
        'tail_wx_band_10_20_power','tail_roll_rad_peak_hz','tail_wx_peak_hz','tail_ez_peak_hz','tail_vz_peak_hz',
        'initial_2s_position_rmse_m','common_reward_A_return','episode_return']
    for k in metrics:
        vals=[x[k] for x in xs if x.get(k) is not None];result[k]=float(np.mean(vals)) if vals else None
    result['max_tilt_deg']=max(x['max_tilt_deg'] for x in xs)
    result['completed_position_rmse_m']=float(np.mean([x['position_rmse_m'] for x in xs if x['completed']])) if result['completed_count'] else None
    result['tail_count']=sum(x['tail_duration_s'] is not None for x in xs)
    result['xyz_rmse_m']=np.mean([x['xyz_rmse_m'] for x in xs],axis=0).tolist()
    result['tail_error_vector_m']=np.mean([x['tail_error_vector_m'] for x in xs if x['tail_error_vector_m'] is not None],axis=0).tolist() if result['tail_count'] else None
    return result
if (r/'metrics_units_verification.json').exists():check('Euler_and_motor_units',read(r/'metrics_units_verification.json')['success'])
for n in plan['order']:
    d=r/n;t=read(d/'training_result.json');training[n]=t
    rows=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()]
    check(n+'_exact_budget',t['success'] and t['additional_iterations']==256 and t['additional_transitions']==8388608 and t['additional_optimizer_updates']==81920 and t['adam_steps']==[184320])
    check(n+'_steps',len(rows)==256 and [x['global_step'] for x in rows]==[32768*(i+1) for i in range(256)])
    check(n+'_all_counters',all(x['Performance/cumulative_transitions']==41943040+x['global_step'] and x['Performance/additional_optimizer_updates']==320*x['additional_iteration'] and x['Performance/optimizer_updates']==102400+320*x['additional_iteration'] for x in rows))
    rv=read(d/'resume_verification.json')
    check(n+'_strict_resume',rv['success'] and rv['rng_restored'] and rv['rng_fingerprints']==rv['parent_rng_fingerprints'] and rv['max_action_error']==0)
    check(n+'_unchanged_environment',not read(d/'config_diff.json')['environment'])
    check(n+'_reward_sum',max(x['Validation/reward_sum_max_abs_error'] for x in rows)<1e-4)
    check(n+'_actual_backward_probe',max(x['Validation/ppo_loss_residual'] for x in rows)<1e-6)
    loss_err=max(abs(x['Loss/ppo_total']-x['Loss/policy']-x['Loss/value_contribution']-x['Loss/entropy_contribution']) for x in rows)
    check(n+'_loss_contributions',loss_err<1e-3,loss_err)
    reward_err=max(abs(sum(x['RewardStep/'+k] for k in TERMS)-x['RewardStep/total']) for x in rows)
    check(n+'_reward_aggregate_sum',reward_err<1e-5,reward_err)
    common_error=max(abs(x['Monitor/common_reward_A']-(-10*x['Physical/xy_error_squared_m2']-6*x['Physical/z_error_squared_m2']-.005*x['Physical/velocity_squared_m2_s2']-6*x['Physical/tilt_quaternion_cost']-.001*x['Physical/body_omega_squared_rad2_s2']-x['Physical/wrapped_yaw_squared_rad2']-10*x['Physical/failure'])) for x in rows)
    check(n+'_common_reward_A',common_error<1e-5,common_error)
    check(n+'_episode_accounting',abs(t['episode_accounting']['reward_residual'])<1e-3 and t['episode_accounting']['step_residual']==0)
    if n!='Q_base':
        err=max(max(abs(x['RewardStep/xy']-.1*(x['Kernel/xy']-1)),abs(x['RewardStep/z']-.015*(x['Kernel/z']-1))) for x in rows)
        check(n+'_kernel_formula',err<1e-6,err)
    else:check(n+'_Q_common_bitwise',all(x['RewardStep/total']==x['Monitor/common_reward_A'] for x in rows))
    with (d/'metrics.csv').open() as f:cr=list(csv.DictReader(f))
    check(n+'_CSV_JSON',len(cr)==len(rows) and all(float(cr[i][k])==v for i,row in enumerate(rows) for k,v in row.items()))
    ev=EventAccumulator(str(d/'tensorboard'),size_guidance={'scalars':0});ev.Reload()
    lookup={tag:{x.step:x.value for x in ev.Scalars(tag)} for tag in ev.Tags()['scalars']}
    mismatches=[];matched=0
    for row in rows:
        for k,v in row.items():
            if '/' not in k:continue
            if lookup.get(k,{}).get(row['global_step'])!=v:mismatches.append([k,row['global_step'],v,lookup.get(k,{}).get(row['global_step'])])
            else:matched+=1
    check(n+'_TensorBoard_all_rows',not mismatches,mismatches[:3]);tb_compare[n]=dict(matched=matched,scalar_tags=len(lookup),first_middle_last_steps=[rows[i]['global_step'] for i in [0,127,255]])
    cp_audit=[]
    for cp in [d/'initial.pt',*sorted(p for p in d.glob('checkpoint_*.pt') if not p.stem.endswith('_reference')),d/'best.pt',d/'final.pt']:
        s=torch.load(cp,map_location='cpu',weights_only=False);steps=sorted({int(v['step']) for v in s['optimizer_state_dict']['state'].values()})
        completed=s['infos']['completed_iterations'];expected=102400+(completed-1280)*320
        check(n+'_'+cp.stem+'_checkpoint_counters',steps==[expected] and s['infos']['optimizer_updates']==expected and s['infos']['collected_transitions']==41943040+(completed-1280)*32768 and s['iter']==completed-1)
        cp_audit.append(dict(path=str(cp),sha256=sha256(cp),iterations=completed,adam_steps=steps,transitions=s['infos']['collected_transitions']))
    counter_audit[n]=cp_audit
    parent8=read(r/'common/evaluation_8s/results.json');periodic=read(d/'evaluation/periodic_8s/results.json')
    for ai in [0,64,128,192,256]:
        xs=parent8 if ai==0 else [x for x in periodic if x['iteration']==1280+ai]
        assert len(xs)==5
        h=dict(branch=n,kind='periodic',additional_iterations=ai,additional_transitions=ai*32768,additional_updates=ai*320,
            training_elapsed_s=0 if ai==0 else rows[ai-1]['Performance/training_elapsed_s'],horizon_s=8,**aggregate(xs))
        history.append(h)
    for horizon in [8,30]:
        xs=read(d/f'evaluation/reload_{horizon}s/results.json')
        for kind in ['best','final']:
            rr=[x for x in xs if x['checkpoint_name']==kind]
            shared=False
            if not rr:
                check(n+f'_{kind}_{horizon}_reuse_exact',kind=='best' and sha256(d/'best.pt')==sha256(d/'final.pt'))
                rr=[x for x in xs if x['checkpoint_name']=='final'];shared=True
            assert len(rr)==5
            s=dict(branch=n,kind=kind,horizon_s=horizon,additional_iterations=rr[0]['iteration']-1280,
                result_shared_with_final=shared,**aggregate(rr));selected.append(s)
            for x in rr:conditions.append(dict(branch=n,kind=kind,result_shared_with_final=shared,**x))
for horizon in [8,30]:
    xs=read(r/f'common/evaluation_{horizon}s/results.json')
    selected.insert(0,dict(branch='Parent',kind='initial',horizon_s=horizon,additional_iterations=0,result_shared_with_final=False,**aggregate(xs)))
    for x in xs:conditions.append(dict(branch='Parent',kind='initial',result_shared_with_final=False,**x))
# All detailed trajectories are audited, including failed periodic candidates.
eval_dirs=[r/'common/evaluation_8s',r/'common/evaluation_30s']+[r/n/'evaluation'/label for n in plan['order'] for label in ['periodic_8s','reload_8s','reload_30s']]
trajectory_count=0;max_eval_reward=0.;reload_count=0
for d in eval_dirs:
    ex=read(d/'execution.json');check(str(d.relative_to(r))+'_backend',ex['actual_backend'].endswith('PhysxManager') and ex['optimizer_updates']==0 and ex['actual_max_episode_steps']==ex['horizon_s']*100)
    for c in read(d/'reload_checks.json'):
        check(str(d.relative_to(r))+'_'+Path(c['checkpoint']).stem+'_reload',c['bitwise_equal'] and c['max_action_error']==0 and c['reference_count']==128);reload_count+=1
    for x in read(d/'results.json'):
        data=read(d/x['checkpoint_name']/(x['condition']+'.json'));rs=data['trajectory'];trajectory_count+=1
        check('trajectory_'+str(trajectory_count),len(rs)==x['steps'] and all(not(y['terminated'] or y['truncated']) for y in rs[:-1]) and all(math.isfinite(float(v)) for y in rs for v in y.values()))
        check('terminal_'+str(trajectory_count),all(y['terminated']==(y['z']<.2 or y['z']>2.5 or y['tilt_deg']>60 or y['position_error_m']>1.5) for y in rs))
        check('tail_'+str(trajectory_count),(x['tail_duration_s'] is None)==(not x['completed']))
        max_eval_reward=max(max_eval_reward,x['reward_sum_max_abs_error'])
        check('reward_'+str(trajectory_count),x['reward_sum_max_abs_error']<1e-4)
replay_comparisons=[]
for n in plan['order']:
    periodic=read(r/n/'evaluation/periodic_8s/results.json')
    for fresh in read(r/n/'evaluation/reload_8s/results.json'):
        if fresh['iteration']==1280:
            oldbase=r/'common/evaluation_8s/initial'
        else:
            old=next(x for x in periodic if x['iteration']==fresh['iteration'] and x['condition']==fresh['condition'])
            oldbase=r/n/'evaluation/periodic_8s'/old['checkpoint_name']
        newbase=r/n/'evaluation/reload_8s'/fresh['checkpoint_name']
        oldrows=read(oldbase/(fresh['condition']+'.json'))['trajectory']
        newrows=read(newbase/(fresh['condition']+'.json'))['trajectory']
        fields=['ex','ey','ez','vx','vy','vz','qw','qx','qy','qz','wx','wy','wz']+[f'action_{i}' for i in range(4)]
        error=max(abs(x[k]-y[k]) for x,y in zip(oldrows,newrows) for k in fields)
        replay_comparisons.append(dict(branch=n,kind=fresh['checkpoint_name'],condition=fresh['condition'],steps_match=len(oldrows)==len(newrows),
            state_and_action_max_abs_error_mixed_units=error,bitwise_equal=error==0 and len(oldrows)==len(newrows)))
dump_json(r/'trajectory_replay_comparison.json',replay_comparisons)
prefix_comparisons=[]
for n in ['Parent',*plan['order']]:
    for kind in (['initial'] if n=='Parent' else ['best','final']):
        b8=r/'common/evaluation_8s/initial' if n=='Parent' else r/n/'evaluation/reload_8s'/kind
        b30=r/'common/evaluation_30s/initial' if n=='Parent' else r/n/'evaluation/reload_30s'/kind
        if not b30.exists():b30=r/n/'evaluation/reload_30s/final'
        for cond in ['nominal','x_plus','x_minus','y_plus','y_minus']:
            s8=read(b8/(cond+'.json'))['trajectory'];s30=read(b30/(cond+'.json'))['trajectory']
            keys=['ex','ey','ez','vx','vy','vz','qw','qx','qy','qz','wx','wy','wz']+[f'raw_action_{i}' for i in range(4)]
            error=max(abs(x[k]-y[k]) for x,y in zip(s8,s30) for k in keys)
            prefix_comparisons.append(dict(branch=n,kind=kind,condition=cond,compared_steps=min(len(s8),len(s30)),
                prefix_max_abs_error_mixed_units=error,bitwise_equal=error==0))
dump_json(r/'horizon_prefix_comparison.json',prefix_comparisons)
check('8s_30s_physical_prefix_equal',all(x['bitwise_equal'] for x in prefix_comparisons))
cfg8=read(r/'common/evaluation_8s/execution.json')['config'];cfg30=read(r/'common/evaluation_30s/execution.json')['config']
ed=differences(cfg8,cfg30);check('30s_only_time_limit_changed',{x['path'] for x in ed}=={'episode_length_s'},ed)
proofs=[read(r/n/'resume_verification.json') for n in plan['order']]
for key in ['actor_sha256','critic_sha256','mean_sha256','std_sha256','critic_output_sha256','rng_fingerprints']:
    check('initial_shared_'+key,all(v[key]==proofs[0][key] for v in proofs))
first=[json.loads((r/n/'metrics.jsonl').read_text().splitlines()[0]) for n in plan['order']]
keys=[k for k in first[0] if k.startswith(('Flight/','Actuator/','Diversity/','Physical/','Kernel/')) or k=='Monitor/common_reward_A']
first_error=max(abs(x[k]-first[0][k]) for x in first for k in keys)
check('first_rollout_physical_samples_identical',first_error==0,first_error)
check('budget_total',sum(t['additional_transitions'] for t in training.values())==25165824 and sum(t['additional_optimizer_updates'] for t in training.values())==245760)
parent_sha=sha256(Path(plan['checkpoint']));check('parent_preserved',parent_sha==plan['parent_sha256'])
preservation={key:preserved(r/'evidence'/fname,ROOT) for key,fname in [('project','project_before_sha256.json'),('prior','prior_runs_sha256.json')]}
check('prior_artifacts_preserved',all(v['success'] for v in preservation.values()),preservation)
source=read(r/'source_sha256.json');check('executed_sources_unchanged',all(sha256(ROOT/k)==v for k,v in source.items()))
# Export evaluation scalars with additional transitions as global_step, preserving original training events.
eval_scalar_rows=[]
for n in plan['order']:
    tagfile=r/n/'evaluation_scalars.json'
    if tagfile.exists():
        local=read(tagfile)
        ev=EventAccumulator(str(r/n/'tensorboard'),size_guidance={'scalars':0});ev.Reload()
        lookup={tag:{x.step:x.value for x in ev.Scalars(tag)} for tag in ev.Tags()['scalars']}
        check(n+'_Eval_TensorBoard_CSV_JSON',all(lookup[z['tag']][z['global_step']]==z['value'] for z in local))
        eval_scalar_rows.extend(local);continue
    local=[]
    def add(prefix,step,data):
        for k,v in data.items():
            if isinstance(v,(int,float)) and v is not None and math.isfinite(v):
                local.append(dict(tag=prefix+'/'+k,global_step=int(step),value=float(np.float32(v))))
    for h in [x for x in history if x['branch']==n]:add('Eval8/summary',h['additional_transitions'],h)
    for x in conditions:
        if x['branch'] not in [n,'Parent']:continue
        if x['branch']=='Parent':prefix=f"Eval{x['horizon_s']}/parent/{x['condition']}";step=0
        else:prefix=f"Eval{x['horizon_s']}/{x['kind']}/{x['condition']}";step=(x['iteration']-1280)*32768
        add(prefix,step,x)
    writer=SummaryWriter(str(r/n/'tensorboard'))
    for z in local:writer.add_scalar(z['tag'],z['value'],z['global_step'])
    writer.flush();writer.close()
    dump_json(tagfile,local);csvwrite(r/n/'evaluation_scalars.csv',local);eval_scalar_rows+=local
    ev=EventAccumulator(str(r/n/'tensorboard'),size_guidance={'scalars':0});ev.Reload()
    lookup={tag:{x.step:x.value for x in ev.Scalars(tag)} for tag in ev.Tags()['scalars']}
    check(n+'_Eval_TensorBoard_CSV_JSON',all(lookup[z['tag']][z['global_step']]==z['value'] for z in local))
spectral_diagnosis=[]
for x in conditions:
    n=x['branch'];h=x['horizon_s'];kind=x['kind'];c=x['condition']
    base=r/f'common/evaluation_{h}s/initial' if n=='Parent' else r/n/f'evaluation/reload_{h}s'/('final' if x['result_shared_with_final'] else kind)
    spectra=read(base/(c+'_spectra.json'))
    rec=dict(branch=n,kind=kind,horizon_s=h,condition=c,completed=x['completed'])
    for segment,sp in spectra.items():
        for sig in ['roll_rad','wx','ez','vz']:
            v=sp[sig]
            if not v:continue
            f=np.array(v['frequency_hz']);psd=np.array(v['psd']);df=v['resolution_hz']
            rec[f'{segment}_{sig}_2_5_power']=float(psd[(f>=2)&(f<=5)].sum()*df)
            band=(f>=1)&(f<=20)
            rec[f'{segment}_{sig}_peak_1_20_hz']=float(f[band][np.argmax(psd[band])])
    data=read(base/(c+'.json'))['trajectory']
    xy=np.array([z['ex']**2+z['ey']**2 for z in data]);zz=np.array([z['ez']**2 for z in data])
    def first_time(pred):
        return next((z['time_s'] for z in data if pred(z)),None)
    rec['first_xy_kernel_below_005_s']=first_time(lambda z: math.exp(-(z['ex']**2+z['ey']**2)/.01)<.05)
    rec['first_z_kernel_below_005_s']=first_time(lambda z: math.exp(-z['ez']**2/.0025)<.05)
    rec['first_tilt_above_20deg_s']=first_time(lambda z:z['tilt_deg']>20)
    rec['first_action_clip_s']=first_time(lambda z:z['action_clipping_fraction']>0)
    rec['termination_reasons']=x['reasons'];rec['duration_s']=x['duration_s']
    rec['xy_kernel_below_005_fraction']=float((np.exp(-xy/.01)<.05).mean())
    rec['z_kernel_below_005_fraction']=float((np.exp(-zz/.0025)<.05).mean())
    spectral_diagnosis.append(rec)
for x in selected:
    ys=[z for z in spectral_diagnosis if z['branch']==x['branch'] and z['kind']==x['kind'] and z['horizon_s']==x['horizon_s']]
    for k in ['tail_ez_2_5_power','tail_ez_peak_1_20_hz','tail_wx_peak_1_20_hz','xy_kernel_below_005_fraction','z_kernel_below_005_fraction']:
        vals=[y[k] for y in ys if k in y];x[k]=float(np.mean(vals)) if vals else None
dump_json(r/'spectral_diagnosis.json',spectral_diagnosis);csvwrite(r/'plots/spectral_diagnosis.csv',spectral_diagnosis)
paired=[]
for h in [8,30]:
    for n,b in [('Q_base','Parent'),('E_position','Q_base'),('E_omega','E_position')]:
        aa=[z for z in conditions if z['branch']==n and z['kind']=='final' and z['horizon_s']==h]
        bb=[z for z in conditions if z['branch']==b and z['kind']==('initial' if b=='Parent' else 'final') and z['horizon_s']==h]
        keep=[z['condition'] for z in aa if z['completed'] and any(y['condition']==z['condition'] and y['completed'] for y in bb)]
        row=dict(branch=n,baseline=b,horizon_s=h,paired_completed_count=len(keep),conditions=keep,
            note='Survivor-conditioned matched starts only; cannot replace all-five completion-rate comparison')
        for key in ['position_rmse_m','tail_position_mean_m','tail_speed_mean_m_s','tail_z_std_m','tail_z_peak_to_peak_m',
            'tail_body_omega_rms_rad_s','tail_roll_rms_rad','tail_roll_rad_band_10_20_power','initial_2s_position_rmse_m']:
            av=[z[key] for z in aa if z['condition'] in keep];bv=[z[key] for z in bb if z['condition'] in keep]
            row[key+'_branch']=float(np.mean(av)) if av else None;row[key+'_baseline']=float(np.mean(bv)) if bv else None
            row[key+'_difference']=float(np.mean(av)-np.mean(bv)) if av else None
        paired.append(row)
dump_json(r/'paired_completed_comparison.json',paired);csvwrite(r/'plots/paired_completed_comparison.csv',paired)
dump_json(r/'training_results.json',training);dump_json(r/'evaluation_history.json',history);dump_json(r/'selected_summary.json',selected)
dump_json(r/'evaluation_conditions.json',conditions);dump_json(r/'checkpoint_audit.json',counter_audit)
dump_json(r/'preservation.json',preservation)
csvwrite(r/'plots/evaluation_history.csv',history);csvwrite(r/'plots/selected_summary.csv',selected);csvwrite(r/'plots/evaluation_conditions.csv',conditions)
windows={}
for n in plan['order']:
    rows=[json.loads(x) for x in (r/n/'metrics.jsonl').read_text().splitlines()]
    tags=[k for k in rows[0] if k.startswith(('RewardStep/','Monitor/','Loss/','Policy/','PPO/','Physical/','Kernel/'))]
    windows[n]={label:{k:float(np.mean([x[k] for x in seq])) for k in tags} for label,seq in [('first64',rows[:64]),('last64',rows[-64:])]}
dump_json(r/'training_window_summary.json',windows)
dump_json(r/'verification.json',dict(success=all(checks.values()),checks=checks,details=details,check_count=len(checks),
    tensorboard_training=tb_compare,eval_scalar_count=len(eval_scalar_rows),trajectory_count=trajectory_count,reload_count=reload_count,
    max_eval_reward_sum_error=max_eval_reward,first_rollout_max_error=first_error,
    horizon_prefix_count=len(prefix_comparisons),horizon_prefix_max_error=max(x['prefix_max_abs_error_mixed_units'] for x in prefix_comparisons),
    analysis_data_wall_s=time.perf_counter()-start))
print('PHASE8_ANALYSIS_VERIFIED',len(checks),trajectory_count,flush=True)


"""Phase7 measured comparisons, event checks, raw plots. No training."""
import argparse,json,csv,math,time,shutil,collections
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ppo_common import ROOT,dump_json,sha256
from phase6_common import preserved,equal_state
START=time.perf_counter()
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def csvwrite(p,rs):
    keys=list(dict.fromkeys(k for x in rs for k in x))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rs)
def avg(xs):return float(np.mean(xs)) if xs else None
def summary(rs,name,kind):
    good=[x for x in rs if x['completed_8s']]
    return dict(branch=name,kind=kind,iteration=rs[0]['iteration'],transitions=rs[0]['transitions'],
        additional_iterations=rs[0]['iteration']-1024,additional_transitions=rs[0]['transitions']-33554432,
        completed_count=len(good),completion_fraction=len(good)/5,mean_duration_s=avg([x['duration_s'] for x in rs]),
        mean_position_rmse_m=avg([x['position_rmse_m'] for x in rs]),
        completed_position_rmse_m=avg([x['position_rmse_m'] for x in good]),
        worst_position_rmse_m=max(x['position_rmse_m'] for x in rs),
        max_tilt_deg=max(x['max_tilt_deg'] for x in rs),mean_tilt_deg=avg([x['mean_tilt_deg'] for x in rs]),
        mean_speed_rms_m_s=avg([x['speed_rms_m_s'] for x in rs]),
        tail2_position_mean_m=avg([x['tail2_position_mean_m'] for x in good]),
        tail2_speed_mean_m_s=avg([x['tail2_speed_mean_m_s'] for x in good]),
        action_clipping_fraction=avg([x['action_clipping_fraction'] for x in rs]),
        allocation_clipping_fraction=avg([x['allocation_clipping_fraction'] for x in rs]),
        checkpoint=rs[0]['checkpoint'])
assert (r/'evaluation_finished.json').exists()
plan=read(r/'experiment_plan.json');parent=Path(plan['parent']);plot=r/'plots';terms=['xy','z','velocity','tilt','angular_velocity','yaw','termination']
initial=read(r/'common/evaluation_initial/results.json');checks={};details={};selected=[];history=[];training={};allconditions=[];audits=[];allreload=[];statuses=[]
parentcp=torch.load(parent/'final.pt',map_location='cpu',weights_only=False)
for name in plan['order']:
    out=r/name
    if not (out/'training_result.json').exists():
        statuses.append(dict(branch=name,success=False,failure=read(out/'failure.json')));continue
    result=read(out/'training_result.json');statuses.append(dict(branch=name,success=True,training=result))
    rows=lines(out/'metrics.jsonl');training[name]=rows;mb=plan['branches'][name]['minibatches'];u=5*mb
    checks[name+'/budget']=len(rows)==256 and result['additional_transitions']==8388608 and result['additional_optimizer_updates']==u*256
    checks[name+'/counters']=all(x['global_step']==33554432+(i+1)*32768 and x['iteration']==1025+i and x['Performance/optimizer_updates']==20480+u*(i+1) and x['Performance/additional_optimizer_updates']==u*(i+1) for i,x in enumerate(rows))
    checks[name+'/first_learning_step']=rows[0]['global_step']==33587200
    checks[name+'/resume']=read(out/'resume_verification.json')['success']
    rv=read(out/'resume_verification.json');checks[name+'/rng_exact']=rv['rng_fingerprints']==rv['parent_rng_fingerprints']
    checks[name+'/finite']=all(math.isfinite(v) for x in rows for v in x.values() if isinstance(v,(int,float)))
    checks[name+'/reward_sample_sum']=max(x['Validation/reward_sum_max_abs_error'] for x in rows)<1e-4
    checks[name+'/backward_loss']=max(abs(x['Validation/ppo_loss_residual']) for x in rows)<1e-6
    rd=max(abs(sum(x['RewardStep/'+k] for k in terms)-x['RewardStep/total']) for x in rows)
    ld=max(abs(x['Loss/policy']+x['Loss/value_contribution']+x['Loss/entropy_contribution']-x['Loss/ppo_total']) for x in rows)
    checks[name+'/reward_aggregate_sum']=rd<1e-4;checks[name+'/loss_aggregate_sum']=ld<1e-4
    scale=plan['branches'][name]['position_scale']
    common_error=max(abs(x['Monitor/common_reward_A']-(x['RewardStep/total']-(1-1/scale)*(x['RewardStep/xy']+x['RewardStep/z']))) for x in rows)
    checks[name+'/common_reward_monitor_only']=common_error<1e-4
    diff=read(out/'config_diff.json');allowed={'experiment_name','run_name','load_run','max_iterations'}|({'algorithm.num_mini_batches'} if name=='M64_minibatch' else set())
    checks[name+'/config_diff_expected']=not diff['environment'] and not diff['optimizer'] and {x['path'] for x in diff['agent']}<=allowed
    checks[name+'/episode_accounting']=abs(result['episode_accounting']['reward_residual'])<1e-3 and result['episode_accounting']['step_residual']==0
    periodic=read(out/'evaluation/periodic/results.json');best=read(out/'evaluation/best_reload/results.json');final=read(out/'evaluation/final_reload/results.json')
    checks[name+'/eval_schedule']=sorted({x['iteration'] for x in periodic})==[1088,1152,1216,1280]
    sh=[summary(initial,name,'initial')]+[summary([x for x in periodic if x['iteration']==i],name,'periodic') for i in [1088,1152,1216,1280]]
    for x in sh:
        x['additional_optimizer_updates']=x['additional_iterations']*u
        x['additional_training_s']=rows[x['additional_iterations']-1]['Performance/training_elapsed_s'] if x['additional_iterations'] else 0.
    history+=sh
    for label,rs in [('best',best),('final',final)]:
        sm=summary(rs,name,label);sm['best_is_parent']=read(out/'selection.json')['best_is_parent'] if label=='best' else False
        sm['additional_optimizer_updates']=sm['additional_iterations']*u;sm['additional_training_s']=rows[sm['additional_iterations']-1]['Performance/training_elapsed_s'] if sm['additional_iterations'] else 0
        selected.append(sm)
        for x in rs:allconditions.append(dict(branch=name,model=label,**{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in x.items()}))
    reloads=[]
    for d in ['periodic','best_reload','final_reload']:reloads+=read(out/('evaluation/'+d+'/reload_checks.json'))
    allreload+=reloads
    checks[name+'/reload_action_exact']=all(x['max_action_error']==0 and x['bitwise_equal'] and x['optimizer_updates_in_process']==0 for x in reloads)
    checks[name+'/fresh_best_final_processes']=read(out/'evaluation/best_reload/reload_checks.json')[0]['pid']!=read(out/'evaluation/final_reload/reload_checks.json')[0]['pid']
    cpfiles=[x for x in out.glob('*.pt') if not x.stem.endswith('_reference')]
    for path in cpfiles:
        cp=torch.load(path,map_location='cpu',weights_only=False);info=cp['infos'];add=info['completed_iterations']-1024
        expected=20480+add*u;steps=sorted({int(v['step']) for v in cp['optimizer_state_dict']['state'].values()})
        good=steps==[expected] and cp['iter']==info['completed_iterations']-1 and info['optimizer_updates']==expected and info['collected_transitions']==33554432+add*32768
        audits.append(dict(branch=name,path=str(path),success=good,additional_iterations=add,adam_steps=steps,sha256=sha256(path)))
    checks[name+'/checkpoint_counters']=all(x['success'] for x in audits if x['branch']==name)
    fc=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    checks[name+'/moments_updated']=all(not torch.equal(v['exp_avg'],parentcp['optimizer_state_dict']['state'][k]['exp_avg']) and not torch.equal(v['exp_avg_sq'],parentcp['optimizer_state_dict']['state'][k]['exp_avg_sq']) for k,v in fc['optimizer_state_dict']['state'].items())
    checks[name+'/best_copy']=sha256(out/'best.pt')==read(out/'selection.json')['source_sha256']
    # Verify terminal recording and last2 vector directly from the saved pre-autoreset trajectory.
    tv=[]
    for folder,rs in [('periodic',periodic),('best_reload',best),('final_reload',final)]:
        for x in rs:
            path=out/'evaluation'/folder/x['checkpoint_name']/(x['condition']+'.csv')
            with path.open() as f:tr=list(csv.DictReader(f))
            istrue=lambda z:z in ('True','1','1.0')
            good=len(tr)==x['steps'] and not any(istrue(z['terminated']) or istrue(z['truncated']) for z in tr[:-1])
            last=tr[-1];physical_failure=float(last['z'])<.2 or float(last['z'])>2.5 or float(last['tilt_deg'])>60 or float(last['position_error_m'])>1.5
            good=good and (istrue(last['terminated'])==physical_failure)
            good=good and abs(len(tr)*.01-x['duration_s'])<1e-9
            if x['completed_8s']:
                tail=tr[-200:];vector=[sum(float(z[k]) for z in tail)/200 for k in ['ex','ey','ez']]
                good=good and len(tr)==800 and x['tail2_duration_s']==2 and max(abs(a-b) for a,b in zip(vector,x['tail2_error_vector_m']))<1e-12
            else:good=good and x['tail2_error_vector_m'] is None
            tv.append(dict(path=str(path),success=good,steps=len(tr),terminated=istrue(last['terminated'])))
    checks[name+'/terminal_and_tail_metrics']=all(x['success'] for x in tv)
    dump_json(out/'evaluation/trajectory_verification.json',tv)
    # Evaluation/start tags are separate from learning metrics.
    erows=[dict(global_step=33554432,**{'Init/Adam_step':20480.,'Init/mean_max_abs_error':float(rv['max_action_error']),'Init/shared_outputs_exact':1.})]
    for sm in sh+[dict(x,event_prefix='EvalReload/'+x['kind']) for x in selected if x['branch']==name]:
        prefix=sm.get('event_prefix','Eval');er={'global_step':int(sm['transitions'])}
        for k,v in sm.items():
            if isinstance(v,(float,int)) and k not in ['transitions']:er[prefix+'/'+k]=float(np.float32(v))
        rs=initial if sm['kind']=='initial' else [x for x in periodic if x['iteration']==sm['iteration']]
        if sm['kind']=='best':rs=best
        if sm['kind']=='final':rs=final
        for x in rs:
            for k in ['duration_s','position_rmse_m','max_tilt_deg','tail2_position_mean_m','tail2_speed_mean_m_s','completed_8s']:
                if x[k] is not None:er[prefix+'/'+x['condition']+'/'+k]=float(np.float32(x[k]))
        erows.append(er)
    ep=out/'evaluation/event_scalars.json'
    if ep.exists():assert read(ep)==erows
    else:
        wr=SummaryWriter(str(out/'tensorboard'),filename_suffix='.phase7_evaluation')
        for x in erows:
            for k,v in x.items():
                if k!='global_step':wr.add_scalar(k,v,x['global_step'])
        wr.add_custom_scalars({'Phase7':{'Reward and shared monitoring':['Multiline',['RewardStep/total','Monitor/common_reward_A']],
            'PPO actual loss':['Multiline',['Loss/policy','Loss/value_contribution','Loss/entropy_contribution','Loss/ppo_total']],
            'Evaluation precision':['Multiline',['Eval/completed_position_rmse_m','Eval/tail2_position_mean_m']],
            'Clipping separation':['Multiline',['PPO/clip_fraction','Actuator/action_clipping_fraction','Actuator/allocation_clipping_fraction']]}})
        wr.flush();wr.close();dump_json(ep,erows)
    csvwrite(out/'evaluation/event_scalars.csv',erows)
    ea=EventAccumulator(str(out/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
    tags=ea.Tags()['scalars'];ev={k:{x.step:x.value for x in ea.Scalars(k)} for k in tags}
    mismatches=[];number=0
    for x in rows+erows:
        for k,v in x.items():
            if '/' not in k:continue
            number+=1
            if ev.get(k,{}).get(x['global_step'])!=v:mismatches.append((k,x['global_step']))
    checks[name+'/event_json_exact']=not mismatches
    checks[name+'/event_steps_increasing']=all(all(a.step<b.step for a,b in zip(ea.Scalars(k),ea.Scalars(k)[1:])) for k in tags)
    with (out/'metrics.csv').open() as f:csvrows=list(csv.DictReader(f))
    checks[name+'/csv_json_exact']=len(csvrows)==256 and all(all((v=='' and k not in row) or (k in row and float(v)==row[k]) for k,v in cr.items()) for cr,row in zip(csvrows,rows))
    details[name]=dict(scalar_tag_count=len(tags),scalar_values_compared=number,event_mismatches=mismatches,
        required_spot_steps=[rows[i]['global_step'] for i in [0,127,255]],
        reward_aggregate_max_error=rd,loss_aggregate_max_error=ld,common_reward_A_max_error=common_error,
        final_log_std=fc['actor_state_dict']['distribution.log_std_param'].tolist())
    csvwrite(out/'evaluation/summary.csv',sh)
# Baseline selection and overall exact budget.
checks['shared_initial_outputs']=len({read(r/n/'resume_verification.json')['mean_sha256'] for n in training})==1 and len({read(r/n/'resume_verification.json')['critic_output_sha256'] for n in training})==1
first=[training[n][0] for n in training]
common_keys=[k for k in first[0] if k.startswith(('Flight/','Actuator/','Diversity/'))]+['Monitor/common_reward_A']
initial_rollout_error=max(abs(x[k]-first[0][k]) for x in first for k in common_keys)
checks['identical_first_rollout_state_statistics']=initial_rollout_error<1e-5
details['initial_rollout_max_abs_error']=initial_rollout_error
checks['diagnosis_before_training']=read(r/'diagnosis_ready.json')['time']<read(r/'training_orchestration_started.json')['time']
checks['diagnosis_unchanged']=read(r/'diagnosis_ready.json')['report_sha256']==sha256(r/'diagnosis_before_training_ko.txt')
checks['all_branches_completed']=len(training)==3
checks['additional_budget_totals']=sum(x['training']['additional_transitions'] for x in statuses if x['success'])==25165824 and sum(x['training']['additional_optimizer_updates'] for x in statuses if x['success'])==92160
dump_json(r/'training_status.json',statuses);dump_json(r/'checkpoint_audit.json',audits)
dump_json(r/'evaluation_history.json',history);dump_json(r/'selected_summary.json',selected)
csvwrite(plot/'evaluation_history.csv',history);csvwrite(plot/'selected_summary.csv',selected);csvwrite(plot/'best_final_conditions.csv',allconditions)
# Summary windows: equal data / 64 iterations per branch.
windows={}
for name,rows in training.items():
    windows[name]={}
    for label,rr in [('first64',rows[:64]),('last64',rows[-64:])]:
        keys=['RewardStep/'+k for k in terms+['total']]+['Monitor/common_reward_A','Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total','Policy/entropy','Policy/std_mean','PPO/approx_kl','PPO/clip_fraction','Actuator/action_clipping_fraction','Actuator/allocation_clipping_fraction']
        val={k:avg([x[k] for x in rr]) for k in keys}
        for k in ['Episode/length_s','Episode/return']:
            good=[x for x in rr if k in x];den=sum(x['Episode/completed_count'] for x in good)
            val[k]=sum(x[k]*x['Episode/completed_count'] for x in good)/den if den else None
        windows[name][label]=val
dump_json(r/'training_window_summary.json',windows)
# Plot raw values; no smoothing, no fabricated evaluation points.
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.22})
colors={'C0_control':'tab:blue','M64_minibatch':'tab:orange','R2_position':'tab:green'}
def save(fig,name):
    fig.tight_layout(rect=(0,0,1,.965));fig.savefig(plot/(name+'.png'),dpi=145);plt.close(fig)
for xkey,xlabel in [('additional_transitions','Additional policy transitions'),('additional_optimizer_updates','Additional measured Adam updates'),('additional_training_s','Additional measured training wall (s)')]:
    fig,axs=plt.subplots(3,2,figsize=(12,11))
    for ax,key,ylabel in zip(axs.flat,['completion_fraction','mean_duration_s','completed_position_rmse_m','max_tilt_deg','tail2_position_mean_m','tail2_speed_mean_m_s'],
          ['8s completion fraction','Mean duration s','Completed-condition RMSE m','Maximum tilt deg','Completed last2 position m','Completed last2 speed m/s']):
        for name in training:
            ss=[x for x in history if x['branch']==name]
            ax.plot([x[xkey] for x in ss],[x[key] if x[key] is not None else np.nan for x in ss],'.-',label=name,color=colors[name])
        if key=='completed_position_rmse_m':ax.axhline(.014310829356538973,color='black',ls=':',label='N64 final, different history')
        ax.set(xlabel=xlabel,ylabel=ylabel);ax.legend(fontsize=7)
    fig.suptitle('Phase7 measured fixed evaluations; failed episode RMSE omitted from precision panels')
    save(fig,'evaluation_vs_'+xkey);csvwrite(plot/('evaluation_vs_'+xkey+'.csv'),history)
def series(name,keys,title):
    fig,axs=plt.subplots(len(keys),1,figsize=(11,2.3*len(keys)),sharex=True,squeeze=False)
    export=[]
    for key,ax in zip(keys,axs[:,0]):
        for branch,rows in training.items():
            ax.plot([x['Performance/additional_transitions'] for x in rows],[x.get(key,np.nan) for x in rows],label=branch,color=colors[branch],lw=.85)
        ax.set_ylabel(key);ax.legend(fontsize=7)
    axs[-1,0].set_xlabel('Additional policy transitions');fig.suptitle(title);save(fig,name)
    for branch,rows in training.items():
        export += [dict(branch=branch,additional_transitions=x['Performance/additional_transitions'],iteration=x['iteration'],**{k:x.get(k) for k in keys}) for x in rows]
    csvwrite(plot/(name+'.csv'),export)
series('reward_terms',['RewardStep/'+k for k in ['total']+terms]+['Monitor/common_reward_A'],'Signed reward terms and common reward A (R2 objective differs)')
series('ppo_loss',['Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total'],'Actual backward losses; R2 value targets differ')
series('ppo_distribution',['PPO/approx_kl','PPO/clip_fraction']+['Policy/std_axis_'+str(i) for i in range(4)],'Probability clipping / KL / Gaussian std')
series('episode_clipping',['Episode/length_s','Episode/end_failure_fraction','Episode/end_timeout_fraction','Actuator/action_clipping_fraction','Actuator/allocation_clipping_fraction'],'Completed episodes and physical clipping; missing averages remain blank')
series('performance',['Performance/transitions_per_s','Performance/rollout_s','Performance/update_s','Performance/cuda_device_used_MiB','Performance/torch_peak_allocated_MiB'],'Measured throughput / time / distinct memory scopes')
# Same axes show parent plus each branch's best/final trajectories.
for label in ['best','final']:
    fig,axs=plt.subplots(5,5,figsize=(19,16));export=[]
    for j,condition in enumerate(['nominal','x_plus','x_minus','y_plus','y_minus']):
        sources=[('Parent',r/'common/evaluation_initial/final'/(condition+'.csv'))]+[(name,r/name/'evaluation'/(label+'_reload')/label/(condition+'.csv')) for name in training]
        for branch,path in sources:
            with path.open() as f:tr=list(csv.DictReader(f))
            t=[float(x['time_s']) for x in tr];data={'ex':[float(x['ex']) for x in tr],'ey':[float(x['ey']) for x in tr],'ez':[float(x['ez']) for x in tr],
                'speed':[math.sqrt(sum(float(x[k])**2 for k in ['vx','vy','vz'])) for x in tr],'tilt':[float(x['tilt_deg']) for x in tr]}
            for ax,k,unit in zip(axs[j],['ex','ey','ez','speed','tilt'],['ex m','ey m','ez m','speed m/s','tilt deg']):
                ax.plot(t,data[k],label=branch,lw=.75,color='black' if branch=='Parent' else colors[branch],ls='--' if branch=='Parent' else '-')
                ax.set(title=condition,ylabel=unit,xlabel='Episode time s')
            export += [dict(model=branch,condition=condition,**x) for x in tr]
        for ax in axs[j]:ax.legend(fontsize=6)
    fig.suptitle('Parent vs phase7 '+label+' policies; actual first-episode trajectories')
    save(fig,label+'_trajectories');csvwrite(plot/(label+'_trajectories.csv'),export)
preservation=dict(project=preserved(r/'evidence/project_before_sha256.json',ROOT),prior_runs=preserved(r/'evidence/prior_runs_sha256.json',ROOT))
dump_json(r/'preservation.json',preservation)
checks['prior_project_unchanged']=preservation['project']['success'];checks['prior_runs_unchanged']=preservation['prior_runs']['success']
details['analysis_wall_s']=time.perf_counter()-START
dump_json(r/'verification.json',dict(success=all(checks.values()),checks=checks,details=details))
print('ANALYSIS_RESULT',json.dumps(dict(success=all(checks.values()),selected=selected,checks=checks),indent=2),flush=True)
assert all(checks.values()),checks


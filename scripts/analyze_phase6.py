"""Post-process phase 6 only: no environment creation or optimizer updates."""
import argparse,csv,json,math,time,collections,shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from ppo_common import ROOT,dump_json,sha256
from phase6_common import preserved
START=time.perf_counter()
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args()
run=Path(args.run_dir).resolve();plot=run/'plots';plot.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def mean(xs):return float(np.mean(xs)) if len(xs) else None
def summarize(rs,source):
    assert len(rs)==5
    completed=[x for x in rs if x['completed_8s']]
    return dict(source=source,iteration=rs[0]['iteration'],transitions=rs[0]['transitions'],
        completed_count=len(completed),completion_rate=len(completed)/5,mean_duration_s=mean([x['duration_s'] for x in rs]),
        mean_position_rmse_m=mean([x['position_rmse_m'] for x in rs]),
        worst_position_rmse_m=max(x['position_rmse_m'] for x in rs),
        completed_mean_position_rmse_m=mean([x['position_rmse_m'] for x in completed]),
        max_tilt_deg=max(x['max_tilt_deg'] for x in rs),mean_tilt_deg=mean([x['mean_tilt_deg'] for x in rs]),
        tail2_position_mean_m=mean([x['tail2_position_mean_m'] for x in completed if x.get('tail2_position_mean_m') is not None]),
        tail2_speed_mean_m_s=mean([x['tail2_speed_mean_m_s'] for x in completed if x.get('tail2_speed_mean_m_s') is not None]),
        action_clipping_fraction=mean([x['action_clipping_fraction'] for x in rs]),
        allocation_clipping_fraction=mean([x['allocation_clipping_fraction'] for x in rs]))
assert (run/'evaluation_finished.json').exists()
train=read(run/'training_result.json');plan=read(run/'experiment_plan.json');parent=Path(plan['parent'])
rows=lines(run/'metrics.jsonl');old=lines(parent/'N1024/metrics.jsonl')
initial=read(run/'evaluation/resume_start/results.json');periodic=read(run/'evaluation/periodic/results.json')
best=read(run/'evaluation/best_reload/results.json');final=read(run/'evaluation/final_reload/results.json')
n64dir=parent/'N64/evaluation/best_final_reload'
reference=[x for x in read(n64dir/'results.json') if x['checkpoint_name']=='final']
assert len(reference)==5
for x in reference:
    src=n64dir/'final'/(x['condition']+'.csv')
    with src.open() as f:tr=list(csv.DictReader(f))
    tail=tr[-200:] if x['completed_8s'] else []
    x['tail2_duration_s']=2.0 if tail else None
    x['tail2_position_mean_m']=mean([float(v['position_error_m']) for v in tail])
    x['tail2_speed_mean_m_s']=mean([math.sqrt(sum(float(v[k])**2 for k in ['vx','vy','vz'])) for v in tail])
    x['tail2_source']=str(src)
summaries=[summarize(initial,'phase6_resume_start')]+[summarize([x for x in periodic if x['iteration']==i],'phase6_periodic') for i in range(320,1025,64)]
selected={'before':summaries[0],'best':summarize(best,'phase6_best_reload'),'final':summarize(final,'phase6_final_reload'),
    'N64_reference':summarize(reference,'phase5_N64_final_saved_evaluation')}
dump_json(run/'evaluation/selected_summary.json',selected)
dump_json(run/'evaluation/N64_reference_derived.json',reference)
csvwrite(run/'evaluation/summary.csv',summaries)
csvwrite(run/'evaluation/best_final_conditions.csv',[dict(model=label,**{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in x.items()}) for label,rs in [('best',best),('final',final),('N64_reference',reference)] for x in rs])
# New eval events use only evaluations executed in this run. Float32 canonical values match event scalars.
evalrows=[]
for s in summaries+[dict(selected['best'],event_prefix='EvalReload/best'),dict(selected['final'],event_prefix='EvalReload/final')]:
    prefix=s.get('event_prefix','Eval')
    r={'global_step':int(s['transitions'])}
    for k,v in s.items():
        if isinstance(v,(int,float)) and k not in ['transitions']:
            r[prefix+'/'+k]=float(np.float32(v))
    rs=initial if s['source']=='phase6_resume_start' else [x for x in periodic if x['iteration']==s['iteration']]
    if prefix.startswith('EvalReload'):rs=best if prefix.endswith('best') else final
    for x in rs:
        for k in ['duration_s','completed_8s','position_rmse_m','max_tilt_deg','speed_rms_m_s','tail2_position_mean_m','tail2_speed_mean_m_s','action_clipping_fraction','allocation_clipping_fraction']:
            if x[k] is not None:r[prefix+'/'+x['condition']+'/'+k]=float(np.float32(x[k]))
    evalrows.append(r)
eventpath=run/'evaluation/event_scalars.json'
if eventpath.exists():assert read(eventpath)==evalrows
else:
    writer=SummaryWriter(str(run/'tensorboard'),filename_suffix='.phase6_evaluation')
    for r in evalrows:
        for tag,v in r.items():
            if tag!='global_step':writer.add_scalar(tag,v,r['global_step'])
    layout={
      'Reward':{'signed contributions':['Multiline',['RewardStep/'+t for t in ['total','xy','z','velocity','tilt','angular_velocity','yaw','termination']]]},
      'PPO':{'actual backward objective':['Multiline',['Loss/policy','Loss/value_contribution','Loss/entropy_contribution','Loss/ppo_total']],
             'probability clipping':['Multiline',['PPO/clip_fraction','PPO/approx_kl','PPO/clip_epsilon']]},
      'Flight':{'physical clipping':['Multiline',['Actuator/action_clipping_fraction','Actuator/allocation_clipping_fraction']],
                'evaluation':['Multiline',['Eval/completion_rate','Eval/mean_duration_s']]},
      'Performance':{'GPU memory scope':['Multiline',['Performance/cuda_device_used_MiB','Performance/torch_peak_allocated_MiB']]}}
    writer.add_custom_scalars(layout);writer.flush();writer.close()
    dump_json(eventpath,evalrows)
csvwrite(run/'evaluation/event_scalars.csv',evalrows)
checks={};details={}
checks['budget_exact']=len(rows)==768 and rows[0]['iteration']==257 and rows[-1]['iteration']==1024 and rows[-1]['global_step']==33554432
checks['transition_steps']=all(r['global_step']==8388608+32768*(i+1) for i,r in enumerate(rows))
checks['optimizer_hook_counters']=all(r['Performance/optimizer_updates']==5120+20*(i+1) and r['Performance/additional_optimizer_updates']==20*(i+1) for i,r in enumerate(rows))
checks['preflight_no_optimizer_updates']=read(run/'preflight.json').get('optimizer_updates',0)==0
resume=read(run/'resume_verification.json');checks['resume_verified']=resume['success']
details['resume_check_keys']=list(resume['checks'])
diff=read(run/'config_diff.json')
checks['configuration_preserved']=not diff['environment'] and not diff['optimizer'] and {x['path'] for x in diff['agent']}<= {'experiment_name','load_run','load_checkpoint','max_iterations','resume','run_name','save_interval'}
checks['reward_sample_sum']=max(r['Validation/reward_sum_max_abs_error'] for r in rows)<1e-4
checks['backward_objective_residual']=max(abs(r['Validation/ppo_loss_residual']) for r in rows)<1e-6
terms=['xy','z','velocity','tilt','angular_velocity','yaw','termination']
details['reward_aggregate_sum_max_abs_error']=max(abs(sum(r['RewardStep/'+t] for t in terms)-r['RewardStep/total']) for r in rows)
checks['reward_aggregate_sum']=details['reward_aggregate_sum_max_abs_error']<1e-4
details['loss_aggregate_sum_max_abs_error']=max(abs(r['Loss/policy']+r['Loss/value_contribution']+r['Loss/entropy_contribution']-r['Loss/ppo_total']) for r in rows)
checks['loss_aggregate_sum']=details['loss_aggregate_sum_max_abs_error']<1e-4
checks['finite_training_scalars']=all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
tags=ea.Tags()['scalars'];events={t:{e.step:e.value for e in ea.Scalars(t)} for t in tags}
missing=[];mismatches=[];compared=0
for r in rows+evalrows:
    for t,v in r.items():
        if '/' not in t:continue
        compared+=1
        if t not in events or r['global_step'] not in events[t]:missing.append((t,r['global_step']))
        elif events[t][r['global_step']]!=v:mismatches.append((t,r['global_step'],v,events[t][r['global_step']]))
checks['tensorboard_exact_json']=not missing and not mismatches
checks['tensorboard_strictly_increasing']=all(all(a.step<b.step for a,b in zip(ea.Scalars(t),ea.Scalars(t)[1:])) for t in tags)
checks['no_past_training_events']=min(events['RewardStep/total'])==8421376
details.update(scalar_tags=tags,scalar_values_compared=compared,missing=missing[:10],mismatches=mismatches[:10])
csvfile=run/'metrics.csv'
with csvfile.open() as f:cr=list(csv.DictReader(f))
checks['csv_exact_json']=len(cr)==len(rows) and all(all((v=='' and k not in r) or (k in r and float(v)==r[k]) for k,v in c.items()) for c,r in zip(cr,rows))
csvwrite(run/'plots/new_training_raw.csv',rows)
# Saved state audits on CPU; no optimizer step.
audits=[]
for cp in sorted(run.glob('checkpoint_*.pt')):
    if cp.stem.endswith('_reference'):continue
    data=torch.load(cp,map_location='cpu',weights_only=False);info=data['infos']
    steps=sorted({int(s['step']) for s in data['optimizer_state_dict']['state'].values()})
    audits.append(dict(path=str(cp),iteration=info['completed_iterations'],framework_iter=data['iter'],transitions=info['collected_transitions'],
        optimizer_updates=info['optimizer_updates'],adam_steps=steps,parameter_states=len(data['optimizer_state_dict']['state']),
        rng_keys=list(data['rng_state']),sha256=sha256(cp)))
for name in ['final','best']:
    cp=run/(name+'.pt');data=torch.load(cp,map_location='cpu',weights_only=False);info=data['infos']
    audits.append(dict(path=str(cp),iteration=info['completed_iterations'],framework_iter=data['iter'],transitions=info['collected_transitions'],
        optimizer_updates=info['optimizer_updates'],adam_steps=sorted({int(s['step']) for s in data['optimizer_state_dict']['state'].values()}),
        parameter_states=len(data['optimizer_state_dict']['state']),rng_keys=list(data['rng_state']),sha256=sha256(cp)))
checks['checkpoint_counters']=all(x['framework_iter']==x['iteration']-1 and x['transitions']==x['iteration']*32768 and x['optimizer_updates']==x['iteration']*20 and x['adam_steps']==[x['iteration']*20] and x['parameter_states']==13 for x in audits)
parent_state=torch.load(parent/'N1024/final.pt',map_location='cpu',weights_only=False)
final_state=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
moment_changes=[]
for k,a in final_state['optimizer_state_dict']['state'].items():
    b=parent_state['optimizer_state_dict']['state'][k]
    moment_changes.append(dict(parameter_id=k,exp_avg_changed=not torch.equal(a['exp_avg'],b['exp_avg']),
        exp_avg_sq_changed=not torch.equal(a['exp_avg_sq'],b['exp_avg_sq'])))
checks['adam_moments_updated']=all(x['exp_avg_changed'] and x['exp_avg_sq_changed'] for x in moment_changes)
checks['actor_updated']=any(not torch.equal(v,parent_state['actor_state_dict'][k]) for k,v in final_state['actor_state_dict'].items())
checks['critic_updated']=any(not torch.equal(v,parent_state['critic_state_dict'][k]) for k,v in final_state['critic_state_dict'].items())
details['final_distribution_std']=final_state['actor_state_dict']['distribution.log_std_param'].exp().tolist()
details['adam_moment_changes']=moment_changes
checks['periodic_iterations']=sorted({x['iteration'] for x in periodic})==list(range(320,1025,64))
checks['best_copy_exact']=read(run/'selection.json')['source_sha256']==sha256(run/'best.pt')
reloads=[]
for name in ['resume_start','periodic','best_reload','final_reload']:reloads+=read(run/('evaluation/'+name+'/reload_checks.json'))
checks['all_reload_action_exact']=all(x['max_action_error']==0 and x['bitwise_equal'] and x['optimizer_updates_in_process']==0 for x in reloads)
checks['best_final_fresh_process']=read(run/'evaluation/best_reload/reload_checks.json')[0]['pid']!=read(run/'evaluation/final_reload/reload_checks.json')[0]['pid']
trajectory_checks=[]
for folder,rs in [('resume_start',initial),('periodic',periodic),('best_reload',best),('final_reload',final)]:
    for x in rs:
        path=run/'evaluation'/folder/x['checkpoint_name']/(x['condition']+'.csv')
        with path.open() as f:tr=list(csv.DictReader(f))
        term=[v for v in tr[:-1] if v['terminated'] in ['True','1','1.0'] or v['truncated'] in ['True','1','1.0']]
        tail=tr[-200:] if x['completed_8s'] else []
        ok=len(tr)==x['steps'] and not term and abs(len(tr)*.01-x['duration_s'])<1e-9
        if tail:ok=ok and len(tr)==800 and x['tail2_duration_s']==2 and abs(mean([float(v['position_error_m']) for v in tail])-x['tail2_position_mean_m'])<1e-12
        else:ok=ok and x['tail2_duration_s'] is None
        trajectory_checks.append(dict(path=str(path),success=ok,rows=len(tr)))
checks['trajectories_first_termination_tail2']=all(x['success'] for x in trajectory_checks)
checks['best_final_trajectory_reproducible']=all(
    sha256(run/'evaluation'/folder/label/(x['condition']+'.csv'))==
    sha256(run/'evaluation/periodic'/next(y['checkpoint_name'] for y in periodic if y['iteration']==x['iteration'])/(x['condition']+'.csv'))
    for folder,label,rs in [('best_reload','best',best),('final_reload','final',final)] for x in rs)
checks['evaluation_reward_sum']=max(x['reward_sum_max_abs_error'] for x in initial+periodic+best+final)<1e-4
dump_json(run/'checkpoint_audit.json',audits);dump_json(run/'evaluation/trajectory_verification.json',trajectory_checks)
# Join existing parent data only in analysis outputs; never insert historical training events.
combined=[]
for r in old+rows:
    phase='phase5' if r['iteration']<=256 else 'phase6'
    combined.append(dict(r,source_phase=phase,cumulative_training_s=r['Performance/training_elapsed_s'] if phase=='phase5' else r['Performance/cumulative_training_elapsed_s']))
csvwrite(plot/'combined_N1024_training.csv',combined)
parent_eval=read(parent/'N1024/evaluation/periodic/results.json')
history=[summarize([x for x in parent_eval if x['iteration']==i],'phase5_saved_evaluation') for i in sorted({x['iteration'] for x in parent_eval}) if i<256]+summaries
old_initial=read(parent/'common/evaluation_initial/results.json')
history=[summarize(old_initial,'phase5_common_initial')]+history
times={r['iteration']:r['cumulative_training_s'] for r in combined};times[0]=0
for s in history:s['cumulative_training_s']=times.get(s['iteration'])
csvwrite(plot/'combined_N1024_evaluation.csv',history);dump_json(run/'evaluation/combined_history.json',history)
comparisons=[]
for n in [64,256,1024]:
    results=read(parent/f'N{n}/evaluation/periodic/results.json')
    metrics=lines(parent/f'N{n}/metrics.jsonl');mt={r['iteration']:r for r in metrics}
    for i in [256,512,768,1024]:
        source='phase5'
        if n==1024 and i>256:rs=[x for x in periodic if x['iteration']==i];source='phase6';m=rows[i-257]
        else:rs=[x for x in results if x['iteration']==i];m=mt.get(i)
        if len(rs)!=5:
            comparisons.append(dict(num_envs=n,iteration=i,status='unmeasured',source=source));continue
        s=summarize(rs,source);s.update(num_envs=n,status='measured',optimizer_updates=i*20,policy_transitions=i*n*32,
            training_s=m['Performance/cumulative_training_elapsed_s'] if source=='phase6' else m['Performance/training_elapsed_s'],
            evaluation_source=rs[0]['checkpoint'])
        comparisons.append(s)
csvwrite(plot/'same_iteration_comparison.csv',comparisons);dump_json(run/'evaluation/same_iteration_comparison.json',comparisons)
checks['same_iteration_values_measured']=len(comparisons)==12 and all(x['status']=='measured' for x in comparisons)
# Matplotlib raw curves, units explicit; no smoothing or invented evaluated points.
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.22})
def savefig(fig,name):
    fig.tight_layout(rect=(0,0,1,.965));fig.savefig(plot/(name+'.png'),dpi=150);plt.close(fig)
def panels(name,groups,title,source=combined,xkey='global_step',xlabel='Cumulative policy transitions'):
    fig,axs=plt.subplots(len(groups),1,figsize=(11,2.45*len(groups)),squeeze=False,sharex=True)
    for ax,(label,keys) in zip(axs[:,0],groups):
        for key in keys:ax.plot([r[xkey] for r in source],[r.get(key,np.nan) if r.get(key) is not None else np.nan for r in source],label=key,linewidth=.9)
        if source is combined:ax.axvline(8388608,color='black',ls='--',label='Resume')
        ax.set_ylabel(label);ax.legend(fontsize=7,ncol=2)
    axs[-1,0].set_xlabel(xlabel);fig.suptitle(title);savefig(fig,name)
    csvwrite(plot/(name+'.csv'),[{k:r.get(k) for k in [xkey,'iteration','source_phase']+list(dict.fromkeys(k for _,ks in groups for k in ks))} for r in source])
panels('reward_terms',[('Reward / policy step',['RewardStep/total'])]+[(t+' contribution',['RewardStep/'+t]) for t in terms],
    'N1024 signed reward contributions (raw iteration averages; no dt multiplier)')
panels('ppo_loss',[(x,['Loss/'+x]) for x in ['policy','value_contribution','entropy_contribution','ppo_total']],
    'Actual PPO backward objective and contributions (20 minibatch updates averaged)')
panels('ppo_probability',[('Fraction',['PPO/clip_fraction','PPO/clip_epsilon']),('KL',['PPO/approx_kl','PPO/kl']),
    ('Action std',['Policy/std_axis_'+str(i) for i in range(4)]),('Preclip gradient norm',['Gradient/actor_preclip_norm','Gradient/critic_preclip_norm'])],
    'PPO probability clipping, KL, Gaussian scale, gradients')
panels('episode_termination',[('Return',['Episode/return']),('Duration s',['Episode/length_s']),
    ('Completed count',['Episode/completed_count']),('Fraction among done',['Episode/end_'+t+'_fraction' for t in ['max_tilt','min_altitude','max_altitude','max_position_error','timeout']])],
    'Completed episodes only; active episode accumulators cross rollout boundaries')
panels('flight_actuation',[('Position RMSE m',['Flight/position_rmse_m']),('Tilt deg',['Flight/tilt_deg','Flight/tilt_deg_max']),
    ('Fraction',['Actuator/action_clipping_fraction','Actuator/allocation_clipping_fraction']),('Rotor thrust N',['Actuator/thrust_native_'+str(i)+'_N' for i in range(4)])],
    'Training population flight and physical saturation')
panels('performance',[('Transitions / s',['Performance/transitions_per_s']),('Seconds / iteration',['Performance/rollout_s','Performance/update_s']),
    ('Device-wide used MiB',['Performance/cuda_device_used_MiB']),('Torch allocator MiB',['Performance/torch_peak_allocated_MiB','Performance/torch_peak_reserved_MiB'])],
    'Throughput restarts denominator at resume; CUDA device memory includes other processes')
panels('phase6_ppo_loss_detail',[(x,['Loss/'+x]) for x in ['policy','value_contribution','entropy_contribution','ppo_total']],
    'Additional phase-6 training only (raw, no smoothing)',source=rows)
fig,axs=plt.subplots(1,3,figsize=(12,4))
precision=[selected['N64_reference'],selected['best'],selected['final']]
for ax,key,lab in zip(axs,['mean_position_rmse_m','tail2_position_mean_m','tail2_speed_mean_m_s'],['8s mean 3D RMSE (m)','Last2s mean position error (m)','Last2s mean speed (m/s)']):
    ax.bar(['N64 reference','P6 best','P6 final'],[x[key] for x in precision]);ax.set_ylabel(lab);ax.tick_params(axis='x',rotation=15)
fig.suptitle('All five conditions completed 8s; precision remains distinct from survival')
savefig(fig,'precision_reference')
csvwrite(plot/'precision_reference.csv',[dict(model=name,**x) for name,x in zip(['N64_reference','best','final'],precision)])
for xkey,xlabel,marker in [('transitions','Cumulative policy transitions',8388608),('iteration','Completed cumulative iterations',256),('cumulative_training_s','Cumulative measured training wall time (s)',read(parent/'N1024/training_result.json')['training_wall_s'])]:
    fig,axs=plt.subplots(4,1,figsize=(11,12),sharex=True)
    for ax,keys,label in zip(axs,[['completion_rate'],['mean_duration_s'],['mean_position_rmse_m','completed_mean_position_rmse_m'],['max_tilt_deg']],
          ['8 s completion fraction','Mean duration (s)','3D RMSE (m)','Maximum tilt (deg)']):
        for k in keys:ax.plot([s[xkey] for s in history],[s[k] if s[k] is not None else np.nan for s in history],'.-',label=k)
        ax.axvline(marker,color='black',ls='--',label='Resume');ax.set_ylabel(label);ax.legend(fontsize=8)
    axs[2].axhline(selected['N64_reference']['mean_position_rmse_m'],color='tab:green',ls=':',label='N64 final reference (5/5 complete)');axs[2].legend(fontsize=8)
    axs[-1].set_xlabel(xlabel);fig.suptitle('N1024 measured evaluations; early-failure RMSE has a shorter horizon')
    savefig(fig,'evaluation_vs_'+xkey)
    csvwrite(plot/('evaluation_vs_'+xkey+'.csv'),history)
fig,axs=plt.subplots(3,2,figsize=(12,11))
for ax,k,label in zip(axs.flat,['completion_rate','mean_duration_s','mean_position_rmse_m','max_tilt_deg','policy_transitions','training_s'],
        ['8 s completion fraction','Mean duration (s)','3D RMSE (m), see duration','Maximum tilt (deg)','Policy transitions','Measured training wall (s)']):
    for n in [64,256,1024]:
        ss=[x for x in comparisons if x['num_envs']==n and x['status']=='measured']
        ax.plot([x['iteration'] for x in ss],[x[k] for x in ss],'.-',label='N'+str(n))
    ax.set(xlabel='Cumulative iterations',ylabel=label);ax.legend()
fig.suptitle('Same iterations: measured saved evaluations, different data budgets');savefig(fig,'same_iteration_comparison')
for label,rs in [('best',best),('final',final)]:
    trajall=[];fig,axs=plt.subplots(5,4,figsize=(16,15))
    for j,x in enumerate(rs):
        path=run/'evaluation'/(label+'_reload')/label/(x['condition']+'.csv')
        with path.open() as f:tr=list(csv.DictReader(f))
        for v in tr:trajall.append(dict(condition=x['condition'],**v))
        for ax,keys,ylabel in zip(axs[j],[['ex','ey','ez'],['tilt_deg'],['vx','vy','vz'],['thrust_'+str(i) for i in range(4)]],
                ['Position error (m)','Tilt (deg)','World velocity (m/s)','Rotor thrust (N)']):
            for k in keys:ax.plot([float(v['time_s']) for v in tr],[float(v[k]) for v in tr],label=k,linewidth=.9)
            ax.set(xlabel='Episode time (s)',ylabel=ylabel,title=x['condition']);ax.legend(fontsize=7)
    fig.suptitle(label+' checkpoint: five deterministic fixed conditions, first episode only');savefig(fig,label+'_trajectories')
    csvwrite(plot/(label+'_trajectories.csv'),trajall)
# Equal 64-iteration windows; episode means weighted by completed count.
def window(rs):
    result={k:mean([r[k] for r in rs if k in r]) for k in ['RewardStep/total','Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total','PPO/clip_fraction','PPO/approx_kl','Policy/std_mean','Flight/position_rmse_m','Diversity/position_spread_m']}
    for k in ['Episode/return','Episode/length_s']:
        denom=sum(r['Episode/completed_count'] for r in rs if k in r)
        result[k]=sum(r[k]*r['Episode/completed_count'] for r in rs if k in r)/denom if denom else None
    for i in range(4):result['Diversity/action_axis_'+str(i)+'_std']=mean([r['Diversity/action_axis_'+str(i)+'_std'] for r in rs])
    return result
dump_json(run/'training_window_summary.json',dict(window_iterations=64,window_transitions=2097152,parent_last64=window(old[-64:]),additional_first64=window(rows[:64]),additional_last64=window(rows[-64:])))
preserve=dict(project=preserved(run/'evidence/project_before_sha256.json',ROOT),parent=preserved(run/'evidence/parent_before_sha256.json',parent))
dump_json(run/'preservation_final.json',preserve);checks['parent_preserved']=preserve['parent']['success'];checks['existing_project_preserved']=preserve['project']['success']
details['analysis_wall_s']=time.perf_counter()-START
details['training_rows']=len(rows);details['new_evaluation_episodes']=len(initial+periodic+best+final)
dump_json(run/'verification.json',dict(success=all(checks.values()),checks=checks,details=details))
print(json.dumps(dict(success=all(checks.values()),checks=checks,selected=selected,analysis_wall_s=details['analysis_wall_s']),indent=2),flush=True)
assert all(checks.values()),checks


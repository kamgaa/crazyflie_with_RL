"""Export raw plots, rank fixed evaluations, and audit TensorBoard against CSV."""
import argparse,csv,json,math,shutil,hashlib,ast
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from ppo_common import dump_json,sha256
from phase4_metrics import TERMS
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args();r=Path(args.run_dir).resolve();plots=r/'plots'
rows=[json.loads(x) for x in (r/'metrics.jsonl').read_text().splitlines()]
train=json.loads((r/'training_result.json').read_text())
evalrows=json.loads((r/'evaluation/checkpoints/results.json').read_text())
baseline=json.loads((r/'evaluation/baseline_50/results.json').read_text())

def csv_write(path,rs):
    keys=list(dict.fromkeys(k for row in rs for k in row))
    with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)

def summarize(rs):
    complete=[x for x in rs if x['completed_8s']]
    return dict(iteration=rs[0]['iteration'],transitions=rs[0]['iteration']*2048,completed_count=len(complete),
        mean_duration_s=np.mean([x['duration_s'] for x in rs]).item(),
        completed_mean_rmse_m=np.mean([x['position_rmse_m'] for x in complete]).item() if complete else None,
        all_mean_rmse_m=np.mean([x['position_rmse_m'] for x in rs]).item(),max_tilt_deg=max(x['max_tilt_deg'] for x in rs))
summary=[summarize([x for x in evalrows if x['iteration']==i]) for i in (100,200,300,400,500)]
best=max(summary,key=lambda x:(x['completed_count'],x['mean_duration_s'],-x['completed_mean_rmse_m'] if x['completed_mean_rmse_m'] is not None else 0))
best_name='final' if best['iteration']==500 else 'checkpoint_'+str(best['iteration'])
shutil.copy2(r/(best_name+'.pt'),r/'best.pt');shutil.copy2(r/(best_name+'_reference.pt'),r/'best_reference.pt')
dump_json(r/'checkpoint_selection.json',dict(best=best,source_checkpoint=str(r/(best_name+'.pt')),best_checkpoint=str(r/'best.pt'),final_checkpoint=str(r/'final.pt'),
    order=['completed_8s_count descending','five-condition mean duration descending','completed-only mean position RMSE ascending; omit if zero completed'],
    tie_break='first chronological checkpoint only if all specified criteria tie',checkpoints=summary))
# All new events use cumulative policy transitions; no legacy iteration-index logger.
if (r/'evaluation/event_scalars.json').exists():
    event_records=json.loads((r/'evaluation/event_scalars.json').read_text())
else:
    event_records=[]
    with SummaryWriter(str(r/'tensorboard'),flush_secs=10) as writer:
        writer.add_custom_scalars({'Reward':{'Signed terms':['Multiline',[f'RewardStep/{k}' for k in (*TERMS,'total')]],'Completed episode':['Multiline',['Episode/return','Episode/length_s']]}, 'Loss':{'Actual objective':['Multiline',['Loss/policy','Loss/value_contribution','Loss/entropy_contribution','Loss/ppo_total']]}, 'Policy':{'Exploration':['Multiline',['Policy/entropy','Policy/std_mean','PPO/kl','PPO/clip_fraction']]}, 'Flight':{'Errors':['Multiline',['Flight/position_rmse_m','Flight/tilt_deg']], 'Saturation':['Multiline',['Actuator/action_clipping_fraction','Actuator/allocation_clipping_fraction']]}, 'Performance':{'Throughput':['Multiline',['Performance/transitions_per_s']], 'Memory':['Multiline',['Performance/gpu_process_sampled_MiB','Performance/torch_peak_allocated_MiB']]}, 'Evaluation':{'Completion':['Multiline',['Eval/completed_count','Eval/mean_duration_s']],
            'Accuracy':['Multiline',['Eval/completed_mean_rmse_m','Eval/all_mean_rmse_m']]}})
        for rs in (baseline,*[[x for x in evalrows if x['iteration']==i] for i in (100,200,300,400,500)]):
            s=summarize(rs);step=s['transitions'];metrics={f'Eval/{k}':v for k,v in s.items() if k not in ('iteration','transitions') and v is not None}
            for e in rs:
                for k in ['duration_s','position_rmse_m','max_position_error_m','tail_position_rmse_m','tail_duration_s','max_tilt_deg','action_clipping_fraction','allocation_clipping_fraction']:
                    metrics[f'Eval/{e["condition"]}/{k}']=e[k]
                metrics[f'Eval/{e["condition"]}/completed_8s']=float(e['completed_8s'])
                for cause in ['min_altitude','max_altitude','max_tilt','max_position_error','timeout']:
                    metrics[f'Eval/{e["condition"]}/end_{cause}']=float(cause in e['reasons'])
                for term,v in e['reward_terms'].items():metrics[f'Eval/{e["condition"]}/reward_{term}']=v
            metrics={k:float(np.float32(v)) for k,v in metrics.items()}
            for k,v in metrics.items():writer.add_scalar(k,v,step)
            event_records.append(dict(global_step=step,**metrics))
        writer.flush()
    dump_json(r/'evaluation/event_scalars.json',event_records)
csv_write(r/'evaluation/event_scalars.csv',event_records)
csv_write(r/'evaluation/checkpoint_summary.csv',summary)
csv_write(r/'evaluation/conditions.csv',[{k:v for k,v in x.items() if k!='reward_terms'} for x in evalrows])

plt.rcParams.update({'font.size':10,'figure.dpi':120,'axes.grid':True,'grid.alpha':.25})
x=np.array([a['global_step'] for a in rows]);color=plt.cm.tab10.colors

def panelplot(filename,groups,title):
    fig,axes=plt.subplots(len(groups),1,figsize=(12,2.8*len(groups)),sharex=True,squeeze=False)
    keys=[]
    for ax,group in zip(axes.flat,groups):
        for key,label in group:
            ax.plot(x,[a.get(key,np.nan) for a in rows],lw=.8,label=label);keys.append(key)
        ax.legend(loc='best',fontsize=8);ax.set_ylabel('raw iteration mean')
    axes[-1,0].set_xlabel('Cumulative policy transitions (all 64 environments)');fig.suptitle(title);fig.tight_layout()
    fig.savefig(plots/(filename+'.png'));plt.close(fig)
    csv_write(plots/(filename+'.csv'),[{k:a.get(k) for k in ['global_step','iteration',*keys]} for a in rows])

panelplot('reward_total_terms', [[('RewardStep/total','total')]]+ [[('RewardStep/'+k,k)] for k in TERMS],
    'Reward A: actual signed contributions, policy-step means (no smoothing)')
panelplot('ppo_losses',[[('Loss/policy','policy')],[('Loss/value','value before coefficient'),('Loss/value_contribution','value contribution')],
    [('Loss/entropy_contribution','entropy contribution')],[('Loss/ppo_total','actual tensor passed to backward')]],'PPO losses: mean over 20 minibatch updates')
panelplot('episode_endings',[[('Episode/return','completed episode return')],[('Episode/length_s','completed episode length [s]')],
    [(f'Episode/end_{k}_count',k) for k in ['min_altitude','max_altitude','max_tilt','max_position_error','timeout']],
    [('Episode/completed_count','completed episodes')]],'Completed episodes per iteration; unfinished episodes carried forward')
panelplot('policy_and_flight',[[('Policy/entropy','differential entropy'),('Policy/std_mean','mean std')],
    [('PPO/kl','KL'),('PPO/clip_fraction','PPO clip fraction')],
    [('Flight/position_rmse_m','position RMSE [m]'),('Flight/tilt_deg','tilt mean [deg]')],
    [('Actuator/action_clipping_fraction','action clipping'),('Actuator/allocation_clipping_fraction','allocation clipping')]],'Policy, flight and distinct saturation metrics')
all_summary=[summarize(baseline),*summary];all_eval=baseline+evalrows
fig,axs=plt.subplots(4,1,figsize=(12,12),sharex=True)
ex=[s['transitions'] for s in all_summary]
axs[0].plot(ex,[s['completed_count'] for s in all_summary],'o-');axs[0].set_ylabel('8 s completions / 5');axs[0].set_ylim(-.2,5.2)
for label in ['nominal','x_plus','x_minus','y_plus','y_minus']:
    ee=[v for v in all_eval if v['condition']==label]
    axs[1].plot([v['iteration']*2048 for v in ee],[v['duration_s'] for v in ee],'o-',label=label)
    axs[3].plot([v['iteration']*2048 for v in ee],[v['max_tilt_deg'] for v in ee],'o-',label=label)
axs[1].set_ylabel('Duration [s]');axs[1].legend(ncol=5)
axs[2].plot(ex,[s['completed_mean_rmse_m'] if s['completed_mean_rmse_m'] is not None else np.nan for s in all_summary],'o-',label='completed-only RMSE')
axs[2].plot(ex,[s['all_mean_rmse_m'] for s in all_summary],'o--',label='all episodes: duration-confounded, not ranking criterion')
axs[2].set_ylabel('Position RMSE [m]');axs[2].legend()
axs[3].set_ylabel('Max tilt [deg]');axs[3].set_xlabel('Cumulative policy transitions')
fig.suptitle('Fixed five-condition evaluation; first termination ends the episode');fig.tight_layout();fig.savefig(plots/'checkpoint_evaluation.png');plt.close(fig)
csv_write(plots/'checkpoint_evaluation.csv',all_summary)
for label in ['nominal','x_plus','x_minus','y_plus','y_minus']:
    source=r/'evaluation/checkpoints'/best_name/(label+'.csv')
    rs=list(csv.DictReader(source.open()));t=[float(v['time_s']) for v in rs]
    fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for k in ['ex','ey','ez']:axs[0].plot(t,[float(v[k]) for v in rs],label=k)
    axs[0].plot(t,[float(v['position_error_m']) for v in rs],label='3D error',color='black',ls='--');axs[0].set_ylabel('Position error [m]');axs[0].legend()
    axs[1].plot(t,[float(v['tilt_deg']) for v in rs]);axs[1].axhline(60,color='red',ls='--',label='failure limit');axs[1].set_ylabel('Tilt [deg]');axs[1].legend()
    for user,native in enumerate([3,2,1,0],1):axs[2].plot(t,[float(v[f'thrust_{native}']) for v in rs],label=f'user_frd motor {user}')
    axs[2].axhline(.289,color='red',ls='--');axs[2].set_ylabel('Actual thrust [N]');axs[2].set_xlabel('Episode time [s]');axs[2].legend(ncol=4)
    fig.suptitle(f'Best checkpoint {best["iteration"]}: {label}, original first-termination boundary');fig.tight_layout();fig.savefig(plots/('best_'+label+'.png'));plt.close(fig)
    shutil.copy2(source,plots/('best_'+label+'.csv'))

# Exact event / CSV audit, required keys, monotonic transitions, objective trace and budgets.
acc=EventAccumulator(str(r/'tensorboard'),size_guidance={'scalars':0,'tensors':0}).Reload()
scalars={k:acc.Scalars(k) for k in acc.Tags()['scalars']}
missing=[];max_error=0.0;ncompare=0
for row in rows+event_records:
    step=row['global_step']
    for k,v in row.items():
        if '/' not in k:continue
        matches=[e for e in scalars.get(k,[]) if e.step==step]
        if not matches:missing.append((k,step));continue
        max_error=max(max_error,abs(matches[-1].value-v));ncompare+=1
with (r/'metrics.csv').open() as f:csvrows=list(csv.DictReader(f))
csv_error=max(abs(float(b[k])-v) for a,b in zip(rows,csvrows) for k,v in a.items() if k in b and b[k]!='')
reward_resid=max(v['Validation/reward_sum_max_abs_error'] for v in rows)
mean_sum_error=max(abs(sum(v['RewardStep/'+k] for k in TERMS)-v['RewardStep/total']) for v in rows)
loss_sum_error=max(abs(v['Loss/policy']+v['Loss/value_contribution']+v['Loss/entropy_contribution']-v['Loss/ppo_total']) for v in rows)
checks=dict(iterations_450=len(rows)==450,counter_sequence=[a['iteration'] for a in rows]==list(range(51,501)),
    transition_sequence=[a['global_step'] for a in rows]==list(range(104448,1024001,2048)),
    event_tags_present=not missing,event_values_equal=max_error==0,csv_values_equal=csv_error==0,
    all_scalar_steps_monotonic=all(all(x.step<y.step for x,y in zip(v,v[1:])) for v in scalars.values()),
    reward_sample_sum=reward_resid<1e-4,reward_aggregate_sum=mean_sum_error<1e-4,
    actual_backward_loss_trace=max(v['Validation/ppo_loss_residual'] for v in rows)==0,
    optimizer_budget=train['additional_optimizer_updates']==9000 and train['total_optimizer_updates']==10000,
    final_new_process=all(v['pid']!=train['pid'] for v in json.loads((r/'evaluation/checkpoints/reload_checks.json').read_text())),
    protected_physics=train['protected_files_unchanged'])
audit=dict(success=all(checks.values()),checks=checks,scalar_tag_count=len(scalars),comparison_count=ncompare,missing=missing,
    event_csv_max_error=max_error,csv_json_max_error=csv_error,reward_per_sample_max_residual=reward_resid,
    reward_aggregate_max_residual=mean_sum_error,loss_aggregate_rounding_max_residual=loss_sum_error,
    scalar_tags=list(scalars),first_new_reward_step=104448,last_step=1024000,baseline_eval_step=102400)
dump_json(r/'verification.json',audit)
assert all(checks.values()),checks
print('BEST',json.dumps(best));print('AUDIT',json.dumps({k:v for k,v in audit.items() if k!='scalar_tags'}))

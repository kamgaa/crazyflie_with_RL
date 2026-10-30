"""Cross-configuration raw plots and event/config/budget audit; no optimizer calls."""
import argparse,csv,json,math,shutil,hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from ppo_common import dump_json,sha256
from phase5_common import TERMS,COUNTS,BUDGET,INTERVAL
p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args();root=Path(args.experiment).resolve();plots=root/'plots'
assert (root/'evaluation_finished.json').exists()

def read(p):return json.loads(Path(p).read_text())
def csv_write(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [item for key in sorted(a.keys()|b.keys()) for item in diff(a.get(key),b.get(key),path+'.'+key if path else key)]
    return [] if a==b else [dict(path=path,a=a,b=b)]
def summary(rs):
    complete=[x for x in rs if x['completed_8s']]
    return dict(transitions=rs[0]['transitions'],completed_count=len(complete),completion_rate=len(complete)/5,
        mean_duration_s=float(np.mean([x['duration_s'] for x in rs])),
        mean_position_rmse_m=float(np.mean([x['position_rmse_m'] for x in rs])),
        completed_position_rmse_m=float(np.mean([x['position_rmse_m'] for x in complete])) if complete else None,
        mean_max_tilt_deg=float(np.mean([x['max_tilt_deg'] for x in rs])),max_tilt_deg=max(x['max_tilt_deg'] for x in rs),
        mean_xyz_rmse_m=np.array([x['xyz_rmse_m'] for x in rs]).mean(0).tolist(),
        mean_speed_rms_m_s=float(np.mean([x['speed_rms_m_s'] for x in rs])),
        mean_omega_rms_rad_s=float(np.mean([x['body_omega_rms_rad_s'] for x in rs])))
base=read(root/'common/evaluation_initial/results.json');all_eval=list(base);data={};comparison=[];audits={};diffs={}
for n in COUNTS:
    r=root/f'N{n}';rows=[json.loads(x) for x in (r/'metrics.jsonl').read_text().splitlines()] if (r/'metrics.jsonl').exists() else []
    tr=read(r/'training_result.json') if (r/'training_result.json').exists() else read(r/'failure.json') if (r/'failure.json').exists() else read(r/'preflight_failure.json')
    periodic=read(r/'evaluation/periodic/results.json') if (r/'evaluation/periodic/results.json').exists() else []
    all_eval+=periodic;data[n]=dict(rows=rows,train=tr,periodic=periodic)
# Equal-time diagnostic uses the shortest actual recorded duration across all policies/conditions.
prefix_steps=min(x['steps'] for x in all_eval);prefix_s=prefix_steps*.01
for n,d in data.items():
    r=root/f'N{n}';rows=d['rows'];tr=d['train'];periodic=d['periodic']
    checkpoint_groups=[base]+[[x for x in periodic if x['transitions']==t] for t in sorted({x['transitions'] for x in periodic})]
    reloads=read(r/'evaluation/periodic/reload_checks.json') if periodic else []
    elapsed_by_step={v['loaded_infos']['collected_transitions']:v['loaded_infos']['training_elapsed_s'] for v in reloads}
    evalsumm=[];evrecords=[]
    for group in checkpoint_groups:
        s=summary(group);s['training_elapsed_s']=elapsed_by_step.get(s['transitions'],0.)
        errors=[]
        for result in group:
            folder=root/'common/evaluation_initial/initial' if result['transitions']==0 else r/'evaluation/periodic'/result['checkpoint_name']
            traj=list(csv.DictReader((folder/(result['condition']+'.csv')).open()))[:prefix_steps]
            errors.append(math.sqrt(sum(float(x['position_error_m'])**2 for x in traj)/prefix_steps))
        s['common_prefix_s']=prefix_s;s['common_prefix_rmse_m']=float(np.mean(errors));evalsumm.append(s)
        values={f'Eval/{k}':v for k,v in s.items() if isinstance(v,(int,float)) and k!='transitions'}
        for result in group:
            label=result['condition']
            for k in ['duration_s','position_rmse_m','max_position_error_m','max_tilt_deg','mean_tilt_deg','speed_rms_m_s','max_speed_m_s','body_omega_rms_rad_s','max_body_omega_rad_s','action_clipping_fraction','allocation_clipping_fraction','evaluation_wall_s']:
                values[f'Eval/{label}/{k}']=result[k]
            for j,axis in enumerate('xyz'):values[f'Eval/{label}/rmse_{axis}_m']=result['xyz_rmse_m'][j]
            values[f'Eval/{label}/completed_8s']=int(result['completed_8s'])
            for cause in ['min_altitude','max_altitude','max_tilt','max_position_error','timeout']:values[f'Eval/{label}/end_{cause}']=int(cause in result['reasons'])
        evrecords.append(dict(global_step=s['transitions'],**{k:float(np.float32(v)) for k,v in values.items()}))
    if not (r/'evaluation/event_scalars.json').exists():
        with SummaryWriter(str(r/'tensorboard'),flush_secs=10) as writer:
            writer.add_custom_scalars({'Reward':{'terms':['Multiline',[f'RewardStep/{k}' for k in (*TERMS,'total')]]},
                'PPO':{'objective':['Multiline',['Loss/policy','Loss/value_contribution','Loss/entropy_contribution','Loss/ppo_total']],
                    'probability ratio':['Multiline',['PPO/clip_fraction','PPO/approx_kl','PPO/clip_epsilon']]},
                'Evaluation':{'success and duration':['Multiline',['Eval/completion_rate','Eval/mean_duration_s']],
                    'precision':['Multiline',['Eval/mean_position_rmse_m','Eval/common_prefix_rmse_m']]},
                'Performance':{'throughput':['Multiline',['Performance/transitions_per_s']],
                    'memory':['Multiline',['Performance/cuda_device_used_MiB','Performance/torch_peak_allocated_MiB']]}})
            for row in evrecords:
                for k,v in row.items():
                    if k!='global_step':writer.add_scalar(k,v,row['global_step'])
            writer.flush()
        dump_json(r/'evaluation/event_scalars.json',evrecords)
    else:assert read(r/'evaluation/event_scalars.json')==evrecords
    csv_write(r/'evaluation/event_scalars.csv',evrecords);csv_write(r/'evaluation/checkpoint_summary.csv',evalsumm)
    dump_json(r/'evaluation/checkpoint_summary.json',evalsumm);d['evalsumm']=evalsumm
    csv_write(r/'metrics.csv',rows)
    # O(number of scalars), not O(number of iterations squared).
    acc=EventAccumulator(str(r/'tensorboard'),size_guidance={'scalars':0,'tensors':0}).Reload()
    scalar={tag:acc.Scalars(tag) for tag in acc.Tags()['scalars']};lookup={k:{v.step:v.value for v in vs} for k,vs in scalar.items()}
    missing=[];maxerr=0.;count=0
    for row in rows+evrecords:
        for k,v in row.items():
            if '/' not in k:continue
            if row['global_step'] not in lookup.get(k,{}):missing.append((k,row['global_step']));continue
            maxerr=max(maxerr,abs(v-lookup[k][row['global_step']]));count+=1
    cr=list(csv.DictReader((r/'metrics.csv').open())) if rows else []
    csv_error=max((abs(float(b[k])-v) for a,b in zip(rows,cr) for k,v in a.items() if b[k]!=''),default=0.)
    init=read(r/'initialization.json') if (r/'initialization.json').exists() else {}
    common=read(root/'common/initialization.json')
    checks=dict(event_values_identical=not missing and maxerr==0,csv_json_identical=csv_error==0,
        monotonic_steps=all(all(a.step<b.step for a,b in zip(v,v[1:])) for v in scalar.values()),
        actor_initial_identical=init.get('actor_sha256')==common['actor_sha256'],critic_initial_identical=init.get('critic_sha256')==common['critic_sha256'],
        new_optimizer=init.get('optimizer_state_count')==0,
        contiguous_iterations=[v['iteration'] for v in rows]==list(range(1,len(rows)+1)),
        transitions_consistent=all(v['global_step']==v['iteration']*n*32 for v in rows),
        measured_updates_consistent=all(v['Performance/optimizer_updates']==v['iteration']*20 for v in rows),
        reward_residual_ok=all(v['Validation/reward_sum_max_abs_error']<1e-4 for v in rows),
        actual_backward_loss_unchanged=all(v['Validation/ppo_loss_residual']==0 for v in rows),
        ppo_epsilon_unchanged=all(abs(v['PPO/clip_epsilon']-.2)<1e-7 for v in rows))
    if tr.get('success'):
        checks.update(full_budget=tr['transitions']==BUDGET and len(rows)==COUNTS[n],
            checkpoint_schedule=[v['transitions'] for v in evalsumm]==list(range(0,BUDGET+1,INTERVAL)))
        verify=read(r/'evaluation/best_final_reload/reload_checks.json')
        checks['best_final_new_process_reload']=all(x['pid']!=tr['pid'] and x['max_action_error']<=1e-6 and x['optimizer_updates_in_process']==0 for x in verify)
        checks['best_final_both_evaluated']={Path(v['checkpoint']).stem for v in verify}=={'best','final'}
        checks['loaded_actor_hashes_match']=all(v['actor_sha256']==v['loaded_infos'].get('actor_sha256',v['actor_sha256']) for v in reloads+verify)
        checkpoint_audit=[]
        for cp in sorted(r.glob('*.pt')):
            if cp.stem.endswith('_reference'):continue
            saved=torch.load(cp,map_location='cpu',weights_only=False)
            saved_steps=sorted({int(v['step']) for v in saved['optimizer_state_dict']['state'].values()})
            expected=saved['infos']['optimizer_updates']
            valid=(saved_steps==[expected]) if expected else not saved_steps
            assert valid,(cp,saved_steps,expected)
            checkpoint_audit.append(dict(checkpoint=cp.name,completed_iterations=saved['infos']['completed_iterations'],transitions=saved['infos']['collected_transitions'],adam_steps=saved_steps,optimizer_parameter_states=len(saved['optimizer_state_dict']['state']),success=valid))
        dump_json(r/'checkpoint_verification.json',checkpoint_audit)
        checks['saved_optimizer_steps_match']=all(v['success'] for v in checkpoint_audit)

    audits[f'N{n}']=dict(success=all(checks.values()),checks=checks,scalar_tags=list(scalar),comparison_count=count,event_max_error=maxerr,csv_max_error=csv_error,
        missing=missing,reward_max_residual=max((v['Validation/reward_sum_max_abs_error'] for v in rows),default=None))
    dump_json(r/'verification.json',audits[f'N{n}'])
    if (r/'env_config.json').exists():
        resolved=dict(agent=read(r/'agent_config.json'),env=read(r/'env_config.json'),optimizer=read(r/'optimizer_config.json'),
            budget=dict(num_envs=n,rollout=32,iterations=COUNTS[n],transitions=BUDGET,minibatch_size=n*8,optimizer_updates=COUNTS[n]*20),seed=42)
        dump_json(r/'resolved_config.json',resolved)
        diffs[f'N{n}_vs_phase4_agent']=diff(read(root/'evidence/phase4_agent_config.json'),resolved['agent'])
        diffs[f'N{n}_vs_phase4_env']=diff(read(root/'evidence/phase4_env_config.json'),resolved['env'])
    entry=dict(config=f'N{n}',success=tr.get('success',False),iterations=tr.get('iterations',tr.get('completed_iterations',0)),
        transitions=tr.get('transitions',tr.get('collected_transitions',0)),optimizer_updates=tr.get('optimizer_updates',tr.get('actual_optimizer_updates',0)),
        training_wall_s=tr.get('training_wall_s'),transitions_per_s=tr.get('transitions_per_s'),
        cuda_device_used_peak_MiB=tr.get('cuda_device_used_peak_sampled_MiB'),torch_peak_allocated_MiB=tr.get('memory_final',{}).get('torch_peak_allocated_MiB'))
    if (r/'selection.json').exists():
        selected=read(r/'selection.json');entry.update(best_transitions=selected['best']['transitions'],best_completed_count=selected['best']['completed_count'],best_mean_duration_s=selected['best']['mean_duration_s'])
        last=evalsumm[-1];entry.update(final_completed_count=last['completed_count'],final_mean_duration_s=last['mean_duration_s'],final_mean_position_rmse_m=last['mean_position_rmse_m'])
        entry['best_mean_position_rmse_m']=next(x['mean_position_rmse_m'] for x in evalsumm if x['transitions']==entry['best_transitions'])
    fresh_path=r/'evaluation/best_final_reload/results.json'
    if fresh_path.exists():
        fresh=read(fresh_path);fresh_summaries=[]
        entry['selection_completed_count']=entry['best_completed_count']
        entry['selection_mean_duration_s']=entry['best_mean_duration_s']
        for label in ['best','final']:
            episodes=[v for v in fresh if v['checkpoint_name']==label]
            assert len(episodes)==5,(n,label,len(episodes))
            fs=summary(episodes);fresh_summaries.append(dict(checkpoint=label,**fs))
            for key in ['completed_count','mean_duration_s','mean_position_rmse_m','max_tilt_deg','mean_max_tilt_deg','mean_speed_rms_m_s','mean_omega_rms_rad_s']:
                entry[f'{label}_{key}']=fs[key]
        dump_json(r/'evaluation/best_final_reload/summary.json',fresh_summaries)
        csv_write(r/'evaluation/best_final_reload/summary.csv',fresh_summaries)
    comparison.append(entry)
for n in [256,1024]:
    if (root/f'N{n}/resolved_config.json').exists() and (root/'N64/resolved_config.json').exists():
        a=read(root/'N64/resolved_config.json');b=read(root/f'N{n}/resolved_config.json');diffs[f'N64_vs_N{n}']=diff(a,b)
        assert a['agent']['actor']==b['agent']['actor'] and a['agent']['critic']==b['agent']['critic'] and a['agent']['algorithm']==b['agent']['algorithm']
        assert [x['path'] for x in diff(a['env'],b['env'])]==['scene.num_envs']
dump_json(root/'config_diff.json',diffs);dump_json(root/'comparison.json',comparison);csv_write(root/'comparison.csv',comparison)
dump_json(root/'verification.json',dict(success=all(x['success'] for x in audits.values()),runs=audits,common_prefix_s=prefix_s,comparison_count=sum(x['comparison_count'] for x in audits.values())))

plt.rcParams.update({'font.size':10,'figure.dpi':130,'axes.grid':True,'grid.alpha':.25})
colors={64:'tab:blue',256:'tab:orange',1024:'tab:green'}
def metric_plot(name,groups,title,xkey='global_step',xlabel='Cumulative policy transitions'):
    fig,axs=plt.subplots(len(groups),1,figsize=(12,2.8*len(groups)),sharex=True,squeeze=False);export=[]
    for ax,keys in zip(axs.flat,groups):
        for n,d in data.items():
            rs=d['rows'];xs=[v[xkey] for v in rs]
            for j,key in enumerate(keys):ax.plot(xs,[v.get(key,np.nan) for v in rs],color=colors[n],lw=.65,ls=['-','--',':','-.'][j%4],label=f'N{n}: {key.split("/")[-1]}')
        ax.legend(fontsize=8,ncol=3)
        label='raw iteration mean'
        if keys[0].startswith('Loss/'):label='mean minibatch loss'
        elif 'transitions_per_s' in keys[0]:label='policy transitions / s'
        elif 'elapsed_s' in keys[0]:label='training wall time [s]'
        elif 'MiB' in keys[0]:label='GPU memory [MiB]'
        elif 'length_s' in keys[0]:label='completed episode [s]'
        elif 'rmse_m' in keys[0]:label='position RMSE [m]'
        ax.set_ylabel(label)
    axs[-1,0].set_xlabel(xlabel);fig.suptitle(title);fig.tight_layout(rect=(0,0,1,.97));fig.savefig(plots/(name+'.png'));plt.close(fig)
    keys=list(dict.fromkeys(k for group in groups for k in group))
    for n,d in data.items():export.extend(dict(config=f'N{n}',global_step=v['global_step'],elapsed_s=v['Performance/training_elapsed_s'],**{k:v.get(k) for k in keys}) for v in d['rows'])
    csv_write(plots/(name+'.csv'),export)
metric_plot('reward_terms_total',[[f'RewardStep/{k}'] for k in ('total',*TERMS)],'Reward A: signed contributions, raw means; identical coefficients')
metric_plot('ppo_losses',[['Loss/policy'],['Loss/value'],['Loss/entropy_contribution'],['Loss/ppo_total']],'Actual PPO optimization losses; equal transitions, unequal update counts')
metric_plot('ppo_clipping_kl',[['PPO/clip_fraction'],['PPO/approx_kl','PPO/kl'],['PPO/ratio_mean'],['PPO/clip_epsilon']],'Probability ratio, out-of-range fraction and fixed epsilon are distinct')
metric_plot('action_std',[[f'Policy/std_axis_{i}'] for i in range(4)],'Gaussian action standard deviation, one panel per action dimension')
metric_plot('throughput_time_memory',[['Performance/transitions_per_s'],['Performance/training_elapsed_s'],['Performance/cuda_device_used_MiB'],['Performance/torch_peak_allocated_MiB']],'Measured throughput and memory; CUDA device usage includes other processes')
metric_plot('reward_vs_elapsed',[['RewardStep/total'],['Episode/length_s'],['Flight/position_rmse_m']],'Training metrics versus measured elapsed training time',xkey='Performance/training_elapsed_s',xlabel='Elapsed training wall time [s], initialization/evaluation excluded')
metric_plot('saturation_gradients',[['Actuator/action_clipping_fraction','Actuator/allocation_clipping_fraction'],['Gradient/actor_preclip_norm','Gradient/critic_preclip_norm']],'Physical saturation and original PPO gradient clipping diagnostics')
for use_time in [False,True]:
    fig,axs=plt.subplots(4,1,figsize=(12,12),sharex=True);export=[]
    for n,d in data.items():
        es=d['evalsumm'];xs=[x['training_elapsed_s'] if use_time else x['transitions'] for x in es]
        for ax,key in zip(axs,['completion_rate','mean_duration_s','mean_position_rmse_m','common_prefix_rmse_m']):
            ax.plot(xs,[v[key] for v in es],'o-',color=colors[n],label=f'N{n}',ms=3)
        export.extend(dict(config=f'N{n}',**e) for e in es)
    axs[0].set_ylabel('8 s completion rate');axs[0].set_ylim(-.05,1.05);axs[0].legend()
    axs[1].set_ylabel('Mean duration [s]');axs[2].set_ylabel('Full observed RMSE [m]\n(duration-confounded)');axs[3].set_ylabel(f'First {prefix_s:.2f} s RMSE [m]')
    axs[3].set_xlabel('Elapsed training wall time [s]' if use_time else 'Cumulative policy transitions')
    fig.suptitle('Fixed five-condition evaluation: precision and survival are separate');fig.tight_layout(rect=(0,0,1,.97))
    name='evaluation_vs_elapsed' if use_time else 'evaluation_vs_transitions';fig.savefig(plots/(name+'.png'));plt.close(fig);csv_write(plots/(name+'.csv'),export)
fig,axs=plt.subplots(4,1,figsize=(12,12),sharex=True);ex=[]
for n,d in data.items():
    es=d['evalsumm'];x=[v['transitions'] for v in es]
    for j,axis in enumerate('xyz'):axs[j].plot(x,[v['mean_xyz_rmse_m'][j] for v in es],'o-',label=f'N{n}',color=colors[n],ms=3);axs[j].set_ylabel(axis.upper()+' RMSE [m]')
    axs[3].plot(x,[v['mean_max_tilt_deg'] for v in es],'o-',label=f'N{n}',color=colors[n],ms=3)
    ex.extend(dict(config=f'N{n}',**v) for v in es)
axs[0].legend();axs[3].set_ylabel('Mean max tilt [deg]');axs[3].set_xlabel('Cumulative policy transitions');fig.suptitle('Errors over actual episode durations; short failure is not accuracy');fig.tight_layout(rect=(0,0,1,.97));fig.savefig(plots/'xyz_tilt.png');plt.close(fig);csv_write(plots/'xyz_tilt.csv',ex)
# Selected-policy trajectory plots use the separate explicit reload evaluation.
for n in COUNTS:
    folder=root/f'N{n}/evaluation/best_final_reload'
    if not (folder/'results.json').exists():continue
    for label in ['best','final']:
        fig,axs=plt.subplots(3,5,figsize=(19,10),sharex='col');export=[]
        for col,condition in enumerate(['nominal','x_plus','x_minus','y_plus','y_minus']):
            file=folder/label/(condition+'.csv')
            if not file.exists():
                available=[x['condition'] for x in read(folder/'results.json') if x['checkpoint_name']==label]
                condition=available[col];file=folder/label/(condition+'.csv')
            tr=list(csv.DictReader(file.open()));keys=list(tr[0]);times=[float(v['time_s']) for v in tr]
            for axis in ['x','y','z']:
                axs[0,col].plot(times,[float(v[axis])-(1 if axis=='z' else 0) for v in tr],label=axis,lw=.8)
            tilt_key='tilt_deg'
            axs[1,col].plot(times,[float(v[tilt_key]) for v in tr],lw=.8)
            motor_keys=[k for k in keys if k.startswith('thrust_')]
            for key in motor_keys:axs[2,col].plot(times,[float(v[key]) for v in tr],label=key,lw=.8)
            axs[0,col].set_title(condition);axs[2,col].set_xlabel('Episode time [s]')
            export.extend(dict(condition=condition,**v) for v in tr)
        axs[0,0].set_ylabel('Position error [m]');axs[1,0].set_ylabel('Tilt [deg]');axs[2,0].set_ylabel('Rotor thrust [N]')
        axs[0,0].legend();axs[2,0].legend(fontsize=7)
        fig.suptitle(f'N{n} {label}: five fixed conditions, first termination only');fig.tight_layout(rect=(0,0,1,.97))
        fig.savefig(plots/f'N{n}_{label}_trajectories.png');plt.close(fig);csv_write(plots/f'N{n}_{label}_trajectories.csv',export)

print('COMPARISON',json.dumps(comparison));print('VERIFIED',all(x['success'] for x in audits.values()),'PREFIX',prefix_s)

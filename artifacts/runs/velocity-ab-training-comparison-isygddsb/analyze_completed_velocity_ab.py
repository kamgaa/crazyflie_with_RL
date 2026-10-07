import argparse,csv,json,collections
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--comparison-dir',type=Path,required=True)
args=parser.parse_args();root=args.comparison_dir
completion=json.loads((root/'completion.json').read_text())
report={};curves={}
for label,run_info in completion['training'].items():
 run=Path(run_info['run_dir'])
 with next((run/'metrics').glob('*training-episodes*.csv')).open() as f:rows=list(csv.DictReader(f))
 ended=[r for r in rows if r['episode_ended']=='True']
 duration=np.array([float(r['duration_sec']) for r in ended])
 reasons=dict(collections.Counter(r['end_reason'] for r in ended))
 windows=[]
 total=run_info['actual_timesteps']
 for lower in range(0,total,100000):
  upper=min(total,lower+100000)
  selected=[r for r in ended if lower<int(r['end_timestep'])<=upper]
  d=np.array([float(r['duration_sec']) for r in selected])
  physical=sum(r['physical_termination']=='True' for r in selected)
  horizon_only=sum(r['time_limit_reached']=='True' and r['physical_termination']=='False' for r in selected)
  windows.append({'step_end':upper,'completed_episode_count':len(selected),'mean_duration_sec':float(d.mean()) if len(d) else None,'median_duration_sec':float(np.median(d)) if len(d) else None,'physical_end_fraction':physical/len(d) if len(d) else None,'time_limit_only_fraction':horizon_only/len(d) if len(d) else None})
 components={key:sum(float(r[key]) for r in rows) for key in rows[0] if key.startswith('reward_sum_')}
 evals=[json.loads(p.read_text()) for p in sorted((run/'metrics').glob('*evaluation-step*.json'))]
 report[label]={'completed_training_episodes':len(ended),'unfinished_records':len(rows)-len(ended),'recorded_training_steps':sum(int(r['length_steps']) for r in rows),'episode_duration_sec':{'mean':float(duration.mean()),'median':float(np.median(duration)),'p95':float(np.percentile(duration,95)),'max':float(duration.max())},'end_reasons':reasons,'reward_component_sums_within_run_only':components,'windows_by_episode_end_timestep':windows,'periodic_evaluations':evals}
 curves[label]=(windows,evals)
(root/'training_episode_summary.json').write_text(json.dumps(report,indent=2)+'\n')
with (root/'training_episode_windows.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['model',*next(iter(curves.values()))[0][0]])
 writer.writeheader()
 for label,(windows,_) in curves.items():
  for row in windows:writer.writerow({'model':label,**row})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(4,1,figsize=(11,12),sharex=True)
for label,(windows,evals) in curves.items():
 x=[r['step_end'] for r in windows]
 axes[0].plot(x,[r['mean_duration_sec'] for r in windows],marker='.',label=label.upper())
 axes[1].plot(x,[r['time_limit_only_fraction'] for r in windows],marker='.',label=label.upper())
 axes[2].plot([r['timestep'] for r in evals],[r['policy_mean_episode_length']/100 for r in evals],label=label.upper())
 axes[3].plot([r['timestep'] for r in evals],[r['policy_disqualifications'] for r in evals],label=label.upper())
for ax,title in zip(axes,['Episode duration [s]','Time-limit-only fraction','Eval duration [s]','Tail-tilt disqualified / 30']):
 ax.set_ylabel(title);ax.grid(alpha=.3);ax.legend()
axes[0].axhline(8,color='gray',ls='--');axes[2].axhline(8,color='gray',ls='--')
axes[-1].set_xlabel('Training timestep (episode windows assigned by end timestep)')
fig.suptitle('A/B training history — time-limit fraction is not policy success rate', fontsize=12);fig.tight_layout(rect=(0,0,1,.97));fig.savefig(root/'training_episode_history.png',dpi=150);plt.close(fig)
print(json.dumps({label:{k:value[k] for k in ['completed_training_episodes','unfinished_records','recorded_training_steps','episode_duration_sec','end_reasons']} for label,value in report.items()},indent=2))

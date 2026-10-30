"""Read-only early-response zooms for completed/failed circle trials."""
import argparse,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir);start=time.perf_counter()
colors={'PID':'#1f77b4','Q_base':'#e17c05','E_position':'#279452'}
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25})
for T,end in [(10,3),(5,2)]:
 fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
 spec=[('position_error_m','Time-aligned position error [m]'),('tilt_deg','Tilt [deg]'),('vy','World Y velocity [m/s]'),('ez','Z error [m]'),('action_clipping_fraction','Action clipping fraction'),('allocation_clipping_fraction','Allocation clipping fraction')]
 for name,col in colors.items():
  data=json.loads((r/'cases'/f'{name}_T{T}'/'trajectory.json').read_text())['trajectory'];data=[x for x in data if x['time_s']<=end]
  for ax,(key,label) in zip(axes.flat,spec):
   ax.plot([x['time_s'] for x in data],[x[key] for x in data],color=col,label=name)
   ax.set_ylabel(label);ax.set_xlim(0,end)
 tt=np.linspace(0,end,301);axes[1,0].plot(tt,(2*np.pi/T)*np.cos(2*np.pi*tt/T),'k--',label='reference vy (metric only)')
 axes[0,1].axhline(60,ls='--',color='gray',label='failure tilt');axes[0,0].legend();axes[1,0].legend(fontsize=8)
 for ax in axes[-1]:ax.set_xlabel('Simulation time [s]')
 fig.suptitle(f'Initial response, T={T}s, zoom 0-{end}s; curves stop at first failure')
 fig.tight_layout();fig.savefig(r/'plots'/f'initial_response_T{T}.png',dpi=170);plt.close(fig)
(r/'diagnostic_plot_metadata.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-start,windows_s={'T10':[0,3],'T5':[0,2]},source_csv='plots/trajectories_T10.csv and trajectories_T5.csv'),indent=2)+'\n')
print('CIRCLE_DIAGNOSTIC_PLOTS_DONE')

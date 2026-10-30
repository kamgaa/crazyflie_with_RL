"""Animate recorded trajectories only; no simulator or optimizer calls."""
import argparse,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib import font_manager
from ppo_common import dump_json
import csv
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir);start=time.perf_counter()
base=Path(json.loads((r/'experiment_plan.json').read_text())['configuration_source']);selected=json.loads((r/'best_selection.json').read_text())['selected']['additional_iteration']
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
records=[]
for T in [10,5]:
 data={}
 for name,path in [('기존 .005 best448',base/'circle/cases'/f'phase13sq_best_T{T}/trajectory.csv'),(f'3배 .015 best{selected}',r/'circle/cases'/f'phase14z_best_T{T}/trajectory.csv')]:
  with path.open() as f:rows=list(csv.DictReader(f))
  data[name]={k:np.array([float(z[k]) for z in rows]) for k in ['time_s','x','y','ez','target_x','target_y']}
 n=max(len(v['time_s']) for v in data.values());frames=list(range(0,n,10))
 if frames[-1]!=n-1:frames.append(n-1)
 fig,(xy,zax)=plt.subplots(1,2,figsize=(10,4.8));theta=np.linspace(0,2*np.pi,400)
 extent=max(1.15,max(max(np.max(np.abs(d['x'])),np.max(np.abs(d['y']))) for d in data.values())*1.05)
 xy.plot(np.cos(theta),np.sin(theta),'k--',lw=1,label='목표 반지름 1m');xy.set(xlim=(-extent,extent),ylim=(-extent,extent),aspect='equal',xlabel='X [m]',ylabel='Y [m]')
 ymax=max(.03,max(np.max(np.abs(d['ez'])) for d in data.values())*1.15)
 zax.axhline(0,color='k',ls='--',lw=1);zax.set(xlim=(0,2*T),ylim=(-ymax,ymax),xlabel='simulation time [s]',ylabel='Z error [m]')
 artists={}
 for (name,d),color in zip(data.items(),['#777777','#0072b2']):
  line,=xy.plot([],[],color=color,label=name);dot,=xy.plot([],[],'o',color=color,ms=5);zl,=zax.plot([],[],color=color,label=name);artists[name]=(line,dot,zl)
 target,=xy.plot([],[],'kx',ms=8,label='현재 목표');cursor=zax.axvline(0,color='k',alpha=.3);title=fig.suptitle('')
 xy.legend(loc='lower right',fontsize=7);zax.legend(fontsize=8);xy.grid(alpha=.25);zax.grid(alpha=.25);fig.tight_layout(rect=(0,0,1,.93))
 def update(i):
  t=(i+1)*.01
  for name,d in data.items():
   j=min(i,len(d['time_s'])-1);line,dot,zl=artists[name];line.set_data(d['x'][:j+1],d['y'][:j+1]);dot.set_data([d['x'][j]],[d['y'][j]]);zl.set_data(d['time_s'][:j+1],d['ez'][:j+1])
  target.set_data([np.cos(2*np.pi*t/T)],[np.sin(2*np.pi*t/T)]);cursor.set_xdata([t,t]);title.set_text(f'기록된 원 추종 / 주기 {T}s / t={t:.2f}s (실시간 배속 1x)')
  return []
 out=r/'plots'/f'06_circle_tracking_T{T}.gif';FuncAnimation(fig,update,frames=frames,interval=100,blit=False,repeat=False).save(out,writer=PillowWriter(fps=10),dpi=85);plt.close(fig)
 records.append(dict(path=str(out),period_s=T,fps=10,source_policy_hz=100,stride=10,frames=len(frames),note='Recorded PhysX trajectories, downsampled display; final frame held for 0.1s. Failed paths stop at first failure; no simulator rerun.'))
dump_json(r/'animation_provenance.json',dict(optimizer_updates=0,simulation_steps=0,wall_s=time.perf_counter()-start,files=records));print('ANIMATIONS_DONE',flush=True)

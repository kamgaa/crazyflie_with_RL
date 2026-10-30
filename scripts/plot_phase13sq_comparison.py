"""Offline physical comparisons, no simulator or optimizer execution."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from ppo_common import dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir);out=r/'plots';t0=time.perf_counter()
f='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(f);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=f).get_name(),'axes.unicode_minus':False,'font.size':8})
sel=json.loads((r/'best_selection.json').read_text());it=sel['selected']['additional_iteration']
plan=json.loads((r/'experiment_plan.json').read_text())
labels={'phase12_baseline':'기존 phase12-1 best512 / Z 복귀속도 제곱','best':f"{plan['branch']} best ({it}회), Z 절대속도 제곱",'final':f"{plan['branch']} final (512회)"}
colors={'phase12_baseline':'#555555','best':'#208b43','final':'#e17019'}
def read(path):
 with path.open() as f:return [{k:float(v) if v not in ('True','False') else v=='True' for k,v in x.items()} for x in csv.DictReader(f)]
def csvout(name,rows):
 keys=list(dict.fromkeys(k for x in rows for k in x))
 with (out/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def save(fig,name):
 fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/(name+'.png'),dpi=145);plt.close(fig)
data={};raw={};allmetrics=[];early=[];tail=[]
for model in labels:
 for suite in ['fixed','random20']:
  folder=r/'evaluation'/f'{model}_{suite}_30s';results=json.loads((folder/'results.json').read_text());data[(model,suite)]=results
  for d in results:
   allmetrics.append(dict(model=model,suite=suite,condition=d['condition'],completed=d['completed'],duration_s=d['duration_s'],rmse_m=d['position_rmse_m'],early_rmse_m=d['early0_5_position_rmse_m'],settling_s=d['settling_time_1cm_1s_s'],**{f'tail_bias_{a}_m':d['tail_error_vector_m'][i] if d['completed'] else None for i,a in enumerate('xyz')},tail_rmse_m=d['tail_position_rmse_m'],early_cross_max_m=max(d['early0_5_cross_axis_max_m'].values(),default=None),full_cross_max_m=max(d['full0_cross_axis_max_m'].values(),default=None),early_overshoot_m=d['early0_5_overshoot_m'],full_overshoot_m=d['full0_overshoot_m']))
   if suite=='fixed':
    src=folder/Path(d['checkpoint']).stem/(d['condition']+'.csv');rows=read(src);j=json.loads(src.with_suffix('.json').read_text());initial=j['initial_observation'];init=dict(rows[0]);init.update(time_s=0.,**dict(zip(['ex','ey','ez','vx','vy','vz','qw','qx','qy','qz','wx','wy','wz'],initial[:13])),roll_rad=0.,pitch_rad=0.,yaw_rad=0.,roll_deg=0.,pitch_deg=0.,tilt_deg=0.)
    raw[(model,d['condition'])]=[init]+rows
    early.extend(dict(model=model,condition=d['condition'],**x) for x in rows if x['time_s']<=5)
    if d['condition']=='nominal':tail.extend(dict(model=model,**x) for x in rows if x['time_s']>=29)
csvout('initial_state_comparison',[dict(model=m,condition=c,time_s=0.,**{k:seq[0][k] for k in ['ex','ey','ez','vx','vy','vz','roll_rad','pitch_rad','yaw_rad','wx','wy','wz']}) for (m,c),seq in raw.items()]);csvout('conditionwise_physical_metrics',allmetrics);csvout('early_trajectory_comparison',early);csvout('nominal_29_30s_comparison',tail)
cases=[d['condition'] for d in data[('phase12_baseline','fixed')] if d['condition']!='nominal'];case_labels={'x_plus':'X +5cm','x_minus':'X −5cm','y_plus':'Y +5cm','y_minus':'Y −5cm','z_plus':'Z +5cm','z_minus':'Z −5cm'}
groups=[('early_position_velocity_0_5s',[('ex','X 오차 [m]'),('ey','Y 오차 [m]'),('ez','Z 오차 [m]'),('vx','Vx [m/s]'),('vy','Vy [m/s]'),('vz','Vz [m/s]')]),('early_attitude_omega_0_5s',[('roll_deg','roll [deg]'),('pitch_deg','pitch [deg]'),('yaw_rad','yaw [rad]'),('wx','body ωx [rad/s]'),('wy','body ωy [rad/s]'),('wz','body ωz [rad/s]')]),('early_rotor_thrust_0_5s',[(f'thrust_{i}',f'rotor {i} [N]') for i in range(4)])]
for name,signals in groups:
 fig,axs=plt.subplots(len(signals),6,figsize=(22,2.05*len(signals)),sharex=True,sharey='row',squeeze=False)
 for j,case in enumerate(cases):
  for i,(key,unit) in enumerate(signals):
   ax=axs[i,j]
   for model in labels:
    rows=[x for x in raw[(model,case)] if x['time_s']<=5 and (x['time_s']>0 or not key.startswith('thrust'))]
    ax.plot([x['time_s'] for x in rows],[x[key] for x in rows],color=colors[model],lw=.7,label=labels[model])
   ax.grid(alpha=.25);ax.set_xlim(0,5)
   if key in ('ex','ey','ez'):ax.axhline(0,color='k',ls=':',lw=.5)
   if j==0:ax.set_ylabel(unit)
   if i==0:ax.set_title(case_labels[case])
   if i==len(signals)-1:ax.set_xlabel('simulation time [s]')
 axs[0,0].legend(fontsize=6);fig.suptitle('정정: Z 절대속도 제곱 비교 / 새 best·final vs phase12-1 / ±5cm 복귀 초기 0~5초');save(fig,name)
for name,n,first,second,units in [('early_raw_applied_actions_0_5s',4,'raw_action','action',['roll action','pitch action','yaw action','Fz action']),('early_requested_actual_torque_0_5s',3,'request','actual',['roll torque [Nm]','pitch torque [Nm]','yaw torque [Nm]'])]:
 fig,axs=plt.subplots(n,6,figsize=(22,2.4*n),sharex=True,sharey='row',squeeze=False)
 for j,case in enumerate(cases):
  for i in range(n):
   ax=axs[i,j]
   for model in labels:
    rows=[x for x in raw[(model,case)] if 0<x['time_s']<=5];t=[x['time_s'] for x in rows]
    ax.plot(t,[x[f'{first}_{i}'] for x in rows],lw=.7,color=colors[model],label=labels[model]+' raw/요청')
    ax.plot(t,[x[f'{second}_{i}'] for x in rows],lw=.7,ls='--',color=colors[model],label=labels[model]+' 적용/실제')
   ax.grid(alpha=.25);ax.set_xlim(0,5)
   if j==0:ax.set_ylabel(units[i])
   if i==0:ax.set_title(case_labels[case])
   if i==n-1:ax.set_xlabel('simulation time [s]')
 axs[0,0].legend(fontsize=5);fig.suptitle('정정: Z 절대속도 제곱 비교 / ±5cm 복귀 초기 0~5초 / 실선 raw·요청, 점선 적용·실제');save(fig,name)
fig,axs=plt.subplots(6,2,figsize=(16,16));signals=[('roll_deg','roll [deg]'),('pitch_deg','pitch [deg]'),('wx','body ωx [rad/s]'),('wy','body ωy [rad/s]'),('wz','body ωz [rad/s]')]+[(f'raw_action_{i}',f'action {i} [normalized]') for i in range(4)]+[(f'actual_{i}',f'actual torque {i} [Nm]') for i in range(3)]
for ax,(key,title) in zip(axs.ravel(),signals):
 for model in labels:
  rows=[x for x in raw[(model,'nominal')] if x['time_s']>=29]
  ax.plot([x['time_s'] for x in rows],[x[key] for x in rows],color=colors[model],lw=.8,label=labels[model])
  if key.startswith('raw_action'):
   ax.plot([x['time_s'] for x in rows],[x[key.replace('raw_action','action')] for x in rows],color=colors[model],lw=.7,ls='--')
 ax.set(title=title,xlabel='simulation time [s]',xlim=(29,30));ax.grid(alpha=.25)
axs[0,0].legend(fontsize=7);fig.suptitle('정정: Z 절대속도 제곱 비교 / nominal 29~30초 / action 실선 raw, 점선 applied');save(fig,'nominal_waveforms_29_30s')
fig,axs=plt.subplots(4,1,figsize=(17,11),sharex=True)
conditions=[(s,d['condition']) for s in ['fixed','random20'] for d in data[('phase12_baseline',s)]]
for mi,model in enumerate(labels):
 points=[next(x for x in allmetrics if x['model']==model and x['suite']==s and x['condition']==c) for s,c in conditions];x=np.arange(27)+(mi-1)*.18
 for i,a in enumerate('xyz'):
  axs[i].scatter(x,[np.nan if z['tail_bias_'+a+'_m'] is None else z['tail_bias_'+a+'_m']*1000 for z in points],s=18,color=colors[model],label=labels[model]);axs[i].axhline(0,color='k',lw=.5);axs[i].set_ylabel(f'{a.upper()} bias [mm]')
 axs[3].scatter(x,[np.nan if z['tail_rmse_m'] is None else z['tail_rmse_m']*1000 for z in points],s=18,color=colors[model]);axs[3].set_ylabel('3D tail RMSE [mm]')
for ax in axs:ax.grid(alpha=.25);ax.axvline(6.5,color='k',ls=':')
axs[0].legend();axs[-1].set_xticks(range(27),[c for s,c in conditions],rotation=55,ha='right');fig.suptitle('조건별 20~30초 편향·위치 오차 / 완료 조건만 표시 / 서로 다른 조건의 편향을 합치지 않음');save(fig,'conditionwise_tail_xyz_bias')
fig,axs=plt.subplots(3,1,figsize=(15,10))
for mi,model in enumerate(labels):
 points=[next(x for x in allmetrics if x['model']==model and x['suite']=='fixed' and x['condition']==c) for c in cases];x=np.arange(6)+(mi-1)*.22
 for ax,key,unit in zip(axs,['early_cross_max_m','early_overshoot_m','settling_s'],['0~5초 다른 축 최대 이탈 [cm]','0~5초 복귀 축 overshoot [cm]','1cm 정착 시간 [s]']):
  vals=[np.nan if z[key] is None else z[key]*(100 if key!='settling_s' else 1) for z in points];ax.bar(x,vals,.2,label=labels[model]);ax.set_ylabel(unit);ax.set_xticks(range(6),[case_labels[c] for c in cases]);ax.grid(axis='y',alpha=.25)
  for xx,v in zip(x,vals):
   if np.isnan(v):ax.text(xx,0,'N/A',rotation=90,fontsize=7)
axs[0].legend(fontsize=7);fig.suptitle('정정: Z 절대속도 제곱 비교 / ±5cm 복귀·축간 이탈·정착 / 실패는 정착 N/A');save(fig,'cross_axis_overshoot_settling')
fig,axs=plt.subplots(2,1,figsize=(17,9),sharex=True)
axis_cases=[(case,j) for case in cases for j in range(3) if j!='xyz'.index(case[0])]
for ai,prefix in enumerate(['early0_5','full0']):
 for mi,model in enumerate(labels):
  lookup={z['condition']:z for z in data[(model,'fixed')]};values=[lookup[c][prefix+'_cross_axis_max_m']['xyz'[j]]*100 for c,j in axis_cases]
  axs[ai].bar(np.arange(len(axis_cases))+(mi-1)*.22,values,.2,label=labels[model])
 axs[ai].set_ylabel(('초기0~5초' if ai==0 else '전체0~30초')+' 다른 축 최대이탈 [cm]');axs[ai].grid(axis='y',alpha=.25)
axs[0].legend(fontsize=7);axs[-1].set_xticks(range(len(axis_cases)),[case_labels[c]+' → '+'XYZ'[j] for c,j in axis_cases],rotation=35,ha='right');fig.suptitle('축간 이탈을 축별로 분리 / X복귀의 Y·Z, Y복귀의 X·Z, Z복귀의 X·Y');save(fig,'cross_axis_individual_comparison')
es=json.loads((r/'evaluation_summary.json').read_text());cr=es['candidates'];baseline=next(z for z in es['final'] if z['model']=='phase12_baseline' and z['suite']=='fixed')
fig,axs=plt.subplots(4,1,figsize=(12,11),sharex=True)
for ax,key,title in zip(axs,['completion_fraction','completed_only_mean_position_rmse_m','completed_only_mean_tail_body_omega_rms_rad_s','completed_only_mean_tail_applied_action_saturation_fraction'],['30초 완료율','완료 조건 3D RMSE [m]','후반 body ω RMS [rad/s]','후반 적용 action 포화 비율']):
 ax.plot([v['additional_transitions']/1e6 for v in cr],[np.nan if v.get(key) is None else v[key] for v in cr],'o-',label='정정: Z 절대속도 제곱 비교');ax.axhline(baseline[key],color=colors['phase12_baseline'],ls='--',label=labels['phase12_baseline']);ax.set_ylabel(title);ax.grid(alpha=.25)
axs[0].legend();axs[-1].set_xlabel('새 policy transitions [million]');fig.suptitle('체크포인트별 위치·진동·포화 / 0,64,...512회 / 실패의 짧은 RMSE와 완료 RMSE를 섞지 않음');save(fig,'checkpoint_rmse_omega_saturation')
training=[json.loads(x) for x in (r/'metrics.jsonl').read_text().splitlines()]
from phase13sq_reward import TERMS
fig,axs=plt.subplots(4,4,figsize=(17,12));x=[v['global_step']/1e6 for v in training]
for ax,key in zip(axs.ravel(),list(TERMS)+['total']):
 ax.plot(x,[v['Reward/'+key] for v in training],lw=.8);ax.set(title='Reward/'+key,xlabel='새 policy transitions [million]');ax.grid(alpha=.25)
for ax in axs.ravel()[14:]:ax.axis('off')
fig.suptitle('정정: Z 절대속도 제곱 비교 / 각 축별 실제 signed 보상 기여도 / 독립 PPO loss가 아님');save(fig,'reward_terms_separate')
summary=[]
for model in labels:
 for suite in ['fixed','random20']:
  ds=data[(model,suite)];done=[d for d in ds if d['completed']];axis=[d for d in done if d['early0_5_cross_axis_max_m']];settled=[d for d in done if d['settling_time_1cm_1s_s'] is not None]
  summary.append(dict(model=model,suite=suite,mean_non_nominal_settling_s=float(np.mean([d['settling_time_1cm_1s_s'] for d in settled if d['condition']!='nominal'])) if any(d['condition']!='nominal' for d in settled) else None,completed=len(done),count=len(ds),settled_count=len(settled),mean_settling_s=float(np.mean([d['settling_time_1cm_1s_s'] for d in settled])) if settled else None,mean_early_rmse_m=float(np.mean([d['early0_5_position_rmse_m'] for d in done])) if done else None,mean_cross_early_m=float(np.mean([max(d['early0_5_cross_axis_max_m'].values()) for d in axis])) if axis else None,max_cross_early_m=max([max(d['early0_5_cross_axis_max_m'].values()) for d in axis],default=None),mean_overshoot_early_m=float(np.mean([d['early0_5_overshoot_m'] for d in axis])) if axis else None))
dump_json(r/'return_metrics_summary.json',summary);csvout('return_metrics_summary',summary);dump_json(r/'comparison_plot_time.json',dict(seconds=time.perf_counter()-t0));print('COMPARISON_PLOTS_DONE')

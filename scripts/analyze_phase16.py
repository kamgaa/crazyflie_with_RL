"""Offline PID payload sweep and comparison to preserved RL results."""
import argparse,csv,json,hashlib,math,shutil,time,sys,tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir);plots=r/'plots';plots.mkdir(exist_ok=True)
font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':9,'axes.grid':True,'grid.alpha':.2})
plan=json.loads((r/'experiment_plan.json').read_text());suites=['hover','circle_T10','circle_T5'];labels={'hover':'고정 목표 호버 30초','circle_T10':'원 추종 T=10초, 두 바퀴','circle_T5':'원 추종 T=5초, 두 바퀴'}
results={s:json.loads((r/'evaluation'/s/'results.json').read_text()) for s in suites};lookup={(s,x['name']):x for s,xs in results.items() for x in xs};flat=[]
for s,xs in results.items():
 for x in xs:
  d={k:x[k] for k in ['name','mass_g','x_cm','suite','horizon_s','completed','duration_s','position_rmse_m','max_position_error_m','max_tilt_deg','speed_rms_m_s','body_omega_rms_rad_s','action_clipping_fraction','allocation_clipping_fraction','tail_position_rmse_m','tail_speed_rms_m_s','tail_body_omega_rms_rad_s','reward_sum_max_error']}
  d['reasons']=';'.join(x['reasons']);d['position_rmse_completed_only_m']=x['position_rmse_m'] if x['completed'] else None
  for i,axis in enumerate('xyz'):
   d[f'{axis}_rmse_m']=x['xyz_rmse_m'][i];d[f'tail_{axis}_bias_m']=x['tail_error_vector_m'][i] if x['tail_error_vector_m'] else None
  for k in ['total_mass_kg','static_level_rotor_feasible','static_level_action_pipeline_feasible']:d[k]=x['physical'][k]
  d['com_x_m']=x['physical']['com_body_m'][0];d['equilibrium_pitch_action']=x['physical']['equilibrium_normalized_action'][1]
  flat.append(d)
def writecsv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
writecsv(r/'summary.csv',flat);dump_json(r/'summary.json',flat)
def save(fig,name):fig.tight_layout(rect=[0,0,1,.96],h_pad=2,w_pad=2);fig.savefig(plots/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
def heat(ax,arr,title,fmt='.2f',vmin=None,vmax=None,cmap='viridis'):
 ax.grid(False);im=ax.imshow(arr,origin='lower',aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax);ax.set(xticks=range(3),xticklabels=[10,20,30],yticks=range(9),yticklabels=range(2,11),xlabel='body +X 거리 [cm]',ylabel='추 질량 [g]',title=title)
 for i in range(9):
  for j in range(3):ax.text(j,i,'실패' if np.isnan(arr[i,j]) else format(arr[i,j],fmt),ha='center',va='center',fontsize=8,color='black',bbox=dict(facecolor='white',alpha=.65,edgecolor='none',pad=.5))
 plt.colorbar(im,ax=ax,shrink=.8)
fig,axs=plt.subplots(1,3,figsize=(12,5))
for ax,s in zip(axs,suites):
 arr=np.array([[lookup[s,f'm{m:02d}g_x{d}cm']['duration_s'] for d in [10,20,30]] for m in range(2,11)]);heat(ax,arr,labels[s]+' / 지속 시간 [s]',vmin=0,vmax=30)
fig.suptitle('기존 CascadePID, 재튜닝 없음 / 첫 실패에서 종료');save(fig,'01_duration_completion');writecsv(plots/'01_duration_completion.csv',flat)
fig,axs=plt.subplots(1,3,figsize=(12,5))
for ax,s in zip(axs,suites):
 arr=np.array([[lookup[s,f'm{m:02d}g_x{d}cm']['position_rmse_m']*100 if lookup[s,f'm{m:02d}g_x{d}cm']['completed'] else np.nan for d in [10,20,30]] for m in range(2,11)]);heat(ax,arr,labels[s]+' / 완료 조건 RMSE [cm]',vmin=0,vmax=70)
fig.suptitle('실패 궤적의 짧은 RMSE는 완료 성능으로 비교하지 않음');save(fig,'02_completed_position_rmse');writecsv(plots/'02_completed_position_rmse.csv',flat)
physical=json.loads((r/'evaluation/hover/engine_properties.json').read_text())['expected'];phys={x['name']:x for x in physical};fig,axs=plt.subplots(1,3,figsize=(12,5))
for ax,key,title in zip(axs,['com_x_m','pitch','feas'],['COM의 전방 이동 [cm]','평형에 필요한 정규화 pitch 행동','정적 수평 평형 가능성\n0불가 / 1추력만 / 2현행출력']):
 arr=np.zeros((9,3))
 for i,m in enumerate(range(2,11)):
  for j,d in enumerate([10,20,30]):
   x=phys[f'm{m:02d}g_x{d}cm'];arr[i,j]=x['com_body_m'][0]*100 if key=='com_x_m' else x['equilibrium_normalized_action'][1] if key=='pitch' else int(x['static_level_rotor_feasible'])+int(x['static_level_action_pipeline_feasible'])
 heat(ax,arr,title,fmt='.2f' if key!='feas' else '.0f')
fig.suptitle('이론상 필요 조건 / 제어 성공을 보장하지 않음 / 원 추종의 동적 가능성 판정 아님');save(fig,'03_static_hover_feasibility');writecsv(plots/'03_static_hover_feasibility.csv',[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in x.items()} for x in physical])
cache={}
def trajectory(s,name):
 key=(s,name)
 if key not in cache:
  with (r/'evaluation'/s/name/'trajectory.csv').open() as f:cache[key]=list(csv.DictReader(f))
 return cache[key]
def col(rows,k):return np.array([float(x[k]) for x in rows])
colors=plt.cm.viridis(np.linspace(.05,.95,9));plotrows=[]
fig,axs=plt.subplots(3,3,figsize=(14,9))
for j,d in enumerate([10,20,30]):
 for mi,m in enumerate(range(2,11)):
  name=f'm{m:02d}g_x{d}cm';rows=trajectory('hover',name);t=col(rows,'time_s')
  for ax,k,title in zip(axs[:,j],['position_error_m','ez','pitch_deg'],['位置誤差'.replace('位置誤差','위치 오차 [m]'),'Z 오차 [m]','실제 pitch [deg]']):ax.plot(t,col(rows,k),color=colors[mi],label=f'{m}g');ax.set(ylabel=title,xlabel='simulation time [s]',xlim=(0,5));ax.grid(alpha=.2)
  for row in rows:
   if float(row['time_s'])<=5:plotrows.append(dict(case=name,**{k:row[k] for k in ['time_s','position_error_m','ex','ey','ez','vx','vy','vz','roll_deg','pitch_deg','wx','wy','wz']}))
 axs[0,j].set_title(f'+X {d}cm / 첫 0~5초');axs[0,j].legend(ncol=3,fontsize=7)
 axs[0,j].set_ylim(0,1.1);axs[1,j].set_ylim(-.32,.02);axs[2,j].set_ylim(-65,65)
fig.suptitle('고정 목표 복귀: 각 궤적은 최초 실패 시 끝남');save(fig,'04_hover_first5s_states');writecsv(plots/'04_hover_first5s_states.csv',plotrows)
fig,axs=plt.subplots(2,3,figsize=(14,9));circlecsv=[]
for i,s in enumerate(suites[1:]):
 for j,d in enumerate([10,20,30]):
  ax=axs[i,j];nom=trajectory(s,'nominal');ax.plot(col(nom,'target_x'),col(nom,'target_y'),'k--',lw=1,label='목표 R=1m');ax.plot(col(nom,'x'),col(nom,'y'),color='gray',lw=1.5,label='무부하')
  for mi,m in enumerate(range(2,11)):
   name=f'm{m:02d}g_x{d}cm';rows=trajectory(s,name);ax.plot(col(rows,'x'),col(rows,'y'),color=colors[mi],label=f'{m}g');ax.plot(float(rows[-1]['x']),float(rows[-1]['y']),'o',ms=3,color=colors[mi])
   for row in rows:circlecsv.append(dict(suite=s,case=name,**{k:row[k] for k in ['time_s','x','y','z','target_x','target_y','target_z','ex','ey','ez','position_error_m','vx','vy','vz','tilt_deg']}))
  ax.set(title=f'{labels[s]} / +X {d}cm',xlabel='로컬 X [m]',ylabel='로컬 Y [m]',aspect='equal',xlim=(-1.5,1.5),ylim=(-1.5,1.5));ax.legend(fontsize=6,ncol=3)
allx=np.concatenate([col(trajectory(s,c['name']),'x') for s in suites[1:] for c in plan['cases']]);ally=np.concatenate([col(trajectory(s,c['name']),'y') for s in suites[1:] for c in plan['cases']]);xl=(min(-1.1,float(allx.min())-.1),max(1.1,float(allx.max())+.1));yl=(min(-1.1,float(ally.min())-.1),max(1.1,float(ally.max())+.1))
for ax in axs.flat:ax.set(xlim=xl,ylim=yl)
fig.suptitle('원 추종: 목표 위치만 이동, yaw=0 / 점은 해당 궤적 종료점');save(fig,'05_circle_paths');writecsv(plots/'05_circle_paths.csv',circlecsv)
selected=['nominal','m02g_x10cm','m04g_x10cm','m05g_x10cm','m02g_x20cm','m02g_x30cm'];fig,axs=plt.subplots(4,2,figsize=(14,12));wavecsv=[]
keys=['raw_action_1','action_1','request_1','actual_1','total_thrust_N','vertical_margin','pitch_deg','wy'];titles=['raw pitch action [-]','clipped pitch action [-]','요청 pitch 토크: 기존 allocator 기준 [Nm]','실제 pitch 토크: 새 COM 기준 [Nm]','실제 총추력 [N]','수직 추력 - 총중량 [N]','실제 pitch [deg]','body Y 각속도 [rad/s]']
for name in selected:
 rows=trajectory('hover',name);t=col(rows,'time_s');weight=phys[name]['total_mass_kg']*9.81
 vertical=col(rows,'total_thrust_N')*np.cos(col(rows,'roll_rad'))*np.cos(col(rows,'pitch_rad'))
 for ax,k,title in zip(axs.flat,keys,titles):ax.plot(t,vertical-weight if k=='vertical_margin' else col(rows,k),label=name);ax.set(title=title,xlim=(0,3),xlabel='simulation time [s]')
 for row,v in zip(rows,vertical):
  if float(row['time_s'])<=3:wavecsv.append(dict(case=name,vertical_thrust_N=v,weight_N=weight,**{k:row[k] for k in ['time_s','raw_action_1','action_1','request_1','actual_1','total_thrust_N','pitch_deg','wy']}))
axs[0,0].legend(fontsize=7,ncol=2);fig.suptitle('고정 목표 초기 0~3초: PID 명령 / 실제 COM 토크 / 수직 추력');save(fig,'06_hover_action_torque_vertical_thrust');writecsv(plots/'06_hover_action_torque_vertical_thrust.csv',wavecsv)
fig,axs=plt.subplots(3,2,figsize=(13,9));tailcsv=[]
for name in selected:
 x=lookup['hover',name]
 if not x['completed']:continue
 rows=trajectory('hover',name);t=col(rows,'time_s')
 for ax,k,label in zip(axs.flat,['ex','ey','ez','pitch_deg','wy','total_thrust_N'],['X 오차 [m]','Y 오차 [m]','Z 오차 [m]','pitch [deg]','body Y 각속도 [rad/s]','총추력 [N]']):ax.plot(t,col(rows,k),label=name);ax.set(title=label,xlabel='simulation time [s]',xlim=(20,30))
 for row in rows:
  if float(row['time_s'])>20:tailcsv.append(dict(case=name,**{k:row[k] for k in ['time_s','ex','ey','ez','pitch_deg','wy','total_thrust_N']}))
axs[0,0].legend(fontsize=7);fig.suptitle('30초 완료 조건만: 후반 20~30초 / 위치 편향과 진동을 구분');save(fig,'07_completed_hover_tail');writecsv(plots/'07_completed_hover_tail.csv',tailcsv)
# Actual event-vs-JSON checks, every exported scalar (identical FP32 rounding).
tb=[]
for s in suites:
 e=EventAccumulator(str(r/'evaluation'/s/'tensorboard_time'),size_guidance={'scalars':0}).Reload();data=json.loads((r/'evaluation'/s/'tensorboard_scalars.json').read_text());expected={(x['tag'],x['step']):x['value'] for x in data};actual={(tag,x.step):x.value for tag in e.Tags()['scalars'] for x in e.Scalars(tag)}
 assert expected.keys()==actual.keys();err=max(abs(expected[k]-actual[k]) for k in expected);assert err==0;tb.append(dict(suite=s,tags=len(e.Tags()['scalars']),scalars=len(expected),max_abs_error=err))
protected=json.loads((r/'evidence/protected_sha256.json').read_text());changed=[path for path,digest in protected.items() if not (ROOT/path).exists() or sha256(ROOT/path)!=digest];assert not changed,changed
executions={s:json.loads((r/'evaluation'/s/'execution.json').read_text()) for s in suites};assert all(x['success'] for x in executions.values())
prior=Path(plan['rl_reference_run']);rl={s:json.loads((prior/'evaluation'/s/'results.json').read_text()) for s in suites};rl_lookup={(s,x['name']):x for s,xs in rl.items() for x in xs}
same={}
for s in suites:
 props=json.loads((r/'evaluation'/s/'engine_properties.json').read_text());oldprops=json.loads((prior/'evaluation'/s/'engine_properties.json').read_text());assert props==oldprops
 init=json.loads((r/'evaluation'/s/'initial_state.json').read_text());oldinit=json.loads((prior/'evaluation'/s/'initial_state.json').read_text());assert init==oldinit
 ec=json.loads((prior/'evaluation'/s/'execution.json').read_text())['env_config'];assert executions[s]['env_config']==ec
 same[s]=dict(engine_properties_identical=True,initial_state_identical=True,env_config_identical=True)
verification=dict(success=True,PID=json.loads((r/'pid_verification.json').read_text()),scalar_checks=tb,executions=executions,RL_matching=same,protected_files_checked=len(protected),protected_files_changed=changed,optimizer_created=False,checkpoint_loaded=False,optimizer_updates=0,reward_sum_max_error=max(x['reward_sum_max_error'] for xs in results.values() for x in xs),headless=True,GUI_verified=False)
dump_json(r/'verification.json',verification)
summary={s:dict(loaded_completed=sum(x['completed'] for x in xs if x['mass_g']),loaded_count=27,completed_names=[x['name'] for x in xs if x['mass_g'] and x['completed']],nominal={k:xs[0][k] for k in ['completed','duration_s','position_rmse_m','max_tilt_deg']}) for s,xs in results.items()};dump_json(r/'result_overview.json',summary)
comparison=[]
for s in suites:
 for x in results[s]:
  z=rl_lookup[s,x['name']];both=x['completed'] and z['completed'];comparison.append(dict(suite=s,case=x['name'],mass_g=x['mass_g'],x_cm=x['x_cm'],PID_completed=x['completed'],RL_completed=z['completed'],PID_duration_s=x['duration_s'],RL_duration_s=z['duration_s'],PID_position_rmse_m=x['position_rmse_m'],RL_position_rmse_m=z['position_rmse_m'],both_completed=both,PID_minus_RL_rmse_m=x['position_rmse_m']-z['position_rmse_m'] if both else None,PID_max_tilt_deg=x['max_tilt_deg'],RL_max_tilt_deg=z['max_tilt_deg'],PID_action_clip=x['action_clipping_fraction'],RL_action_clip=z['action_clipping_fraction']))
writecsv(r/'PID_vs_RL.csv',comparison);dump_json(r/'PID_vs_RL.json',comparison)
fig,axs=plt.subplots(2,3,figsize=(13,10))
for row,(who,source) in enumerate([('PID',lookup),('RL',rl_lookup)]):
 for ax,s in zip(axs[row],suites):
  arr=np.array([[source[s,f'm{m:02d}g_x{d}cm']['duration_s'] for d in [10,20,30]] for m in range(2,11)]);heat(ax,arr,who+' / '+labels[s]+' [s]',vmin=0,vmax=30)
fig.suptitle('같은 물성/초기화/출력 제한의 PID vs RL / 최초 종료까지 지속 시간');save(fig,'08_PID_vs_RL_duration');writecsv(plots/'08_PID_vs_RL_duration.csv',comparison)
fig,axs=plt.subplots(1,3,figsize=(14,5))
for ax,s in zip(axs,suites):
 paired=[x for x in comparison if x['suite']==s and x['both_completed']];idx=np.arange(len(paired));ax.bar(idx-.18,[x['PID_position_rmse_m']*100 for x in paired],.36,label='PID');ax.bar(idx+.18,[x['RL_position_rmse_m']*100 for x in paired],.36,label='RL');ax.set(xticks=idx,xticklabels=[x['case'] for x in paired],ylabel='3D 위치 RMSE [cm]',title=labels[s]);ax.tick_params(axis='x',rotation=35);ax.legend()
fig.suptitle('두 제어기가 모두 끝까지 완료한 같은 조건만 비교');save(fig,'09_PID_vs_RL_common_completed_rmse');writecsv(plots/'09_PID_vs_RL_common_completed_rmse.csv',[x for x in comparison if x['both_completed']])
fig,axs=plt.subplots(3,3,figsize=(14,10));compare_wave=[]
for j,name in enumerate(['m02g_x10cm','m04g_x10cm','m02g_x20cm']):
 for who,path in [('PID',r),('RL',prior)]:
  with (path/'evaluation/hover'/name/'trajectory.csv').open() as f:rows=list(csv.DictReader(f))
  for ax,k,title in zip(axs[:,j],['position_error_m','pitch_deg','actual_1'],['위치 오차 [m]','pitch [deg]','실제 COM pitch 토크 [Nm]']):ax.plot(col(rows,'time_s'),col(rows,k),label=who);ax.set(title=name+' / '+title,xlabel='simulation time [s]',xlim=(0,5));ax.legend()
  for row in rows:
   if float(row['time_s'])<=5:compare_wave.append(dict(controller=who,case=name,**{k:row[k] for k in ['time_s','ex','ey','ez','position_error_m','pitch_deg','wy','raw_action_1','action_1','actual_1']}))
fig.suptitle('PID vs RL 초기 응답 / 실패한 궤적을 이후 시간까지 연장하지 않음');save(fig,'10_PID_vs_RL_hover_transient');writecsv(plots/'10_PID_vs_RL_hover_transient.csv',compare_wave)
print(json.dumps(summary,indent=2),flush=True)

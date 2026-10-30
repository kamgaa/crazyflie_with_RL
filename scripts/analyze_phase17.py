"""Offline fault-response analysis and artifact checks; no training/simulation."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir);plots=r/'plots';plan=json.loads((r/'experiment_plan.json').read_text());ctrls=['RL','PID'];results={k:json.loads((r/'evaluation'/k/'results.json').read_text()) for k in ctrls};lookup={(k,x['name']):x for k,xs in results.items() for x in xs}
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':9,'axes.grid':True,'grid.alpha':.2})
def save(fig,name):fig.tight_layout(rect=[0,0,1,.96],h_pad=2,w_pad=2);fig.savefig(plots/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
def writecsv(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
cache={}
def rows(k,name):
 if (k,name) not in cache:
  with (r/'evaluation'/k/name/'trajectory.csv').open() as f:cache[k,name]=list(csv.DictReader(f))
 return cache[k,name]
def col(q,key):return np.array([float(x[key]) for x in q])
flat=[]
for k,xs in results.items():
 for x in xs:
  z=x['response']['postfault'];t=x['response']['tail20_30'];d={f:x[f] for f in ['controller','name','motor_user','native_index','efficiency','completed','duration_s']};d['reasons']=';'.join(x['reasons']);d['postfault_duration_s']=x['response']['postfault_duration_s'];d['first_failure_after_fault_s']=x['response']['first_failure_after_fault_s'];d['full_position_rmse_m']=x['position_rmse_m']
  for field in ['position_rmse_m','max_position_error_m','speed_rms_m_s','omega_rms_rad_s','max_tilt_deg','max_abs_yaw_deg','action_clipping_fraction','allocation_clipping_fraction']:d['post_'+field]=z[field] if z else None
  for field in ['position_rmse_m','speed_rms_m_s','omega_rms_rad_s']:d['tail_'+field]=t[field] if t else None
  for i,axis in enumerate('xyz'):d['tail_bias_'+axis+'_m']=t['bias_xyz_m'][i] if t else None;d['tail_std_'+axis+'_m']=t['std_xyz_m'][i] if t else None
  d['first_fault_substep_index']=x['first_fault_substep_index'];d['settling_1cm_after_fault_s']=x['response']['settling_1cm_for_final1s_after_fault_s'];flat.append(d)
writecsv(r/'summary.csv',flat);dump_json(r/'summary.json',flat)
# Motor numbers are user-facing; the physics array is explicitly reversed.
fig,ax=plt.subplots(figsize=(7,7));motor=[]
for user,native,(x,y),sgn in [(1,3,(.03536,.03536),1),(2,2,(-.03536,.03536),-1),(3,1,(-.03536,-.03536),1),(4,0,(.03536,-.03536),-1)]:
 px,py=-y,x;ax.plot([0,px],[0,py],color='gray',lw=7);ax.scatter([px],[py],s=2200,c='lightblue',edgecolors='black');ax.text(px,py,f'M{user}',ha='center',va='center',fontsize=17,weight='bold');ax.text(px,py+(.014 if x>0 else -.017),f'native[{native}], body({x:+.5f},{y:+.5f},0)m\n반토크 body +Z 부호 {sgn:+d}',ha='center',va='center',fontsize=8);motor.append(dict(user_motor=user,native_index=native,body_x_m=x,body_y_m=y,body_z_m=0,reaction_sign_body_z=sgn))
ax.annotate('+X 앞',xy=(0,.068),xytext=(0,.015),ha='center',arrowprops=dict(arrowstyle='->'));ax.annotate('+Y 왼쪽',xy=(-.068,0),xytext=(-.015,0),ha='center',arrowprops=dict(arrowstyle='->'));ax.set(xlim=(-.08,.08),ylim=(-.08,.08),aspect='equal');ax.axis('off');fig.suptitle('위에서 본 user_frd 모터 번호 / 계산 좌표 body FLU 유지\n추 없음, 한 번에 한 모터만 고장');save(fig,'01_motor_numbering');writecsv(plots/'01_motor_numbering.csv',motor)
def matrix(ax,values,title,fmt='.2f',vmin=0,vmax=None):
 ax.grid(False);im=ax.imshow(values,aspect='auto',cmap='viridis',vmin=vmin,vmax=vmax);ax.set(xticks=range(3),xticklabels=['90%','80%','70%'],yticks=range(4),yticklabels=['M1','M2','M3','M4'],xlabel='정상 대비 남는 추력',ylabel='고장 모터',title=title)
 for i in range(4):
  for j in range(3):ax.text(j,i,'실패' if np.isnan(values[i,j]) else format(values[i,j],fmt),ha='center',va='center',bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=1))
 plt.colorbar(im,ax=ax,shrink=.85)
for name,key,title,scale,only in [('02_postfault_duration','postfault_duration_s','고장 후 지속 시간 [s]',1,False),('03_postfault_position_rmse','position_rmse_m','고장 후 위치 RMSE [cm]',100,True),('04_postfault_body_omega','omega_rms_rad_s','고장 후 body 각속도 RMS [rad/s]',1,True)]:
 fig,axs=plt.subplots(1,2,figsize=(10,4));arrays=[]
 for k in ctrls:
  data=[]
  for motor_id in range(1,5):
   row=[]
   for eta in [90,80,70]:
    x=lookup[k,f'M{motor_id}_eta{eta:03d}'];v=x['response'][key] if key=='postfault_duration_s' else x['response']['postfault'][key];row.append(v*scale if (x['completed'] or not only) else np.nan)
   data.append(row)
  arrays.append(np.array(data))
 vmax=25 if key=='postfault_duration_s' else max(np.nanmax(x) for x in arrays)
 for ax,k,values in zip(axs,ctrls,arrays):matrix(ax,values,k+' / '+title,vmax=max(vmax,1e-6))
 fig.suptitle('0~5초 정상, 5~30초 부분 고장 / '+('완료 조건만 오차 비교' if only else '실제 최초 실패에서 종료'));save(fig,name);writecsv(plots/(name+'.csv'),flat)
fig,axs=plt.subplots(3,4,figsize=(16,10));curve=[]
for i,eta in enumerate([90,80,70]):
 for j,m in enumerate(range(1,5)):
  name=f'M{m}_eta{eta:03d}';ax=axs[i,j]
  for k in ctrls:
   q=rows(k,name);ax.plot(col(q,'time_s'),col(q,'position_error_m')*100,label=k)
   for z in q:curve.append(dict(controller=k,case=name,**{f:z[f] for f in ['time_s','position_error_m','ex','ey','ez','roll_deg','pitch_deg','wx','wy','wz','tilt_deg']}))
  ax.axvline(5,ls='--',color='gray');ax.set(title=f'M{m}, {eta}% 유지',xlabel='simulation time [s]',ylabel='위치 오차 [cm]',xlim=(0,30));ax.legend()
fig.suptitle('모터별 PID/RL 위치 오차 시간 이력 / 점선5초에 고장 / 실패 이후 데이터 없음');save(fig,'05_position_error_all_conditions');writecsv(plots/'05_position_error_all_conditions.csv',curve)
name='M1_eta070';fig,axs=plt.subplots(3,2,figsize=(13,10));statecsv=[]
for j,k in enumerate(ctrls):
 q=rows(k,name);t=col(q,'time_s')
 for i,(fields,title,factor) in enumerate([(['ex','ey','ez'],'XYZ 위치 오차 [cm]',100),(['roll_rad','pitch_rad','yaw_rad'],'ZYX Euler roll/pitch/yaw [deg]',180/np.pi),(['wx','wy','wz'],'body 각속도 [rad/s]',1)]):
  for f in fields:axs[i,j].plot(t,col(q,f)*factor,label=f)
  axs[i,j].axvline(5,ls='--',c='gray');axs[i,j].set(title=k+' / '+title,xlabel='simulation time [s]',xlim=(0,30));axs[i,j].legend()
 for z in q:statecsv.append(dict(controller=k,**{f:z[f] for f in ['time_s','ex','ey','ez','vx','vy','vz','roll_rad','pitch_rad','yaw_rad','wx','wy','wz']}))
for i in range(3):
 lo=min(ax.get_ylim()[0] for ax in axs[i]);hi=max(ax.get_ylim()[1] for ax in axs[i]);[ax.set_ylim(lo,hi) for ax in axs[i]]
fig.suptitle('대표 M1 추력70% 유지: 위치·자세·각속도 / 5초 이후 고장');save(fig,'06_M1_eta070_states');writecsv(plots/'06_M1_eta070_states.csv',statecsv)
fig,axs=plt.subplots(3,2,figsize=(13,10));motorcsv=[];colors=['tab:blue','tab:orange','tab:green','tab:red']
for j,k in enumerate(ctrls):
 q=rows(k,name);t=col(q,'time_s')
 for user,c in zip(range(1,5),colors):
  axs[0,j].plot(t,col(q,f'user_actual_thrust_N_{user}'),color=c,label=f'M{user} 실제');axs[0,j].plot(t,col(q,f'user_nominal_thrust_N_{user}'),'--',color=c,alpha=.6)
  axs[1,j].plot(t,col(q,f'user_rpm_{user}'),color=c,label=f'M{user}');axs[2,j].plot(t,col(q,f'user_efficiency_{user}'),color=c,label=f'M{user}')
 for ax,title in zip(axs[:,j],['모터별 추력 [N]: 실선실제/점선정상모델','모터 RPM [rev/min]','추력·반토크 효율 [-]']):ax.axvline(5,ls='--',c='gray');ax.set(title=k+' / '+title,xlabel='simulation time [s]',xlim=(4.5,10));ax.legend(fontsize=7,ncol=2)
 for z in q:
  if 4.5<=float(z['time_s'])<=10:motorcsv.append(dict(controller=k,**{f:z[f] for f in ['time_s']+[f'user_{field}_{i}' for field in ['actual_thrust_N','nominal_thrust_N','rpm','efficiency'] for i in range(1,5)]}))
fig.suptitle('대표 M1 추력70% 유지 / 4.5~10초 확대 / RPM 자체를70%로 줄이는 모델이 아님');save(fig,'07_M1_eta070_motor_response');writecsv(plots/'07_M1_eta070_motor_response.csv',motorcsv)
fig,axs=plt.subplots(4,2,figsize=(13,13));commandcsv=[]
for j,k in enumerate(ctrls):
 q=rows(k,name);t=col(q,'time_s')
 for i,c in enumerate(colors):
  axs[0,j].plot(t,col(q,f'raw_action_{i}'),c=c,label=['roll','pitch','yaw','collective'][i]);axs[0,j].plot(t,col(q,f'action_{i}'),'--',c=c,alpha=.7)
  if i<3:
   axs[1,j].plot(t,col(q,f'request_{i}'),c=c,label=['roll','pitch','yaw'][i]);axs[1,j].plot(t,col(q,f'actual_{i}'),'--',c=c)
 axs[2,j].plot(t,col(q,'request_3'),label='요청');axs[2,j].plot(t,col(q,'actual_3'),'--',label='실제')
 axs[3,j].plot(t,col(q,'action_clipping_fraction'),label='action 범위초과');axs[3,j].plot(t,col(q,'allocation_clipping_fraction'),label='allocation 제한')
 for ax,title in zip(axs[:,j],['정규화 명령 [-]: 실선raw/점선applied','body torque [Nm]: 실선요청/점선실제','body Z 총추력 [N]','4축/모터 평균 제한 비율 [-]']):ax.axvline(5,ls='--',c='gray');ax.set(title=k+' / '+title,xlabel='simulation time [s]',xlim=(4.5,10));ax.legend(fontsize=7,ncol=2)
 for z in q:
  if 4.5<=float(z['time_s'])<=10:commandcsv.append(dict(controller=k,**{f:z[f] for f in ['time_s','action_clipping_fraction','allocation_clipping_fraction']+[f'{field}_{i}' for field in ['raw_action','action','request','actual'] for i in range(4)]}))
fig.suptitle('대표 M1 추력70% 유지 / 4.5~10초 확대 / PID 또는 frozen RL만 제어');save(fig,'08_M1_eta070_commands_wrench');writecsv(plots/'08_M1_eta070_commands_wrench.csv',commandcsv)
# Explicit worst RL postfault-position-RMSE condition, without changing any trial or selection.
worst=max((x for x in results['RL'] if x['motor_user']),key=lambda x:x['response']['postfault']['position_rmse_m']);name=worst['name'];fig,axs=plt.subplots(4,2,figsize=(13,12));zoom=[]
for j,k in enumerate(ctrls):
 q=[z for z in rows(k,name) if float(z['time_s'])>=29];t=col(q,'time_s')
 for f in ['roll_deg','pitch_deg']:axs[0,j].plot(t,col(q,f),label=f)
 for i,c in enumerate(colors[:3]):
  axs[1,j].plot(t,col(q,['wx','wy','wz'][i]),c=c,label=['wx','wy','wz'][i]);axs[2,j].plot(t,col(q,f'raw_action_{i}'),c=c,label=['roll','pitch','yaw'][i]);axs[2,j].plot(t,col(q,f'action_{i}'),'--',c=c)
  axs[3,j].plot(t,col(q,f'actual_{i}'),c=c,label=['roll','pitch','yaw'][i])
 for ax,title in zip(axs[:,j],['roll/pitch [deg]','body 각속도 [rad/s]','명령 [-]: 실선raw/점선applied','실제 body torque [Nm]']):ax.set(title=k+' / '+title,xlabel='simulation time [s]',xlim=(29,30));ax.legend(fontsize=7)
 for z in q:zoom.append(dict(controller=k,case=name,**{f:z[f] for f in ['time_s','roll_deg','pitch_deg','wx','wy','wz']+[f'{field}_{i}' for field in ['raw_action','action','actual'] for i in range(4)]}))
for i in range(4):
 lo=min(ax.get_ylim()[0] for ax in axs[i]);hi=max(ax.get_ylim()[1] for ax in axs[i]);[ax.set_ylim(lo,hi) for ax in axs[i]]
fig.suptitle(name+' / RL 고장 후 위치RMSE 최대 조건, 29~30초 확대 / 동일 Y축');save(fig,'09_worst_RL_tail_29_30s');writecsv(plots/'09_worst_RL_tail_29_30s.csv',zoom)
# Verify all TensorBoard values and preserved files.
tb=[]
for k in ctrls:
 ev=EventAccumulator(str(r/'evaluation'/k/'tensorboard_time'),size_guidance={'scalars':0}).Reload();exp={(x['tag'],x['step']):x['value'] for x in json.loads((r/'evaluation'/k/'tensorboard_scalars.json').read_text())};act={(tag,x.step):x.value for tag in ev.Tags()['scalars'] for x in ev.Scalars(tag)};assert exp.keys()==act.keys();err=max(abs(exp[x]-act[x]) for x in exp);assert err==0;tb.append(dict(controller=k,tags=len(ev.Tags()['scalars']),scalars=len(exp),max_error=err))
protected=json.loads((r/'evidence/protected_sha256.json').read_text());changed=[p for p,h in protected.items() if not (ROOT/p).exists() or sha256(ROOT/p)!=h];assert not changed,changed
ex={k:json.loads((r/'evaluation'/k/'execution.json').read_text()) for k in ctrls};assert all(x['success'] for x in ex.values());assert ex['RL']['env_config']==ex['PID']['env_config'];assert json.loads((r/'evaluation/RL/engine_properties.json').read_text())==json.loads((r/'evaluation/PID/engine_properties.json').read_text());assert json.loads((r/'evaluation/RL/initial_state.json').read_text())==json.loads((r/'evaluation/PID/initial_state.json').read_text())
prefault={}
for k in ctrls:
 healthy=rows(k,'healthy');ref=np.array([[float(z[f]) for f in ['ex','ey','ez','wx','wy','wz']] for z in healthy[:500]]);errors=[]
 for c in plan['cases'][1:]:
  q=rows(k,c['name']);val=np.array([[float(z[f]) for f in ['ex','ey','ez','wx','wy','wz']] for z in q[:500]]);assert val.shape==ref.shape;errors.append(dict(case=c['name'],position_max_error_m=float(np.abs(val[:,:3]-ref[:,:3]).max()),omega_max_error_rad_s=float(np.abs(val[:,3:]-ref[:,3:]).max())))
 assert max(x['position_max_error_m'] for x in errors)<1e-4 and max(x['omega_max_error_rad_s'] for x in errors)<.01;prefault[k]=errors
verify=dict(success=True,physics=json.loads((r/'physics_verification.json').read_text()),PID=json.loads((r/'pid_verification.json').read_text()),reload=json.loads((r/'evaluation/RL/reload.json').read_text()),same_env_config=True,same_initial_state=True,same_engine_properties=True,prefault_same_controller_comparison=prefault,scalar_checks=tb,protected_files_checked=len(protected),protected_files_changed=changed,executions=ex,optimizer_updates=0,GUI_verified=False)
dump_json(r/'verification.json',verify)
overview={}
for k in ctrls:
 overview[k]={}
 for eta in [.9,.8,.7]:
  xx=[x for x in results[k] if x['motor_user'] is not None and x['efficiency']==eta];good=[x for x in xx if x['completed']];overview[k][str(eta)]=dict(completed=len(good),count=4,mean_post_position_rmse_cm=float(np.mean([x['response']['postfault']['position_rmse_m'] for x in good]))*100 if good else None,max_post_position_rmse_cm=max(x['response']['postfault']['position_rmse_m'] for x in good)*100 if good else None,mean_tail_omega_rms_rad_s=float(np.mean([x['response']['tail20_30']['omega_rms_rad_s'] for x in good])) if good else None,mean_tail_position_rmse_cm=float(np.mean([x['response']['tail20_30']['position_rmse_m'] for x in good]))*100 if good else None)
dump_json(r/'overview.json',overview);print(json.dumps(overview,indent=2),flush=True)

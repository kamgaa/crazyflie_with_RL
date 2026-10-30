"""Read-only paired metrics, implementation audit, TensorBoard checks and plots."""
import csv,json,sys,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ppo_common import ROOT,dump_json,sha256
from external_integral_metrics import segment
r=Path(sys.argv[1]);plan=json.loads((r/'experiment_plan.json').read_text());plots=r/'plots';labels={'off':'PPO 단독 (OFF)','on':'PPO + 외부 적분 (ON)'};cache={};results={};checks={};errors={};scalar_checks={}
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':9,'axes.grid':True,'grid.alpha':.25})
def readcsv(p):
 with p.open() as f:
  rows=list(csv.DictReader(f))
 return [{k:(v=='True' if v in ['True','False'] else float(v)) for k,v in z.items()} for z in rows]
def arr(rows,keys):return np.array([[z[k] for k in keys] for z in rows],dtype=float)
def wc(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check(k,v):
 checks[k]=bool(v)
 if not checks[k]:raise AssertionError(k)
flat=[]
for task in ['hover','T10','T5']:
 for mode in ['off','on']:
  d=r/'evaluation'/mode/task;x=json.loads((d/'result.json').read_text());q=readcsv(d/'trajectory.csv');cache[mode,task]=q;results[mode,task]=x
  check(mode+'_'+task+'_execution',x['success']);check(mode+'_'+task+'_correct_checkpoint',x['checkpoint_sha256']==plan['checkpoint_sha256']);check(mode+'_'+task+'_optimizer0',x['optimizer_updates']==0)
  used=arr(q,['xi_used_'+a for a in 'xyz']);e=arr(q,['e_actual_'+a for a in 'xyz']);candidate=used-.002*e;projected=candidate.copy();projected[:,:2]*=.4/np.maximum(np.linalg.norm(projected[:,:2],axis=1,keepdims=True),.4);projected[:,2]=np.clip(projected[:,2],-.15,.15)
  al=arr(q,[f'allocation_{i}' for i in range(4)]);clipped=((al<0)|(al>.289)).any(1);expected=np.where(((~clipped)&(mode=='on'))[:,None],projected,used)
  done=np.array([z['episode_ended'] for z in q]);next_state=np.where(done[:,None],0,expected)
  observed=arr(q,['xi_next_'+a for a in 'xyz']);obs_end=arr(q,['xi_interval_end_'+a for a in 'xyz'])
  er=float(np.max(np.abs(next_state-observed)));errors[mode+'_'+task+'_integral_formula_max_m']=er
  check(mode+'_'+task+'_integral_formula',er<2e-7 and np.max(np.abs(expected-obs_end))<2e-7)
  check(mode+'_'+task+'_allocator_freeze_exact',np.array_equal(clipped,np.array([z['allocator_clipped'] for z in q])))
  check(mode+'_'+task+'_xy_z_bounds',np.max(np.linalg.norm(obs_end[:,:2],axis=1))<=.4+1e-6 and np.max(np.abs(obs_end[:,2]))<=.15+1e-6)
  check(mode+'_'+task+'_policy_error',np.max(np.abs(e-used-arr(q,['e_policy_'+a for a in 'xyz'])))<1e-7)
  check(mode+'_'+task+'_cross_step_state',len(q)<2 or np.max(np.abs(observed[:-1]-used[1:]))<1e-7)
  check(mode+'_'+task+'_pre_error_timing',len(q)<2 or np.max(np.abs(e[1:]-arr(q,['ex','ey','ez'])[:-1]))<1e-7)
  # Independent NumPy reward reconstruction from actual end-of-step state.
  ep=arr(q,['ex','ey','ez']);vel=arr(q,['vx','vy','vz']);vd=-4*ep;vd*=1.5/np.maximum(np.linalg.norm(vd,axis=1,keepdims=True),1.5)
  ang=arr(q,['roll_rad','pitch_rad','yaw_rad']);om=arr(q,['wx','wy','wz'])
  reward=.1*np.expm1(-ep[:,:2]**2/.1**2).sum(1)+.015*np.expm1(-ep[:,2]**2/.05**2)-.005*((vel[:,:2]-vd[:,:2])**2).sum(1)-.015*vel[:,2]**2-.075*(ang[:,:2]**2).sum(1)-ang[:,2]**2-.001*(om**2).sum(1)-10*np.array([z['terminated'] for z in q])
  er=float(np.max(np.abs(reward-arr(q,['reward'])[:,0])));errors[mode+'_'+task+'_actual_reward_max']=er;check(mode+'_'+task+'_reward_actual_target',er<2e-5)
  ea=EventAccumulator(str(d/'tensorboard_time'),size_guidance={'scalars':0});ea.Reload();logged=json.loads((d/'tensorboard_scalars.json').read_text());lookup={(tag,z.step):z.value for tag in ea.Tags()['scalars'] for z in ea.Scalars(tag)}
  er=max(abs(lookup[z['tag'],z['step']]-z['value']) for z in logged);scalar_checks[mode+'_'+task]=dict(tags=len(ea.Tags()['scalars']),scalars=len(logged),max_error=er);check(mode+'_'+task+'_tensorboard_json',er==0 and len(lookup)==len(logged))
  check(mode+'_'+task+'_tensorboard_csv',all(abs(z['value']-float(np.float32(q[int(z['step'])-1][z['tag'].split('/',1)[1]])))==0 for z in logged))
  m=x['metrics'];z=dict(task=task,controller=labels[mode],mode=mode,completed=m['completed'],duration_s=m['duration_s'],reason=';'.join(m['termination_reasons']))
  for seg in ['full','tail']:
   for k in ['position_rmse_m','body_omega_rms_rad_s','max_tilt_deg','action_clipping_fraction','allocation_clipping_fraction','integral_frozen_fraction','integral_xy_limit_fraction','integral_z_limit_fraction']:
    z[seg+'_'+k]=None if m[seg] is None else m[seg][k]
   z[seg+'_xyz_rmse_m']=None if m[seg] is None else json.dumps(m[seg]['xyz_rmse_m']);z[seg+'_bias_xyz_m']=None if m[seg] is None else json.dumps(m[seg]['bias_xyz_m']);z[seg+'_std_xyz_m']=None if m[seg] is None else json.dumps(m[seg]['std_xyz_m'])
   for k in ['radial_rmse_m','phase_lag_mean_deg']:z[seg+'_'+k]=None if m[seg] is None else m[seg].get(k)
  z['final_xi_before_reset_m']=json.dumps(m['final_xi_before_episode_reset_m']);flat.append(z)
 for f in ['initial_state.json','engine_properties.json','resolved_config.json']:
  check(task+'_'+f+'_same',json.loads((r/'evaluation/off'/task/f).read_text())==json.loads((r/'evaluation/on'/task/f).read_text()))
 check(task+'_same_first_action',np.max(np.abs(arr(cache['off',task][:1],[f'raw_action_{i}' for i in range(4)])-arr(cache['on',task][:1],[f'raw_action_{i}' for i in range(4)])))==0)
wc(r/'summary.csv',flat);dump_json(r/'summary.json',flat)
common={}
for task in ['hover','T10','T5']:
 n=min(len(cache['off',task]),len(cache['on',task]));common[task]={mode:segment(cache[mode,task][:n],task!='hover') for mode in ['off','on']}
dump_json(r/'common_interval_comparison.json',common)
# Compare OFF with the existing phase14 recordings (not demanding bitwise trajectory identity).
src=Path(plan['source_run']);hist={}
for task in ['hover','T10','T5']:
 old=src/'evaluation/best_fixed_30s/best/nominal.csv' if task=='hover' else src/'circle/cases'/f'phase14z_best_{task}'/'trajectory.csv'
 a=readcsv(old);b=cache['off',task];n=min(len(a),len(b));ep=float(np.max(np.abs(arr(a[:n],['ex','ey','ez'])-arr(b[:n],['ex','ey','ez']))));ew=float(np.max(np.abs(arr(a[:n],['wx','wy','wz'])-arr(b[:n],['wx','wy','wz']))))
 tol=plan['comparison_tolerances'];hist[task]=dict(source=str(old),sha256=sha256(old),samples_before=len(a),samples_now=len(b),position_max_abs_m=ep,body_omega_max_abs_rad_s=ew,passed=len(a)==len(b) and ep<tol['off_historical_position_max_m'] and ew<tol['off_historical_body_omega_max_rad_s'])
 check(task+'_off_reproduces_historical',hist[task]['passed'])
dump_json(r/'off_historical_comparison.json',hist)
protected=json.loads((r/'evidence/protected_sha256.json').read_text());changed=[k for k,v in protected.items() if sha256(ROOT/k)!=v];check('existing_files_unchanged',not changed)
verification=dict(success=all(checks.values()),checks=checks,errors=errors,scalars=scalar_checks,protected_count=len(protected),protected_changed=changed,historical_comparison=hist,optimizer_updates=0,GUI_verified=False)
dump_json(r/'verification.json',verification)
# Shared-axis figures and matching source CSV.
def save(fig,name,task):
 fig.tight_layout(rect=[0,0,1,.965],h_pad=1.7);fig.savefig(plots/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
 paired=[dict(mode=mode,**z) for mode in ['off','on'] for z in cache[mode,task]];wc(plots/(name+'.csv'),paired)
def title(task,extra=''):
 return f'phase14 best448 동결 | PPO 단독 vs PPO + 외부 적분 | {task} {extra}'
def lines(ax,task,keys,unit,start=0,control=False):
 colors=plt.get_cmap('tab10').colors
 for mode in ['off','on']:
  q=[z for z in cache[mode,task] if z['time_s']>start-1e-9];t=[z['control_time_s'] if control else z['time_s'] for z in q]
  for i,key in enumerate(keys):ax.plot(t,[z[key] for z in q],color=colors[i],ls='-' if mode=='off' else '--',lw=1,label=f'{mode.upper()} {key}')
 ax.set_ylabel(unit);ax.set_xlabel('시간 [s]');ax.legend(fontsize=7,ncol=2)
fig,axs=plt.subplots(3,2,figsize=(13,10));fig.suptitle(title('hover','0~30초'))
for i,x in enumerate('xyz'):
 lines(axs[i,0],'hover',['e'+x],'실제 '+x+' 오차 [m]');lines(axs[i,1],'hover',['xi_used_'+x],'사용 xi_'+x+' [m]',control=True)
save(fig,'01_hover_error_integral','hover')
for j,task in enumerate(['T10','T5']):
 fig,ax=plt.subplots(figsize=(8,8));theta=np.linspace(0,2*np.pi,500);ax.plot(np.cos(theta),np.sin(theta),'k:',label='실제 목표 원')
 for mode in ['off','on']:q=cache[mode,task];ax.plot([z['x'] for z in q],[z['y'] for z in q],label=labels[mode]);ax.scatter(q[0]['x'],q[0]['y'],s=20)
 ax.set_aspect('equal');ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.legend();fig.suptitle(title(task,'전체 XY 경로'));save(fig,f'{2+j:02d}_{task}_xy_path',task)
 fig,axs=plt.subplots(3,1,figsize=(12,9));fig.suptitle(title(task,'실제 위치와 실제 목표, 전체 구간'))
 for i,x in enumerate('xyz'):
  lines(axs[i],task,[x],x+' [m]');q=cache['off',task];axs[i].plot([z['time_s'] for z in q],[z['target_'+x] for z in q],'k:',label='실제 목표');axs[i].legend()
 save(fig,f'{4+j:02d}_{task}_xyz_reference',task)
 fig,axs=plt.subplots(3,1,figsize=(12,9));fig.suptitle(title(task,'실제 목표 오차·반경·위상'))
 lines(axs[0],task,['ex','ey','ez'],'위치 오차 [m]');lines(axs[1],task,['radial_error_m'],'반경 오차 [m]');lines(axs[2],task,['phase_lag_deg'],'위상 지연 [deg]');save(fig,f'{6+j:02d}_{task}_errors_phase',task)
 fig,axs=plt.subplots(3,3,figsize=(16,11));fig.suptitle(title(task,'제어 시각 t_k: 적분·가상 목표·정책 입력'))
 for i,x in enumerate('xyz'):
  lines(axs[i,0],task,['xi_used_'+x],'xi_'+x+' [m]',control=True)
  q=cache['on',task];t=[z['control_time_s'] for z in q]
  axs[i,1].plot(t,[z['control_target_'+x] for z in q],'k:',label='실제 목표');axs[i,1].plot(t,[z['virtual_target_'+x] for z in q],label='ON 가상 목표');axs[i,1].set_ylabel(x+' [m]');axs[i,1].legend();axs[i,1].set_xlabel('시간 [s]')
  axs[i,2].plot(t,[z['e_actual_'+x] for z in q],label='ON 실제 오차');axs[i,2].plot(t,[z['e_policy_'+x] for z in q],label='ON 정책 입력 오차');axs[i,2].set_ylabel(x+' 오차 [m]');axs[i,2].legend();axs[i,2].set_xlabel('시간 [s]')
 save(fig,f'{8+j:02d}_{task}_integral_virtual_policy',task)
for j,task in enumerate(['hover','T10','T5']):
 horizon=next(x['horizon_s'] for x in plan['tasks'] if x['name']==task)
 for zoom in ([False] if task=='hover' else [False,True]):
  start=horizon-1 if zoom else 0;fig,axs=plt.subplots(5,2,figsize=(16,16));fig.suptitle(title(task,f'{start:g}~{horizon:g}초 자세·행동·wrench·모터'))
  fields=[(['roll_deg','pitch_deg'],'Euler 각도 [deg]'),(['wx','wy','wz'],'body 각속도 [rad/s]'),([f'raw_action_{i}' for i in range(4)],'raw action [정규화]'),([f'action_{i}' for i in range(4)],'applied action [정규화]'),([f'request_{i}' for i in range(3)],'요청 토크 [Nm]'),([f'actual_{i}' for i in range(3)],'실제 토크 [Nm]'),(['request_3','actual_3'],'합추력 [N]'),([f'thrust_{i}' for i in range(4)],'로터 추력 [N]'),([f'motor_rpm_{i}' for i in range(4)],'로터 속도 [RPM]'),(['tilt_deg'],'기울기 [deg]')]
  for ax,(keys,unit) in zip(axs.flat,fields):lines(ax,task,keys,unit,start)
  name=f'{10+j:02d}_{task}_dynamics' if not zoom else f'{12+j:02d}_{task}_last1s_dynamics'
  save(fig,name,task)
print('ANALYSIS_OK',len(checks),json.dumps(hist),flush=True)

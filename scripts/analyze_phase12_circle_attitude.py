"""Offline position/velocity-derived attitude and vertical thrust; no simulator/training."""
import argparse,csv,hashlib,importlib.util,json,math,shutil,sys,tarfile,textwrap
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def writecsv(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def euler(R):
 return np.stack((np.arctan2(R[...,2,1],R[...,2,2]),np.arcsin(np.clip(-R[...,2,0],-1,1)),np.arctan2(R[...,1,0],R[...,0,0])),axis=-1)
def matrix(q):
 q=np.asarray(q,dtype=np.float64);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
 w,x,y,z=np.moveaxis(q,-1,0)
 return np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),axis=-1).reshape(q.shape[:-1]+(3,3))
def independent_outer(pos,vel,target,I,cfg,dt):
 s=cfg['settings'];v=s['kp_position']*(target-pos);v*=min(1.,s['velocity_limit']/max(np.linalg.norm(v),1e-30));ve=v-vel
 I=np.clip(I+dt*ve,-s['integrator_limit'],s['integrator_limit'])
 a=s['kp_velocity']*ve+s['ki_velocity']*I;F=cfg['vehicle']['mass']*(a+np.array([0,0,cfg['vehicle']['gravity']]))
 bz=F/max(np.linalg.norm(F),1e-30) if np.linalg.norm(F)>1e-6 else np.array([0.,0.,1.])
 tilt=math.acos(np.clip(bz[2],-1,1));limit=math.radians(s['max_tilt_deg'])
 if tilt>limit:
  xy=bz[:2];bz=np.r_[math.sin(limit)*xy/np.linalg.norm(xy),math.cos(limit)]
 by=np.cross(bz,[1.,0.,0.]);by/=max(np.linalg.norm(by),1e-6);bx=np.cross(by,bz)
 return v,I,a,F,np.column_stack((bx,by,bz))

p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output');args=p.parse_args()
r=Path(args.run_dir).resolve();out=Path(args.output).resolve() if args.output else r/('attitude_vertical_thrust_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
assert not out.exists(),out
old_files={str(p.relative_to(r)):sha(p) for p in r.rglob('*') if p.is_file()}
plan=read(r/'experiment_plan.json');pidcfg=plan['pid'];dt=float(plan['control_dt_s']);assert dt==.01
controller=Path(plan['historical_run'])/'source_snapshot/original_controllers.py'
assert sha(controller)==pidcfg['sha256']
spec=importlib.util.spec_from_file_location('original_circle_pid',controller);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
constants_path=ROOT/'src/crazyflie_isaaclab/tasks/crazyflie/constants.py'
cs=importlib.util.spec_from_file_location('circle_constants',constants_path);const=importlib.util.module_from_spec(cs);cs.loader.exec_module(const)
assert const.TOTAL_MASS==.043384 and const.GRAVITY==9.81
mass=const.TOTAL_MASS;g=const.GRAVITY;mg=mass*g
out.mkdir(parents=True)
for d in ['plots','data','source_snapshot']: (out/d).mkdir()
shutil.copy2(controller,out/'source_snapshot/original_controllers.py');shutil.copy2(constants_path,out/'source_snapshot/constants.py')
shutil.copy2(__file__,out/'source_snapshot'/Path(__file__).name)
checks={'pid_source_sha256_matches_previous_protocol':True,'no_simulation_or_optimizer_updates':True}
# Analytic projection tests for known quaternions.
quats=np.array([[1,0,0,0],[math.sqrt(.5),math.sqrt(.5),0,0],[math.sqrt(.5),0,math.sqrt(.5),0],[math.sqrt(.5),0,0,math.sqrt(.5)]])
checks['identity_and_90deg_projection']=bool(np.allclose(matrix(quats)[:,2,2],[1,0,0,1],atol=1e-14))
zero=np.zeros(3);R0=independent_outer(zero,zero,zero,zero,pidcfg,dt)[-1]
Rp=independent_outer(np.array([.05,0,0]),zero,zero,zero,pidcfg,dt)[-1]
Ry=independent_outer(np.array([0,.05,0]),zero,zero,zero,pidcfg,dt)[-1]
checks['neutral_reference_is_level']=bool(np.allclose(R0,np.eye(3)))
checks['FLU_return_attitude_signs']=bool(euler(Rp)[1]<0 and euler(Ry)[0]>0)
all_series={};summaries=[];details={}
for period in [10,5]:
 case=r/'cases'/f'phase12_best_T{period}'
 data=read(case/'trajectory.json');rows=data['trajectory'];res=read(case/'result.json')
 A=lambda keys:np.array([[row[k] for k in keys] for row in rows],dtype=np.float64)
 t=A(['time_s'])[:,0];pos=A(['x','y','z']);vel=A(['vx','vy','vz']);target=A(['target_x','target_y','target_z']);q=A(['qw','qx','qy','qz']);omega=A(['wx','wy','wz'])
 actual_R=matrix(q);actual_e=euler(actual_R);force=A(['actual_3'])[:,0];sumforce=A([f'thrust_{i}' for i in range(4)]).sum(1)
 factor=actual_R[:,2,2];vertical=force*factor;projection_euler=force*np.cos(actual_e[:,0])*np.cos(actual_e[:,1])
 pid=mod.CascadePID(dt,vehicle=SimpleNamespace(**pidcfg['vehicle']),settings=SimpleNamespace(**pidcfg['settings']));pid.reset();captured={}
 wanted=['velocity_setpoint','velocity_error','acceleration_setpoint','desired_force','desired_rotation','collective_force','tilt']
 def trace(frame,event,arg):
  if event=='return' and frame.f_code is mod.CascadePID.__call__.__code__:
   captured.clear()
   for key in wanted:
    value=frame.f_locals[key];captured[key]=value.copy() if isinstance(value,np.ndarray) else value
   captured['integral']=frame.f_locals['self']._i_vel.copy()
 old_profile=sys.getprofile();sys.setprofile(trace)
 refs=[];I=np.zeros(3);max_reference_residual=0.
 try:
  # The original circle starts exactly at its first target, at rest, with zero integrator.
  init=res['initial'];io=np.array(init['observation']);ip=np.array(init['position'])-np.array(init['env_origin'])
  pid(ip,io[6:10],io[3:6],io[10:13],np.array(plan['initial_position_local_m']),0.)
  assert np.allclose(pid._i_vel,0)
  for k in range(len(rows)):
   pid(pos[k],q[k]/np.linalg.norm(q[k]),vel[k],omega[k],target[k],0.)
   v,I,a,F,refR=independent_outer(pos[k],vel[k],target[k],I,pidcfg,dt)
   for expected,observed in [(v,captured['velocity_setpoint']),(I,captured['integral']),(a,captured['acceleration_setpoint']),(F,captured['desired_force']),(refR,captured['desired_rotation'])]:
    max_reference_residual=max(max_reference_residual,float(np.max(np.abs(expected-observed))))
   refs.append({key:(value.copy() if isinstance(value,np.ndarray) else value) for key,value in captured.items()})
 finally:sys.setprofile(old_profile)
 ref_R=np.stack([x['desired_rotation'] for x in refs]);ref_e=euler(ref_R);vdes=np.stack([x['velocity_setpoint'] for x in refs]);aref=np.stack([x['acceleration_setpoint'] for x in refs]);fref=np.stack([x['desired_force'] for x in refs]);integral=np.stack([x['integral'] for x in refs])
 series=[]
 for k,row in enumerate(rows):
  item=dict(time_s=float(t[k]),step=int(row['step']),actual_total_thrust_N=float(force[k]),rotor_thrust_sum_N=float(sumforce[k]),requested_collective_thrust_N=float(row['request_3']),cos_roll_cos_pitch=float(factor[k]),actual_vertical_thrust_N=float(vertical[k]),weight_total_mass_N=mg,vertical_thrust_minus_weight_N=float(vertical[k]-mg),thrust_projection_loss_N=float(force[k]-vertical[k]),level_equilibrium_total_thrust_at_actual_tilt_N=float(mg/factor[k]),reference_unlimited_tilt_deg=math.degrees(refs[k]['tilt']),reference_tilt_limit_active=bool(refs[k]['tilt']>pid.max_tilt))
  for i,axis in enumerate('xyz'):
   item.update({f'position_{axis}_m':float(pos[k,i]),f'target_{axis}_m':float(target[k,i]),f'error_{axis}_m':float(pos[k,i]-target[k,i]),f'velocity_{axis}_m_s':float(vel[k,i]),f'return_velocity_{axis}_m_s':float(vdes[k,i]),f'velocity_PI_integral_{axis}_m':float(integral[k,i]),f'reference_acceleration_{axis}_m_s2':float(aref[k,i]),f'reference_force_{axis}_N':float(fref[k,i])})
  for i,axis in enumerate(['roll','pitch','yaw']):
   item[f'actual_{axis}_rad']=float(actual_e[k,i]);item[f'actual_{axis}_deg']=math.degrees(actual_e[k,i]);item[f'reference_{axis}_rad']=float(ref_e[k,i]);item[f'reference_{axis}_deg']=math.degrees(ref_e[k,i])
  series.append(item)
 all_series[period]=series
 writecsv(out/'data'/f'attitude_vertical_thrust_T{period}.csv',series);dump(out/'data'/f'attitude_vertical_thrust_T{period}.json',series)
 with (out/'data'/f'attitude_vertical_thrust_T{period}.csv').open() as f:
  cr=list(csv.DictReader(f));csv_error=max(abs(float(cr[k][field])-value) for k,item in enumerate(series) for field,value in item.items() if not isinstance(value,bool))
 metrics=dict(period_s=period,duration_s=float(t[-1]),samples=len(rows),actual_vertical_thrust_mean_N=float(vertical.mean()),actual_vertical_thrust_min_N=float(vertical.min()),actual_vertical_thrust_max_N=float(vertical.max()),actual_total_thrust_mean_N=float(force.mean()),weight_total_mass_N=mg,vertical_thrust_below_weight_fraction=float(np.mean(vertical<mg)),projection_loss_mean_N=float(np.mean(force-vertical)),projection_loss_max_N=float(np.max(force-vertical)),actual_roll_rms_deg=float(np.sqrt(np.mean(np.degrees(actual_e[:,0])**2))),actual_pitch_rms_deg=float(np.sqrt(np.mean(np.degrees(actual_e[:,1])**2))),reference_roll_rms_deg=float(np.sqrt(np.mean(np.degrees(ref_e[:,0])**2))),reference_pitch_rms_deg=float(np.sqrt(np.mean(np.degrees(ref_e[:,1])**2))),reference_vs_actual_roll_rmse_deg=float(np.sqrt(np.mean(np.degrees(ref_e[:,0]-actual_e[:,0])**2))),reference_vs_actual_pitch_rmse_deg=float(np.sqrt(np.mean(np.degrees(ref_e[:,1]-actual_e[:,1])**2))),reference_tilt_limit_fraction=float(np.mean([x['tilt']>pid.max_tilt for x in refs])),z_error_max_abs_m=float(np.max(np.abs(pos[:,2]-target[:,2]))))
 summaries.append(metrics)
 details[str(period)]=dict(reference_vs_independent_equations_max_abs=max_reference_residual,projection_identity_max_abs_N=float(np.max(np.abs(vertical-projection_euler))),rotor_sum_vs_actual_wrench_max_abs_N=float(np.max(np.abs(sumforce-force))),logged_roll_pitch_vs_normalized_quaternion_max_abs_rad=float(np.max(np.abs(actual_e[:,:2]-A(['roll_rad','pitch_rad'])))),csv_json_max_abs=csv_error,quaternion_norm_max_deviation=float(np.max(np.abs(np.linalg.norm(q,axis=1)-1))))
 checks[f'T{period}_reference_matches_original_PID']=max_reference_residual<1e-12
 checks[f'T{period}_vertical_projection_identity']=details[str(period)]['projection_identity_max_abs_N']<1e-12
 checks[f'T{period}_actual_wrench_matches_rotor_sum']=details[str(period)]['rotor_sum_vs_actual_wrench_max_abs_N']<2e-7
 checks[f'T{period}_normalized_euler_matches_logged']=details[str(period)]['logged_roll_pitch_vs_normalized_quaternion_max_abs_rad']<1e-6
 checks[f'T{period}_100Hz_samples']=bool(np.allclose(np.diff(t),dt,atol=1e-12))
 checks[f'T{period}_finite']=all(math.isfinite(value) for item in series for value in item.values())
 checks[f'T{period}_reference_rotation_orthonormal']=bool(np.max(np.abs(np.swapaxes(ref_R,1,2)@ref_R-np.eye(3)))<1e-12 and np.allclose(np.linalg.det(ref_R),1,atol=1e-12))
 checks[f'T{period}_csv_json_exact']=csv_error==0
 checks[f'T{period}_no_autoreset_contamination']=not any(x['terminated'] or x['truncated'] for x in rows[:-1])

plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'axes.grid':True,'grid.alpha':.25})
colors={'reference':'#D55E00','actual':'#0072B2','thrust':'#009E73','weight':'#333333'}
def vals(seq,key):return np.array([x[key] for x in seq])
def save(fig,name):
 fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'plots'/name,dpi=180);plt.close(fig)
for period in [10,5]:
 rows=all_series[period];t=vals(rows,'time_s')
 for suffix,start,number in [('full',0,1 if period==10 else 2),('last_lap',period,3 if period==10 else 4)]:
  fig,axs=plt.subplots(4,1,figsize=(13,12),sharex=True)
  for ax,axis in zip(axs[:2],['roll','pitch']):
   ax.plot(t,vals(rows,f'reference_{axis}_deg'),'--',color=colors['reference'],label='위치·속도로 사후 계산한 PID 참고 목표')
   ax.plot(t,vals(rows,f'actual_{axis}_deg'),color=colors['actual'],label='실제 RL 자세')
   ax.set_ylabel(axis+' (deg)');ax.legend(loc='upper right',fontsize=9)
  ax=axs[2];ax.plot(t,vals(rows,'actual_total_thrust_N'),color=colors['thrust'],alpha=.8,label='실제 총추력 T')
  ax.plot(t,vals(rows,'actual_vertical_thrust_N'),color=colors['actual'],label='수직 성분 T cos(roll) cos(pitch)')
  ax.axhline(mg,color=colors['weight'],ls=':',label=f'전체 질량의 무게 mg = {mg:.8f} N');ax.set_ylabel('추력 (N)');ax.legend(loc='upper right',fontsize=9)
  ax=axs[3];l1=ax.plot(t,vals(rows,'error_z_m')*100,color=colors['actual'],label='Z 위치오차');ax.set_ylabel('Z 오차 (cm)',color=colors['actual']);ax.axhline(0,color='gray',lw=.6)
  twin=ax.twinx();l2=twin.plot(t,vals(rows,'velocity_z_m_s')*100,color=colors['reference'],alpha=.8,label='실제 수직 속도');twin.set_ylabel('Z 속도 (cm/s)',color=colors['reference']);twin.grid(False)
  ax.legend(l1+l2,[x.get_label() for x in l1+l2],loc='upper right',fontsize=9)
  for ax in axs:ax.set_xlim(start,2*period)
  if suffix=='last_lap':
   mask=t>=start
   def zoom_y(ax,arrays):
    yy=np.concatenate([np.ravel(x) for x in arrays]);lo=float(yy.min());hi=float(yy.max());pad=max((hi-lo)*.12,1e-8);ax.set_ylim(lo-pad,hi+pad)
   for ax,axis in zip(axs[:2],['roll','pitch']):zoom_y(ax,[vals(rows,f'reference_{axis}_deg')[mask],vals(rows,f'actual_{axis}_deg')[mask]])
   zoom_y(axs[2],[vals(rows,'actual_total_thrust_N')[mask],vals(rows,'actual_vertical_thrust_N')[mask],np.array([mg])])
   zoom_y(axs[3],[vals(rows,'error_z_m')[mask]*100,np.array([0.])]);zoom_y(twin,[vals(rows,'velocity_z_m_s')[mask]*100])
  axs[-1].set_xlabel('simulation time (s)')
  fig.suptitle(f'{plan['branch']} best {plan['selected_iteration']}회 · 반지름 1m / 주기 {period}s · {start}–{2*period}s\n참고 목표는 기존 PID 외부루프의 사후 계산값이며 RL 내부 목표가 아님',fontsize=14)
  save(fig,f'{number:02d}_attitude_vertical_thrust_T{period}_{suffix}.png')
 # Detailed vertical balance, with no differentiation of sampled velocity.
 fig,axs=plt.subplots(3,1,figsize=(13,10),sharex=True)
 axs[0].plot(t,vals(rows,'actual_total_thrust_N'),label='실제 총추력 T',color=colors['thrust'])
 axs[0].plot(t,vals(rows,'level_equilibrium_total_thrust_at_actual_tilt_N'),'--',label='수직 평형 필요 총추력 mg / (cos roll cos pitch)',color=colors['reference']);axs[0].set_ylabel('총추력 (N)')
 axs[1].plot(t,vals(rows,'vertical_thrust_minus_weight_N')*1000,label='T cos roll cos pitch − mg',color=colors['actual']);axs[1].axhline(0,color='black',lw=.7);axs[1].set_ylabel('수직 추력 − 무게 (mN)')
 axs[2].plot(t,vals(rows,'thrust_projection_loss_N')*1000,label='기울기로 줄어든 수직 성분: T − T cos roll cos pitch',color=colors['reference']);axs[2].set_ylabel('수직 성분 감소 (mN)');axs[2].set_xlabel('simulation time (s)')
 for ax in axs:ax.set_xlim(0,2*period);ax.legend(fontsize=9)
 fig.suptitle(f'{plan['branch']} best {plan['selected_iteration']}회 · 주기 {period}s · 실제 모터 추력 / 수직 투영 / 중력\n100Hz 저장 표본, 요청 추력 및 500Hz 구간 평균과 구분',fontsize=14)
 save(fig,f'{5 if period==10 else 6:02d}_vertical_balance_T{period}.png')
writecsv(out/'summary.csv',summaries);dump(out/'summary.json',summaries)
checks['all_previous_circle_files_unchanged']=all((r/key).is_file() and sha(r/key)==value for key,value in old_files.items())
verification=dict(success=all(checks.values()),checks=checks,details=details,simulation_runs=0,optimizer_updates=0)
dump(out/'verification.json',verification);assert verification['success'],verification
metadata=dict(source_run=str(r),output=str(out),source_checkpoint=plan['models']['phase12_best'],pid_source=str(controller),pid_source_sha256=sha(controller),pid_parameters=pidcfg,source_files_sha256={str(case.relative_to(r)):old_files[str(case.relative_to(r))] for T in [10,5] for case in [r/f'cases/phase12_best_T{T}/trajectory.json',r/f'cases/phase12_best_T{T}/result.json']},total_mass_kg=mass,gravity_m_s2=g,weight_N=mg,sampling_hz=100,reference_description='Original CascadePID outer-loop replayed on actual RL states; initial velocity integral zero, original velocity PI integral carried along saved RL trajectory. Counterfactual diagnostic, not a recorded RL setpoint and not a new PID flight.',reference_timing='At state t and target t; sampled actual thrust is last physics-substep force buffer with post-policy-step attitude. No next reset state. Not exact 500Hz force history or interval average.',gui_executed=False,original_files_count_preserved=len(old_files))
dump(out/'analysis_config.json',metadata)
report=['Phase12 원 궤적: 위치·속도로 계산한 참고 목표 자세와 실제 자세 및 수직 추력',str(out),'',
'분석 범위: 해당 phase12 best의 T10(20초), T5(10초) 원 비행 기록만 후처리. 시뮬레이션 재실행/학습/optimizer update 없음. 기존 파일/모델/압축파일을 변경하지 않았다.',
'주의: RL은 4D wrench를 직접 출력한다. RL 내부에는 별도의 목표 roll/pitch가 없으므로 이 그래프의 목표 자세는 기존 CascadePID 위치/속도 외부루프를 RL의 저장 상태에 적용한 사후 참고값이다. 이를 정책이 명시적으로 명령한 자세 또는 실제 PID 비행의 자세라고 해석하지 않는다.',
'', '계산식 및 단위:',
'e = p_link_world - env_origin - p_ref(t) [m]',
'v_ref = norm_clip(-4 e, 1.5) [m/s]. 원의 접선 목표속도/가속도 feedforward는 추가하지 않는다.',
'I_k = clip(I_(k-1) + 0.01 * (v_ref - v_world), -2, +2), 축별 제한. 초기 I=0. 기존 PID와 같은 속도 PI이며 현재 위치/속도뿐 아니라 그 오차 이력이 포함된다.',
'a_ref = 4*(v_ref-v_world) + 1*I [m/s^2]. D항 없음.',
'F_ref = 0.04338 * (a_ref + 9.81*[0,0,1]) [N]. 이 질량은 기존 PID의 본체 질량이다.',
'목표 body Z축 = F_ref/||F_ref||, 기존 35도 기울기 제한 적용. heading=[1,0,0], bodyY=normalize(bodyZ cross heading), bodyX=bodyY cross bodyZ. R_ref의 열은 bodyX,Y,Z. 이 회전행렬에서 ZYX Euler roll/pitch/yaw를 계산한다.',
'기존 heading 기반 구성에서 목표 yaw heading=0과 Euler yaw=0은 모든 복합 기울기에서 완전히 같지는 않다. CSV에 reference_yaw도 제공한다. 기존 알고리즘을 변경하지 않았다.',
'실제 quaternion(wxyz)을 정규화하고 body-to-world 회전행렬을 구성한다. 실제 Euler 각도는 ZYX, 그래프 deg / CSV rad 및 deg 병기.',
'T = actual_3 [N]: 모터 RPM 지연/추력 다항식 이후 실제 rotor force의 합, PhysX body-force buffer 값. rotor thrust 합과 수치 비교. 요청 collective force 또는 action scale 출력이 아니다.',
'F_vertical = T*R[2,2] = T*cos(roll)*cos(pitch) [N]. yaw에 무관하다. T는 body +Z 방향 총추력이다.',
f'무게 비교는 실제 전체 질량 {mass} kg: mg={mg:.8f} N. 기존 중력 보상의 본체 질량 {const.BODY_MASS} kg과 구분한다.',
'F_vertical-mg는 중력을 뺀 수직 힘의 표본 재구성값이다. mg/(cos roll cos pitch)는 해당 자세에서 수직 평형에 필요한 총추력이며 새로운 제어 명령이 아니다.',
'', '표본/시점 한계:',
'원시 기록은 100Hz policy 종료 시점의 자세와 마지막 500Hz substep에 적용한 모터/힘 버퍼이다. 이 표본의 자세로 힘을 수직 투영했다. 500Hz pre-step 자세 전체가 저장되어 있지 않으므로 모든 적분 시점의 실제 world force나 10ms 평균 힘을 정확하게 복원한 것은 아니다.',
'목표 자세는 같은 저장 시각 t의 위치/속도와 p_ref(t)에서 계산한다. RL raw action은 직전 policy 시각에서 계산되어 적용되었으므로 목표 자세와 RL action 사이를 동일시하지 않는다.',
'평가 원점은 result.json의 env_origin을 따른다. e는 local link position-target이며 COM 위치와 바꾸지 않았다. 수직 힘 평형은 전체 강체 COM 기준이다.',
'원래 float32 상태가 JSON에 저장되었으며 후처리는 float64이다. 원래 데이터의 정밀도가 증가하는 것은 아니다. smoothing/시간 지연 보정 없이 원 표본을 그렸다.',
'', '요약:']
for s in summaries:
 report.append(f"T{s['period_s']}s: 수직 추력 평균 {s['actual_vertical_thrust_mean_N']:.8f}N, 범위 [{s['actual_vertical_thrust_min_N']:.8f},{s['actual_vertical_thrust_max_N']:.8f}]N, 평균 투영 감소 {s['projection_loss_mean_N']*1000:.4f}mN, 목표-실제 roll/pitch RMSE {s['reference_vs_actual_roll_rmse_deg']:.3f}/{s['reference_vs_actual_pitch_rmse_deg']:.3f}deg, PID 참고 기울기 제한 비율 {s['reference_tilt_limit_fraction']:.4f}.")
report+=['','위 RMSE는 선택한 기존 PID 참고식과의 차이이며 RL 내부 자세 추종 오차가 아니다. 수직 추력과 고도/속도를 함께 표시하지만 표본 상관만으로 고도 변화 원인을 확정하지 않는다.','검증: 원본 PID 구현의 내부 desired_rotation과 독립 계산식 일치, quaternion/90도 투영, Euler 곱과 Rzz 투영 일치, 실제 motor 합과 wrench 일치, CSV/JSON 일치, 기존 파일 SHA256 보존. 상세값 verification.json.','plots/: 01/02 전체 구간, 03/04 마지막 한 바퀴 구간, 05/06 수직 힘 평형. 모든 그래프의 대응 수치는 data/attitude_vertical_thrust_T10.csv 및 T5.csv.']
(out/'REPORT_attitude_vertical_thrust_ko.txt').write_text('\n'.join(report)+'\n')
(out/'COMMANDS.txt').write_text(f'cd {ROOT}\n/home/mrl_6534/miniconda3/envs/isaaclab/bin/python scripts/analyze_circle_attitude_vertical.py --run-dir {r}\n# Creates a new timestamped output directory. Offline only; no simulation or training.\n# Windows PowerShell\nscp "mrl_6534@100.79.103.47:{out}/review.tar.gz" .\n')
files=sorted(x for x in out.rglob('*') if x.is_file())
(out/'FILES.txt').write_text('\n'.join(str(x.relative_to(out)) for x in files)+'\nFILES.txt\nMANIFEST.sha256\n');files.append(out/'FILES.txt')
manifest={str(x.relative_to(out)):sha(x) for x in files};(out/'MANIFEST.sha256').write_text(''.join(v+'  '+k+'\n' for k,v in manifest.items()))
with tarfile.open(out/'review.tar.gz','w:gz') as f:
 for x in files+[out/'MANIFEST.sha256']:f.add(x,arcname=str(x.relative_to(out)),recursive=False)
with tarfile.open(out/'review.tar.gz','r:gz') as f:
 for key,value in manifest.items():assert hashlib.sha256(f.extractfile(key).read()).hexdigest()==value
h=sha(out/'review.tar.gz');(out/'review.tar.gz.sha256').write_text(h+'  review.tar.gz\n')
dump(out/'archive_verification.json',dict(success=True,files=len(manifest),sha256=h,bytes=(out/'review.tar.gz').stat().st_size))
print('OUTPUT',out,flush=True)
for s in summaries:
 print('PERIOD',s['period_s'])
 for k,v in s.items():print(k,v)
print('VERIFICATION',verification['success'])

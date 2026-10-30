"""Numerical checks, comparisons, TensorBoard, plots and Korean report; no training."""
import argparse,csv,json,math,time,datetime,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ppo_common import ROOT,sha256,dump_json,protected_hashes
from circle_eval_common import reference,segment_metrics
from phase8_evalmetrics import spectral
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();start=time.perf_counter()
plan=json.loads((r/'experiment_plan.json').read_text());finished=json.loads((r/'evaluation_finished.json').read_text())
assert finished['jobs']==6
out=r/'plots';out.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
def csvwrite(p,rows):
 if not rows:return
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
checks={};results={};trajectories={};summary=[];scalars=[]
for T in [10,5]:
 for name in plan['controllers']:
  key=f'{name}_T{T}';case=r/'cases'/key
  if not (case/'result.json').exists():checks[key+'_execution']=False;continue
  z=read(case/'result.json');rows=read(case/'trajectory.json')['trajectory'];results[key]=z;trajectories[key]=rows;m=z['metrics']
  checks[key+'_execution']=z['success'];checks[key+'_optimizer_zero']=z['optimizer_updates']==0
  checks[key+'_reference_time']=all(abs(x['time_s']-(i+1)*.01)<1e-10 for i,x in enumerate(rows))
  ref=reference(np.array([x['time_s'] for x in rows]),T)[0]
  target=np.array([[x['target_x'],x['target_y'],x['target_z']] for x in rows])
  checks[key+'_reference_path']=np.max(np.abs(ref-target))<1e-12
  e=np.array([[x['x']-x['target_x'],x['y']-x['target_y'],x['z']-x['target_z']] for x in rows]);observed=np.array([[x['ex'],x['ey'],x['ez']] for x in rows])
  checks[key+'_pre_reset_error']=np.max(np.abs(e-observed))<2e-6
  checks[key+'_CSV_JSON']=len(list(csv.DictReader((case/'trajectory.csv').open())))==len(rows)
  if name!='PID':checks[key+'_reload']=read(case/'reload.json')['bitwise_equal']
  row=dict(controller=name,period_s=T,requested_s=2*T,duration_s=m['duration_s'],completed=m['completed_two_reference_laps'],reason=','.join(m['reasons']),
   reference_laps=m['reference_laps_elapsed'],measured_azimuth_revolutions=m['measured_azimuth_revolutions'],
   position_rmse_m=m['position_rmse_m'],x_rmse_m=m['axis_rmse_m'][0],y_rmse_m=m['axis_rmse_m'][1],z_rmse_m=m['axis_rmse_m'][2],
   max_position_error_m=m['max_position_error_m'],radial_rmse_m=m['radial_error_rmse_m'],mean_radial_error_m=m['mean_radial_error_m'],phase_lag_mean_deg=m['phase_lag_mean_deg'],
   speed_mean_m_s=m['speed_mean_m_s'],velocity_tracking_rmse_m_s=m['velocity_tracking_rmse_m_s'],max_tilt_deg=m['max_tilt_deg'],
   body_omega_rms_rad_s=m['body_omega_rms_rad_s'],z_peak_to_peak_m=m['z_peak_to_peak_m'],action_clipping_fraction=m['action_clipping_fraction'],
   allocation_clipping_fraction=m['allocation_clipping_fraction'],lap2_complete=m['laps']['2']['complete'],
   lap2_position_rmse_m=m['laps']['2']['metrics']['position_rmse_m'] if m['laps']['2']['complete'] else None,
   initialization_s=z['initialization_s'],evaluation_wall_s=z['evaluation_wall_s'],optimizer_updates=z['optimizer_updates'])
  summary.append(row)
  marker=case/'tensorboard_written.json'
  writer=None if marker.exists() else SummaryWriter(str(r/'tensorboard'/key))
  tags={'Position/error_m':'position_error_m','Position/radial_error_m':'radial_error_m','Position/z_error_m':'ez',
   'Tracking/phase_lag_deg':'phase_lag_deg','Attitude/tilt_deg':'tilt_deg','Attitude/roll_deg':'roll_deg','Attitude/body_roll_rate_rad_s':'wx',
   'Actuator/action_clipping_fraction':'action_clipping_fraction','Actuator/allocation_clipping_fraction':'allocation_clipping_fraction',
   'Actuator/total_thrust_N':'total_thrust_N','Reward/actual':'reward','Reward/common_A':'common_reward_A'}
  for i in range(4):tags[f'Actuator/thrust_native_{i}_N']=f'thrust_{i}';tags[f'Action/raw_{i}']=f'raw_action_{i}';tags[f'Action/applied_{i}']=f'action_{i}'
  for tag,col in tags.items():
   for x in rows:
    value=float(np.float32(x[col]));
    if writer is not None:writer.add_scalar(tag,value,x['step'])
    scalars.append(dict(case=key,tag=tag,global_step=x['step'],sim_time_s=x['time_s'],value=value))
  for field in ['position_rmse_m','radial_error_rmse_m','phase_lag_mean_deg','max_tilt_deg','body_omega_rms_rad_s','duration_s']:
   tag='Summary/'+field;value=float(np.float32(m[field]));
   if writer is not None:writer.add_scalar(tag,value,rows[-1]['step'])
   scalars.append(dict(case=key,tag=tag,global_step=rows[-1]['step'],sim_time_s=rows[-1]['time_s'],value=value))
  if writer is not None:
   writer.flush();writer.close();dump_json(marker,dict(written=True))
  ea=EventAccumulator(str(r/'tensorboard'/key),size_guidance={'scalars':0});ea.Reload();lookup={tag:{x.step:x.value for x in ea.Scalars(tag)} for tag in ea.Tags()['scalars']}
  matched=[x for x in scalars if x['case']==key];checks[key+'_TensorBoard_scalars']=all(lookup[x['tag']][x['global_step']]==x['value'] for x in matched)
  spect={}
  for label,seq in [('full',rows),('lap2',[x for x in rows if x['time_s']>T] if m['laps']['2']['complete'] else [])]:
   if seq:spect[label]={col:spectral([x[col] for x in seq]) for col in ['roll_rad','wx','ez','vz']}
  dump_json(case/'spectra.json',spect)
common=[]
for T in [10,5]:
 keys=[f'{n}_T{T}' for n in plan['controllers'] if f'{n}_T{T}' in trajectories]
 n=min(len(trajectories[k]) for k in keys)
 for key in keys:
  met=segment_metrics(trajectories[key][:n]);common.append(dict(controller=key.rsplit('_T',1)[0],period_s=T,common_window_s=n*.01,position_rmse_m=met['position_rmse_m'],radial_rmse_m=met['radial_error_rmse_m'],max_tilt_deg=met['max_tilt_deg']))
initials=[z['initial'] for z in results.values()];checks['identical_initial_state']=all(x==initials[0] for x in initials)
checks['physics_source_preserved']=protected_hashes()==read(r/'evidence/protected_physics_sha256.json')
checks['existing_sources_preserved']=all(sha256(ROOT/f)==h for f,h in read(r/'evidence/existing_sources_sha256.json').items())
checks['source_checkpoints_preserved']=all(sha256(x['source'])==x['sha256']==sha256(x['copy']) for x in plan['models'].values())
checks['six_trials']=len(results)==6

first_step=[]
parent=ROOT/'runs/phase8_exponential_20261012_164652'
for name in ['Q_base','E_position']:
 original=read(parent/name/'evaluation/reload_8s/best/nominal.json')['trajectory'][0]
 for T in [10,5]:
  key=f'{name}_T{T}'
  if key not in trajectories:continue
  current=trajectories[key][0]
  fields=[f'{prefix}_{i}' for prefix in ['raw_action','action','omega','actual'] for i in range(4)]+['qw','qx','qy','qz','wx','wy','wz','vx','vy','vz']
  delta={k:abs(original[k]-current[k]) for k in fields}
  first_step.append(dict(case=key,max_difference=max(delta.values()),differences=delta))
  checks[key+'_first_step_equals_hover']=max(delta.values())==0.
dump_json(r/'first_step_vs_hover.json',first_step)

csvwrite(r/'summary.csv',summary);dump_json(r/'summary.json',summary);csvwrite(r/'common_window_comparison.csv',common);dump_json(r/'common_window_comparison.json',common)
csvwrite(r/'tensorboard_scalars.csv',scalars);dump_json(r/'tensorboard_scalars.json',scalars)
for fn in ['summary.csv','common_window_comparison.csv']:shutil.copy2(r/fn,out/fn)
# Graphs use common axes within each period; failures end at the actual first terminal sample.
colors={'PID':'#1f77b4','Q_base':'#e17c05','E_position':'#279452'}
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25,'figure.dpi':120})
def save(fig,name):fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
def arr(rows,key):return np.array([x[key] for x in rows])
fig,axs=plt.subplots(1,2,figsize=(12,5.5))
for ax,T in zip(axs,[10,5]):
 rr=reference(np.linspace(0,T,501),T)[0];ax.plot(rr[:,0],rr[:,1],'k--',label='reference R=1m')
 for name in plan['controllers']:
  seq=trajectories.get(f'{name}_T{T}')
  if not seq:continue
  ax.plot([1]+arr(seq,'x').tolist(),[0]+arr(seq,'y').tolist(),label=name,color=colors[name]);ax.scatter(seq[-1]['x'],seq[-1]['y'],color=colors[name],marker='o' if results[f'{name}_T{T}']['metrics']['completed_two_reference_laps'] else 'x')
 ax.scatter([1],[0],marker='s',color='black',s=30);ax.set(aspect='equal',xlabel='X [m]',ylabel='Y [m]',title=f'T={T}s, requested 2 reference laps');ax.legend()
save(fig,'xy_paths')
for T in [10,5]:
 tt=np.arange(round(T*2*100)+1)*.01;ref=reference(tt,T)[0]
 fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
 for j,ax in enumerate(axs):
  ax.plot(tt,ref[:,j],'k--',label='reference')
  for name in plan['controllers']:
   seq=trajectories.get(f'{name}_T{T}')
   if seq:ax.plot(arr(seq,'time_s'),arr(seq,['x','y','z'][j]),color=colors[name],label=name)
  ax.set_ylabel(['X [m]','Y [m]','Z [m]'][j]);ax.axvline(T,color='gray',ls=':',label='lap boundary' if j==0 else None)
 axs[0].legend(ncol=5);axs[-1].set_xlabel('Simulation time [s]');fig.suptitle(f'Position tracking, T={T}s; no continuation after failure');save(fig,f'position_T{T}')
 fig,axs=plt.subplots(5,1,figsize=(11,11),sharex=True)
 for ax,col,label in zip(axs,['position_error_m','radial_error_m','phase_lag_deg','ez','tilt_deg'],['Time-aligned error [m]','Radial error [m]','Wrapped phase lag [deg]','Z error [m]','Tilt [deg]']):
  for name in plan['controllers']:
   seq=trajectories.get(f'{name}_T{T}')
   if seq:ax.plot(arr(seq,'time_s'),arr(seq,col),label=name,color=colors[name])
  ax.set_ylabel(label);ax.axvline(T,color='gray',ls=':')
 axs[0].legend();axs[-1].set_xlabel('Simulation time [s]');fig.suptitle(f'Circle tracking error and attitude, T={T}s');save(fig,f'errors_T{T}')
 fig,axs=plt.subplots(5,1,figsize=(11,11),sharex=True)
 for ax,col,label in zip(axs,['roll_deg','wx','raw_action_0','action_0','actual_0'],['Euler roll [deg]','Body roll rate [rad/s]','Raw roll action [-]','Applied roll action [-]','Actual roll torque [N m]']):
  for name in plan['controllers']:
   seq=trajectories.get(f'{name}_T{T}')
   if seq:ax.plot(arr(seq,'time_s'),arr(seq,col),label=name,color=colors[name],lw=.8)
  ax.set_ylabel(label)
 axs[0].legend();axs[-1].set_xlabel('Simulation time [s]');fig.suptitle(f'Roll attitude, rate, action and torque are different quantities, T={T}s');save(fig,f'roll_action_T{T}')
 fig,axs=plt.subplots(4,1,figsize=(11,9),sharex=True)
 for i,ax in enumerate(axs):
  for name in plan['controllers']:
   seq=trajectories.get(f'{name}_T{T}')
   if seq:ax.plot(arr(seq,'time_s'),arr(seq,f'thrust_{i}'),label=name,color=colors[name],lw=.8)
  ax.set_ylabel(f'Rotor {i} thrust [N]');ax.set_ylim(-.01,.30);ax.axhline(.289,color='black',ls=':')
 axs[0].legend();axs[-1].set_xlabel('Simulation time [s]');fig.suptitle(f'Actual rotor thrusts in native index order, T={T}s');save(fig,f'motors_T{T}')
 # Full trajectories alongside each plot family are available as matching CSV files.
 combined=[dict(controller=name,**row) for name in plan['controllers'] for row in trajectories.get(f'{name}_T{T}',[])]
 csvwrite(out/f'trajectories_T{T}.csv',combined)
fig,axs=plt.subplots(2,3,figsize=(14,8))
metrics=[('position_rmse_m','Time-aligned position RMSE [m]'),('radial_rmse_m','Radial RMSE [m]'),('phase_lag_mean_deg','Mean wrapped phase lag [deg]'),('duration_s','Actual duration [s]'),('body_omega_rms_rad_s','Body omega RMS [rad/s]'),('action_clipping_fraction','Action clipping fraction')]
for ax,(field,label) in zip(axs.flat,metrics):
 for j,name in enumerate(plan['controllers']):
  vals=[next(x for x in summary if x['controller']==name and x['period_s']==T)[field] for T in [10,5]]
  ax.bar(np.arange(2)+(j-1)*.24,vals,width=.24,label=name,color=colors[name])
 ax.set_xticks([0,1],['T=10s','T=5s']);ax.set_ylabel(label)
axs[0,0].legend();fig.suptitle('One deterministic trial per controller and speed; inspect completion before comparing RMSE');fig.tight_layout();save(fig,'comparison_metrics')
verification=dict(success=all(checks.values()),checks=checks,check_count=len(checks),preflight=read(r/'preflight_verification.json'),optimizer_updates=0,
 tensorboard_scalar_count=len(scalars),analysis_wall_s=time.perf_counter()-start,case_integrity_checks={k:v['checks'] for k,v in results.items()})
dump_json(r/'verification.json',verification)
commands=f'''cd {ROOT}
PY=/home/mrl_6534/miniconda3/envs/isaaclab/bin/python
RUN={r}
# GUI, two laps; choose PID / Q_base / E_position and period 10 / 5.
"$PY" scripts/evaluate_circle.py --run-dir "$RUN" --controller E_position --period 10 \\
  --output "$RUN/manual_gui_E_position_T10_$(date +%Y%m%d_%H%M%S)" \\
  --device cuda:0 --visualizer kit --physics isaacsim_physx
# Headless replay: replace --visualizer kit with --headless. No optimizer update.
# TensorBoard: keep existing phase8 service at 6006; use free port 6007.
"$PY" -m tensorboard.main --logdir "$RUN/tensorboard" --host 127.0.0.1 --port 6007
# Windows PowerShell tunnel, then open http://127.0.0.1:6007
ssh -N -L 6007:127.0.0.1:6007 mrl_6534@100.79.103.47
# Download plots/report/checkpoints/source in the review archive:
scp mrl_6534@100.79.103.47:{r}/review.tar.gz .
# Fresh-run reproduction, only when requested by the user:
"$PY" scripts/prepare_circle_eval.py
NEW_RUN=$(cat /tmp/crazyflie_circle_current_run)
"$PY" scripts/validate_circle_eval.py --run-dir "$NEW_RUN"
"$PY" scripts/run_circle_eval.py --run-dir "$NEW_RUN"
"$PY" scripts/analyze_circle_eval.py --run-dir "$NEW_RUN"
"$PY" scripts/plot_circle_diagnostics.py --run-dir "$NEW_RUN"
"$PY" scripts/package_circle_eval.py --run-dir "$NEW_RUN"
'''
(r/'COMMANDS.txt').write_text(commands)
def table(headers,rows):return '\n'.join([' | '.join(headers),' | '.join(['---']*len(headers))]+[' | '.join(map(str,x)) for x in rows])
metric_table=table(['controller','period s','complete','duration s','3D RMSE m','radial RMSE m','phase lag deg','lap2 RMSE m','max tilt deg','action clip'],
 [[x['controller'],x['period_s'],x['completed'],f"{x['duration_s']:.2f}",f"{x['position_rmse_m']:.6f}",f"{x['radial_rmse_m']:.6f}",f"{x['phase_lag_mean_deg']:.3f}",str(x['lap2_position_rmse_m']),f"{x['max_tilt_deg']:.3f}",f"{x['action_clipping_fraction']:.6f}"] for x in summary])
interpret=[]
for T in [10,5]:
 ss=[x for x in summary if x['period_s']==T]
 interpret.append(f'주기 {T}s: '+', '.join(f"{x["controller"]} {x["duration_s"]:.2f}/{2*T}s, 완료={x["completed"]}, 위치RMSE={x["position_rmse_m"]*100:.2f}cm" for x in ss))
 if all(x['completed'] for x in ss):
  ranked=sorted(ss,key=lambda x:x['position_rmse_m']);interpret.append('같은 전체 시간 구간의 위치RMSE 순서: '+' < '.join(x['controller'] for x in ranked))
 else:interpret.append('일찍 실패한 궤적의 낮은 RMSE를 우수한 추종으로 해석하지 않는다. common_window_comparison.csv의 동일 시간 구간도 함께 본다.')
report=f'''Crazyflie 원형 궤적 평가: PID / Q_base best / E_position best
작성 {datetime.datetime.now().isoformat()}
서버 MRL-server / 실행 {r}

1. 평가 조건
반지름1m, 중심[0,0,1]m, 반시계 방향, yaw0. 주기10초는20초(두바퀴), 주기5초는10초(두바퀴).
p_ref(t)=[cos(2*pi*t/T),sin(2*pi*t/T),1]. 속력은각각{2*math.pi/10:.6f},{2*math.pi/5:.6f}m/s,
구심가속도는각각{(2*math.pi/10)**2:.6f},{(2*math.pi/5)**2:.6f}m/s².
각기체는[1,0,1],수평자세,속도0,기존hover-equilibrium모터상태에서 출발한다. 시작직후목표속력으로이동하며 ramp/이륙/정착구간없음.
서로 독립된 새 프로세스6개, seed42, 한조건당1회 deterministic평가. 물리500Hz/명령100Hz,기존질량/관성/모터지연/allocation/clip유지.
학습과gain튜닝은실행하지않았고optimizer update는0이다.

2. 실제 모델과 PID
Q_base best: {plan['models']['Q_base']['source']}
E_position best: {plan['models']['E_position']['source']}
Q_base best는phase8의초기부모, E_position best는phase8추가128회모델이다. 같은최종학습량의대조실험으로해석하지않는다.
고정128관측출력을새프로세스에서대조했고reload.json에실제오차/해시를저장했다.
PID는기존MuJoCo소스의CascadePID를그대로import한다. 복사본과SHA256은source_snapshot/original_controllers.py및experiment_plan.json에있다.
P(position)->PI(velocity)->P(attitude)->PI(rate); D계수는원본코드에서도실제로적용하지않는다.
기존gain: {json.dumps(plan['pid']['settings'],ensure_ascii=False)}
기존참조구현의500Hz호출과달리이번비교에서는PID dt=.01,100Hz로맞췄다. 이변경과명령제한까지포함한PID기준이다.
PID는자체max_force1N/max_torque.02Nm클립후wrench를정규화action으로변환한다.
a=[tau_x/.0075,tau_y/.0075,tau_z/.001,(Fz-.04338*9.81)/.5].
그뒤세제어기모두동일action[-1,1] clip->allocation->모터추력[0,.289]clip->ESC역변환->50msRPM지연->실제힘/반토크PhysX경로를사용한다.
PID에속도/가속도feedforward를추가하지않았으며gain자동튜닝없음. 따라서trajectory전용최적PID의한계성능을뜻하지않는다.

3. 관측과 시간 정합
정책의15D입력은그대로유지: 위치오차=기체link원점world위치-env_origin-움직이는목표,속도는절대world속도,
quaternion wxyz,body각속도,yaw오차sin/cos. 목표속도/가속도는정책과PID입력에추가하지않고평가지표에만쓴다.
컨트롤러는t의목표로행동을계산한다. pre_physics에서명령을만든뒤목표를t+.01로갱신하여step끝reward/done/오차와실제시각이일치한다.
terminal전상태는reward hook에서clone하므로autoreset뒤의위치가궤적에섞이지않는다.
모든조건의실제첫종료에서멈춘다. 시간제한만2*T로변경,고도[.2,2.5]m/기울기60deg/목표대비3D오차1.5m실패조건유지.

4. 결과
{metric_table}

{chr(10).join(interpret)}
2바퀴완료는지정된두주기의참조시간을실패없이진행했다는뜻이다. 실제기체가반지름1m를정확히두번돌았다는뜻은아니다.
기하학적반지름오차=norm([x,y])-1,시간정합위치오차=p(t)-p_ref(t)를구분한다.
phase lag=wrap(theta_ref-atan2(y,x)),양수는지연,[-180,180]deg. 작은radial오차라도큰phase lag가있으면목표점추종은부정확하다.
measured_azimuth_revolutions는원중심기준unwrap각도의순변화이며정확한원주완주인증이아니다.
1바퀴/2바퀴지표를나눠초기출발과후반추종을구분했다. 두바퀴만으로정상상태또는다중seed일반성은주장하지않는다.
두 RL 정책은 hover에서 학습되었으므로본평가는미학습이동목표에대한전이성능이다. 원형추종학습을한정책간비교가아니다.

5. 검증과원시자료
사전검증 {len(read(r/'preflight_verification.json')['checks'])}개 및후처리 {len(checks)}개; success={all(checks.values())}.
기존 hover 평가와 첫 step의 정책행동/모터/힘/자세/속도가 같은지 별도로 대조했다(first_step_vs_hover.json).
각case의물리step수/모터갱신5회/실제PhysX event시간/target오차/reward합/기존물리파일해시/optimizer0를검사했다.
TensorBoard와CSV/JSON의scalar {len(scalars)}개대조. 가로축은평가policy step이며시간은step*.01초다. 학습전이수와혼동하지않는다.
Reward/actual은조건별reward,Reward/common_A는동일기존A수식으로계산한진단값이다. reward정의가다른조건의return으로추종우열을정하지않는다.
전체궤적은cases/<controller>_T<period>/trajectory.csv,json. 위치,목표위치/속도,자세,각속도,raw/appliedaction,모터RPM/추력,requested/actualwrench포함.
원본roll스펙트럼은100Hz,mean제거,symmetricHann,one-sidedPSD; lap2는두번째바퀴완료시에만계산한다.
기존phase3~8코드/체크포인트와설치/드라이버변경없음. GUI는아래명령으로실행가능하나이번성능수치는headless실측이다.

6. 실행시간
'''+table(['case','initialization s','evaluation wall s'],[[k,f"{z['initialization_s']:.3f}",f"{z['evaluation_wall_s']:.3f}"] for k,z in results.items()])+f'''
초기화/평가벽시계시간은시뮬레이션시간과다르다. evaluation_jobs.json에프로세스전체시간도보존.

7. 재생과전달
plots/에XY궤적,XYZ목표추종,위치/반지름/위상오차,roll/각속도/action/토크,모터추력및비교PNG와대응CSV저장.
GUI는전체원의카메라와비물리적참조원/빨간이동목표점을표시한다. 이표시는강체/충돌/모터를바꾸지않는다.
TensorBoard기존6006서비스는보존한다. 새데이터서비스는별도6007사용(시작전포트확인).
review.tar.gz에보고서/설정/원시결과/모델/코드/PNG와manifest/files.sha256을포함한다.

{commands}
'''
if (r/'measured_findings_ko.txt').exists():report+=(r/'measured_findings_ko.txt').read_text()
(r/'REPORT_circle_ko.txt').write_text(report)
(r/'TENSORBOARD_TAGS_ko.txt').write_text('각run=controller/period별독립평가. global_step=100Hz policy step, simulation seconds=step/100. Summary는episode최종집계. Position은m, Attitude는태그표기deg/rad/s, action은무차원, thrust는N. episode return/학습loss없음. 원시per-stepscalar를보존하며smoothing없이CSV동일값. Reward actual과common_A는별도모니터지표.\n')
print('CIRCLE_ANALYSIS',len(checks),all(checks.values()),len(summary),len(scalars),flush=True)
assert all(checks.values()),[k for k,v in checks.items() if not v]

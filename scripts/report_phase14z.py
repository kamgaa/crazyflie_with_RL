"""Phase13 정정 final comparison/report; offline only. All values originate from saved evaluations."""
import argparse,csv,json,math,time,shutil,socket
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ppo_common import ROOT,dump_json,sha256,protected_hashes
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args()
r=Path(a.run_dir).resolve();out=r/'plots';tick=time.perf_counter()
read=lambda p:json.loads(Path(p).read_text())
plan=read(r/'experiment_plan.json');base=Path(plan['configuration_source'])
train=read(r/'training_result.json');sel=read(r/'best_selection.json');it=sel['selected']['additional_iteration']
summary=read(r/'evaluation_summary.json');verify=read(r/'verification.json')
def csvout(name,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (out/(name+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def loadcsv(path):
 with Path(path).open() as f:
  return [{k:float(v) if v not in ('True','False') else v=='True' for k,v in row.items()} for row in csv.DictReader(f)]

# Z-specific hover comparison, retained per case before averaging.
hover_vertical=[]
for model in ['phase13_baseline','best','final']:
 for suite in ['fixed','random20']:
  folder=r/'evaluation'/f'{model}_{suite}_30s'
  for d in read(folder/'results.json'):
   rows=loadcsv(folder/Path(d['checkpoint']).stem/(d['condition']+'.csv'))
   tail=rows[-1000:] if d['completed'] else []
   hover_vertical.append(dict(model=model,suite=suite,condition=d['condition'],completed=d['completed'],duration_s=d['duration_s'],
    position_rmse_m=d['position_rmse_m'],z_rmse_m=d['xyz_rmse_m'][2],max_abs_z_error_m=d['max_abs_z_error_m'],
    full_vz_rms_m_s=float(np.sqrt(np.mean([x['vz']**2 for x in rows]))),
    tail_vz_rms_m_s=float(np.sqrt(np.mean([x['vz']**2 for x in tail]))) if tail else None,
    tail_z_mean_m=d['tail_z_mean_m'],tail_z_std_m=d['tail_z_std_m'],tail_z_peak_to_peak_m=d['tail_z_peak_to_peak_m']))
csvout('hover_vertical_per_condition',hover_vertical)
hover_vertical_summary=[]
for model in ['phase13_baseline','best','final']:
 for suite in ['fixed','random20']:
  ds=[x for x in hover_vertical if x['model']==model and x['suite']==suite];done=[x for x in ds if x['completed']]
  hover_vertical_summary.append(dict(model=model,suite=suite,completed=len(done),conditions=len(ds),
   **{k:float(np.mean([x[k] for x in done])) if done else None for k in ['position_rmse_m','z_rmse_m','max_abs_z_error_m','full_vz_rms_m_s','tail_vz_rms_m_s','tail_z_std_m','tail_z_peak_to_peak_m']},
   mean_abs_condition_tail_z_bias_m=float(np.mean([abs(x['tail_z_mean_m']) for x in done])) if done else None))
csvout('hover_vertical_summary',hover_vertical_summary);dump_json(r/'hover_vertical_summary.json',hover_vertical_summary)

font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':9,'axes.unicode_minus':False,'axes.grid':True,'grid.alpha':.25})
def save(fig,name):
 fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/(name+'.png'),dpi=155);plt.close(fig)
labels={'baseline':'phase13sq best448 / Z 속도 가중치 0.005','new':f'phase14z best{it} / Z 속도 가중치 0.015'}
colors={'baseline':'#777777','new':'#0072b2'}
# Reward coefficient comparison, same world Vz reference and square.
v=np.linspace(-1.5,1.5,601)
curve=[dict(vz_m_s=float(x),baseline_reward=-.005*x*x,new_reward=-.015*x*x,baseline_gradient=-.010*x,new_gradient=-.030*x) for x in v]
fig,axs=plt.subplots(1,2,figsize=(13,4))
for w,label,color in [(.005,'기존: -0.005 Vz²','#777777'),(.015,'3배: -0.015 Vz²','#0072b2')]:
 axs[0].plot(v,-w*v*v,label=label,color=color)
 axs[1].plot(v,-2*w*v,label=label,color=color)
axs[0].set(xlabel='world Vz [m/s]',ylabel='signed reward contribution',title='Z 속도 제곱 벌점')
axs[1].set(xlabel='world Vz [m/s]',ylabel='d reward / d Vz',title='속도에 대한 기울기')
for ax in axs:ax.legend()
fig.suptitle('Z 속도 가중치만 0.005 -> 0.015 / 위치 및 나머지 12항 동일')
save(fig,'00_vertical_velocity_reward_curves');csvout('00_vertical_velocity_reward_curves',curve)
# Compare learning curves against historical phase13sq at the same new transitions.
training={k:[json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines()] for k,path in [('baseline',base),('new',r)]}
groups=[('Z 속도 보상 기여도',['Reward/velocity_z']),('실제 총보상 (정의가 달라 직접 순위화 금지)',['Reward/total']),
('진단용 기존 13항 reward',['Reward/common_axis_baseline']),('실제 PPO total loss',['Loss/ppo_total']),
('Policy loss',['Loss/policy']),('Value loss',['Loss/value']),('PPO ratio clip fraction',['PPO/clip_fraction']),('Approximate KL',['PPO/approx_kl'])]
fig,axs=plt.subplots(4,2,figsize=(14,13))
for ax,(title,keys) in zip(axs.ravel(),groups):
 for model,rows in training.items():
  for key in keys:
   actual_key='Reward/total' if model=='baseline' and key=='Reward/common_axis_baseline' else key
   ax.plot([x['global_step']/1e6 for x in rows],[x.get(actual_key,np.nan) for x in rows],color=colors[model],lw=.6,label=labels[model].split(' best')[0])
 ax.set(title=title,xlabel='fresh policy transitions [million]');ax.legend(fontsize=7)
fig.suptitle('동일 초기화·동일 예산 / iteration 평균 / common reward도 상태 분포는 서로 다름')
save(fig,'01_training_comparison')
# All saved circle metrics and rows. Baseline is reused, never retrained.
circle_rows=[];trajectories={};circlechecks={};sources=[];tb_records=[]
cw=None if any((r/'tensorboard_circle').glob('events.out.tfevents.*')) else SummaryWriter(str(r/'tensorboard_circle'))
for model,path,prefix in [('baseline',base/'circle','phase13sq_best'),('new',r/'circle','phase14z_best')]:
 for T in [10,5]:
  folder=path/'cases'/f'{prefix}_T{T}';res=read(folder/'result.json');ck=read(folder/'reload.json')
  assert res['success'] and res['optimizer_updates']==0
  if model=='new':
   circlechecks[f'circle_T{T}_all_checks']=all(res['checks'].values())
   circlechecks[f'circle_T{T}_selected_best']=ck['sha256']==sha256(r/'best.pt')
   circlechecks[f'circle_T{T}_reload']=ck['max_action_error']<=1e-6
  rows=loadcsv(folder/'trajectory.csv');trajectories[(model,T)]=rows
  sources.append(dict(model=model,period_s=T,result=str(folder/'result.json'),result_sha256=sha256(folder/'result.json'),trajectory_sha256=sha256(folder/'trajectory.csv'),checkpoint=ck['checkpoint'],checkpoint_sha256=ck['sha256'],historical_reused=model=='baseline'))
  met={k:v for k,v in res['metrics'].items() if k!='laps'}
  circle_rows.append(dict(model=model,label=labels[model],period_s=T,**met))
  for row in rows:
   for key in ['position_error_m','ex','ey','ez','vx','vy','vz','roll_rad','pitch_rad','wx','wy','wz','actual_3','action_clipping_fraction','allocation_clipping_fraction']:
    val=float(np.float32(row[key]));tag=f'{model}/T{T}/{key}';step=int(row['step'])
    if cw is not None:cw.add_scalar(tag,val,step)
    tb_records.append(dict(tag=tag,global_step=step,time_s=row['time_s'],value=val))
if cw is not None:cw.flush();cw.close()
csvout('circle_comparison',circle_rows);dump_json(r/'circle_comparison.json',circle_rows);dump_json(r/'circle_sources.json',sources)
csvout('circle_scalars',tb_records);dump_json(r/'circle_scalars.json',tb_records)
ea=EventAccumulator(str(r/'tensorboard_circle'),size_guidance={'scalars':0});ea.Reload()
lookup={}
for row in tb_records:lookup.setdefault(row['tag'],[]).append((row['global_step'],row['value']))
circlechecks['circle_tb_csv_json']=all([(e.step,e.value) for e in ea.Scalars(tag)]==expected for tag,expected in lookup.items())
# Plot full trajectories and explicitly mark first failure or end.
fig,axs=plt.subplots(1,2,figsize=(13,6))
for ax,T in zip(axs,[10,5]):
 theta=np.linspace(0,2*np.pi,400);ax.plot(np.cos(theta),np.sin(theta),'k--',label='목표 R=1m')
 for model in labels:
  rows=trajectories[(model,T)];x=[z['x'] for z in rows];y=[z['y'] for z in rows]
  ax.plot(x,y,color=colors[model],lw=1,label=labels[model]);ax.plot(x[-1],y[-1],'o',color=colors[model],ms=4)
 ax.set(title=f'주기 {T}s / 최대 {2*T}s / 원형 목표 위치만 제공',xlabel='X [m]',ylabel='Y [m]',aspect='equal');ax.legend(fontsize=7)
fig.suptitle('최종 선정 best 원 궤적 / 점=실제 기록 종료 (실패 또는 시간제한)')
save(fig,'02_circle_xy_comparison')
common=[]
for T in [10,5]:
 seq={m:trajectories[(m,T)] for m in labels}
 csvout(f'circle_trajectories_T{T}',[dict(model=m,**row) for m,rows in seq.items() for row in rows])
 n=min(map(len,seq.values()))
 for m,rows in seq.items():common.append(dict(model=m,period_s=T,common_duration_s=n*.01,common_rmse_m=float(np.sqrt(np.mean([z['ex']**2+z['ey']**2+z['ez']**2 for z in rows[:n]])))))
 for name,signals in [
 ('03_circle_position_velocity',[('ex','X 오차 [m]'),('ey','Y 오차 [m]'),('ez','Z 오차 [m]'),('vx','Vx [m/s]'),('vy','Vy [m/s]'),('vz','Vz [m/s]')]),
 ('04_circle_attitude_vertical',[('roll_deg','실제 roll [deg]'),('pitch_deg','실제 pitch [deg]'),('wx','body roll rate [rad/s]'),('wy','body pitch rate [rad/s]'),('actual_3','실제 총추력 T [N]'),('vertical_thrust','T cos roll cos pitch [N]')]),
 ('05_circle_action_torque',[('raw_action_0','raw roll action [normalized]'),('action_0','applied roll action [normalized]'),('raw_action_1','raw pitch action [normalized]'),('action_1','applied pitch action [normalized]'),('actual_0','실제 roll torque [Nm]'),('actual_1','실제 pitch torque [Nm]')])]:
  fig,axs=plt.subplots(3,2,figsize=(14,10),sharex=True)
  for ax,(key,title) in zip(axs.ravel(),signals):
   for m,rows in seq.items():
    vals=[z[key] if key!='vertical_thrust' else z['actual_3']*math.cos(z['roll_rad'])*math.cos(z['pitch_rad']) for z in rows]
    ax.plot([z['time_s'] for z in rows],vals,color=colors[m],lw=.8,label=labels[m])
   if key=='vertical_thrust':ax.axhline(.043384*9.81,color='k',ls=':',label='mg 총질량')
   ax.set(title=title,xlabel='simulation time [s]',xlim=(0,2*T))
  axs[0,0].legend(fontsize=7);fig.suptitle(f'주기 {T}s 원 평가 / 0~{2*T}s / baseline vs 새 best{it}')
  save(fig,f'{name}_T{T}')
csvout('circle_common_interval',common)
# Verify unchanged physical sources and actual runtime scripts captured before optimizer updates.
runtime=read(r/'runtime_source_sha256.json')
needed=['train_phase14z.py','phase14z_common.py','phase14z_metrics.py','phase14z_reward.py','evaluate_phase14z.py']
runtime_checks={n:sha256(ROOT/'scripts'/n)==runtime['scripts/'+n] for n in needed}
circlechecks.update(runtime_training_sources_unchanged=all(runtime_checks.values()),physics_unchanged=protected_hashes()==read(r/'evidence/physics_before_sha256.json'),initialization=read(r/'baseline_initialization_verification.json')['success'],first_rollout=read(r/'first_rollout_comparison.json')['success'])
verify['checks'].update(circlechecks);verify['success']=all(verify['checks'].values());verify['runtime_source_checks']=runtime_checks
verify['checks']['pre_optimizer_first_rollout']=read(r/'pre_optimizer_first_rollout_verification.json')['success']
oldrc=read(base/'reward_config.json');newrc=read(r/'reward_config.json')
expected=json.loads(json.dumps(oldrc));expected['reward_id']=newrc['reward_id'];expected['axis_weights']['velocity'][2]=.015
verify['checks']['only_vz_weight_changed']=expected==newrc and oldrc['axis_weights']['velocity']==[.005]*3
verify['success']=all(verify['checks'].values())
dump_json(r/'verification.json',verify);assert verify['success'],verify
# Explicit timing definitions; no evaluation wall time in training throughput.
final_exec=[read(r/'evaluation'/f'{model}_{suite}_30s/execution.json') for model in ['best','final'] for suite in ['fixed','random20']]
circle_exec=[read(r/'circle/cases'/f'phase14z_best_T{T}/result.json') for T in [10,5]]
timing=dict(training_initialization_s=train['initialization_s'],training_s=train['training_wall_s'],transitions_per_training_s=train['transitions_per_s'],
 initial_evaluation_wall_s=train['initial_evaluation_wall_s'],intermediate_evaluation_wall_s=train['intermediate_evaluation_wall_s'],
 final_hover_evaluation_total_wall_s=sum(x['total_wall_s'] for x in final_exec),final_hover_evaluation_initialization_s=sum(x['initialization_s'] for x in final_exec),
 circle_initialization_s=sum(x['initialization_s'] for x in circle_exec),circle_simulation_wall_s=sum(x['evaluation_wall_s'] for x in circle_exec),
 note='training throughput denominator excludes initialization and evaluation pauses; circle initialization/simulation exclude some file export and process shutdown')
dump_json(r/'timing_summary.json',timing)
def fmt(v):return 'N/A' if v is None else f'{v:.7g}'
report=[
'Phase14 — Z 속도 제곱 벌점 가중치 3배',str(r),'',
'사용자 확정: 위치 가중치가 아닌 Z 속도 벌점 계수만 0.005에서 0.015로 변경한다.',
'기준: '+str(base)+'; 이 경로는 설정과 비교자료의 출처이며 학습된 가중치는 불러오지 않았다.',
'새 actor/critic, log_std=-1.5, seed42, 새 Adam, 카운터0. 동일 네트워크 초기화 및 reset RNG. 기존 실험 결과를 보존한다.',
'실제 Z 속도 보상: -0.015 * world_vz^2. v_des_z를 빼지 않는다. X/Y는 -0.005*(v-v_des)^2 유지.',
'v_des=norm_clip(-4*e,1.5m/s), e=link world position-env_origin-target, target=[0,0,1]. v_des는 보상/진단 기준이며 외부 힘 명령이 아니다.',
'나머지 항: X/Y 각0.100*expm1(-e_axis^2/.10^2); Z위치0.015*expm1(-ez^2/.05^2); roll/pitch 각-0.075*angle_rad^2; yaw -error_rad^2; body omega 각-0.001*omega^2; 실패-10.',
'13개 signed 항을 합산하며 dt/생존상수/필터/PID/추가 벌점을 도입하지 않았다. 시간 제한은 실패 벌점이 아니다.',
'1024환경, rollout32, minibatches64(512표본), epochs5, actor/critic[64,64]tanh, lr3e-4 fixed, clip.2, gamma.99, lambda.95, entropy.003, value1, valueclip, grad1, rollout advantage 정규화. 관측/보상 정규화 추가없음.',
'500Hz physics /100Hz policy, decimation5. 물성/모터 지연0.05s/allocation/clip/15D관측/4D행동/좌표계/중력보상 동일. 8초 학습episode, XYZ각축 독립±.05m 초기위치, 정상동역학 유지.',
'관측 속도는 절대 world 속도이며 quaternion wxyz. roll/pitch는 정규화한 quaternion의 ZYX Euler(rad). body 각속도는 Euler각 미분과 구분.',
f"실제 예산: {train['iterations']}회, {train['transitions']}전이, optimizer.step {train['optimizer_updates']}회. Adam 파라미터별 step의 고유값={train['adam_steps']}이며 파라미터별 step을 합산하지 않았다.",
f"초기화 {train['initialization_s']:.3f}s; 학습 {train['training_wall_s']:.3f}s; 처리량 {train['transitions_per_s']:.2f}전이/s. 초기평가 {train['initial_evaluation_wall_s']:.3f}s; 중간평가 {train['intermediate_evaluation_wall_s']:.3f}s.",
'최종평가 시간 및 범위: '+json.dumps(timing,ensure_ascii=False),
f"장치 전체 sampled peak {train['cuda_device_used_peak_sampled_MiB']:.3f}MiB (프로세스 단독 사용량 아님). Torch allocator 최종/peak 통계: {train['memory_final']}",
f"Best={it}회, checkpoint={sel['selected']['checkpoint']}; best/final 동일파일해시={sel['best_final_identical']}; final=512회.",
'선택기준: 고정7조건30초 완료개수 우선; 7미만 동률은 평균지속시간, 7/7 후보는 평균3D RMSE, 최종동률은 이른checkpoint. 초기모델 포함. random20 및 원추종은 선택에 미사용.',
'호버 결과: 조건별 원시값과 완료율을 먼저 저장. 아래 RMSE/후반지표는 완료조건만의 평균이며 실패수도 함께 확인한다. RMS=제곱평균제곱근, RMSE=오차의 제곱평균제곱근.']
for row in summary['final']:
 report.append(f"{row['label']} {row['suite']}: {row['completed_count']}/{row['conditions']}, 3D RMSE={fmt(row.get('completed_only_mean_position_rmse_m'))}m, tail speed RMS={fmt(row.get('completed_only_mean_tail_speed_rms_m_s'))}m/s, tail body omega RMS={fmt(row.get('completed_only_mean_tail_body_omega_rms_rad_s'))}rad/s.")
report+=['','Z 호버 지표: 후반=완료episode 마지막1000표본(20.01~30초), 편향 상쇄를 피하기 위해 조건별 절대편향을 평균.']
for row in hover_vertical_summary:
 report.append(f"{row['model']} {row['suite']}: {row['completed']}/{row['conditions']}, Z RMSE={fmt(row['z_rmse_m'])}m; Vz RMS(full/tail)={fmt(row['full_vz_rms_m_s'])}/{fmt(row['tail_vz_rms_m_s'])}m/s; tail Z std={fmt(row['tail_z_std_m'])}m; tail Z peak-to-peak={fmt(row['tail_z_peak_to_peak_m'])}m; mean abs bias={fmt(row['mean_abs_condition_tail_z_bias_m'])}m.")
report+=['','원 추종: R1m, center[0,0,1], T10s/5s 각각 두바퀴분20s/10s, CCW, yaw0, start[1,0,1]정지/수평/hoverRPM. 시간에 따른 위치목표만 제공하고 속도/가속도 feedforward는 추가하지 않는다.',
'실제 첫실패에서 종료하며 autoreset 후 상태를 연결하지 않는다. 목표 두바퀴분 시간을 완료한 것과 기체가 정확히 두바퀴의 기하학적 경로를 돈 것은 구분한다. 실제 방위각 회전수도 원시 JSON에 보존한다.']
for row in circle_rows:
 report.append(f"{row['label']} T{row['period_s']}: 완료={row['completed_two_reference_laps']}, {row['duration_s']}s; 3D RMSE={row['position_rmse_m']:.6f}m, Z RMSE={row['axis_rmse_m'][2]:.6f}m, Z p-p={row['z_peak_to_peak_m']:.6f}m, mean signed radial error={row['mean_radial_error_m']:.6f}m, max tilt={row['max_tilt_deg']:.3f}deg, omega RMS={row['body_omega_rms_rad_s']:.6f}rad/s, action/allocation clip={row['action_clipping_fraction']}/{row['allocation_clipping_fraction']}.")
for T in [10,5]:
 old=next(x for x in circle_rows if x['model']=='baseline' and x['period_s']==T);new=next(x for x in circle_rows if x['model']=='new' and x['period_s']==T)
 if old['completed_two_reference_laps'] and new['completed_two_reference_laps']:
  report.append(f"T{T} 동일시간: 3D RMSE 변화 {(new['position_rmse_m']/old['position_rmse_m']-1)*100:+.2f}%, Z RMSE {(new['axis_rmse_m'][2]/old['axis_rmse_m'][2]-1)*100:+.2f}%, Z p-p {(new['z_peak_to_peak_m']/old['z_peak_to_peak_m']-1)*100:+.2f}%. 반경 RMS {old['radial_error_rmse_m']:.6f}->{new['radial_error_rmse_m']:.6f}m; 위상지연 {old['phase_lag_mean_deg']:.3f}->{new['phase_lag_mean_deg']:.3f}deg. 음수 반경오차는 원 안쪽 편향.")
 else:report.append(f'T{T}: 생존시간/완료여부가 다르므로 짧은 궤적 RMSE로 우수성을 판단하지 않는다. 공통구간 결과는 circle_common_interval.csv 참조.')
report+=['',
f"검증: fixed128 저장/재로드 action 최대오차={verify['max_reload_action_error']}; reward 합 오차={verify['reward_max_error']}; 실제 backward loss 구성 residual={verify['loss_max_residual']}. TB/CSV/JSON 및 실제 optimizer 카운터 검증 통과.",
'첫 optimizer 전 초기 actor/critic/std/빈Adam/RNG/reset/첫rollout의 상태·행동·물리량이 기존과 일치하며, Z속도항만 3배이고 나머지12항은 동일함을 확인했다. pre_optimizer_first_rollout_verification.json 참조.',
'평가는 별도프로세스, 학습정지 중 순차실행하며 관측/모터/episode/rollout/RNG/Adam 보존을 검사했다. 사전검증 및 평가 optimizer=0.',
'common_axis_baseline은 현재 상태에 기존 phase13sq의 .005 Z속도 보상을 적용한 진단값이다. 실제 학습 reward에 중복가산하지 않는다. legacy common_stop/common_return은 이전9단계 정의의 진단값으로 별도 구분.',
'학습 보상 정의가 다르므로 총 reward와 PPO loss 크기를 직접 성능 순위로 해석하지 않는다. 물리 지표와 동일 생존 구간을 비교한다.',
'수직추력 T*cos(roll)*cos(pitch)는 실제 rotor 추력합을 저장된 poststep자세로 투영한100Hz표본이다. 마지막500Hz force buffer를 사용하며 전체 substep 평균은 아니다.',
'circle/attitude_vertical_thrust_* 참고자세는 기존 PID 외부루프를 저장된 RL상태에 사후 적용한 비교기준이다. RL 내부 목표자세가 아니며 실제 PID제어를 추가하지 않았다.',
'위치 원시상태/환경원점은 float32. 후처리 float64 및 CSV 소수자릿수는 엔진 정밀도를 높이지 않는다. 매우 작은 std를 실제기체 정밀도로 확대해석하지 않는다.',
'동일 초기화·seed42·동일512예산의 단일 비교다. best선택 iteration이 다르면 평가모델의 실제학습량이 다르다. 다중seed 우월성, OOD 안정성, 가중치 최적성을 입증하지 않는다.',
'기존 코드/물리파일/체크포인트/결과/설치/드라이버/TensorBoard서비스 보존. GUI는 실행하지 않았으며 재생명령만 제공한다. 지정된512회후 추가학습/튜닝 없음.',
'Git 저장소에는 유효한 HEAD commit이 없고 기존 프로젝트 파일은 untracked이다. 코드 버전은 source_snapshot, source_changes.patch, SHA256 목록으로 재현한다.',
'산출물: verification.json, condition_diff.json, source_snapshot/, source_changes.patch, checkpoint/best/final, TensorBoard, CSV/JSON, plots/, COMMANDS.txt, review.tar.gz 및 SHA256.'
]
(r/'REPORT_phase14_ko.txt').write_text('\n'.join(report)+'\n')
# Loopback port is only suggested; no service is launched or stopped.
port=6008
while True:
 s=socket.socket()
 try:s.bind(('127.0.0.1',port));s.close();break
 except OSError:s.close();port+=1
cmds=f"""cd {ROOT}
PY=/home/mrl_6534/miniconda3/envs/isaaclab/bin/python
RUN={r}
# Actual completed training command: do not rerun inside this completed directory.
"$PY" scripts/train_phase14z.py --run-dir "$RUN" --iterations 512 --device cuda:0 --headless --physics isaacsim_physx
# Hover evaluation only
"$PY" scripts/evaluate_phase14z.py --checkpoints "$RUN/best.pt" "$RUN/final.pt" --run-dir "$RUN" --output "$RUN/evaluation/manual_$(date +%Y%m%d_%H%M%S)" --positions "$RUN/evaluation/initial_positions.json" --suite fixed --horizon 30 --device cuda:0 --headless --physics isaacsim_physx
# Circle evaluation only. Use period 10 or 5; always two reference laps.
"$PY" scripts/evaluate_circle_phase14z.py --run-dir "$RUN/circle" --controller phase14z_best --period 10 --output "$RUN/circle/manual_T10_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --headless --physics isaacsim_physx
# GUI not executed/verified in this experiment.
"$PY" scripts/evaluate_circle_phase14z.py --run-dir "$RUN/circle" --controller phase14z_best --period 5 --output "$RUN/circle/gui_T5_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --visualizer kit --physics isaacsim_physx
# Training/validation axes are transitions; exclude separate circle-time logs from this view.
"$PY" -m tensorboard.main --logdir_spec baseline_train:{base}/tensorboard,vz_weight3_train:{r}/tensorboard,vz_weight3_validation:{r}/tensorboard_validation --host 127.0.0.1 --port {port}
# PowerShell tunnel, then http://127.0.0.1:{port}
ssh -N -L {port}:127.0.0.1:{port} mrl_6534@100.79.103.47
# PowerShell download
scp "mrl_6534@100.79.103.47:{r}/review.tar.gz" .
"""
(r/'COMMANDS.txt').write_text(cmds)
import subprocess,sys
if not (r/'animation_provenance.json').exists():
 subprocess.run([sys.executable,str(ROOT/'scripts/animate_phase14z_circle.py'),'--run-dir',str(r)],check=True)
dump_json(r/'report_execution.json',dict(report_and_comparison_plot_s=time.perf_counter()-tick,tensorboard_suggested_free_port=port,gui_executed=False,training_transitions=train['transitions'],optimizer_calls=train['optimizer_updates']))
print('REPORT_DONE',flush=True)

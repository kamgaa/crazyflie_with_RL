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
for model in ['phase12_baseline','best','final']:
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
for model in ['phase12_baseline','best','final']:
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
labels={'baseline':'phase12-1 best512 / Z 복귀속도 제곱','new':f'phase13sq best{it} / Z 절대속도 제곱'}
colors={'baseline':'#777777','new':'#0072b2'}
# Numeric reward curves; no implicit dimensional equivalence.
v=np.linspace(-1.5,1.5,601);curve=[]
fig,axs=plt.subplots(1,2,figsize=(13,4))
for ez in [0.,.05,-.05]:
 vd=-4*ez
 axs[0].plot(v,-.005*(v-vd)**2,label=f'기존: ez={ez:+.2f}m, vdes_z={vd:+.2f}m/s')
 curve.extend(dict(vz_m_s=float(x),ez_m=ez,old_velocity_z=-.005*(x-vd)**2,new_velocity_z=-.005*x**2) for x in v)
axs[0].plot(v,-.005*v**2,'k',lw=2,label='새 항: -0.005 vz²')
axs[0].set(xlabel='world vz [m/s]',ylabel='signed reward contribution',title='Z 속도 항: 위치가 바뀌어도 새 항은 동일')
axs[0].legend(fontsize=7)
for ez in [0.,.05,-.05]:axs[1].plot(v,-.01*(v+4*ez),label=f'기존 ez={ez:+.2f}m')
axs[1].plot(v,-.01*v,'k',lw=2,label='새 항 gradient');axs[1].set(xlabel='world vz [m/s]',ylabel='d reward / d vz',title='같은 제곱 계수 / 기준속도만 0으로 변경');axs[1].legend(fontsize=7)
fig.suptitle('Z 속도 reward만 변경 / 나머지 12개 항 동일 / e_x=e_y=0, norm 제한 미활성 예시')
save(fig,'00_vertical_velocity_reward_curves');csvout('00_vertical_velocity_reward_curves',curve)
# Compare learning curves against historical phase12-1 at the same new transitions.
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
for model,path,prefix in [('baseline',base/'circle','phase12_best'),('new',r/'circle','phase13sq_best')]:
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
needed=['train_phase13sq.py','phase13sq_common.py','phase13sq_metrics.py','phase13sq_reward.py','evaluate_phase13sq.py']
runtime_checks={n:sha256(ROOT/'scripts'/n)==runtime['scripts/'+n] for n in needed}
circlechecks.update(runtime_training_sources_unchanged=all(runtime_checks.values()),physics_unchanged=protected_hashes()==read(r/'evidence/physics_before_sha256.json'),initialization=read(r/'baseline_initialization_verification.json')['success'],first_rollout=read(r/'first_rollout_comparison.json')['success'])
verify['checks'].update(circlechecks);verify['success']=all(verify['checks'].values());verify['runtime_source_checks']=runtime_checks
provenance=read(r/'correction_provenance.json')
prior=Path(provenance['preserved_prior_run'])
verify['checks']['previous_quartic_run_preserved']=all(sha256(prior/name)==h for name,h in provenance['prior_files'].items())
verify['success']=all(verify['checks'].values())
dump_json(r/'verification.json',verify);assert verify['success'],verify
# Exact applied definitions and limitations.
report=[
'Phase13 정정 — Z축 절대 속도 제곱 보상 / phase12-1 기준 단일 새 학습',str(r),'',
'사용자 확정 기준: phase12-1, roll/pitch=.075. phase11/phase12 모델을 이어받지 않고 동일 seed42 새 actor/critic/log_std=-1.5, 빈 Adam, 카운터0부터 학습.',
'유일한 학습 보상 변경: velocity_z = -0.005*(world_vz[m/s])**2. v_des_z를 빼지 않는다. X/Y 속도 항은 -0.005*(v_axis-v_des_axis)**2 유지.',
'v_des=norm_clip(-4*position_error,1.5 m/s) 3D 식 자체는 유지. 관측은 절대 world velocity, quaternion wxyz를 포함한 기존15D. 4D wrench 및 clip/scale/allocation/RPM 경로 유지.',
'나머지 위치 x/y/z expm1, roll/pitch Euler(rad) .075, yaw1, 각축 body omega .001, 물리실패10 등12항 동일. 총13항 signed 합; dt/상수/추가항 없음.',
'계수 .005와 제곱 차수는 기존과 동일하다. Z 속도의 기준만 v_des_z에서 0으로 변경했으며 -0.005*abs(vz)^2와 -0.005*vz^2는 같다. z 위치 복귀 유도는 위치 보상으로 계속 남는다.',
'사용자가 뜻한 -|vz|^2에 맞춘 정정 실험이다. 기존 phase12-1 대비 Z 속도의 기준(vz-v_des_z→vz)만 바꾸고 제곱 차수와 .005 가중치는 유지한다. 앞서 실행한 4제곱 모델은 보존하며 이어 학습하지 않는다.',
'물리/환경/PPO: N1024,rollout32,epochs5,minibatches64,size512,actor/critic[64,64]tanh,lr3e-4 fixed,clip.2,gamma.99,lambda.95,entropy.003,value1,valueclip,actor/critic grad1,rolloutadvnorm,관측/보상 normalization 없음.',
'physics .002s/decimation5,500Hz/100Hz,8초학습episode,XYZ각축 독립±.05m box reset,정상질량/COM/모터/고장없음/DR없음/기존실패조건 유지.',
f"실제 학습: {train['iterations']} iterations, {train['transitions']} transitions, optimizer.step {train['optimizer_updates']}회; Adam state step {train['adam_steps']}. 초기/사전검증/평가는 예산과 별개이며 optimizer 0.",
f"시간: 초기화 {train['initialization_s']:.3f}s, 학습 {train['training_wall_s']:.3f}s, 처리량 {train['transitions_per_s']:.2f}transitions/s, 초기평가 {train['initial_evaluation_wall_s']:.3f}s, 중간평가 {train['intermediate_evaluation_wall_s']:.3f}s.",
f"장치 전체 메모리 sampled peak {train['cuda_device_used_peak_sampled_MiB']:.3f}MiB; 프로세스 전용 사용량 아님. Torch allocator 통계: {train['memory_final']}",
f"Best: 추가학습이 아닌 새 실행 {it}회, selected={sel['selected']['checkpoint']}; best/final 동일 SHA256={sel['best_final_identical']}.",
'Best규칙 phase11/12와 동일: 고정7조건30초 완료개수 우선;7미만 동률 평균생존;7/7끼리 평균3D RMSE;동률 이른checkpoint. 0회도 새 초기모델만 포함. 원평가/20개초기위치는 선정에 미사용.',
'호버: best/final 별도 프로세스 explicit reload, fixed7 및 기존random20 각30초. 과거baseline 자료는 출처를 표시해 재사용, 새 학습으로 복제하지 않음.',
'호버 결과 (완료조건만 평균; 완료수도 함께 표시):']
for row in summary['final']:
 report.append(f"{row['label']} / {row['suite']}: {row['completed_count']}/{row['conditions']}, 3D RMSE={row['completed_only_mean_position_rmse_m']}m, 후반 평균 Z 위치 편향={row.get('completed_only_mean_tail_error_vector_m_2')}m, tail speed RMS={row['completed_only_mean_tail_speed_rms_m_s']}m/s, body omega RMS={row['completed_only_mean_tail_body_omega_rms_rad_s']}rad/s.")
report+=['','Z축 호버 세부 비교: 후반은 완료한 episode의 마지막1000개100Hz표본(20.01~30초). 서로 다른 조건의 Z편향은 절댓값 평균으로 비교하여 상쇄 방지.']
for row in hover_vertical_summary:
 report.append(f"{row['model']} {row['suite']}: {row['completed']}/{row['conditions']}, Z RMSE={row['z_rmse_m']}m, full Vz RMS={row['full_vz_rms_m_s']}m/s, tail Vz RMS={row['tail_vz_rms_m_s']}m/s, tail Z std={row['tail_z_std_m']}m, tail Z p-p={row['tail_z_peak_to_peak_m']}m, mean abs condition Z bias={row['mean_abs_condition_tail_z_bias_m']}m.")
report+=['','원 프로토콜: R1m,center[0,0,1],T10/T5 각두바퀴분20/10s,CCW,yaw0,start[1,0,1]정지/수평/hoverRPM. 위치목표만 바꾸고 원의 속도/가속도 feedforward를 관측/제어에 추가하지 않는다. 실제첫실패종료,autoreset뒤표본연결없음. 원평가optimizer0.']
for row in circle_rows:
 report.append(f"{row['label']} T{row['period_s']}: completed={row['completed_two_reference_laps']}, {row['duration_s']}s, RMSE={row['position_rmse_m']:.6f}m, Z RMSE={row['axis_rmse_m'][2]:.6f}m, Z p-p={row['z_peak_to_peak_m']:.6f}m, max tilt={row['max_tilt_deg']:.3f}deg, phase lag={row['phase_lag_mean_deg']:.3f}deg, action/allocation clip={row['action_clipping_fraction']}/{row['allocation_clipping_fraction']}.")
for T in [10,5]:
 old=next(x for x in circle_rows if x['model']=='baseline' and x['period_s']==T);new=next(x for x in circle_rows if x['model']=='new' and x['period_s']==T)
 if old['completed_two_reference_laps'] and new['completed_two_reference_laps']:
  report.append(f"T{T} 동일완료시간 비교: 3D RMSE 변화 {(new['position_rmse_m']/old['position_rmse_m']-1)*100:+.2f}%, Z RMSE 변화 {(new['axis_rmse_m'][2]/old['axis_rmse_m'][2]-1)*100:+.2f}%, Z p-p 변화 {(new['z_peak_to_peak_m']/old['z_peak_to_peak_m']-1)*100:+.2f}%.")
 else:report.append(f"T{T}: 완료여부/시간이 다르므로 짧은 RMSE만으로 우수판단하지 않음. 공통시간 CSV 별도.")
 report.append(f"T{T} 반경/자세 상충관계: 평균 signed 반경오차(음수=원 안쪽) {old['mean_radial_error_m']:.6f}m -> {new['mean_radial_error_m']:.6f}m; 반경오차 RMS {old['radial_error_rmse_m']:.6f}m -> {new['radial_error_rmse_m']:.6f}m; 전체 body 각속도 RMS {old['body_omega_rms_rad_s']:.6f}rad/s -> {new['body_omega_rms_rad_s']:.6f}rad/s. 전체 위치 RMSE 감소와 반경/각속도 지표 개선은 같지 않다.")
report+=['원에서 목표두바퀴분 시간을 비행했다는 뜻과 실제 방위각회전수/정확한원추종을 구분. 위치 추종 지연/반경오차도 함께 보아야 한다.',
'수직추력 그래프의 T는 실제 rotor합/actual wrench; T*cos(roll)*cos(pitch)는 저장된100Hz poststep자세로 마지막500Hz force buffer를 투영한 값이다. 500Hz전체평균 힘 복원은 아니다.',
'circle/attitude_vertical_thrust_* 참고자세는 기존 PID 외부루프를 저장 RL상태에 사후적용한 값이며 RL 내부 목표자세가 아니다. 별도 PID학습/새제어기 적용 없음.',
f"검증: fixed128 reload 최대action오차 {verify['max_reload_action_error']}; reward합 최대오차 {verify['reward_max_error']}; 실제backward loss residual {verify['loss_max_residual']}; TB/CSV/JSON 일치. 최초rollout 상태/action/motor/물리 동일, 보상차이는 Z속도항뿐.",
'모델 선택 iteration이 다르면 best원평가모델의 실제학습량도 다르다. 동일512예산의 모델선택 절차 비교이며 같은bestiteration을 강제하지 않았다.',
'seed42 하나, 정상동역학의국소초기위치학습이다. OOD 질량/고장 검증이나 일반적squared우월성/계수최적성으로 확대하지 않는다.',
'기존코드/phase1~12결과/설치/드라이버/TensorBoard서비스 보존. 새entrypoint만사용. GUI미실행,명령만제공. 지정512회완료후추가학습/튜닝없음.',
'학습그래프X축=이번새실행전이수0~16777216. 평가trajectoryX축=simulationtime/step. common_axis_baseline은기존phase12-1의13항을새상태에서진단한값이며학습보상에가산하지않음.',
'검토자료: verification.json, initial_position_validation.json, first_rollout_comparison.json, source_snapshot/, plots/*.csv와PNG, COMMANDS.txt, review.tar.gz/SHA256.']
report+=['','추가 해석 및 저장 정밀도:']
bv=next(x for x in hover_vertical_summary if x['model']=='phase12_baseline' and x['suite']=='fixed')
nv=next(x for x in hover_vertical_summary if x['model']=='best' and x['suite']=='fixed')
if bv['completed']==nv['completed']==7:
 report.append(f"고정7조건 best 비교: 3D RMSE 변화 {(nv['position_rmse_m']/bv['position_rmse_m']-1)*100:+.2f}%, Z RMSE 변화 {(nv['z_rmse_m']/bv['z_rmse_m']-1)*100:+.2f}%, 전체 Vz RMS 변화 {(nv['full_vz_rms_m_s']/bv['full_vz_rms_m_s']-1)*100:+.2f}%, 조건별 최대Z오차 평균 변화 {(nv['max_abs_z_error_m']/bv['max_abs_z_error_m']-1)*100:+.2f}%. 정확도와 속도/과도응답은 서로 다른 지표이다.")
report.append('후반 Z 표준편차가 원래 float32 world/link 위치와 환경원점 차분의 저장 해상도에 근접하는 경우, 수치 표현의 영향을 함께 고려해야 한다. CSV의 소수점 자릿수나 float64 후처리가 원래 물리 상태 정밀도를 높이지 않는다. 이를 실제 기체의 나노미터/마이크로미터 제어 정밀도로 해석하지 않는다.')
report.append('원 평가의 위상지연·Z오차·자세·힘 변동은 위 실측 표로 각각 판단한다. 특정지표의 개선만으로 진동없는 비행이나 모든 조건의 우월성을 선언하지 않는다.')
(r/'REPORT_phase13_squared_ko.txt').write_text('\n'.join(report)+'\n')
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
"$PY" scripts/train_phase13sq.py --run-dir "$RUN" --iterations 512 --device cuda:0 --headless --physics isaacsim_physx
# Hover evaluation only
"$PY" scripts/evaluate_phase13sq.py --checkpoints "$RUN/best.pt" "$RUN/final.pt" --run-dir "$RUN" --output "$RUN/evaluation/manual_$(date +%Y%m%d_%H%M%S)" --positions "$RUN/evaluation/initial_positions.json" --suite fixed --horizon 30 --device cuda:0 --headless --physics isaacsim_physx
# Circle evaluation only. Use period 10 or 5; always two reference laps.
"$PY" scripts/evaluate_circle_phase13sq.py --run-dir "$RUN/circle" --controller phase13sq_best --period 10 --output "$RUN/circle/manual_T10_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --headless --physics isaacsim_physx
# GUI not executed/verified in this experiment.
"$PY" scripts/evaluate_circle_phase13sq.py --run-dir "$RUN/circle" --controller phase13sq_best --period 5 --output "$RUN/circle/gui_T5_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --visualizer kit --physics isaacsim_physx
# Training/validation axes are transitions; exclude separate circle-time logs from this view.
"$PY" -m tensorboard.main --logdir_spec baseline_train:{base}/tensorboard,squared_train:{r}/tensorboard,squared_validation:{r}/tensorboard_validation --host 127.0.0.1 --port {port}
# PowerShell tunnel, then http://127.0.0.1:{port}
ssh -N -L {port}:127.0.0.1:{port} mrl_6534@100.79.103.47
# PowerShell download
scp "mrl_6534@100.79.103.47:{r}/review.tar.gz" .
"""
(r/'COMMANDS.txt').write_text(cmds)
dump_json(r/'report_execution.json',dict(report_and_comparison_plot_s=time.perf_counter()-tick,tensorboard_suggested_free_port=port,gui_executed=False,training_transitions=train['transitions'],optimizer_calls=train['optimizer_updates']))
print('REPORT_DONE',flush=True)

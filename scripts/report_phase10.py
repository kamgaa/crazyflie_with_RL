import argparse,json,time,socket,subprocess,shlex,shutil
from pathlib import Path
import phase10_tables as tables
import numpy as np
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();tick=time.perf_counter()
plan=json.loads((r/'experiment_plan.json').read_text());tr=json.loads((r/'training_result.json').read_text());sel=json.loads((r/'best_selection.json').read_text());ver=json.loads((r/'verification.json').read_text());summ=json.loads((r/'evaluation_summary.json').read_text());resume=json.loads((r/'resume_verification.json').read_text());df=tables.read_csv(r/'iterations.csv')
python='/home/mrl_6534/miniconda3/envs/isaaclab/bin/python';initial_eval=json.loads((r/'evaluation/validation_000/execution.json').read_text());final_exec=json.loads((r/'final_evaluation_execution.json').read_text());isolated=json.loads((r/'evaluation_isolation.json').read_text());best_it=sel['selected']['additional_iteration']
# Inspect sockets without changing existing services; reserve no listening port.
port=None
for candidate in range(6009,6030):
 with socket.socket() as sock:
  try:sock.bind(('127.0.0.1',candidate));port=candidate;break
  except OSError:pass
assert port is not None
cmd=f'''# Run from server project directory. This file contains no credentials.
cd {ROOT}
PY={python}
RUN={r}

# Actual 512-iteration continuation command; already completed.
# Duplicate execution against this run is rejected. Do not run again to replay.
"$PY" scripts/train_phase10.py --run-dir "$RUN" --additional-iterations 512 --device cuda:0 --headless --physics isaacsim_physx

# Reproduce in a newly prepared run, retaining the same phase9 final parent.
# This starts a new experiment only if the user explicitly chooses to execute it.
# "$PY" scripts/prepare_phase10.py
# NEW_RUN=$(cat runs/.phase10_current_path)
# "$PY" scripts/train_phase10.py --run-dir "$NEW_RUN" --preflight --additional-iterations 512 --device cuda:0 --headless --physics isaacsim_physx
# "$PY" scripts/evaluate_phase10.py --checkpoints "$NEW_RUN/initial.pt" --run-dir "$NEW_RUN" --output "$NEW_RUN/evaluation/validation_000" --positions "$NEW_RUN/evaluation/initial_positions.json" --suite fixed --horizon 30 --device cuda:0 --headless --physics isaacsim_physx
# "$PY" scripts/train_phase10.py --run-dir "$NEW_RUN" --additional-iterations 512 --device cuda:0 --headless --physics isaacsim_physx
# "$PY" scripts/finalize_phase10.py --run-dir "$NEW_RUN"

# Headless replay/evaluation: zero optimizer updates, unique output directory.
"$PY" scripts/evaluate_phase10.py --checkpoints "$RUN/best.pt" "$RUN/final.pt" --run-dir "$RUN" --output "$RUN/evaluation/manual_$(date +%Y%m%d_%H%M%S)" --positions "$RUN/evaluation/initial_positions.json" --suite fixed --horizon 30 --device cuda:0 --headless --physics isaacsim_physx

# GUI nominal replay: GUI not executed/verified in this work. No driver changes.
"$PY" scripts/evaluate_phase10.py --checkpoints "$RUN/best.pt" "$RUN/final.pt" --run-dir "$RUN" --output "$RUN/evaluation/gui_$(date +%Y%m%d_%H%M%S)" --positions "$RUN/evaluation/initial_positions.json" --suite fixed --case nominal --horizon 30 --device cuda:0 --visualizer kit --physics isaacsim_physx

# TensorBoard: port {port} was free when checked; existing 6006/6007/6008 untouched.
"$PY" -m tensorboard.main --logdir "$RUN" --host 127.0.0.1 --port {port}

# Windows PowerShell SSH tunnel:
ssh -N -L {port}:127.0.0.1:{port} mrl_6534@100.79.103.47
# Browser: http://127.0.0.1:{port}

# Windows PowerShell download:
$R = "{r}"
scp "mrl_6534@100.79.103.47:${{R}}/review.tar.gz" .
scp "mrl_6534@100.79.103.47:${{R}}/review.tar.gz.sha256" .
Get-FileHash .\\review.tar.gz -Algorithm SHA256
'''
(r/'COMMANDS.txt').write_text(cmd)
lineage=[]
for path in sorted((r/'evaluation').glob('validation_*/reload_checks.json')):
 d=json.loads(path.read_text())[0];lineage.append(dict(evaluation=str(path),checkpoint=d['checkpoint'],sha256=d['sha256'],infos=d['loaded_infos'],reload_max_error=d['max_action_error']))
dump_json(r/'lineage.json',dict(parent=plan['checkpoint'],parent_sha256=plan['parent_sha256'],base_iterations=1664,base_transitions=54525952,base_adam_updates=225280,candidates=lineage,best_selection=sel,final_counters=dict(iterations=2176,transitions=71303168,adam_updates=389120)))
lines=[f'Crazyflie Isaac Lab phase10: 복귀 속도 모델 추가 학습\n실행 폴더: {r}',
'RMS는 제곱평균제곱근, RMSE는 오차의 제곱평균제곱근이다. 고정 목표 local [0,0,1]m; e=link world position-env_origin-target. 속도 관측은 link 원점의 절대 world 속도. body FLU/world Z-up, 정책 quaternion wxyz를 유지한다.',
'1. 출발점·계보·설정',f"부모: {plan['checkpoint']}\nSHA256: {plan['parent_sha256']}\n고정 비교: {plan['control_checkpoint']}\nSHA256: {plan['control_sha256']}",
'이번 부모는 phase9 track_return FINAL이며 best가 아니다. control_stop FINAL은 평가만 했다. 부모 누적1664회/54,525,952전이/Adam225,280에서 이어 최종2176회/71,303,168전이/Adam389,120. Native framework iter는 마지막2175이며 완료 횟수와 1 차이가 있다. 다른 분기의 예산은 계보에 더하지 않는다.',
'PPO N1024, rollout32, batch32768, minibatches64(각512), epochs5, iteration당320 Adam calls. Actor/critic [64,64],tanh; LR3e-4 fixed,epsilon.2,gamma.99,lambda.95,entropy.003,value coef1,value clipping true,grad limit1,rollout 전체 advantage 정규화. 관측/보상 정규화 추가 없음. Timeout bootstrap/원래 reward의 구분 및 RSL-RL 연결 그대로.',
'물리 dt.002s, decimation5, 정책.01s. 15D 관측=e3+world v3+wxyz4+body omega3+yaw sin/cos2; action=[roll,pitch,yaw torque,collective thrust]. clip[-1,1] 후 scale=[.0075,.0075,.001,.5], 앞3단위Nm/마지막N, Fz에 본체질량 .04338*9.81=.4255578N 보상. Allocation→모터추력[0,.289]N→ESC역변환→RPM tau.050s→실제 COM wrench. 총질량.043384kg,COM/관성/반토크 등 기존 물리파일 유지. 기존 sleep_threshold=0 및 stabilization_threshold=0 유지; PID나 상태 강제고정 추가 없음.',
'학습8초, XYZ 각축 Uniform[-.05,.05]m 상자. 목표고정, 자세/속도/모터 호버초기화 동일. 고도[.2,2.5]m,tilt60deg,3D오차1.5m 실패 유지. 30초 평가만 시간제한 변경. DR/센서노이즈/OOD/필터/PID/관측추가/출력gain변경 없음.',
'2. 실제 보상',
'r_xy=.100*expm1(-(ex²+ey²)/.10²), r_z=.015*expm1(-ez²/.05²). v_des=norm_clip(-4e,1.5m/s); velocity=-.005||v-v_des||²; tilt=-6(qx²+qy²); omega=-.001||omega_body||²; yaw=-wrapped_yaw_error²; failure=-10*failure. 이미 부호가 있는 7항을 합하며 추가 마이너스/dt/양수상수 없음. 위치m,속도m/s,각속도rad/s,yawrad. v_des는 reward/진단용이며 외부 제어명령이 아니다. 실제 함수 phase9_reward.py를 변경 없이 재사용한다. JSON의 일부 위치/gain 필드는 기존 코드의 상수 설명용이며 이번에도 동일 값이다.',
'3. 복원과 검증',
'Actor/critic/Gaussian log_std 및 Adam 전체 state_dict/moments/param groups strict 복원. 13개 파라미터별 Adam step225280 확인; parameter step의 합을 update 횟수로 세지 않았다. Python/NumPy/Torch CPU/CUDA RNG 모두 첫 rollout 직전 복원. 저장된 전용 reset Generator도 복원한 다음 새 reset 표본을 생성. seed42의 계보이며 대체 난수 seed 없음. 물리상태·episode·rollout은 저장되지 않아 새 reset에서 시작; 중단없는 물리궤적 재개가 아니다.',
'사전 점검16 policy batch steps=16384전이는 학습 예산 외, optimizer0. 실제 부분reset/80물리갱신/관측유한성/reward합/부모Adam불변 확인. 학습 전 부모30초 평가도 별도 프로세스. 매64회 부모 학습 프로세스는 동기적으로 기다리며 별도 평가 프로세스가 실행; 복귀후 RNG/obs/motor/reset RNG/episode/Adam step/rollout index 불변 확인. 학습·평가 GPU 작업 중첩 없음.',
'4. 실제 예산·시간·메모리',
f"추가 iterations={tr['additional_iterations']}; 전이={tr['additional_transitions']:,}; 실제 optimizer post-hook 호출={tr['additional_optimizer_updates']:,}; 최종 Adam 내부 steps={tr['adam_steps']}. 계측한 physics batch steps={tr['additional_physics_batch_steps']}. 학습초기화={tr['initialization_s']:.3f}s; 추가학습={tr['training_wall_s']:.3f}s; 처리량={tr['transitions_per_s']:.1f} transitions/s.",
f"학습 전 부모 평가 프로세스 wall={initial_eval['total_wall_s']:.3f}s; 중간 평가 subprocess wall={tr['intermediate_evaluation_wall_s']:.3f}s; 최종평가 subprocess wall={sum(x['wall_s'] for x in final_exec):.3f}s. 각 평가 execution.json에서 초기화와 simulation/후처리 포함 평가시간 구분. 추가학습 시간은 rollout/update/log/checkpoint 및 격리확인 오버헤드 포함, 평가 subprocess 대기시간 제외. 처리량 분자16,777,216/그 추가학습시간. 부모누적학습초는 부모 checkpoint infos의1485.13664에서 이어진다. 초기화와 평가를 처리량 분모에 섞지 않았다.",
f"장치전체 sampled peak={tr['cuda_device_used_peak_sampled_MiB']:.2f}MiB(cudaMemGetInfo; 모든프로세스/PhysX포함, 프로세스 단독 아님). Torch allocator 통계: {json.dumps(tr['memory_final'],ensure_ascii=False)}. memory 정의는 재사용 phase5_common.memory, 해당 프로세스의 CUDA 전체 메모리(NVML)는 별도로 측정하지 않았다. Torch allocator는 프로세스 Torch만 포함하고 PhysX 할당 전체를 포함하지 않는다.",
'5. 사전 고정 best 선택·평가',
'고정7조건30초 완료개수 우선. 7미만끼리는 평균지속시간,7/7끼리는 평균3D RMSE,동률은 추가iteration이른순. 부모0회 포함. 20조건은 이미 phase9에서 평가된 고정 목록을 그대로 사용했고 선택/튜닝에 사용하지 않았다. OOD나 미공개테스트가 아니다.',
f"best=추가 {best_it}회, 부모선택={sel['parent_selected']}; best/final 동일해시={sel['best_final_identical']}. best 선택은 위치중심이며 최소진동 모델을 뜻하지 않는다.\nbest SHA256={sel['best_sha256']}\nfinal SHA256={sel['final_sha256']}",
'checkpoint | 완료/7 | 평균 지속s | 완료 평균3D RMSEm | 후반 omegaRMS rad/s | 후반 rollRMS rad | 후반 action포화']
for row in summ['candidates']:
 lines.append(' | '.join(str(row[k]) for k in ['additional_iteration','completed_count','mean_duration_s','completed_only_mean_position_rmse_m','completed_only_mean_tail_body_omega_rms_rad_s','completed_only_mean_tail_roll_rms_rad','completed_only_mean_tail_applied_action_saturation_fraction']))
lines+=['모델 | suite | 완료 | RMSEm | 후반속력m/s | 후반속도오차m/s | 후반omega rad/s | 후반XYZ bias m | 후반XYZ std m | action포화']
for row in summ['final']:
 lines.append(f"{row['model']} | {row['suite']} | {row['completed_count']}/{row['conditions']} | "+' | '.join(str(row.get(k)) for k in ['completed_only_mean_position_rmse_m','completed_only_mean_tail_speed_rms_m_s','completed_only_mean_tail_velocity_error_rms_m_s','completed_only_mean_tail_body_omega_rms_rad_s'])+' | '+str([row.get(f'completed_only_mean_tail_error_vector_m_{i}') for i in range(3)])+' | '+str([row.get(f'completed_only_mean_tail_position_error_m_std_{i}') for i in range(3)])+' | '+str(row.get('completed_only_mean_tail_applied_action_saturation_fraction')))
lookup={(x['model'],x['suite']):x for x in summ['final']}
lines+=['6. 추가 학습 효과 및 한계']
for suite in ['fixed','random20']:
 parent=lookup['parent',suite]
 for name in ['best','final']:
  new=lookup[name,suite];lines.append(f"{suite} {name} vs 부모: 완료 {parent['completed_count']}/{parent['conditions']}→{new['completed_count']}/{new['conditions']}")
  if parent['completed_count']==parent['conditions'] and new['completed_count']==new['conditions']:
   for label,key in [('위치 RMSE','position_rmse_m'),('후반 속력 RMS','tail_speed_rms_m_s'),('후반 각속도 RMS','tail_body_omega_rms_rad_s'),('후반 roll RMS','tail_roll_rms_rad'),('후반 pitch RMS','tail_pitch_rms_rad'),('후반 Z centered std','tail_z_std_m')]:
    old=parent['completed_only_mean_'+key];newv=new['completed_only_mean_'+key];lines.append(f'  {label}: {old:.9g}→{newv:.9g}; {(newv/old-1)*100:+.2f}%')
  else:lines.append('완료율 차이가 있으므로 완료집합 평균 RMSE만으로 우열 판단하지 않음. paired_completed_comparison.json 참조.')
for suite in ['fixed','random20']:
 for model in ['parent','control_stop','best','final']:
  row=lookup[model,suite]
  lines.append(f'{suite} {model} 후반 제어명령/모터 변동 (조건별 지표 평균):')
  for name,n in [('tail_raw_action_std',4),('tail_applied_action_rate_rms_s_inv',4),('tail_actual_wrench_std',4),('tail_rpm_std',4),('tail_thrust_N_std',4)]:
   lines.append(name+' = '+str([row.get(f'completed_only_mean_{name}_{i}') for i in range(n)]))
lines+=['학습 reward 첫64회→마지막64회 평균: '+str(df['Reward/total'].head(64).mean())+'→'+str(df['Reward/total'].tail(64).mean()),'실제 total PPO loss 첫64회→마지막64회 평균: '+str(df['Loss/ppo_total'].head(64).mean())+'→'+str(df['Loss/ppo_total'].tail(64).mean()),'Reward/loss 변화는 학습중 확률적 상태분포 통계이며 deterministic 비행성능의 대리판정으로 쓰지 않는다. 위치/자세/각속도/action포화를 별도로 비교한다. 고주파 파형만으로 high gain 원인이나 폐루프 안정성이 입증된 것은 아니다. 고정 control_stop은 추가 학습하지 않았으므로 그 모델 대비 차이로 속도 보상 방식만의 독립 효과를 판단할 수 없다. 공통 단일 부모·seed42·정상동역학의 추가 학습 관찰이며 다른 seed/처음부터 학습/OOD에 일반화하지 않는다. 추가 reward 튜닝/학습/조건은 실행하지 않았다.',
'7. 지표·스펙트럼·로그 정의',
'표의 모델별 수치는 완료한 각 평가조건에서 계산한 지표의 산술평균이며 여러 trajectory를 이어 붙인 RMS가 아니다. 실패가 있으면 완료개수와 paired 동일완료조건 표를 함께 본다. Full은 첫 step0.01~30.00s, tail은 완료episode의20.01~30.00s 1000표본(20~30초구간). 실패episode에는 없는 후반을 null/N/A로 기록. XYZ bias는 오차mean,centered std는 population std(ddof0),RMS는 mean 제거 전 제곱평균제곱근이다. Z bias와 진동std를 구분. Tilt는 quaternion에서 bodyZ-worldZ 각도이며 Euler roll로 대체하지 않는다. 각 rotor RPM=omega(rad/s)*60/(2pi). 요청 wrench 순서[tx,ty,tz,Fz],단위[Nm,Nm,Nm,N].',
'PSD는 기존 phase8 spectral 재사용: fs100Hz,구간mean제거,symmetric numpy.hanning(N),one-sided abs(rfft)^2/(fs*sum(window²)),비DC/nonNyquist2배,df=.1Hz(후반10초),band power=sum(PSD)*df. roll/pitch단위rad²/Hz,omega (rad/s)²/Hz. JSON에설정/원시RMS/centeredRMS/PTP/전체적분파워/10~20Hzpower를 함께 보존;최대peak나부분대역만으로개선판정하지 않음.',
'raw action 초과: abs(raw)>1(추가epsilon0); applied포화: abs(applied)>=1-1e-6. 축별분모=해당구간step수,전체비율분모=step수*4. Allocation clip시도: unconstrained rotor force<0 or>.289N. command bound는 clippedcommand<=1e-7 or>=.289-1e-7; actual thrust bound도 같은허용오차이며 clipping시도와구별. 후반action차분은 동일episode구간내diff/dt. 원시CSV에 raw/applied/action·allocation·command·actual torque/force·RPM/thrust 저장.',
'훈련 TensorBoard X=추가 transitions32768..16777216. 추가/누적iteration,누적전이/Adam별도태그. Reward/* signed기여,Reward/failure는기존termination별칭. Reward/step/* 기존태그도유지. Loss/value는계수전,Loss/value_contribution계수후,Loss/entropy_contribution=-entropycoef*entropy,Loss/ppo_total은실제backward직전계측. AST비교로probe제거시원본PPO계산과동일함을확인. Policy/std는마지막minibatch/최종update직전,logstd파라미터는최종update후. PPO clipfraction은확률비clipping,TrainStochastic/Actuator/*은학습명령clipping. evaluation JSON full/tail_action지표는deterministic이다. Episode통계는환경별완료시에만정산하고미완료episode를rollout경계에서초기화하지않는다.',
'8. 최종 검증',json.dumps(ver,ensure_ascii=False,indent=2),
'실제 학습·평가 backend는각execution/resolved JSON참조. 추가학습결과와고정관측네트워크reload일치,PhysXtrajectory의수치적재현성은서로다른검증이다. 기존부모·고정비교·관측/물리/보상코드해시보존확인. phase10 새스크립트만추가했으며기존기본동작유지. GUI 실행안함/미검증,드라이버변경없음.',
'9. 파일·명령',
'후처리 최초 실행에서 pandas 미설치로 import가 중단됐다. 학습/평가에는 영향이 없으며 패키지 설치 없이 phase10_tables.py의 CSV/NumPy 처리로 수정했다. 학습/평가 재실행이나 추가 optimizer update 없음. 새 파일: prepare_phase10.py(부모/예산/보상 점검 및 새 run), phase10_common.py(부모 누적 카운터와 strict 복원 검증), train_phase10.py(기존 수집/PPO/로깅 재사용,512회 및 중간 평가 격리), evaluate_phase10.py(기존 평가 경로+추가 진단), phase10_evalmetrics.py(후반 자세/모터/clipping/PSD), finalize_phase10.py(고정 선택 및 새 프로세스 최종 평가), analyze_phase10.py(CSV/TensorBoard검사/플롯), report_phase10.py, package_phase10.py, complete_phase10.py(학습 종료 후 평가/전달 순차 실행). source_snapshot과 evidence/train_phase9_to_phase10.patch에 실제 코드 보존.',f'보고서 {r}/REPORT_phase10_ko.txt\nbest {r}/best.pt\nfinal {r}/final.pt\n전달 {r}/review.tar.gz\n명령 COMMANDS.txt. TensorBoard loopback {port} 확인시빈포트,기존6006/6007/6008서비스미변경.',
'주요플롯:validation_position.png,validation_oscillation.png,nominal_states_30s.png,nominal_attitude_29_30s.png,nominal_actuator_{parent,control_stop,best,final}.png,nominal_tail_spectra.png,training_reward_loss.png. 대응CSV:checkpoint_comparison.csv,checkpoint_conditions.csv,final_comparison.csv,all_evaluation_metrics.csv,nominal_30s_comparison.csv,nominal_tail_spectra.csv,training_metrics.csv. 전체조건원시trajectory는evaluation/에보존.']
fixed_parent=lookup['parent','fixed'];fixed_final=lookup['final','fixed']
conclusion=('핵심 결론: 설정을 유지한 추가512회 학습에서 부모/고정비교/best/final 모두 고정7조건과 기존20조건30초를 완주했다. '
 f"위치 중심 best는 추가{best_it}회로 final과 같은 SHA256이다. 고정7조건 평균 위치 RMSE는 "
 f"{fixed_parent['completed_only_mean_position_rmse_m']*100:.4f}→{fixed_final['completed_only_mean_position_rmse_m']*100:.4f}cm, "
 f"후반 각속도 RMS는 {fixed_parent['completed_only_mean_tail_body_omega_rms_rad_s']:.6f}→{fixed_final['completed_only_mean_tail_body_omega_rms_rad_s']:.6f}rad/s로 감소했다. "
 '이번 관측 범위에서는 위치 정확도와 자세/제어명령 진동 모두 추가 학습 효과가 있었다. '
 '초기192회 후보처럼 위치 오차는 줄지만 각속도가 커진 구간도 있어 두 지표는 별개다. '
 '최종 후반 XYZ 평균 편향은 약[+0.577,+0.417,+1.932]mm로 완전히0은 아니며, Z 진동std 감소를 Z 오차 제거로 해석하지 않는다. '
 '고정 비교 control_stop은 학습량이 달라 보상 방식만의 인과 비교가 아니다. 단일계보/seed42/정상동역학/고정목표30초 검증 범위이다.')
lines.insert(1,conclusion)
lines.append('Nominal 후반 PSD: 부모 roll peak15.2Hz→final12.7Hz. roll 10~20Hz power2.93539922e-5→1.19077136e-10 rad²; 전체 PSD적분2.93746403e-5→1.22434720e-10 rad²; mean-removed roll RMS .00542069268→.0000110605109rad. 중심 주파수 이동만이 아니라 원본 RMS와 전체 파워도 함께 감소했다. 주파수별/축별 자료는 spectra JSON/CSV 참조.')
relations=[]
for suite in ['fixed','random20']:
 b=json.loads((r/'evaluation'/f'best_{suite}_30s/results.json').read_text());f=json.loads((r/'evaluation'/f'final_{suite}_30s/results.json').read_text())
 assert [x['condition'] for x in b]==[x['condition'] for x in f]
 relations.append(dict(suite=suite,same_checkpoint_sha256=sel['best_final_identical'],separate_processes=True,max_position_rmse_difference_m=max(abs(x['position_rmse_m']-y['position_rmse_m']) for x,y in zip(b,f)),completion_equal=all(x['completed']==y['completed'] for x,y in zip(b,f))))
dump_json(r/'best_final_relation.json',relations)
(r/'REPORT_phase10_ko.txt').write_text('\n\n'.join(lines))
dump_json(r/'timing_summary.json',dict(training_initialization_s=tr['initialization_s'],additional_training_s=tr['training_wall_s'],initial_parent_evaluation_process_s=initial_eval['total_wall_s'],intermediate_evaluation_process_s=tr['intermediate_evaluation_wall_s'],final_evaluation_process_s=sum(x['wall_s'] for x in final_exec),postprocessing=json.loads((r/'postprocessing_time.json').read_text()),report_generation_s=time.perf_counter()-tick,tensorboard_port=port))
for name in ['scripts','src']:shutil.copytree(ROOT/name,r/'source_snapshot'/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
subprocess.run(['git','diff','--no-index',str(ROOT/'scripts/train_phase9.py'),str(ROOT/'scripts/train_phase10.py')],stdout=(r/'evidence/train_phase9_to_phase10.patch').open('w'))
print('REPORT_DONE',flush=True)

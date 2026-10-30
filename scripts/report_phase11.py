import argparse,json,csv,time,subprocess
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir);tick=time.perf_counter()
def load(n):return json.loads((r/n).read_text())
plan=load('experiment_plan.json');tr=load('training_result.json');sel=load('best_selection.json');summary=load('evaluation_summary.json');ret=load('return_metrics_summary.json');ver=load('verification.json');reset=load('initial_position_validation.json');fresh=load('fresh_verification.json');focused=load('focused_verification.json')
rows=[json.loads(s) for s in (r/'metrics.jsonl').read_text().splitlines()];ev=load('final_evaluation_execution.json')
fmt=lambda x:'N/A' if x is None else f'{x:.7g}' if isinstance(x,(float,int)) else json.dumps(x,ensure_ascii=False)
lines=[]
def add(x=''):lines.append(str(x))
add('기준 제어기 실험 1 — 축별 보상과 완화된 자세 벌점')
add('Crazyflie Isaac Lab phase11 / 정상 동역학·고정 목표·seed42 새 정책 학습')
add('실행 폴더: '+str(r));add()
add(f"512회 학습과 실제 Adam {tr['optimizer_updates']:,}회 완료. 새 실행 카운터는 0에서 시작, 기존 모델 학습량을 더하지 않았다. best는 이번 실행 {sel['selected']['additional_iteration']}회이며 final은 512회. 동일 파일 해시 여부: {sel['best_final_identical']}. best는 위치 중심 규칙이며 최소 진동 모델을 뜻하지 않는다.")
add('1~10단계는 환경 이식·예비 학습 기록으로 보존. 이번은 새 초기화·첫 episode부터 XYZ 독립 ±5cm 위치 분포·보상 변경을 함께 적용한 기준 제어기 설계 실험이다. 보상 분리 또는 자세 가중치 감소 하나만의 독립적 인과 효과가 아니다. phase10은 더 긴 다른 학습 이력을 가진 고정 참고 모델이며 재학습하지 않았다. 단일 seed42·정상 동역학·고정 목표의 국소 복귀 평가이며 OOD/원형 추종 검증이 아니다.')
add('\n1. 실제 학습과 평가 성능')
add('RMS=제곱평균제곱근; RMSE=오차의 제곱평균제곱근. 표의 RMSE·후반 지표는 완료 조건만의 평균이다. 완료 수가 다르면 같은 품질로 직접 비교하지 않는다. 모든 조건별 수치는 plots/all_evaluation_metrics.csv 및 conditionwise_physical_metrics.csv, 공통 완료 조건 비교는 paired_completed_comparison.json.')
add('model | suite | 완료 | 평균3D RMSE m | 초기0~5s RMSE m | 후반 속력 m/s | 후반 omega rad/s | 후반 적용action포화 | 정착 조건 수/전체 | 정착 평균s')
for x in summary['final']:
 q=next(z for z in ret if z['model']==x['model'] and z['suite']==x['suite'])
 add(' | '.join(map(str,[x['model'],x['suite'],f"{x['completed_count']}/{x['conditions']}",fmt(x.get('completed_only_mean_position_rmse_m')),fmt(q['mean_early_rmse_m']),fmt(x.get('completed_only_mean_tail_speed_rms_m_s')),fmt(x.get('completed_only_mean_tail_body_omega_rms_rad_s')),fmt(x.get('completed_only_mean_tail_applied_action_saturation_fraction')),f"{q['settled_count']}/{q['count']}",fmt(q['mean_settling_s'])])))
add('고정7조건 ±5cm 복귀 축 6조건의 초기0~5초 다른축 이탈/overshoot:')
for q in ret:
 if q['suite']=='fixed':add(f"{q['model']}: 평균 cross最大 {fmt(q['mean_cross_early_m'])}m; 최악 cross最大 {fmt(q['max_cross_early_m'])}m; 평균 초기축 overshoot {fmt(q['mean_overshoot_early_m'])}m")
for q in ret:
 if q['suite']=='fixed':add(f"{q['model']} nominal 제외 ±5cm 6조건 정착시간 평균={fmt(q['mean_non_nominal_settling_s'])}s")
add('정착=3D오차<=1cm 진입 후 최소1초 유지하면서 평가 끝까지 다시 이탈하지 않는 최초시점. 실패는 N/A. 전체30초와 초기0~5초 구분. 초기표본 t=0은 저장한 첫 관측을 사용하고 이후100Hz .01~30초 표본이다. 기존 선택 RMSE는 .01~30초 표본 정의를 유지하며 full0/early0_5 지표는 t=0을 포함한다.')
add('\n체크포인트 | 완료/7 | 평균 지속s | 완료조건 RMSEm | 후반 omegaRMS | 후반 action포화')
for x in summary['candidates']:add(' | '.join(fmt(x.get(k)) for k in ['additional_iteration','completed_count','mean_duration_s','completed_only_mean_position_rmse_m','completed_only_mean_tail_body_omega_rms_rad_s','completed_only_mean_tail_applied_action_saturation_fraction']))
add('\nphase10 대비 물리 비교(완료 조건 수 동일하고 모두 완료한 경우에만 전체 평균 변화율):')
base=next(x for x in summary['final'] if x['model']=='phase10' and x['suite']=='fixed')
for label in ['best','final']:
 x=next(x for x in summary['final'] if x['model']==label and x['suite']=='fixed')
 add(label+':')
 if x['completed_count']==base['completed_count']==7:
  for key,title in [('position_rmse_m','전체 위치 RMSE'),('tail_body_omega_rms_rad_s','후반 body각속도'),('tail_roll_rms_rad','후반roll RMS'),('tail_pitch_rms_rad','후반pitch RMS'),('tail_speed_rms_m_s','후반속력')]:
   k='completed_only_mean_'+key;b=base[k];v=x[k];add(f'  {title}: {b:.9g} -> {v:.9g}; {(v/b-1)*100:.3f}% 변화')
 else:add('  완료 수가 달라 실패의 짧은 궤적 RMSE로 우열을 매기지 않는다. paired_completed_comparison.json을 함께 확인.')
 add('  후반 XYZ signed bias m='+str([x.get(f'completed_only_mean_tail_error_vector_m_{i}') for i in range(3)]))
 add('  후반 roll/pitch 평균제거 std rad='+str([x.get(f'completed_only_mean_tail_attitude_rad_std_{i}') for i in range(2)]))
 add('  후반 실제 roll/pitch 토크 std Nm='+str([x.get(f'completed_only_mean_tail_actual_wrench_std_{i}') for i in range(2)]))
 add('  축별 조건의 편향은 conditionwise_tail_xyz_bias.png에 개별 표시. 서로 반대편향 상쇄를 정확도 개선으로 해석하지 않는다.')
new=next(x for x in summary['final'] if x['model']=='final' and x['suite']=='fixed')
if new['completed_count']==base['completed_count']==7:
 br=next(x for x in ret if x['model']=='phase10' and x['suite']=='fixed');nr=next(x for x in ret if x['model']=='final' and x['suite']=='fixed')
 add('물리적 해석: 전체 위치 RMSE와 초기 복귀·정착은 '+('개선' if new['completed_only_mean_position_rmse_m']<base['completed_only_mean_position_rmse_m'] else '개선되지 않음')+'으로 관측됐다. '+f"±5cm 6조건 평균 정착 {fmt(br['mean_non_nominal_settling_s'])}→{fmt(nr['mean_non_nominal_settling_s'])}s, 초기 다른축 최악 이탈 {fmt(br['max_cross_early_m'])}→{fmt(nr['max_cross_early_m'])}m.")
 for i,axis in enumerate(['roll','pitch']):
  k=f'completed_only_mean_tail_attitude_rad_std_{i}';add(f"후반 {axis} 평균제거 std {base[k]:.8g}→{new[k]:.8g}rad: "+('증가' if new[k]>base[k] else '감소')+'했다. 이는 정적인 자세 편향을 뺀 변동이다.')
 add('이 결과는 위치 정확도·빠른 복귀와 미세한 후반 자세/속도 변동 사이의 손익을 보여준다. 실제 roll 토크 변동이 줄어도 Euler 자세변동까지 줄었다는 뜻은 아니다. 고주파 또는 특정 PSD peak만으로 high gain 원인을 확정하지 않는다.')
add('\n2. 신규 초기화와 설정')
add('학습 코드에는 runner.load 호출이 없다. resume=false, load_run/load_checkpoint 빈 문자열. Actor/critic은 기존 RSL-RL OnPolicyRunner/MLPModel 생성자 방식으로 새로 생성. 환경 생성 후 seed_all(42)로 Python/NumPy/Torch CPU/CUDA seed를 설정해 네트워크 초기화 시 환경의 난수 소비와 분리. 초기 Gaussian log_std 네 축 모두 -1.5, std=exp(-1.5). 새 Adam state count=0; 첫 optimizer post-hook에서 파라미터별 step1 및 13개 moments 생성 확인. initial.pt에는 이번 새 모델과 빈 optimizer만 저장. 새학습0회 모델도 선택 후보이며 phase10은 best 후보가 아니다.')
add('초기 actor SHA256='+fresh['actor_sha256']+'; critic SHA256='+fresh['critic_sha256'])
add('학습물리 상태·이전 episode·기존 RNG/weights/Adam은 복원하지 않음. 독립 CUDA reset Generator seed42를 RSL wrapper 최초 reset 전에 설치. 최초부터 랜덤 위치. 사전 검증은 별도 프로세스 optimizer0, 본학습은 다시 새 모델·새 optimizer·새 난수로 시작; preflight와 초기모델/정책RNG/실제초기관측 일치 확인.')
add('N1024, rollout32, batch32768, epochs5, minibatches64, size512, 320 Adam calls/iteration. [64,64]tanh actor/critic, LR3e-4 fixed, epsilon.2,gamma.99,lambda.95,entropy.003,value1; value clipping true, actor/critic 각 grad limit1; rollout 전체 advantage정규화, obs/reward normalization 추가 없음. Adam 기본값과 param groups는 optimizer_config.json. Timeout bootstrap은 기존process_env_step 그대로이며 환경 원래 reward/episode return에 보정 reward를 섞지 않는다. KL조기중단/적응LR/episode시작시간분산 없음.')
add('환경 metadata: evidence/environment_versions.json 참조. 로드된 IsaacLab은 지정 IsaacLab-3.0.0-EA/source/isaaclab editable 경로이며 설치 distribution metadata는17.0.2로 표시된다(폴더/릴리스명과 metadata버전표기가다름). PyTorch2.11.0+cu128, IsaacSim6.1.0.0, RSL-RL5.4.1, Newton1.5.2. 이번작업에서패키지/드라이버/설치코드변경없음.')
add('물리/관측/행동은 phase10과 같은 코드 및 설정: isaacsim_physx cuda0, physics.002s/decimation5, policy.01s, 학습8초. 15D=e3+absolute world v3+wxyz4+bodyω3+yaw sin/cos2. Link origin 상태 사용, env_origin 제거. worldZup/bodyFLU. Action=[rollτ,pitchτ,yawτ,Fz], clip±1, scale=[.0075,.0075,.001,.5] (앞3 Nm,마지막N), Fz중력보상 .04338*9.81N. bodymass.04338,total.043384kg; COM/관성/배치/반토크/모터추력상한.289N/지연.050s 그대로. Allocation→clip→ESC역변환→RPM지연→actual COMwrench. PID/필터/gain조정/DR 없음. 실패 고도[.2,2.5],tilt60deg,오차1.5m; timeout은 실패벌점 제외.')
add('\n3. 13개 signed reward와 단위')
add('e=link world position-env_origin-[0,0,1], m. v_des=norm_clip(-4e,1.5m/s), 방향 보존·norm 제한, reward/진단 전용. 관측의 절대world속도 유지.')
add('position_x=.100*expm1(-ex²/.10²); position_y=.100*expm1(-ey²/.10²); position_z=.015*expm1(-ez²/.05²)')
add('velocity_x/y/z=-.005*(v_axis-v_des_axis)²; roll=-.15*roll²; pitch=-.15*pitch²; yaw=-wrapped_yaw_error²; omega_x/y/z=-.001*body_omega_axis²; failure=-10*physical_failure.')
add('총보상=위13항합. 추가 minus/항수나눔/dt/양수상수/생존보상 없음. 기존 통합XY/tilt는 중복합산되지 않는다. m,m/s,rad,rad/s; roll/pitch는 정규화wxyz quaternion의 ZYX Euler. body각속도는 Euler미분과 다르다.')
add('X/Y는 각각.1, 목표근처 각각-10e²로 기존과 같으나 먼곳합은-.2로 기존통합XY-.1보다 강하다. 기존 -6(qx²+qy²)는 작은기울기에서 약-1.5roll²-1.5pitch²; .15는 근처벌점 약10배 완화 시작값이다. 속도/각속도 합계는 같은가중치이면 이전 제곱합과 동일. sigma는 길이척도이지 성공허용오차가 아니다. 이 계수 최적성/일반성 입증 아님.')
add('실제 계산용 설정: reward_config.json의 position.{x,y,z}.{weight,sigma_m},axis_weights.velocity/attitude/angular_velocity,failure_weight,return_gain_s_inv,return_speed_limit_m_s. legacy weights/sigma_xy 등은 기존 common reward 모니터링 호환용이며 실제13항에 추가되지 않는다. legacy RewardConfig/exponential_active=0은 통합XY exponential 모드가 아님을 뜻하며 새축별 exponential이 꺼졌다는 뜻이 아니다. 실제모드는 reward_id=phase11_axis/position_mode=axis_exponential.')
add('\n4. 초기 분포·정밀도')
add(json.dumps(reset,ensure_ascii=False,indent=2))
add('각축독립 Uniform±.05m 상자, 모서리거리 최대.0866m. 표본강제균등/축복사 없음. 초기 수평/yaw0, 선속도·각속도0, 기존 bodymass 기준모터호버 초기화. 첫행동 전 rawworld/env_origin/actualpolicy오차를 initial_reset.pt에 보존. 이후부분reset actual도 preflight_subsequent_reset.pt. 학습전체 이후reset표본 통계는 Reset/*에 누적 기록(생성표본 기준), 실제검증표본과 구분.')
add('PhysX/link 위치는float32, CSV는Python float round-trip 표현으로 불필요한 소수점반올림 없음. world좌표와env_origin 및 position_precision/world float32 ULP를 평가JSON/CSV에 저장. 큰env_origin에서 뺄셈 전 위치양자화가 생길 수 있고 local subtraction으로 사라지지 않는다. 극소std/0std는 float32해상도 영향을 받을 수 있어 실제기체 정밀도로 확대하지 않는다.')
add('\n5. 학습·평가·시간·메모리')
add(json.dumps({k:tr[k] for k in ['iterations','transitions','optimizer_updates','adam_steps','framework_iteration','initialization_s','training_wall_s','transitions_per_s','initial_evaluation_wall_s','intermediate_evaluation_wall_s','cuda_device_used_peak_sampled_MiB','memory_final']},ensure_ascii=False,indent=2))
add('실제 optimizer.step post-hook 횟수와 각 parameter Adam step 확인. parameter step을 합산하지 않았다. framework마지막 index511은 완료512회. 물리batch갱신81920, env별물리전이83886080. 처리량 분자16777216 / 학습시간(rollout/update/log/checkpoint/격리검증 포함, 평가subprocess대기 제외). 초기화·평가·후처리 분리.')
add('cudaMemGetInfo sampled peak는 장치전체(다른프로세스/PhysX포함), 프로세스 단독아님. torch allocator allocated/reserved/peak는 본Torch만, PhysX전체아님. NVML프로세스GPU메모리 미측정. framework내CPU동기화는 유지, 프레임워크 최적화 안함.')
add('최종 평가 process wall 합='+fmt(sum(x['wall_s'] for x in ev))+'s. 개별 execution.json은 초기화와 평가 구간을 분리. 학습중평가 프로세스 실행시 학습은대기하며GPU학습/평가중첩없음. 평가전후RNG/관측/모터/resetRNG/episode/Adamstep/storage index불변을 evaluation_isolation.json에 기록.')
add('\n6. 선택·재로드·검증')
add('사전고정규칙: 완료수우선; 7미만동률이면지속시간; 7/7끼리RMSE; 동률이른시점. 후보0,64,...512; phase10제외. 기존20조건은 phase10 실제JSON SHA동일, 이미사용한 조건이며 미공개/OOD가 아니다. best선택에사용하지않음. best/final/phase10을 별도프로세스에명시로드하여fixed7/random20 평가. 같은best/final도 이번에는 독립평가해 확인했다. 실패첫시점에서CSV종료하며 autoreset상태미포함. GUI는실행안함.')
add('best SHA256='+sha256(r/'best.pt'));add('final SHA256='+sha256(r/'final.pt'));add('phase10 final SHA256='+plan['comparison_sha256'])
add(json.dumps(ver,ensure_ascii=False,indent=2));add(json.dumps(focused,ensure_ascii=False,indent=2))
add('후처리 첫 실행에서 PNG 색상코드 오타를 수정했다. 실패한 분석로그·이벤트는 evidence/postprocess_attempt1에 보존했고 후처리만 재실행했다. 학습/평가/optimizer는 재실행하지 않았다. TensorBoard 실행명령은 검증된 train/validation 디렉터리만 지정하므로 실패한 후처리 이벤트를 섞지 않는다.')
add('기존소스/체크포인트/보고서/설정SHA 보존. 새phase11 전용스크립트만추가; source_snapshot 및 source_change_manifest.json으로재현. 실제loss를backward직전에기록하는기존AST검증probe재사용. loss=policy+value_coef*value-entropy_coef*mean_entropy (이설정에서추가loss없음); 실측잔차로검증. reward항은Loss가아니다.')
add('\n7. TensorBoard와 플롯')
add('학습X축 새policy transitions=32768..16777216, iteration/Adam별태그. Reward/{13항,total}=계수·부호적용된env reward 평균; Reward/episode=환경별완료episode누적평균,rollout경계에서reset하지않음. 완료0이면평균누락. Reward/common_stop/common_return은기존phase9정의진단용,학습보상에더하지않음. Loss/*실제목적함수,Policy/entropy,std,PPO/KL,ratio clip fraction과TrainStochastic/Actuator/action clipping구분. Gradient actor/critic preclip norm 및clip비율. 평가TensorBoard는새전이축 validation과시뮬레이션step축trajectory를별도폴더.')
add('핵심플롯: validation_position.png,validation_oscillation.png,early_position_velocity_0_5s.png,early_attitude_omega_0_5s.png,early_raw_applied_actions_0_5s.png,early_requested_actual_torque_0_5s.png,early_rotor_thrust_0_5s.png,nominal_waveforms_29_30s.png,conditionwise_tail_xyz_bias.png,cross_axis_overshoot_settling.png,training_reward_loss.png. 대응CSV 및모든원시trajectory보존. baseline수평선은학습이력다른phase10참고. 실패누락후반은N/A,0채움안함.')
add('위치 정확도와 복귀속도 개선이 자세진동 감소를 의미하지 않는다. 특히 평균제거 roll/pitch std, body omega RMS, 후반 속력, 실제 토크 std를 별개로 비교한다. 부드러운 토크와 작은 각속도만으로 모든 자세변동이 줄었다고 쓰지 않는다.')
add('위치·진동·축간 이탈·정착을 별도물리지표로 판단. reward정의다른phase10과총reward/loss크기순위비교안함. 학습reward상승/loss감소만으로비행개선주장안함. 지정512회종료; 추가학습/튜닝/다른seed/OOD실행없음.')
add('\n초기/말기64회 학습지표 평균(환경분포도변하므로 제어성능 대체 아님):')
for k in ['Reward/total','Reward/position_x','Reward/position_y','Reward/roll','Reward/pitch','Loss/policy','Loss/value','Loss/ppo_total','PPO/approx_kl','PPO/clip_fraction']:
 add(f'{k}: {sum(x[k] for x in rows[:64])/64:.8g} -> {sum(x[k] for x in rows[-64:])/64:.8g}')
(r/'REPORT_phase11_ko.txt').write_text('\n'.join(lines)+'\n')
# Read-only port discovery, no service startup or termination.
ss=subprocess.run(['ss','-ltn'],capture_output=True,text=True).stdout
port=next(v for v in range(6009,6020) if f':{v} ' not in ss)
PYTHON='/home/mrl_6534/miniconda3/envs/isaaclab/bin/python'
commands=f'''# MRL-server; no credentials stored.
cd {ROOT}
PY={PYTHON}
RUN={r}
# Actual fresh training command (already completed; same run refuses rerun):
"$PY" scripts/train_phase11.py --run-dir "$RUN" --iterations 512 --device cuda:0 --headless --physics isaacsim_physx
# Reproduce a NEW independent fixed-budget run only when explicitly desired:
# "$PY" scripts/prepare_phase11.py; then use NEW run path, validate reward, preflight, train.
# Evaluation only, no optimizer; output must not already exist:
"$PY" scripts/evaluate_phase11.py --checkpoints "$RUN/best.pt" "$RUN/final.pt" --run-dir "$RUN" --positions "$RUN/evaluation/initial_positions.json" --suite fixed --horizon 30 --output "$RUN/evaluation/manual_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --headless --physics isaacsim_physx
# GUI command only, NOT tested; run on server graphical session:
"$PY" scripts/evaluate_phase11.py --checkpoints "$RUN/best.pt" --run-dir "$RUN" --positions "$RUN/evaluation/initial_positions.json" --suite fixed --case nominal --horizon 30 --output "$RUN/evaluation/gui_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --visualizer kit --physics isaacsim_physx
# TensorBoard port {port} was free at report generation; existing services untouched.
"$PY" -m tensorboard.main --logdir_spec "train:$RUN/tensorboard,validation:$RUN/tensorboard_validation" --host 127.0.0.1 --port {port}
# Windows PowerShell tunnel:
ssh -N -L {port}:127.0.0.1:{port} mrl_6534@100.79.103.47
# Browser: http://127.0.0.1:{port}
# Windows PowerShell download:
scp "mrl_6534@100.79.103.47:{r}/review.tar.gz" .
scp "mrl_6534@100.79.103.47:{r}/review.tar.gz.sha256" .
'''
(r/'COMMANDS.txt').write_text(commands)
timing=dict(initialization_s=tr['initialization_s'],training_s=tr['training_wall_s'],initial_evaluation_s=tr['initial_evaluation_wall_s'],intermediate_evaluation_s=tr['intermediate_evaluation_wall_s'],final_evaluation_s=sum(x['wall_s'] for x in ev),analysis_plots_s=load('postprocessing_time.json')['analysis_and_plots_s'],additional_comparison_plots_s=load('comparison_plot_time.json')['seconds'],report_generation_s=time.perf_counter()-tick)
dump_json(r/'timing_summary.json',timing);print('REPORT_DONE',r)

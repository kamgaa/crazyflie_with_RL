import argparse,json,csv,shlex,subprocess,sys,time,shutil
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
def load(p):return json.loads(Path(p).read_text())
data=load(r/'analysis_result.json');summary=data['summary'];ver=load(r/'verification.json');parent=load(r/'parent_info.json');selections={b:load(r/b/'best_selection.json') for b in ['control_stop','track_return']}
def fmt(x):return '미해당/실패 후반 없음' if x is None else f'{x:.6f}'
lines=['Crazyflie phase9: 고정 목표 ±5cm 상자 복귀 / 위치 오차 기반 속도 보상','실행 폴더: '+str(r),'서버: MRL-server / cuda:0 / isaacsim_physx / headless','',
'RMS는 제곱평균제곱근, RMSE는 오차의 제곱평균제곱근이다. 위치 오차 e = link world position - env_origin - [0,0,1] m. 양의 X 오차는 목표보다 +X에 있다는 뜻이다. 속도는 link 원점의 절대 world 속도이며 관측에서 속도 오차로 교체하지 않았다.','',
'1. 부모 및 실행 예산','부모: '+parent['checkpoint'],'SHA256: '+parent['sha256'],'선택된 best.pt는 phase8 추가 128회이며 실제 누적 1408 iterations / 46,137,344 transitions / Adam step 143,360 (native iter=1407). 부모 training_result.json의 수치는 final.pt를 설명하므로 best.pt 카운터로 사용하지 않았다.',
'각 조건: 정확히 추가 256 iterations / 8,388,608 policy transitions / 실제 Adam optimizer.step 81,920회. 두 조건 합계 추가 16,777,216 transitions / 163,840 Adam updates. 공통 부모를 두 번 합산하지 않는다. 최종 조건별 누적 1664 iterations / 54,525,952 transitions / Adam step 225,280 (native iter=1663).','']
for b,t in data['training'].items():
 lines.append(f"{b}: 실제 추가 전이 {t['additional_transitions']:,}; Adam calls {t['additional_optimizer_updates']:,}; 13개 파라미터 Adam 내부 step {t['adam_steps']}; 초기화 {t['initialization_s']:.3f}s; 학습 {t['training_wall_s']:.3f}s; {t['transitions_per_s']:.1f} transitions/s; device 전체 sampled peak {t['cuda_device_used_peak_sampled_MiB']:.2f} MiB; PyTorch allocator peak allocated {t['memory_final']['torch_peak_allocated_MiB']:.2f} MiB / reserved {t['memory_final']['torch_peak_reserved_MiB']:.2f} MiB.")
lines+=['학습 시간은 main rollout/update/log/checkpoint 구간 wall time이며 초기화와 별도 평가 프로세스는 제외한다. iteration 수집/최적화 시간은 Performance/rollout_s, update_s에 기록한다. 학습 누적 초는 체크포인트 부모 cumulative_training_elapsed_s 1158.348048에서 이어진다. CUDA mem_get_info 기반 device 전체 사용량은 다른 프로세스와 PhysX 등을 포함한다. 이를 해당 프로세스 단독 메모리라고 표시하지 않는다. PyTorch allocator 통계는 해당 프로세스의 Torch 할당만 포함하며 PhysX 전체를 포함하지 않는다.','',
'2. 동일성, 초기 상태와 물리','actor/critic 각각 [64,64], tanh; N1024, rollout32, batch32768, minibatches64(512/sample batch), epochs5, 320 Adam calls/iteration. LR=3e-4 fixed, gamma=.99, lambda=.95, clip epsilon=.2, entropy=.003, value coefficient=1, value clipping=true, grad norm limit=1, 전체 rollout advantage 정규화; observation/reward normalization 추가 없음. 실제 agent/env/optimizer/reward 설정은 조건별 resolved_config.json에 있다.',
'부모 actor/critic/Gaussian log_std, Adam moments/param_groups/step과 전체 state_dict를 strict 복원하고 128개 고정 관측의 actor mean/std 및 critic 출력을 비교했다. Python/NumPy/Torch CPU/CUDA RNG를 첫 rollout 직전에 부모 상태로 복원했다. 초기 위치용 GPU Generator는 별도 seed 42를 두 조건에 동일하게 사용하며 global 학습 RNG를 소비하지 않는다. 조건별 checkpoint에는 reset_generator_state도 저장한다.',
'부모의 물리 상태, 진행 중 episode, rollout buffer는 복원하지 않는다. 새 XYZ 위치 분포에서 시작한 가중치·Adam 재개이며 중단 없는 물리 궤적 재개가 아니다. 환경 시작 시각을 무작위로 분산하지 않았다. 새 reward/초기상태 분포에 기존 critic이 적응하는 과도기가 포함된다.',
'초기 위치는 매 최초/episode/부분 reset에서 env_origin+[0,0,1]+delta. 각 delta 축은 독립 Uniform(-.05,.05)m. 축별 ±5cm 상자이며 모서리 거리는 약8.66cm이다. 목표는 고정된다. 자세/초기 속도/모터 평형 초기화/force composer는 기존 reset을 재사용하고 위치만 변경했다. partial reset시 선택 환경의 물리와 episode 누적값만 초기화한다.',
'정상 질량·COM·관성·motor delay .050s·allocation·action scale/clip·좌표 변환을 유지했다. physics dt .002s / decimation5 / policy .01s, 15D 관측,4D action. payload0,모터효율1,DR 없음. training episode8s, altitude .2~2.5m/tilt60deg/3D 위치오차1.5m 실패 조건 유지. tilt는 body Z와 world Z 사이 각도이며 roll로 대체하지 않았다. Lab/PhysX 기존 CPU 동기화는 유지했다.','']
for b in data['training']:
 d=load(r/b/'initial_reset.json');lines.append(b+' 초기 1024개 위치 표본: '+json.dumps(d,ensure_ascii=False))
lines+=['','3. reward 및 PPO loss 정의',
'공통 xy = .100*expm1(-(ex²+ey²)/.10²), z=.015*expm1(-ez²/.05²). sigma는 길이 척도이며 성공 허용오차가 아니다. far-field 위치 벌점 포화는 유지된다.',
'공통 tilt=-6*(qx²+qy²), quaternion wxyz; angular_velocity=-.001*||omega_body_FLU||² (rad/s); yaw=-wrapped_yaw_error²(rad²); termination=-10*failure. 실제 환경 reward는 signed 7개 항의 합, 추가 dt 배율/생존 보너스 없음.',
'정지 속도 보상 control_stop: velocity=-.005*||v_world||². 복귀 속도 추종 보상 track_return: velocity=-.005*||v_world-v_des||². gain4 s^-1, v_unsat=-4*e, v_des=v_unsat*min(1,1.5/||v_unsat||), m/s, 3D norm cap이며 축별 clamp 아님. zero vector는 안전하다. 기존 reward B의 inline 속도 생성식과 일치 검증했으며 전체 reward_variant B로 전환하지 않았다.',
'v_des는 고정 목표 미분이나 PID 명령이 아니다. reward와 진단만 사용하고 기존 PPO가 force/torque를 직접 결정한다. 관측 속도는 실제 절대 world 속도 그대로이다.',
'양 조건에서 common_stop/common_return reward를 같은 상태로 계산하여 모니터링하지만 실제 reward에 중복 합산하지 않는다. 계수가 다른 reward 숫자나 total loss만으로 제어 성능을 판단하지 않는다.',
'Loss/policy는 실제 clipped surrogate; Loss/value는 coefficient 적용 전 value loss; Loss/value_contribution은 value coefficient 적용 후; Loss/entropy_contribution=-entropy_coef*entropy; Loss/ppo_total은 실제 backward 직전에 계측한 loss이다. code probe가 실제 objective와 항 합의 residual을 기록한다. reward 항들은 optimizer loss가 아니다.',
'Policy/std_axis_*는 기존 로거가 읽는 마지막 PPO minibatch의 분포 std이며 최종 Adam step 직전 값이다. Policy/log_std_axis_*는 그 iteration의 최종 Adam step 이후 파라미터다. 측정 시점은 TAG_DEFINITIONS_ko.txt에 명시했다.',
'PPO/clip_fraction은 importance probability ratio가 [1-epsilon,1+epsilon] 밖에 있는 비율이고 epsilon=.2 설정값과 다르다. Actuator/action_clipping_fraction 및 allocation_clipping_fraction은 물리 명령 제한 비율이며 PPO clipping과 다르다.','',
'4. 평가 및 best 선택','8초 validation은 nominal,X±.05,Y±.05,Z±.05의 7조건이다. 부모 additional0 및 64/128/192/256회 모델을 평가했다. 7/7 완료 후보만 있으면 평균3D RMSE→마지막2초 평균속력→이른 checkpoint; 7/7이 없으면 완료수→평균지속→이른 checkpoint를 사용한다. 실패 궤적의 낮은 RMSE로 best를 선택하지 않는다.']
for b,s in selections.items():lines.append(f"{b} best: {s['source']} (추가 {s['additional_iteration']}회, 부모 선택={s['parent_selected']}); SHA256 {s['sha256']}")
lines+=['best를 확정한 뒤 부모와 두 조건 best/final 각각을 새 프로세스에서 다시 로드했다. 고정7조건 8초/30초 및 미리 저장한 seed20261013 XYZ box random20조건 30초를 평가했다. 30초/random20 결과는 선택이나 튜닝에 쓰지 않았다. random20은 20개 학습 seed가 아니다.',
'평가 환경은 고정7개 또는 random20개를 한 배치에서 독립 실행한다. 부모의 기존 1환경5조건 결과와 차이 검증은 parent_reproduction.json에 있다. env_origin을 제거한 link 상태를 사용하고 첫 terminated/truncated 직전 reward hook snapshot을 보존한다. 실패 후 autoreset된 상태는 해당 궤적에서 제외한다. 30초는 episode 시간제한만 변경한다. 환경 생성 수와 평가 초기 위치 지정 외 물리/실패 조건 동일.',
'8초 마지막2초,30초 마지막10초 지표는 완료 episode에만 제공한다. 실패 후반은 null/빈 값이며 0으로 채우지 않았다. 모든 모델이 같은 초기조건에서 완료한 paired comparison은 plots/paired_completed_comparison.csv에 있다.',
'각 row에는 raw actor mean/applied normalized action, requested/actual wrench(torque Nm,force N),motor force N/RPM,3D 위치와 절대속도/v_des/error/body각속도 등이 있다. action rate RMS는 같은 마지막 구간 내부 diff(action)/.01s이며 episode 경계를 넘지 않는다. Z bias(mean)와 centered std/peak-to-peak는 구분한다.','',
'모델 | 평가 | 완료 | 평균 지속s | 3D RMSE m(완료만) | 후반속력 RMS m/s | 후반속도오차 RMS m/s | 후반 omega RMS rad/s | 후반 Z std m']
for s in summary:lines.append(f"{s['model']} | {s['suite']} {s['horizon_s']}s | {s['completed']}/{s['total']} | {s['mean_duration_s']:.3f} | {fmt(s['position_rmse_m_completed_mean'])} | {fmt(s['tail_speed_rms_m_s_completed_mean'])} | {fmt(s['tail_velocity_error_rms_m_s_completed_mean'])} | {fmt(s['tail_body_omega_rms_rad_s_completed_mean'])} | {fmt(s['tail_z_std_m_completed_mean'])}")
lines+=['','5. 정량 해석 (같은 평가조건과 완료율을 함께 사용)']
for suite,h in [('fixed',8),('fixed',30),('random20',30)]:
 def get(m):return next(x for x in summary if x['model']==m and x['suite']==suite and x['horizon_s']==h)
 for kind in ['best','final']:
  c,t=get('control_stop_'+kind),get('track_return_'+kind)
  if c['completed']==c['total']==t['completed']==t['total']:
   vals=[]
   for k,name in [('position_rmse_m_completed_mean','위치 RMSE'),('tail_speed_rms_m_s_completed_mean','후반 속력 RMS'),('tail_velocity_error_rms_m_s_completed_mean','후반 속도오차 RMS'),('tail_body_omega_rms_rad_s_completed_mean','후반 각속도 RMS'),('tail_z_std_m_completed_mean','후반 Z std')]:
    vals.append(f"{name}: 정지 {c[k]:.6f} -> 복귀 {t[k]:.6f} ({(t[k]/c[k]-1)*100:+.2f}%)")
   lines.append(f'{suite} {h}s {kind}: 양 조건 모두 완료; '+ '; '.join(vals))
  else:lines.append(f"{suite} {h}s {kind}: 정지 {c['completed']}/{c['total']}, 복귀 {t['completed']}/{t['total']}. 완료 조건 집합이 다르면 aggregate RMSE를 직접 우열로 해석하지 않는다. paired_completed_comparison.csv 참조.")
for suite,h in [('fixed',8),('fixed',30),('random20',30)]:
 base=next(x for x in summary if x['model']=='parent' and x['suite']==suite and x['horizon_s']==h)
 for b in ['control_stop','track_return']:
  cur=next(x for x in summary if x['model']==b+'_best' and x['suite']==suite and x['horizon_s']==h)
  if base['completed']==base['total']==cur['completed']==cur['total']:
   ratio=(cur['position_rmse_m_completed_mean']/base['position_rmse_m_completed_mean']-1)*100
   lines.append(f"부모 대비 {b} best / {suite} {h}s: 위치 RMSE {ratio:+.2f}% 변화. 후반 속력 RMS {base['tail_speed_rms_m_s_completed_mean']:.6f}->{cur['tail_speed_rms_m_s_completed_mean']:.6f} m/s.")
for b in ['control_stop','track_return']:
 rows=[json.loads(x) for x in (r/b/'metrics.jsonl').read_text().splitlines()]
 for key in ['Reward/step/total','Reward/common_stop','Reward/common_return','Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total','PPO/approx_kl','PPO/clip_fraction','Flight/position_rmse_m']:
  before=sum(x[key] for x in rows[:64])/64;after=sum(x[key] for x in rows[-64:])/64
  lines.append(f"{b} 학습 첫64회/마지막64회 평균 {key}: {before:.8g} -> {after:.8g}. 서로 다른 학습 상태분포의 집계이며 deterministic 평가를 대체하지 않음.")
c30=next(x for x in summary if x['model']=='control_stop_final' and x['suite']=='fixed' and x['horizon_s']==30)
t30=next(x for x in summary if x['model']=='track_return_final' and x['suite']=='fixed' and x['horizon_s']==30)
lines.append('이번 실행의 해석: 두 조건 모두 고정7/무작위20 초기상태의 30초 비행을 완료해 생존율 차이는 없었다. 정지 조건은 부모보다 위치 RMSE가 개선됐다. 복귀 속도 조건은 8초 선택 best와 final 모두 정지 조건보다 위치 RMSE가 컸다. 반면 복귀 final은 후반 속력과 Z 변동이 감소했다. 따라서 위치 정확도 우위가 아닌 후반 움직임 감소와 위치/복귀 과도응답 사이의 손익으로 해석한다.')
lines.append(f"고정7조건 30초 final 비교: 복귀 위치 RMSE {(t30['position_rmse_m_completed_mean']/c30['position_rmse_m_completed_mean']-1)*100:+.2f}%, 후반 속력 RMS {(t30['tail_speed_rms_m_s_completed_mean']/c30['tail_speed_rms_m_s_completed_mean']-1)*100:+.2f}%, 후반 Z std {(t30['tail_z_std_m_completed_mean']/c30['tail_z_std_m_completed_mean']-1)*100:+.2f}%, 후반 body각속도 RMS {(t30['tail_body_omega_rms_rad_s_completed_mean']/c30['tail_body_omega_rms_rad_s_completed_mean']-1)*100:+.2f}%. Z 변동 감소가 roll/body각속도 진동의 제거를 의미하지 않는다.")
p30=next(x for x in summary if x['model']=='parent' and x['suite']=='fixed' and x['horizon_s']==30)
lines.append(f"전체30초 RMSE와 후반 위치편향을 구분: 마지막10초 평균 위치오차 크기는 부모 {p30['tail_position_mean_m_completed_mean']*1000:.3f}mm, 정지 final {c30['tail_position_mean_m_completed_mean']*1000:.3f}mm, 복귀 final {t30['tail_position_mean_m_completed_mean']*1000:.3f}mm였다. 전체 RMSE가 개선됐어도 후반 위치 편향까지 부모보다 개선된 것은 아니다. 축별 편향은 evaluation/summary.json 및 all_results.csv에 있다.")
lines.append('복귀 best는 8초 기준128회 모델이며 30초 성능은 final256회보다 나빴다. 사전 지정 8초 선택규칙을 유지했고 30초 평가를 보고 best를 바꾸지 않았다. 마지막10초 위치 편향 및 각속도 진동은 여전히 남는다. 보상 기준속도를 바꾼 것만으로 정밀 복귀가 확보됐다고 결론내릴 수 없다. 무엇이 원인인지는 추가 통제 실험 없이 단정하지 않는다.')
lines+=['부모와 control_stop 차이는 추가 학습 및 초기 위치 분포 변화가 함께 포함된다. 초기 위치 변화만의 인과 효과를 분리한 실험이 아니다. 두 조건 차이는 속도 reward 기준만 변경한 단일 부모·seed42 후속 비교다. 보상 변화는 critic target 및 학습 상태분포도 변화시킨다.',
'학습 reward 개선 또는 loss 감소만으로 제어 성능 개선을 주장하지 않는다. 8초 완료,30초 완료,위치 정확도,속력 및 진동을 각각 판단한다. 안정적인 지속 비행과 무진동 호버는 동일하지 않다. 정상 동역학 국소 고정목표 복귀 실험이며 추가 질량/COM변화/모터고장 OOD 대응을 검증하지 않았다. 원 추종 평가/학습은 실행하지 않았다.',
'다음 검토는 결과를 기반으로 별도 승인된 실험에서만 진행한다. 이번에는 추가 학습·가중치 탐색·seed/조건 추가를 실행하지 않았다.','',
'6. 검증 및 데이터','reward 수식/부호/zero/vector cap,기존 exponential stop reward 일치,환경 축 및 origin 독립성,velocity 외 설정 동일,1024 독립XYZ reset분포/부분reset,부모 actor/critic/Gaussian/Adam/RNG,첫32step rollout 동일성,실제optimizer hook/내부step,원래reward 및 실제 backward loss 합산,TB/CSV/JSON,128고정관측 재로드,first terminal capture를 검증했다. 세부 verification.json/evidence를 참조한다.',
'기존 task/common 기본 파일과 prior run 산출물 SHA256을 재검증했다. 기존 TensorBoard 이벤트는 live service 보존을 위해 prior hash 목록에서 제외했으며 수정하지 않았다. GUI는 실행하지 않았으며 검증 완료라고 주장하지 않는다. NVIDIA/패키지/GUI 환경을 변경하지 않았다.','']
for b,d in ver['training_tb'].items():lines.append(f"{b}: TensorBoard/CSV/JSON {d['checked_scalar_values']:,} scalar 일치, {d['tag_count']} tags, first/middle/last step={d['first_middle_last_steps']}, max error={d['max_error']}.")
lines.append('평가 TensorBoard scalar도 validation_scalars.csv/json의 float32 값과 정확히 일치했다. 원래 float64 평가 지표와 이벤트의 반올림 차이는 verification.json에 별도로 기록했다. 8초와30초 평가의 공통 첫800step 상태/action 차이는0이었다.')
lines.append('고정128관측 재로드 최대 action 오차: '+str(max(x['max_action_error'] for x in ver['reloads'])))
executions=[load(p) for p in (r/'evaluation').glob('*/execution.json')]+[load(p) for p in r.glob('*/evaluation/validation_*/execution.json')]
lines.append(f"평가 프로세스 {len(executions)}개: 초기화 합 {sum(x['initialization_s'] for x in executions):.3f}s, 별도 평가 구간 합 {sum(x['evaluation_s'] for x in executions):.3f}s. analysis/plot wall {data['analysis_wall_s']:.3f}s. 사전 학습검사는 각 조건16*1024전이,optimizer0으로 본 학습 예산 밖이다.")
lines+=['','7. 파일 및 재현','변경은 scripts/*phase9*.py의 새 전용 파일만 추가. phase9_reward.py(속도 기준),phase9_reset.py(별도XYZ RNG),train_phase9.py(엄격한 resume/예산/첫rollout),evaluate_phase9.py(고정7/random20,terminal snapshot),phase9_metrics/evalmetrics.py(집계),prepare/validate/ready/run/analyze/report/package(재현·검증·산출물). source_snapshot/에는 사용한 src/scripts와 baseline git 상태 및 파일 해시가 포함된다.',
'설치 확인: IsaacLab source는 /home/mrl_6534/rl_crazyflie/IsaacLab-3.0.0-EA이며 release VERSION=3.0.0이다. Python 배포 metadata의 isaaclab 버전은17.0.2로 다른 라벨이며 실제 읽은 값을 environment_versions.json에 함께 기록했다. PyTorch2.11.0+cu128/IsaacSim6.1.0.0/RSL-RL5.4.1/TensorBoard2.21.0을 변경하지 않았다. 프로젝트 git HEAD는 아직 유효한 commit이 없으므로 존재하지 않는 commit을 기재하지 않고 baseline status,source snapshot 및 SHA256을 코드버전 근거로 사용한다.',
'TensorBoard 학습 기본 X축은 추가 policy transitions(첫학습32768,마지막8388608). 부모 포함카운터 및추가 iteration은 별도Performance태그. EvalValidation의0은 실제 부모평가이며 학습0행을 조작하지 않았다. 평가 trajectory TB는 tensorboard_time에 policy evaluation step(100Hz) 축과time_s태그로 분리한다. Reward/step은1024*32표본 평균,Reward/episode는이번iteration에끝난 episode만 정산한다. rollout경계에서 미완료 episode누적을 버리지 않는다. 종료없는iteration은count0만 기록하고 episode평균0을 만들지 않는다. timeout bootstrap은환경reward/return기록에 섞지 않는다.',
'원시 scalar는 iterations.csv/metrics.jsonl, 평가 전체 결과는 evaluation/all_results.csv,집계는 plots/evaluation_summary.csv. PNG 대표플롯에 대응하는 training/validation/trajectory CSV도 plots에 보존한다. 실제 명령은 actual_commands.json과 COMMANDS.txt에 있다. review.tar.gz 및 MANIFEST.sha256 검증목록으로 전달한다.']
text='\n'.join(lines)+'\n';(r/'REPORT_phase9_ko.txt').write_text(text);shutil.copy2(r/'REPORT_phase9_ko.txt',ROOT/'REPORT_phase9_ko.txt')
py='/home/mrl_6534/miniconda3/envs/isaaclab/bin/python'
commands=f'''# 서버에서 평가만 실행 (추가 학습 없음)
cd {ROOT}
PY={py}
RUN={r}
BRANCH=track_return
"$PY" scripts/evaluate_phase9.py --checkpoints "$RUN/$BRANCH/best.pt" "$RUN/$BRANCH/final.pt" --run-dir "$RUN/$BRANCH" --output "$RUN/$BRANCH/evaluation/manual_$(date +%Y%m%d_%H%M%S)" --positions "$RUN/evaluation/initial_positions.json" --suite fixed --horizon 30 --device cuda:0 --headless --physics isaacsim_physx

# GUI nominal 재생; 검증은 headless로만 수행했음
"$PY" scripts/evaluate_phase9.py --checkpoints "$RUN/$BRANCH/best.pt" --run-dir "$RUN/$BRANCH" --output "$RUN/$BRANCH/evaluation/gui_$(date +%Y%m%d_%H%M%S)" --positions "$RUN/evaluation/initial_positions.json" --suite fixed --case nominal --horizon 30 --device cuda:0 --visualizer kit --physics isaacsim_physx
# final.pt로 바꾸거나 BRANCH=control_stop으로 변경 가능. GUI는 서버 그래픽 세션에서 실행.

# TensorBoard: 기존6006/6007 보존. 6008 점유 여부를 먼저 확인하고 빈 경우만 실행
ss -ltnp | grep ':6008'
"$PY" -m tensorboard.main --logdir "$RUN" --host 127.0.0.1 --port 6008

# Windows PowerShell 터널
ssh -N -L 6008:127.0.0.1:6008 mrl_6534@100.79.103.47
# http://127.0.0.1:6008

# Windows PowerShell 결과 다운로드
scp mrl_6534@100.79.103.47:{r}/review.tar.gz .
scp mrl_6534@100.79.103.47:{r}/review.tar.gz.sha256 .
Get-FileHash .\\review.tar.gz -Algorithm SHA256

# 전체실험 재현은 새폴더에서만; 이미 완료된 run 재학습을 거부함. 아래 명령은 문서화용이며 실행하지 않음.
# python scripts/prepare_phase9.py --run-dir NEW_RUN
# python scripts/validate_phase9_reward.py --run-dir NEW_RUN
# 부모 preflight8 평가 -> 양 조건 --preflight -> ready_phase9.py -> run_phase9.py
# 각 단계 실제 실행 인자 전체는 actual_commands.json 및 조건별 preflight_command.json 참조.
'''
(r/'COMMANDS.txt').write_text(commands)
dump_json(r/'timing_summary.json',dict(training={b:{k:v[k] for k in ['initialization_s','training_wall_s','transitions_per_s']} for b,v in data['training'].items()},evaluation_processes=len(executions),evaluation_initialization_s=sum(x['initialization_s'] for x in executions),evaluation_s=sum(x['evaluation_s'] for x in executions),analysis_plot_s=data['analysis_wall_s']))
print(text[-1500:])

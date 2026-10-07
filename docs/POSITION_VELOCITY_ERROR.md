# 위치 기반 목표 속도를 사용하는 E2E PPO

2026-10-02 후속: [비학습 원인 진단 및 독립 A/B 설정](VELOCITY_AB_DIAGNOSIS.md).
관측/reward 모드가 분리됐으며, 아래 기존 D 학습 설정과 결과의 의미는 유지된다.

새 규약은 `configs/e2e_train_position_velocity_error.yaml`에서만 명시적으로 켠다.
기존 YAML의 관측과 속도 reward는 absolute velocity다. E2E 전용 설정은 PID 게인과 독립적이다.

```
e_p = p_world - p_target
v_raw = -4.0 * e_p
v_des = v_raw * min(1, 1.5 / ||v_raw||)   # zero vector => zero
e_v = v_world - v_des
obs = [e_p(3), e_v(3), quaternion_wxyz(4), omega_body(3), sin(yaw_error), cos(yaw_error)]
velocity reward = -0.005 * dot(e_v,e_v)
```

position target/yaw target이라는 외부 인터페이스, 관측 15차원, 행동 4차원 wrench를 유지한다.
적분기, history, GRU, 목표점 시간 미분, 필터, 가속도 제한은 추가하지 않았다.
`crazyflie_rl/velocity_reference.py:desired_velocity`가 공통 계산 함수다.
환경 `_obs`와 `step`은 각각 현재/제어 후 물리 상태에서 이 함수를 호출한다.
제어 전 목표 속도를 reward에 재사용하지 않는다. 정책 입력 전처리/정규화보다 먼저 적용된다.

## 변경 경로

- 공통 규약/환경: `crazyflie_rl/velocity_reference.py`, `config.py`, `environment.py`.
- 학습 기록/체크포인트 계약: `training.py`, `artifacts.py`.
- 평가·로딩·로그: `dr_policy.py`, `dr_transfer.py`, `eval_cli.py`,
  기존 `interactive_eval.py`, `plotting.py`, `reward_balance.py`.
- 신규 설정: `configs/e2e_train_position_velocity_error.yaml`,
  `configs/e2e_train_position_velocity_error_smoke.yaml`, `configs/eval_position_velocity_error.yaml`.
- 검증: `tests/test_position_velocity_error.py` 추가; COM 및 진단 의미가 바뀐 범위의
  `test_environment_contracts.py`, `test_initial_pose.py`, `test_reward_balance.py`,
  `test_reward_diagnostics.py` 회귀 기대값 갱신.
- 문서: 이 문서 및 이전 COM 감사/interactive 문서의 물리 모델 설명 갱신.

기존 학습 YAML과 과거 체크포인트/결과를 수정하지 않았다. 기존 interactive 구현과
motor degradation 변경도 보존했다. commit/push는 수행하지 않았다.

## 실제 학습 설정

| 항목 | 값 |
|---|---|
| e2e_velocity.mode | position_error |
| position_gain / max_speed | 4.0 s⁻¹ / 1.5 m/s |
| position_xy_weight / position_z_weight | 10 / 6 |
| velocity_weight | 0.005 (기존 nominal 유지) |
| action scale | [.0075,.0075,.001,.5] |
| policy / physics Hz | 100 / 500 |
| 초기 위치 | legacy 각 축 독립 Uniform(-.05,.05) m |
| 초기 자세 perturbation | 0° |
| payload / 모터 효율 | 0 kg / 모두 1 |
| actuator 모델 | 기존 1차 지연, 파라미터 randomization off |
| seed / 요청 학습 스텝 | 42 / 1,000,000 |
| PPO | 기존 n_steps=2048, batch_size=256, n_epochs=10, 기타 설정 그대로 |

현재 `e2e_train_position_split_equal.yaml`은 이름과 달리 반경 .15 m reset DR로 변경돼 있었다.
이 때문에 nominal인 `e2e_train.yaml`을 상속하고 위치 가중치 10/6을 명시했다.
초기 분포 선택은 사용자에게 선택지를 제시했으며, 답변 없이 작업을 진행하는 동안에는
요청의 nominal 기준을 적용했다. 기존 YAML은 변경하지 않았다.

SB3는 n_steps 단위 rollout을 완료하므로 요청 스텝보다 실제 스텝이 커질 수 있다.
manifest에 requested/actual_total_timesteps를 모두 기록한다. best는 기존 평가 조건
(tail tilt disqualification=0, position score 개선)으로 선택하며 return으로 비교하지 않는다.

## COM 수정

수정 전 감사는 [PAYLOAD_MOMENT_AUDIT.md](PAYLOAD_MOMENT_AUDIT.md)에 남아 있다.
이제 `_set_com_bias`가 합성 질량·COM·전체 관성 텐서를 계산하고 principal inertia 및
inertia frame quaternion으로 MuJoCo 모델에 기록한 뒤 `mj_setConst`를 호출한다.
비대각 관성도 보존한다. payload는 body xy 평면의 점질량이며 별도 충돌 형상은 없다.
명시적인 `offset × payload_gravity` 추가 토크를 제거했으며,
`R @ dist_torque_body` 외란 토크는 그대로 유지했다.
기존 payload/DR 학습 모델의 물리 궤적은 이 수정 후 달라질 수 있다. 과거 결과는 보존했다.
무게추 0 nominal의 100-step 관측·reward는 변경 전 HEAD 코드와 정확히 일치했다.

## 저장·로딩 규약

manifest와 PPO zip 자체에 `observation_contract`, `velocity_semantics`를 기록한다.
새 규약은 mode뿐 아니라 Kp와 속도 상한도 일치해야 로드된다.
명시적 규약이 없는 과거 모델은 absolute로 해석하므로 새 프로파일에서 거부한다.
새 모델도 구형 absolute 프로파일에서 거부한다. 단순 15/4 shape 일치로 통과하지 않는다.

`dr_policy`/`compare_dr_policies.py`의 `--velocity-inputs error`는 기존 모델에 대한
별도 v_ref 차감 실험이다. 새 규약에서는 오류로 거부해 이중 차감을 방지한다.
새 규약 평가의 `--velocity-inputs absolute` 기본값은 **추가 변환 없이 환경 관측을 사용**한다는
뜻이며 실제 입력은 내부 목표 속도에 대한 오차다. manifest의 환경 규약에 이를 명시한다.
기존 frozen normalization 경로는 유지한다. 이번 새 학습은 VecNormalize를 사용하지 않는다.

## 로그 및 시간 규약

- TensorBoard: 기존 reward_terms/velocity를 실제 속도 오차 비용으로 기록한다.
  reward_raw/velocity_sq는 실제 속도 제곱을 유지하며 velocity_error_sq, desired_velocity_sq를 추가했다.
- interactive CSV: 실제 linear_velocity, desired_velocity, velocity_error를 별도로 기록한다.
  desired_velocity_before/velocity_error_before/raw_observation은 policy_input_time의 값이다.
  반환 상태/desired_velocity/velocity_error는 time_post 값이며 구간 동안 적용된 target 기준이다.
- view_live: 동일 의미의 RolloutTrace/NPZ를 저장하고 속도 패널에 실제 속도와 내부 목표 속도를 함께 표시한다.
  기존 time_sec는 구간 시작 시각이다. 새 time_post는 반환 상태 시각이다.
- reward_balance: 실제 속도 RMS는 유지하고, 비용 계산에는 새 규약의 velocity_error를 사용한다.
  velocity_reward_signal, velocity_reward_sq_mean, velocity_error_rms_mps로 구분한다.
- DR 비교 CSV: 기존 trajectory reference velocity와 internal desired_velocity를 구분한다.
  internal_velocity_error_rmse_m_s가 새 내부 속도 오차 지표다. 기존 velocity_rmse는
  실제 속도와 미션 참조 속도의 차이로 유지한다. 이번 세 정지 목표 케이스에서 미션 참조 속도는 0이다.

## 실행

저장소 루트에서:

```bash
python train_ppo_02.py --config configs/e2e_train_position_velocity_error.yaml --dry-run

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_ppo_02.py \
  --config configs/e2e_train_position_velocity_error_smoke.yaml

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_ppo_02.py \
  --config configs/e2e_train_position_velocity_error.yaml
```

smoke는 2048 step, evaluation 3 episodes/주기 평가 없음으로 별도 best/final을 저장한다.
본 학습은 기존 30 episodes/20,000 step 평가 주기를 유지한다. 기존 모델을 resume하지 않는다.
GUI 평가에는 아래와 같이 새 규약의 평가 설정을 지정한다:

```bash
python view_e2e_interactive.py --config configs/eval_position_velocity_error.yaml --model CHECKPOINT.zip
python view_live.py --config configs/eval_position_velocity_error.yaml --model CHECKPOINT.zip --mode hover
```

## 검증 및 학습 산출물

집중 검증 합계 149 passed, 1 skipped (핵심 경로 126 + reward 진단 21 + reset RNG 회귀 2).
전체 저장소 테스트 통과를 뜻하지 않는다. 현재 YAML과 맞지 않는 기존 config/position-split/
view-live/initial-pose 테스트 기대값은 이번 작업에서 임의로 변경하지 않았다.
skip은 과거 DR 비교 결과 디렉터리를 지정하는 환경변수가
없는 경우의 실험 재현 테스트다. 새 규약 13개 검사에는 목표/방향/norm 제한, reward=0,
반대 방향 비용, 제어 후 reward, 관측 유한성/차원, 기존 nominal 정확한 회귀,
일반 평가/interactive 동일 관측, 목표 이벤트 상태 보존, 계약 불일치/이중 차감 거부,
COM·full inertia·정적 평형·외란 보존을 포함한다.

smoke run: `artifacts/runs/ppo_e2e_hover_position-velocity-error-smoke_seed42_20261001-205901`.
2048 step 학습·best/final 저장·재로딩 후 각 10 step 추론 완료. 구형 설정으로의 로드 거부 확인.
별도 view_live/interactive headless smoke에서도 로그·그래프 생성 완료.

본 학습 run: `artifacts/runs/ppo_e2e_hover_position-velocity-error-nominal_seed42_20261001-210046`.
학습 완료. 요청 1,000,000 / 실제 1,001,472 steps, seed=42.
final: `models/ppo_e2e_hover_position-velocity-error-nominal_seed42_final_20261001-210046.zip`.
SHA256: `badb391d8eeaafeb19cbc6d981ea4aabc2cec3ab1d41d7366295b224c8206be9`.
zip metadata에서도 1,001,472 steps, seed=42, 새 관측 규약과 Kp=4/vmax=1.5를 확인했다.
**본 학습에서는 best가 생성되지 않았다.** 모든 주기 평가에 tail-tilt disqualification이
있어 기존 best 선정 조건을 통과하지 못했다. final을 best로 바꾸어 저장하지 않았다.
smoke의 best/final 저장 검증은 저장 경로 확인이며 제어 성능 합격을 의미하지 않는다.

### 실제 최종 평가 (seed=42, deterministic, 공중 시작)

결과: `artifacts/runs/dr-transfer-39rkr7zg`.
초기 평가 `artifacts/runs/dr-transfer-l3pneew4`도 보존했다. 범례의 기존 `absolute`
표기를 새 입력 규약인 `position velocity error (native)`로 명확히 한 뒤 독립 실행했다.
세 rollout CSV와 summary CSV/JSON이 바이트 단위로 일치했다.
`final_evaluation_record.json`에 두 결과 경로와 해시를 기록했다.

| 케이스 | 실제 시간(s) | XY RMSE(m) | Z RMSE(m) | 전체 위치 RMSE(m) | 실제 속도 RMS(m/s) | 내부 속도 오차 RMS(m/s) |
|---|---:|---:|---:|---:|---:|---:|
| hover | 2.68 | 0.038036 | 0.033724 | 0.050834 | 0.258704 | 0.346827 |
| step-005 | 2.65 | 0.068509 | 0.029227 | 0.074483 | 0.417644 | 0.524281 |
| step-050 | 0.14 | 0.497232 | 0.001738 | 0.497235 | 0.162998 | 1.407113 |

모두 `max_tilt`로 조기 종료했다. 위 RMSE는 관측된 부분 구간 값이며 완주 지표가 아니다.
정착 시간과 고정 마지막 2초 지표는 모두 null, settled=false다.
**학습 실행은 완료했지만 안정적인 새 기준 정책을 얻지는 못했다.**
실패 원인을 속도 규약 변경 하나로 자동 확정하지 않는다. 추가 튜닝이나 재학습은 실행하지 않았다.

기존 baseline을 자신의 absolute 규약으로 새로 평가한 결과는 `artifacts/runs/dr-transfer-u9kp4hvv`다.

| 케이스 | 완료/실제 시간(s) | 전체 위치 RMSE(m) | 정착(s) | 실제 속도 RMS(m/s) |
|---|---|---:|---:|---:|
| hover | 완료 / 8.00 | 0.001928 | 0.00 | 0.000585 |
| step-005 | 완료 / 8.00 | 0.011038 | 1.44 | 0.020399 |
| step-050 | max_tilt / 0.40 (partial) | 0.418164 | null | 1.204433 |

baseline의 hover/step-005 마지막 2초 위치 RMSE는 각각 0.00210988/0.00210996 m다.
서로 다른 길이의 partial RMSE나 서로 다른 reward return을 우열 판단에 사용하지 않는다.
이전 baseline과 새 정책의 seed/학습 이력이 같지 않으므로 통제된 인과효과 비교는 아니다.

최종 CSV 전체에서 정책 입력 시각의 `v-v_des`, 제어 후 `v_des`,
`reward_terms_velocity=-.005*||v-v_des||²`, dt=.01을 재검증했다.
첫 raw velocity는 hover `[0,0,0]`, step-005 `[-.2,0,0]`, step-050 `[-1.5,0,0]`이었다.
학습 핵심 소스/설정과 기존 baseline 체크포인트 해시 불변 확인도 통과했다.
증거는 본 학습 run의 `task_validation/`, stdout, TensorBoard, manifests에 남겼다.
GUI에서 새 정책의 실제 키보드 조작을 다시 수행하지는 않았다. 해당 경로는 이번 변경의
단위 검사와 별도 headless view_live/interactive smoke로 확인했다.

### 저장된 final 모델 재평가

```bash
MODEL=artifacts/runs/ppo_e2e_hover_position-velocity-error-nominal_seed42_20261001-210046/models/ppo_e2e_hover_position-velocity-error-nominal_seed42_final_20261001-210046.zip

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
python compare_dr_policies.py --config configs/eval_position_velocity_error.yaml \
  --model "final=$MODEL" --cases hover step-005 step-050 --seed 42

python view_live.py --config configs/eval_position_velocity_error.yaml \
  --model "$MODEL" --mode hover

python view_e2e_interactive.py --config configs/eval_position_velocity_error.yaml \
  --model "$MODEL"
```

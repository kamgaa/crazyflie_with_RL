# 속도 관측·reward 분리 진단 (2026-10-02, 추가 학습 없음)

이 문서는 학습 전 진단 기록이다. 이후 승인된 A/B 학습은 [별도 실행 보고서](VELOCITY_AB_TRAINING.md)에 기록한다.

이번 작업은 비학습 진단과 A/B 설정 준비다. PPO 학습, optimizer update, checkpoint 수정,
commit/push를 실행하지 않았다. 이전 정상 체크포인트와 D의 모든 run 파일 및 이전 평가 결과
86개 파일의 SHA256 불변을 확인했다.

## 결론과 해석 범위

현재 물리 모델에서 이전 정상 nominal 정책은 호버와 5cm 이동을 모두 완주했다.
관측 복원 및 실제 정책 행동 일치도 float32 허용오차 안에서 통과했다.
D의 50회 주기 평가에서는 안정적인 8초 호버를 달성한 구간이 확인되지 않는다.
이 증거만으로 D의 실패 원인을 관측 변환, 속도 reward, PPO 탐색 중 하나로 확정할 수 없다.
특히 이전 정상 정책은 **seed=42로 새로 학습한 A의 결과가 아니다**.

## 이전 정상 정책: 수정된 환경에서 비학습 평가

체크포인트:
`artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip`

설정은 기존 `configs/eval_dr_transfer.yaml`, evaluator는 `compare_dr_policies.py`를 사용했다.
결과: `artifacts/runs/dr-transfer-ps3oquze` (manifest, CSV, summary, 기존 형식 그래프 포함).

- 초기 상태 `[0,0,1] m`, level quaternion `[1,0,0,0]`, 선속도·각속도 0, yaw target 0.
- 환경 reset 후 기존 EvaluationAdapter의 공중 초기화, forward 및 actuator reset을 사용.
  5cm 목표 변경 후 관측을 재생성하며 사전 호버 구간은 없다.
- 모터별 초기 command/actual thrust 약 `0.10638945 N`, airborne hover equilibrium.
  기존 `cf21b_first_order`, nominal time constant 0.05 s, 모터 효율 모두 1.
- payload 0, payload/actuator parameter/reset 위치·자세 난수화 off. 100/500 Hz 유지.
- COM·합성 전체 관성·mj_setConst 및 명시적 payload 중력 토크 제거는 이미 반영돼 있었다.
  이번에는 물리 구현을 다시 바꾸지 않았다. `_set_com_bias`, `_apply_control`, `reset`,
  `reset_actuator_state`의 AST가 작업 시작 시 코드와 동일하다. 별도 외란 토크도 유지한다.
- 종료 기준: altitude 0.2–2.5 m, tilt 60°, position error 1.5 m; horizon 8초.
- 원래 absolute velocity 관측, deterministic=True, seed=42.
  로더가 저장소 학습 명령과 PPO `_last_original_obs=null`을 확인해 VecNormalize 없음으로 처리했다.
  zip 하나만으로 외부 사용자 정의 전처리의 부재까지 증명할 수 있다는 뜻은 아니다.

| 시험 | 완료 시간(s) | XY RMSE(m) | Z RMSE(m) | 전체 RMSE(m) | 정착(s) | 실제 속도 RMS(m/s) | 마지막 2초 위치 RMSE(m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| hover | 8.00 | 0.001910 | 0.000267 | 0.001928 | 0.00 | 0.000585 | 0.002110 |
| step-005 | 8.00 | 0.010982 | 0.001119 | 0.011038 | 1.44 | 0.020399 | 0.002110 |

두 시험 모두 terminated=false, horizon에서 truncated=true, completed=true다.
동일 시각 post-state/reference 오차로 계산하며 정착은 기존 5mm/0.02m/s/최소 1초 기준이다.
이는 nominal 환경·초기화·로더가 모든 정책을 무조건 실패시키는 상태는 아니라는 증거다.
payload가 있는 모든 조건이나 D의 학습 경로 전체가 옳다는 보장은 아니다.

## 관측 복원과 deterministic 행동 수치 검증

`diagnose_velocity_ab.py`의 `RestoredAbsolutePolicy`는 기존 absolute 정책을 정식 frozen loader로
로드한 후 사용하는 **명시적인 비학습 진단 adapter**다. 호환성 검사를 해제하지 않는다.
raw error 관측 사본에 v_des를 더해 복원한 뒤 기존 frozen normalization/clipping/predict 경로로 넘긴다.
실제 기체 상태나 원본 배열은 바꾸지 않는다.

512개 synthetic physical state 표본에 실제 baseline PPO를 사용했다. 같은 RNG를 seed=42로
한 번 초기화한 뒤 진행했다. 제한 미적용 256개, 제한 적용 256개이며 quaternion·각속도·속도도
변화시켰다. 원래 관측과 변환 관측은 각각 실제 환경 `_obs`로 생성했다.

| 표본 | 최대 관측 복원 절대오차 | 최대 정책 행동 절대오차 |
|---|---:|---:|
| 제한 미적용 | 1.19209290e-7 | 2.38418579e-7 |
| norm 제한 적용 | 1.19209290e-7 | 1.78813934e-7 |

사전 허용오차는 raw 관측 5e-7, 행동 5e-6 (rtol=0)이며 모두 통과했다.
속도 외 관측 성분은 정확히 같고, 전후 simulator/actuator snapshot도 동일하다.
여기서 행동 단위는 4차원 정규화된 policy action이다. 새 관측을 **복원하지 않고** 구형 정책에
넣어도 행동이 같다는 주장이 아니다. 표현 변환이 학습 최적화에 미치는 영향도 이 검사로 확정할 수 없다.

수치/NPZ: `artifacts/runs/velocity-ab-nonlearning-fhgv0_t1/observation_restoration.json` 및 `.npz`.

## D 학습 기록 분석

대상 run: `artifacts/runs/ppo_e2e_hover_position-velocity-error-nominal_seed42_20261001-210046`.
요청 1,000,000 / 실제 1,001,472 steps, seed=42. best 없음, final zip 한 개만 존재한다.
중간 체크포인트가 없으므로 재평가나 재생성 학습을 하지 않았다.

20,000 step 간격 평가 50회, 매회 30 episode, seed 1000–1029, deterministic.

| 구간 | 평균 episode 길이 | 평균 생존 시간 | tail tilt 탈락 |
|---|---:|---:|---:|
| 최초 기록, 20,000 | 42.667 steps | 0.4267 s | 30/30 |
| 최대 평균 생존, 880,000 | 297.967 steps | 2.9797 s | 30/30 |
| 최소 탈락 수, 840,000 | 259.800 steps | 2.5980 s | 20/30 |
| 마지막 주기 평가, 1,000,000 | 275.200 steps | 2.7520 s | 30/30 |

50회 모두 평균 episode 길이가 800 steps 미만이며 탈락 0회 평가는 없다.
**관측된 주기 평가에서는 처음부터 안정적 호버를 확보하지 못했고, 이후 평균 생존 시간이
늘어났지만 안정화 달성을 확인할 구간은 없다.** 평가 사이의 미저장 정책 상태나 개별 episode가
전혀 완주하지 못했다고 단정할 수는 없다. 집계만으로 개별 종료 원인도 복원할 수 없다.

학습 TensorBoard에 `rollout/ep_len_mean`, 학습 episode별 길이/생존 시간/종료 사유가 없다.
콘솔의 `surv`는 학습 episode 길이가 아니라 위 주기 평가의 평균 길이다.
기록된 최종 세 시험은 max_tilt 종료지만 이것으로 모든 과거 종료 원인을 추정하지 않는다.

| 기록된 optimizer 진단 | 첫 값 | 마지막 값 | 전체 범위 |
|---|---:|---:|---:|
| approx_kl | 0.001734 | 0.004495 | 0.001427–0.005879 |
| clip_fraction | 0.097217 | 0.126611 | 0.049902–0.198193 |
| std | 0.224032 | 0.399046 | 0.224032–0.449679 |
| explained_variance | -0.014758 | 0.818418 | -0.183004–0.849387 |

488개 기록을 읽었다. `std`는 SB3의 평균 `exp(log_std)`로, 실측 action 표준편차나 모터 포화율이 아니다.
학습률 기록은 3e-4로 일정하다. 기록 범위에서 KL 폭증은 관찰되지 않지만 제어 안정성을 의미하지 않는다.
탐색 분산 증가는 확인된 사실이며 실패의 원인이라는 결론은 미확인이다.

best의 실제 선정 조건은 `score < best_score AND disqualifications == 0`이다.
score는 episode별 tail 위치 거리 평균을 다시 평균한 값이다. tail은 episode 마지막 30%,
최소 1 sample이고, 그 구간의 최대 tilt가 30°보다 크면 탈락한다.
이는 환경의 **60° 기울기 종료 한계와 다르다**. floor보다 좋은 score나 8초 완주는 추가 저장 조건이 아니다.
모든 평가에 20–30회 탈락이 있어 저장 조건을 통과하지 못했다. 조건을 완화하지 않았다.

원자료/CSV/그래프: `artifacts/runs/velocity-ab-nonlearning-fhgv0_t1/`의
`d_training_audit.json`, `d_evaluation_history.csv`, `d_training_scalars.json`,
`d_training_scalars.csv`, `d_training_diagnostics.png`.

## 독립 설정과 하위 호환

`environment.e2e_velocity`에 `observation_mode`, `reward_mode`를 추가했다.
각 필드를 생략하면 기존 `mode`로 대체한다. 따라서 기존 YAML과 D의
`mode: position_error` 해석은 변하지 않는다. 새 필드가 생략된 resolved 설정도 이전과 동일하다.
각 필드 값은 `absolute` 또는 `position_error`다.

| 실험 | observation_mode | reward_mode | 이번 상태 |
|---|---|---|---|
| A | absolute | absolute | 설정만 준비 |
| B | absolute | position_error | 설정만 준비 |
| C | position_error | absolute | 정의/단위 검증만, 학습 설정 파일 없음 |
| D | 생략 → position_error | 생략 → position_error | 기존 설정·모델 보존 |

A/B는 D의 nominal 기반 설정을 상속하되 두 채널을 명시적으로 덮어쓴다.
A에서도 공통 v_des를 진단용으로 계산하지만 관측과 reward에는 실제 속도를 사용한다.
속도 reference는 기존 `desired_velocity` 함수 하나로 생성하며 pre-observation과 post-reward에서
각각 해당 시각 상태로 재계산한다. PID 게인·물리 속도·wrench·관측 차원은 바꾸지 않는다.

`velocity_semantics`는 이제 명시적으로 **관측 계약만** 나타낸다. observation이 같으면 reward만
다른 설정에서도 정책을 로드할 수 있으며, position error 관측의 Kp/vmax 불일치는 계속 거부한다.
모델/manifest/evaluation에 `velocity_reward_semantics`도 따로 기록한다.
기존 v_ref 차감 실험은 observation이 이미 error인 C/D에서 거부하며, reward-only B와 혼동하지 않는다.
reward_balance도 reward 모드를 사용하고, raw `velocity_reward_sq`를 추가해 실제 속도 비용 피연산자를
확인할 수 있게 했다. `velocity_sq`, `velocity_error_sq`, 실제 물리 속도 로그는 유지한다.

## A/B 공통 설정과 실행 명령 (실행하지 않음)

- 수정된 동일 COM/관성 물리 모델, 초기 위치 축별 Uniform(-0.05,0.05) m, 자세 perturbation 0.
- payload=0, 모터 효율 1, payload/actuator parameter randomization off; 새 DR 없음.
- XY/Z=10/6, velocity_weight=0.005, action scale=[0.0075,0.0075,0.001,0.5].
- Kp=4 s⁻¹, vmax=1.5 m/s, 100/500 Hz, seed=42, 예정 total_timesteps=1,000,000.
- PPO 네트워크·초기화·하이퍼파라미터는 동일한 기존 nominal 설정을 상속한다.

두 설정을 공식 train CLI의 `--dry-run`으로 검증했다. resolved 차이는 정확히 다음 네 개다:
`environment.e2e_velocity.reward_mode`, `experiment.condition`, `experiment.description`, `source_path`.
메타데이터를 제외하면 **reward_mode 하나만 다르다**.
실험 condition이 다르므로 기존 ArtifactManager가 각각 독립적인 run/model/log 경로를 만든다.
기존 모델을 resume하지 않는 명령이다.

```bash
# 향후 학습 명령. 이번 진단에서는 실행하지 않았다.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
python train_ppo_02.py --config configs/e2e_train_velocity_ab_a.yaml

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
python train_ppo_02.py --config configs/e2e_train_velocity_ab_b.yaml
```

비학습 진단 재실행:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
python compare_dr_policies.py --config configs/eval_dr_transfer.yaml \
  --model baseline=artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip \
  --cases hover step-005 --seed 42

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
python diagnose_velocity_ab.py
```

## 검증과 변경 파일

관련 검증 **110 passed, 1 skipped** (104개 통과 후 새 검증 6개 추가, A/B 모듈 15개 재확인).
기존 DR 재현 경로 환경변수가 없는 선택적 검사는 skip이다. 전체 저장소 테스트 통과 주장이 아니다.
4개 조합의 pre-observation/post-reward와 reward_balance, A/B resolved 단일 차이,
기존 D 설정 보존, 실제 구형/D 모델의 입력 호환 로드 및 잘못된 로드 거부,
비단위 전처리 이전 raw 복원, 목표 변경 상태 보존, 기존 COM·reward·로깅 경로를 검사했다.

D final을 새 코드에서 비학습 재평가한 `artifacts/runs/dr-transfer-g08lsp1_`은
이전 `dr-transfer-39rkr7zg`와 세 CSV의 공통 147개 열이 label 외 정확히 일치했다.
hover/5cm/0.5m은 그대로 2.68/2.65/0.14초 max_tilt 종료다. 신규 raw 진단 열은 별도다.

이번 수정 파일:

- `crazyflie_rl/config.py`, `velocity_reference.py`, `environment.py`: 모드 분리와 공통 계산.
- `training.py`, `artifacts.py`, `dr_policy.py`, `eval_cli.py`, `dr_transfer.py`,
  `interactive_eval.py`: 별도 reward 계약 기록; 관측 기준 호환성.
- `reward_balance.py`: 실제 reward 모드와 진단 피연산자 일치.
- `configs/e2e_train_velocity_ab_a.yaml`, `configs/e2e_train_velocity_ab_b.yaml`: 신규 A/B 설정.
- `diagnose_velocity_ab.py`: 비학습 복원 adapter, 실제 추론 검증, 기존 학습 기록 분석/그래프.
- `tests/test_velocity_ab.py`, `tests/test_reward_diagnostics.py`: 집중 검증과 신규 raw 열 반영.
- 이 문서 및 이전 보고서의 후속 진단 링크.

통합 증거 디렉터리: `artifacts/runs/velocity-ab-diagnosis-v7o58b14`.
`diagnostic_manifest.json`, A/B resolved/dry-run, `ab_resolved_diff.json`,
`d_replay_regression.json`, 테스트 로그와 수정 전 소스 snapshot을 포함한다.

미확인 가설은 관측 표현의 최적화 영향, 속도 reward 변경의 영향, 탐색·seed 민감성 등이다.
이들을 확인하기 전에 가중치·게인·학습률을 조정하지 않았다. A/B 학습 결과는 아직 없으며,
C/D 추가 학습도 실행하지 않았다.

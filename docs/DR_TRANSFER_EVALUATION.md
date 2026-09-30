# 기존 PPO 체크포인트의 DR 전이 비교

`compare_dr_policies.py`는 명시한 체크포인트를 공통 환경에서 평가한다. PPO
학습·optimizer update·checkpoint 저장은 하지 않는다. 기존 train YAML,
`environment.reset`, 환경의 raw 관측/행동, allocator, actuator 및 `view_live` 루프는
변경하지 않았다. 2026-09-30 추가한 `error` 모드는 평가에서 정책에 전달할 속도의
의미만 바꾸는 별도 추론 실험이다(하단 설명). 기본 `absolute` 경로는 유지한다.
체크아웃에서 적용할 AGENTS.md는 발견되지 않았다.

## 구현 파일

- `compare_dr_policies.py`: import-safe CLI 진입점.
- `configs/eval_dr_transfer.yaml`: 기존 E2E 설정 상속, 공통 평가 조건만 명시.
- `crazyflie_rl/dr_transfer.py`: 평가 adapter, 케이스, 시각 정렬, 지표, 파일 저장.
- `crazyflie_rl/dr_policy.py`: 모델/manifest 계약 확인, 고정 정규화, SHA256 기록.
- `crazyflie_rl/plotting.py`: 기존 headless backend·경로 검사·quaternion 변환을
  재사용하는 N개 정책 비교 plot 함수 추가. 기존 plot API 동작은 유지.
- `tests/test_dr_transfer.py`: 비학습 검증.

## 실행

저장소 루트에서 사용자가 지정한 두 정책을 실행하는 정확한 명령:

```bash
python compare_dr_policies.py \
  --config configs/eval_dr_transfer.yaml \
  --model baseline=artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip \
  --model posdr=artifacts/runs/ppo_e2e_hover_position-only-dr_seedunset_20260929-132036/models/ppo_e2e_hover_position-only-dr_seedunset_best_20260929-132036-04.zip \
  --cases step-005 step-050 circle --seed 42
```

`--dry-run`을 덧붙이면 모델 경로/해시/입출력 계약, 공통 resolved 설정, 초기
조건, phase 및 horizon을 출력한다. 환경을 생성하거나 rollout 파일을 쓰지 않는다.
모델 메타데이터·shape 검사를 위해 PPO는 로드하지만 학습은 하지 않는다.

`--cases step-005`처럼 한 케이스만 지정할 수 있다. `--model LABEL=ZIP`은
두 개 이상이며 label은 중복할 수 없다. 같은 파일을 서로 다른 label로 지정하는
것은 재현성 검증용이다. `latest`/`latest-best`는 지원하지 않는다.

`--output-dir /원하는/부모경로` 아래에 고유한 `dr-transfer-*` 하위 디렉터리를
새로 만든다. 기본은 `artifacts/runs/` 아래이며 기존 결과를 덮어쓰지 않는다.
정착 조건은 `--position-band 0.005 --speed-band 0.02 --settle-hold-sec 1`
옵션으로 바꿀 수 있고 manifest에 기록한다.

## 공통 설정 및 초기화

- E2E, 관측 15/행동 4, action scale `[0.0075,0.0075,0.001,0.5]`,
  xy/z 가중치 10/6, 정책 100Hz/물리 500Hz.
- nominal payload, payload/actuator 파라미터 랜덤화 off. 1차 모터 모델과
  기존 시상수/물성/종료 한계는 유지한다.
- legacy perturbation=0 및 새 `initial_pose_randomization.enabled=false`.
  새 설정이 존재하면 legacy 분기보다 우선한다. 양쪽 모두 명시적으로 껐다.
- 각 모델×케이스마다 새 환경을 만든다. 모델의 학습 설정은 검증·기록에만 쓰고
  환경을 만드는 데 사용하지 않는다.
- step은 reset 기준점을 `[0,0,1]`로 두고 정상 reset 후 qpos/zero qvel을
  고정한다. `mj_forward` 후 airborne hover actuator reset, PID/previous-action/
  step-counter 초기화를 확인한 뒤 목표만 이동한다. 모터 파라미터를 다시 뽑지 않는다.
- 실제 reset은 `options`를 무시하므로 존재하지 않는 옵션은 사용하지 않는다.
  adapter는 현재 `_read_state()`와 부작용 없는 `_obs()`를 재사용한다.
  목표 설정은 pos_des/yaw_des만 바꾸며, 매 predict 직전 관측을 새로 만든다.
  첫 위치 오차는 float32 표현의 `[-0.05,0,0]` / `[-0.5,0,0]`다.
- 절대 선속도, wxyz, body omega, sin/cos yaw error 의미를 유지한다.
  관측 재생성에는 history 업데이트나 추가 RNG 소비가 없다.
- snapshot에는 qpos/qvel/ctrl, 관측/목표, PID 적분, 이전 action/step,
  모터 RPM/추력/command/wrench, 물성도 저장하고 모델 사이의 정확한 일치를 확인한다.

## 원 미션과 종료 한계

기존 `mission_from_experiment`/명시적 `CircleMission`을 그대로 사용한다.
center=(0.5,0), radius=0.5, period=5, laps=2, altitude=1, ccw,
start_angle=0, ramp=2다. TAKEOFF 4초 → SETTLE1 2초 → GOTO 4초 →
SETTLE2 2초 → CIRCLE 12초(ramp 2 + 2×5) → HOLD 2초로 총 **26초**다.
평가 환경의 horizon만 26초/2,600 policy steps로 설정한다.

바닥 시작 `[0,0,0.02]`, identity quaternion/zero velocity, ground-zero RPM은
기존 `EvaluationRunner._force_floor_start`를 재사용한다. 이 시작 방식과 phase
시간은 현재 `view_live_circle_eval.yaml` 및 과거
`artifacts/view_live/ppo_e2e_circle_c-small-T5-L2_seed42_20260922-221434/manifests/ppo_e2e_circle_c-small-T5-L2_seed42_manifest_20260922-221434.json`
의 runtime mission 기록과 일치함을 확인했다.

**기존 view_live는 원 미션에서 env.terminated를 기록한 뒤 계속 진행한다.**
새 평가에서는 요청된 계약대로 첫 terminated/truncated에서 멈춘다.
바닥 시작은 그대로 둔 종료 하한 min_altitude=0.2보다 낮으므로, 이번 두 모델은
0.01초에 종료됐다. 원 궤적 phase를 실행한 성능 비교로 해석할 수 없다.
지표를 얻기 위해 종료 플래그를 무시하거나 한계를 완화하지 않았다.

## 샘플 시각·지표

각 CSV row는 하나의 제어 전이이며:

- `time=t`, `phase`, `*_before`, `reference`, `observation`, `action`은 추론 시점.
- `reference`는 실제 `[t,t+dt]` 제어 구간에 유지한 참조다.
- `time_post=t+dt`, `phase_post`, `position/velocity/quaternion/omega`,
  `reference_post=mission.reference(t+dt)`는 평가 시점이다.
- RMSE는 **동일 시각 post-state와 reference_post**의 오차로 계산한다.
- `motor_thrust_command/motor_thrust` 및 `wrench_command/wrench_actual`은
  해당 구간 마지막 물리 substep의 요청/실제 값이다. 구간 평균이 아니다.
- reward 및 reward 진단은 환경의 실제 held-control-reference 기준이다.
  post-reference 기준 평가 오차와 혼동하지 않는다.

기존 view_live는 post-state를 `t`로 기록하고 held reference와 비교했다.
새 루프는 이 혼합을 재사용하지 않고 pre/post 시각을 구분한다.

위치 RMSE는 xy=`sqrt(mean(ex²+ey²))`, z=`sqrt(mean(ez²))`,
total=`sqrt(mean(ex²+ey²+ez²))`다. 수평은 축 수로 나누지 않는 거리 기준이다.
기존 전체 3D 정의를 유지하며 circle phase 및 다른 기존 phase도 별도로 기록한다.

step의 마지막 2초 지표는 완주한 경우에만 고정 구간 **(6,8]** 표본에서 구한다.
위치 RMSE, 속도 norm RMS, x/y/z 위치 표준편차(ddof=0)를 기록한다.
조기 종료되면 해당 값 전부와 settling은 null이다. 부분 RMSE/overshoot는 실제
관측 구간만의 값이며 `partial=true`/`metric_scope=partial_observed_only`로 표시한다.
누락 구간은 채우지 않는다.

first position entry는 속도와 무관한 첫 위치-band 진입 시각이다. settling은
위치 norm≤band, 속도 norm≤band가 동시에 만족한 뒤 종료까지 한 번도 벗어나지
않는 suffix의 최초 시각이며 최소 1초 길이가 필요하다. 이벤트는 t=0 초기 표본도
포함하고 100Hz 표본에서 판정한다. 연속 시간 사이의 교차 시각을 추정하지 않는다.
미달이면 `settled=false`, `settling_time_s=null`이다. circle에는 이 정지 목표
지표를 넣지 않는다. horizon에 정상 도달한 step은 completed=true와 truncated=true가
동시에 가능하며, 물리적 terminated와 구분한다.

## 체크포인트와 정규화

명시한 zip 절대 경로·SHA256, 연결 가능한 run manifest/sidecar/명시적
`--manifest LABEL=JSON`의 내용·해시를 남긴다. 알려진 action scale, control mode,
입출력 차원, 관측 의미가 공통 조건과 다르면 오류다. 알려지지 않은 학습 scale은
`unknown_no_training_scale_metadata`로 기록하며 검증됐다고 표시하지 않는다.

정규화가 알려져 있거나 PPO `_last_original_obs`가 non-null이면 저장 통계가
필요하다. `--normalization LABEL=/path/vecnormalize.pkl`로 제공한다.
통계는 VecNormalize로 읽고 training=false/norm_reward=false로 고정한다.
환경을 wrapper로 step/reset하지 않고 부작용 없는 `normalize_obs`만 사용한다.
파일 해시, obs shape, 평균/분산의 유효성을 검사한다.

이 저장소의 학습 진입점 manifest와 PPO의 null `_last_original_obs`는 현재 raw
관측 경로의 근거로 기록한다. 임의의 외부 전처리는 zip만으로 증명할 수 없다.
이 근거도 없는 모델은 `--normalization LABEL=none` 명시적 선언 또는 통계 파일이
필요하다. 알려진 정규화 모델에 `none`을 주면 실패한다. FrameStack 등 미지원
wrapper가 manifest에 있으면 조용히 맞추지 않고 실패한다. PPO의
`normalize_advantage`는 관측 정규화가 아니다.

## 출력 및 이번 실행

고유 결과 디렉터리:
`artifacts/runs/dr-transfer-8rfoihws/`

- `manifest.json`: 공통 설정, 초기 snapshot, 케이스/phase/horizon,
  checkpoint 및 manifest hash, 실제 override, seed, threshold, 시각 규약.
- `step-005-baseline.csv` 등 6개 전이 로그. 벡터 suffix는 순서대로
  xyz, quaternion은 wxyz, action/wrench는 tau_x/tau_y/tau_z/Fz다.
- `*-reference.npz`: 조기 종료 여부와 무관한 완전한 공통 reference sequence.
- `summary.csv`, `summary.json`: 모델×케이스의 완주/부분 지표 및 종료 원인.
- `*-comparison.png`: 같은 축에서 두 정책의 위치·절대속도·자세·각속도를
  비교하며 전체 위치 참조와 phase 경계도 표시한다.

| 케이스 | 모델 | 관측 시간 | 전체 위치 RMSE (m) | 최종 2초 위치 RMSE (m) | settling (s) |
|---|---|---:|---:|---:|---:|
| step-005 | baseline (legacy position reset) | 8.00 | 0.0110384 | 0.0021100 | 1.44 |
| step-005 | POS-only DR (new reset) | 8.00 | 0.0245290 | 0.0171895 | null |
| step-050 | baseline (legacy position reset) | 0.40, max_tilt | 0.4181637 (partial) | null | null |
| step-050 | POS-only DR (new reset) | 0.23, max_tilt | 0.4927105 (partial) | null | null |
| circle | 두 모델 | 0.01, min_altitude | TAKEOFF 부분 표본만 | 해당 없음 | 해당 없음 |

baseline SHA256: `c054ba76c7813256a84e038040b905d23c7962c1063de3f464cc27365169108b`

posdr SHA256: `a9c4521a7249c1049107f3d8aeb43c4c0b80111782493b014a7fd53db0a7d925`

두 모델의 학습 manifest에서 action scale과 xy/z=10/6을 확인했다.
**baseline도 legacy 위치 perturbation=0.05가 있었으므로 완전히 reset 난수가 없는
학습은 아니다.** 새 pose DR은 없었고, POS-DR 모델은 새 sampler의 반경 0.15m,
자세 랜덤화 off였다. 평가에서는 두 종류 reset 난수를 모두 껐다.
위 결과로 DR 자체를 원인으로 자동 확정하거나 여러 seed를 독립 학습 반복으로
간주하지 않는다. 특히 0.5m의 서로 다른 부분 구간 RMSE는 완주 점수처럼 비교할 수 없다.

## 검증

최초 구현 당시 관련 비학습 테스트 **58개 통과**. `git diff --check` 및 새 모듈의
Python 컴파일 검사도 통과했다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
  python -m pytest -q tests/test_dr_transfer.py tests/test_plotting_comparison.py \
  tests/test_missions.py tests/test_import_safety.py
```

첫 predict 관측, 목표 변경의 상태 불변성, 두 정책의 초기 snapshot/reference,
26초 horizon, post-reference 시각 정렬, RMSE/정착/null 처리, normalization
통계 고정·누락 실패, scale/차원/속도 의미 불일치 실패를 검증한다.
실제 baseline checkpoint를 두 label로 모든 케이스에 실행하는 테스트에서는
CSV가 byte 단위로 일치한다. PPO.learn/PPO.train을 호출하면 실패하도록 막고
checkpoint 해시 불변도 확인한다. 기존 plotting/mission/import 회귀도 포함한다.

## 2026-09-29 후속 점검: circle-air

기존 결과 `dr-transfer-8rfoihws`의 manifest/summary/CSV를 직접 읽었다.
기존 `circle` 및 그 결과는 보존했다. 새 `circle-air`는 기존 생성기의
`settle2_end=12`부터 `total=26`까지를 시간 이동하여 재사용한다:
`reference_air(t) = reference_circle(t + 12)`.
따라서 TAKEOFF/GOTO/SETTLE을 제외하고 **CIRCLE 12초(ramp 2초 포함) + HOLD 2초 = 14초**다.
원을 다시 근사하거나 원의 시간 매개화를 수정하지 않았다.

초기 위치는 생성기 `reference_air(0)`의 실제 출력 **[1,0,1] m**다.
level/yaw=0, 선속도·각속도=0, airborne hover equilibrium으로 초기화한다.
새 관측의 위치 오차는 0이다. 종료 한계는 min_altitude=0.2 m 및 max_tilt=60°를
포함해 그대로다. 원래 26초 바닥 시작 전체 미션과 다른 시험이다.
CLI의 기본 케이스 목록은 종전 세 개를 유지하며 `circle-air`는 명시적으로 선택한다.

실제 실행 결과: `artifacts/runs/dr-transfer-x4ib_6hk/`.
두 정책 모두 1,400 step, 14초 완료: `completed=true`, `terminated=false`,
`truncated=true`(horizon 도달), `partial=false`다.
초기 snapshot 전체와 공통 reference가 정확히 일치했으며 checkpoint SHA256도 유지됐다.

| 지표 (m) | baseline (legacy position reset) | POS-only DR (new reset) |
|---|---:|---:|
| 전체 xy RMSE | 0.294178 | 0.388092 |
| 전체 z RMSE | 0.056207 | 0.029363 |
| 전체 total RMSE | 0.299500 | 0.389201 |
| CIRCLE xy RMSE | 0.313180 | 0.404896 |
| CIRCLE z RMSE | 0.056030 | 0.031047 |
| CIRCLE total RMSE | 0.318152 | 0.406085 |
| CIRCLE 평균 고도 편차 | +0.002474 | +0.007765 |
| CIRCLE 고도 편차 min / max | -0.086367 / +0.085324 | -0.039180 / +0.056846 |
| CIRCLE 최대 절대 고도 편차 | 0.086367 | 0.056846 |

고도 편차는 동일 시각의 `z-z_ref`다. CIRCLE 지표는 기존 post-phase 규약대로
local t=0.01…11.99의 1,199개 표본을 사용한다. t=12의 경계 표본부터 HOLD에
포함되므로 HOLD는 t=12…14의 201개 표본이다. 실제 제어 CIRCLE 구간은 [0,12)다.
새 summary에는 전체 및 CIRCLE의 고도 min/max, 평균·min/max·최대 절대 편차를 추가했다.
두 모델의 수평/고도 성능 차이를 관찰했지만 그 원인을 DR로 확정하지 않는다.

정확한 후속 실행 명령(저장소 루트, MuJoCo/SB3 설치 환경):

```bash
MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_dr_policies.py \
  --config configs/eval_dr_transfer.yaml \
  --model baseline=artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip \
  --model posdr=artifacts/runs/ppo_e2e_hover_position-only-dr_seedunset_20260929-132036/models/ppo_e2e_hover_position-only-dr_seedunset_best_20260929-132036-04.zip \
  --cases circle-air --seed 42
```

`--dry-run`도 해당 케이스의 14초 horizon/초기 위치/원래 미션 시간과 시간 이동을
출력한다. 파일명 및 CLI label은 `baseline`/`posdr`를 유지하고 manifest/summary/
콘솔/비교 plot의 표시명은 위 표의 이름을 사용한다.

## 학습 reset 감사

최종 감사 산출물: `artifacts/runs/dr-reset-audit-ckgo78r8/`.
`reset_audit.json`, `reset_comparison.csv`, `baseline-reset-samples.npz`,
`posdr-reset-samples.npz`, `existing_rollout_audit.json`을 저장했다.
각 run의 보관된 resolved YAML과 manifest의 설정이 일치하는지 검사하고,
현재 train YAML을 상속하지 않고 보관된 모든 값을 검증 가능한 config로 재구성했다.
재구성 후 `resolved_dict()`의 정확한 일치도 확인한다.

| 확인 항목 | baseline (legacy position reset) | POS-only DR (new reset) |
|---|---|---|
| sampler 선택 근거 | 새 config 없음 → legacy | 새 config 존재 → legacy보다 우선 |
| 위치 분포 | 축별 독립 Uniform(-0.05,0.05) | Gaussian 벡터 정규화 방향, r=0.15×U, U~Uniform(0,1) |
| 축별 절대 상한 (m, 이론) | 각각 0.05 | 각각 0.15 |
| norm 상한 (m, 이론) | √3×0.05 = 0.086603 | 0.15 |
| norm RMS (m, 이론) | 0.05 | 0.15/√3 = 0.086603 |
| legacy 필드 잔존값 | 0.05 (적용) | 0.05 (새 설정에 의해 무시) |
| 자세 난수 | off, legacy 각도=0 | off, new attitude.enabled=false |
| actuator 초기화 | auto → airborne hover equilibrium | 동일 |
| 모터 모델/시상수/난수화 | cf21b_first_order / 0.05 s / off | 동일 |
| payload | nominal, mass=0, offset=[0,0], 난수화 off | 동일 |
| 관측/행동 | 15/4, 절대속도 입력, raw 관측 근거 존재 | 동일 |
| action scale | [0.0075,0.0075,0.001,0.5] | 동일 |
| 위치 가중치 | legacy 4, 유효 xy=10 / z=6 | 동일 |
| 나머지 보상 가중치 | velocity=.005, tilt=3, omega=.001, yaw=1, action=0, action_rate=0, crash=10 | 동일 |
| 학습 seed | config와 checkpoint 모두 null; 실제 난수 seed 미확인 | 동일 |
| 선택 checkpoint timestep | **980000** (zip metadata와 manifest 일치) | **980000** (동일 근거) |

두 manifest의 git SHA는 `9a5419e5a3dc73921121bdc8f08ea8241354bebd`이고
`dirty=true`다. 따라서 **보관된 설정과 현재 코드의 sampler 선택 경로는 확인했지만,
당시 실행된 dirty 소스 전체가 현재 소스와 정확히 같았는지는 미확인**이다.
이번 표본은 보관 설정을 현재 실제 환경 reset으로 재생한 값이다. 실행에 사용한
environment/initial_pose/config 소스와 YAML/manifest/NPZ의 해시를 감사 JSON에 남겼다.
체크포인트 suffix `-13`/`-04`는 timestep 근거로 사용하지 않았다.
관측 정규화는 기존 로더가 확인한 학습 command 및 checkpoint의 null
`_last_original_obs` 근거를 기록했으며, 별도 외부 전처리의 부재까지 증명하지 않는다.

새 sampler는 **반지름 균등이며 구 내부 부피 균등이 아니다**.
가정적으로 R=.05라면 norm RMS=.028868 m지만, 이 POS-only DR run의 보관값은
**R=.15**이므로 그 가정의 수치를 이번 모델의 실측값으로 사용하지 않는다.

각 조건 실제 `env.reset()` **10,000회**, 첫 reset만 seed=42, 이후 seed=None으로
난수열을 진행했다. 두 조건 각각 서로 다른 위치 표본 10,000개, level 자세/zero
velocity/고정 모터 RPM 및 hover 추력을 확인했다. 학습 rollout의 방문 상태 분포가 아니다.

| norm 통계 (m, 재생 실측) | baseline (legacy position reset) | POS-only DR (new reset) |
|---|---:|---:|
| max | 0.085020 | 0.149967 |
| mean | 0.047992 | 0.074749 |
| RMS | 0.049969 | 0.086413 |
| P50 | 0.049285 | 0.075290 |
| P95 | 0.069111 | 0.142635 |

| 모델·축 | min (m) | max (m) | mean (m) | std (m) |
|---|---:|---:|---:|---:|
| baseline (legacy position reset), x | -0.049994 | 0.049996 | -0.000176 | 0.028795 |
| baseline (legacy position reset), y | -0.049999 | 0.049994 | -0.000028 | 0.028797 |
| baseline (legacy position reset), z | -0.049994 | 0.049995 | 0.000051 | 0.028957 |
| POS-only DR (new reset), x | -0.148028 | 0.147088 | 0.000077 | 0.050312 |
| POS-only DR (new reset), y | -0.148950 | 0.147103 | 0.000270 | 0.049719 |
| POS-only DR (new reset), z | -0.147646 | 0.146092 | 0.000323 | 0.049638 |

축별 RMS/P50/P95도 JSON에 기록했다. 이 두 보관 설정에서 POS-only DR의 reset
오차 norm RMS는 더 크다. 이것만으로 평가 결과의 원인을 특정할 수는 없다.

감사 재실행:

```bash
python -m crazyflie_rl.dr_reset_audit \
  --source-run artifacts/runs/dr-transfer-8rfoihws --samples 10000 --seed 42
```

`--output-dir` 아래 고유 `dr-reset-audit-*` 디렉터리를 생성한다. 기존 비교의
manifest가 가리키는 명시적인 모델 경로/해시를 확인하고 그 run의 보관 설정을 읽는다.

## 기존 step CSV 재분석

step-005의 고정 tail (6,8]에 대해 `MSE = ||평균 오차||² + Σ축별 분산`으로 분해했다.

| 지표 | baseline (legacy position reset) | POS-only DR (new reset) |
|---|---|---|
| 평균 오차 xyz (mm) | [-1.97635,-0.67489,+0.30073] | [-11.74411,-2.73004,+12.24960] |
| 위치 표준편차 xyz (mm) | [0.00583,0.00408,0.00160] | [0.11582,0.13951,0.12378] |
| 평균 offset의 MSE 비중 | 99.9988% | **99.9837%** |

이 구간의 POS-only DR 위치 RMSE 17.1895 mm는 대부분 평균 offset이다.
작은 위치 변동도 존재하며, 이 분석으로 다른 시간대나 자세·속도의 진동까지 부정하지 않는다.

step-050의 최종 제어 전이(종료 전 pre-state와 종료 post-state를 분리):

| 항목 | baseline (legacy position reset) | POS-only DR (new reset) |
|---|---|---|
| pre → post 시각 (s) | 0.39 → 0.40 | 0.22 → 0.23 |
| pre 위치 오차 xyz (m) | [-.173096,.128533,-.022557] | [-.472551,.007027,-.014262] |
| pre roll/pitch (deg) | [35.132,-31.474] | [-16.734,55.604] |
| pre omega xyz (rad/s) | [14.8398,-18.4415,-1.5963] | [-2.5669,11.9544,-.0047] |
| post 위치 오차 xyz (m) | [-.157150,.135547,-.020953] | [-.467492,.008173,-.017738] |
| post roll/pitch (deg) | [50.378,-39.326] | [-22.322,62.399] |
| post omega xyz (rad/s) | [16.3200,-20.4625,-1.5465] | [-2.8154,12.7175,.0168] |
| action (tau_x,tau_y,tau_z,Fz) | [.17027,-.42889,.65581,-.19997] | [-.06643,.05292,.19602,-.26771] |
| 모터 요청 추력 (N, 4개) | [.122710,.022017,.095279,.085568] | [.081893,.071005,.080459,.058346] |
| 모터 실제 추력 (N, 4개) | [.128422,.046348,.114752,.159108] | [.073041,.089434,.088327,.057734] |
| 요청/실제 총추력 (N) | .325574 / .448630 | .291702 / .308536 |
| 종료 | max_tilt | max_tilt |

환경의 tilt는 roll 또는 pitch 하나의 절댓값과 같지 않다. 추력은 최종 구간의
마지막 물리 substep 값이며 시간 평균이 아니다. 요청/실제 wrench 전체는 JSON에
함께 기록했다. **allocator의 clipping 전 할당값·포화 flag는 기존 CSV에 없어 미측정**이다.
추력 차이나 종료 자세만으로 allocator 포화를 원인으로 확정하지 않는다.
0.5 m 순간 목표 변경은 부드러운 goto/원 추종과 같은 시험이 아니다.

## 후속 수정 범위와 검증

이번 수정: `crazyflie_rl/dr_transfer.py`(circle-air/고도 지표/표시명),
`crazyflie_rl/dr_reset_audit.py`(신규 감사), `tests/test_dr_transfer.py`,
`tests/test_dr_reset_audit.py`, 이 문서. 기존 첫 구현 파일 및 사용자의 변경을 보존했다.
공용 환경/미션 생성기/학습 YAML/학습 코드/체크포인트는 수정하지 않았다.

관련 비학습 테스트 **66개**: 공중 초기 p=p_ref/유효 고도/모터 상태,
두 환경 snapshot 일치, 1,401개 참조의 원본 시간 이동 관계, 14초 horizon,
첫 추론 관측, 목표 설정의 상태 불변성, 고도 편차 정의, RMSE/실패 null,
실제 동일 checkpoint 두 label의 circle-air CSV byte 일치,
보관 설정 재구성/난수열 진행/legacy 우선순위/표본 재현성과 기존 회귀를 포함한다.
학습 호출은 테스트에서 금지했으며 실제 평가와 감사에서도 실행하지 않았다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
  python -m pytest -q tests/test_dr_transfer.py tests/test_dr_reset_audit.py \
  tests/test_plotting_comparison.py tests/test_missions.py tests/test_import_safety.py
```

`circle-air --dry-run`, `git diff --check`, 변경 Python 컴파일 검사도 수행했다.
**이는 전체 저장소 테스트 통과나 기존 실패 테스트 해결을 뜻하지 않는다.**
이번 후속 작업에서 commit/push는 하지 않았으며 원격 반영 완료로 보고하지 않는다.

## 2026-09-30: 절대 속도 / 속도 오차 입력 실험

새 실제 결과: **`artifacts/runs/dr-transfer-u_pc02_3/`**.
두 checkpoint × `absolute/error` × `circle-air/step-005`의 8개 평가를 실행했다.
기존 `dr-transfer-x4ib_6hk`와 `dr-transfer-8rfoihws`는 읽기 및 회귀 비교에만 사용했다.
reset 분포 감사는 반복하지 않았다. 학습 YAML/보상/action scale/물성/초기화/종료
한계를 변경하지 않았고 학습, optimizer update, checkpoint 저장, commit/push도 하지 않았다.

### 입력 변환과 해석적 참조 속도

실제 환경 `_read_state()`는 `qvel[:3]` world-frame 선속도를 읽으며, `_obs()`는
`[p-p_ref, v_world, quaternion(wxyz), omega_body, sin(yaw-yaw_ref), cos(yaw-yaw_ref)]`
순서의 float32 15차원 관측을 만든다. 기존 wrapper에는 참조 속도 차감이 없다.
`crazyflie_rl/dr_policy.py`의 **`VELOCITY_SLICE`와 `policy_raw_observation()`**에서
slice [3:6]과 변환을 정의한다. `dr_transfer.run_case()`는 다음 순서로 호출한다.

1. 같은 추론 시각 t의 상태와 위치/yaw 참조로 원래 raw 관측 생성.
2. `absolute`는 원래 관측을 그대로 사용. `error`는 사본의 속도만 `v_world-v_ref_world`로 교체.
3. `FrozenPolicy.predict()`의 기존 고정 normalization/clip_obs(있는 경우) 적용.
4. 기존 deterministic 정책 추론과 action 처리.

raw 관측을 float32로 만드는 기존 계약을 유지한다. 그 raw 속도에서 물리 단위
참조 속도를 차감하고 float32로 저장한다. 원본 관측, MuJoCo qvel/센서, 로그용
물리 상태, reward에 쓰는 실제 속도에는 변환을 적용하지 않는다.
두 실제 checkpoint는 기존 raw 관측 경로이며, 별도 비단위 VecNormalize 통계 테스트에서
차감 → 정규화 → clipping 순서 및 mean/var/count·통계 파일 불변을 확인했다.

`CircleMission.reference_velocity()`와 `ramped_phase_velocity()`를 추가했다.
기존 ramp는 위치 보간이 아니라 half-cosine 각속도의 적분이다. CIRCLE 시작 이후
시간 s, ramp R=2, omega=2π/5일 때:

```text
theta_dot(s) = omega * (1 - cos(pi*s/R))/2   (0 <= s < R)
             = omega                      (R <= s < CIRCLE 끝)
v_ref = radius * theta_dot * [-sin(theta), cos(theta), 0]
```

theta는 기존 `circle_phase()`를 그대로 사용한다. `circle-air`의 위치와 속도 모두
원래 mission time **t+12**를 사용한다. 시작 속도는 0, local t=1에는 speed=π/10,
t≥2의 정상 원 구간에는 speed=π/5 m/s다. local **t=12부터 HOLD의 v_ref=0**이다.
이 경계의 위치는 연속이지만 속도는 불연속이므로 양측 미분은 정의되지 않는다.
기존 phase 판정과 같은 오른쪽 구간값 0을 적용하며 impulse나 가짜 spike를 만들지 않는다.
TAKEOFF/GOTO의 기존 cosine 위치식도 해석적으로 미분했다. 이번 실제 실행에는 이 구간이 없다.
고정 목표와 step은 regulation command이므로 v_ref가 항상 0이다.
위치 생성식, phase 길이, 샘플 시각, 종료 조건은 변경하지 않았다.

### 실제 결과

아래는 이번 실행의 **CIRCLE phase** 지표다. 네 조합 모두 local t=0.01…11.99의
1,199개 CIRCLE post-state 표본을 확보했다.

| 모델 | 입력 모드 | xy RMSE (m) | z RMSE (m) | total RMSE (m) | 최대 절대 고도 편차 (m) | velocity RMSE (m/s) | episode 결과 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline (legacy position reset) | absolute | .313180 | .056030 | .318152 | .086367 | .401647 | 14초 완료 |
| baseline (legacy position reset) | error | .031386 | .014295 | .034488 | .022449 | .046753 | 14초 완료 |
| POS-only DR (new reset) | absolute | .404896 | .031047 | .406085 | .056846 | .494146 | 14초 완료 |
| POS-only DR (new reset) | error | .041837 | .014045 | .044131 | .024310 | .059088 | **12.16초 max_tilt 종료 (HOLD)** |

| 모델 | 입력 모드 | 실제 수평 속도 RMS (m/s) | roll RMS (deg) | pitch RMS (deg) |
|---|---|---:|---:|---:|
| baseline (legacy position reset) | absolute | .522399 | 2.572427 | 2.810940 |
| baseline (legacy position reset) | error | .631977 | 3.304979 | 3.360826 |
| POS-only DR (new reset) | absolute | .280254 | 1.306569 | 1.540246 |
| POS-only DR (new reset) | error | .641292 | 3.414185 | 3.422430 |

velocity_rmse=`sqrt(mean(sum((v_world-v_ref_world)^2)))`이며, 수평 속도 RMS는
오차가 아닌 `sqrt(mean(vx²+vy²))`다. roll/pitch는 world 자세 quaternion에서 얻은
Euler 각의 RMS(deg)다. 모두 위치 지표와 같은 post-state 시각의 참조를 사용한다.
전체 episode 지표도 summary에 별도로 저장했다.

POS-only DR/error는 CIRCLE 구간은 관측했지만 HOLD에서 실패했다. 기존 summary의
`partial`은 episode 완주 여부를 따른다. 따라서 이 조합은 전체와 circle_phase 모두
`partial=true`이며, CIRCLE 표본 수가 다른 조합과 같다는 사실과 구분해 해석한다.
종료 이후 표본을 채우지 않았고, 종료 한계를 완화하지 않았다.
나머지 세 circle-air 조합은 `completed=true, terminated=false, truncated=true`다.

step-005는 모델별 두 입력 모드의 raw 관측/정책 raw 입력/행동/물리 상태/보상/종료가
모두 정확히 같았다. 네 조합 모두 8초 완료했다.

| 모델 | 입력 모드 | 마지막 2초 total 위치 RMSE (m) | settling (s) |
|---|---|---:|---:|
| baseline (legacy position reset) | absolute / error 동일 | .002109961 | 1.44 |
| POS-only DR (new reset) | absolute / error 동일 | .017189480 | null (미달) |

이번 원 구간에서는 **동일 정책의 속도 입력 변환으로 추종 오차가 감소**했다.
다만 POS-only DR/error의 HOLD 종료가 있으므로 전체 제어 성능·안정성 개선으로
일괄 판정하지 않는다. DR의 인과효과, 다른 궤적 일반화, 안정성 보장 또는
가속도 feedforward 효과에 대한 결과가 아니다. 보상 합계로 우열을 판정하지 않았다.

### 출력과 재실행

기본 `--velocity-inputs`는 `absolute` 하나다. 명시적으로 `absolute error` 또는
`error`만 선택할 수 있다. 모드 중복은 오류다. 두 모드 실행 파일명은
`circle-air-baseline--velocity-absolute.csv`처럼 조합을 포함한다. absolute만 실행하면
종전 파일명 규약을 유지하며, 모든 CSV와 summary에 입력 모드를 명시한다.

manifest schema v2에는 모드 조합, 환경 raw 계약과 입력 변환의 의미, frozen
normalization 순서, 참조 속도 수식/시간 규약, 공통 config/실제 override/미션,
모델 hash, 초기 snapshot을 기록했다. 기존 `_git_metadata()`로 commit/branch/dirty를
기록하고 실행 소스 SHA256도 저장했다. `error`는 원래 정책의 동일 조건 평가가 아닌
추론 입력 변환 실험으로 명시한다.

각 CSV row의 시각/속도 열은 다음과 같다(모든 선속도 world frame, m/s).

| 시각 | 실제 속도 | 참조 속도 | 속도 오차 | 정책 입력 |
|---|---|---|---|---|
| `time=policy_input_time=t` | `velocity_before_0..2` | `reference_velocity_0..2` | `velocity_error_before_0..2` | `policy_raw_velocity_0..2`, `policy_raw_observation_0..14` |
| `time_post=t+dt` | `velocity_0..2` | `reference_velocity_post_0..2` | `velocity_error_0..2` | 해당 없음 |

`observation_0..14`는 변환 전 환경 raw 관측이다. action은 t에서 추론해 [t,t+dt]에
적용된다. position RMSE와 velocity RMSE는 t+dt끼리 비교한다.
모든 row에 `model_label`, `observation_velocity_mode`, `case`가 기록된다.
reference NPZ에는 전체 위치·속도 참조를 저장한다.

- `circle-air-comparison-xy.png`: 동일 XY 축의 참조와 네 조합 경로.
- `circle-air-comparison.png`: 같은 시간축의 위치, 실제/참조 속도, 자세, 각속도.
- `step-005-comparison*.png`: 고정 목표 회귀 결과. 같은 모델의 두 모드가 겹친다.
- 모델별 색(파랑 baseline, 주황 POS-only DR), 모드별 선(실선 absolute, 점선 error).
- `velocity_validation.json`: 이전 결과/체크포인트 24개 파일의 실행 전·후 hash 및 검증 기록.

```bash
MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_dr_policies.py \
  --config configs/eval_dr_transfer.yaml \
  --model baseline=artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip \
  --model posdr=artifacts/runs/ppo_e2e_hover_position-only-dr_seedunset_20260929-132036/models/ppo_e2e_hover_position-only-dr_seedunset_best_20260929-132036-04.zip \
  --cases circle-air step-005 --velocity-inputs absolute error --seed 42
```

`--dry-run`을 덧붙이면 동일 8개 조합과 모델 경로, 공통 설정, raw 변환 계약,
참조 속도 정의를 출력한다. 이번에도 별도로 실행하여 확인했다.

### 이번 변경의 검증 범위

수정: `crazyflie_rl/dr_policy.py`, `dr_transfer.py`, `missions.py`, `plotting.py`,
`tests/test_dr_transfer.py`, 신규 `tests/test_velocity_input.py`, 이 문서.
CLI entrypoint는 기존 import-safe wrapper를 유지한다. 환경/학습 YAML은 수정하지 않았다.

이번 관련 단위·회귀 테스트 **68개 통과**, 위 실제 출력에 대한 별도 검증 **1개 통과**다.
이전 reset 감사 및 66개 테스트를 다시 수행했다는 의미가 아니다.
기존 전체 checkpoint smoke 두 개는 이번에 필요 없는 step-050/바닥 circle 실행 등을
피하기 위해 제외했다. 대신 지정된 실제 8개 조합을 실행하고 그 산출물을 검증했다.

- absolute의 이전 CSV 공통 열과 기존 summary 수치 **정확히 일치**.
- step-005의 모드 간 모든 CSV 열은 모드 식별자 외 **정확히 일치**.
- 매 추론의 raw 입력이 선언한 변환과 일치, 원본 관측·환경 상태 불변.
- 비단위 frozen 정규화 및 clipping 순서, v_ref=0의 raw/정규화 입력·행동 일치.
- ramp/정상 원/HOLD 및 cw/ccw 참조의 해석적 미분을 위치 중앙차분과 검증.
- 두 모델×두 모드의 초기 snapshot 및 전체 위치/속도 reference 일치.
- 체크포인트 및 이전 결과 24개 파일 hash 불변, 실행 소스 hash 일치.
- Python 컴파일 및 `git diff --check` 통과. 전체 저장소 테스트 통과를 의미하지 않는다.

재검증(평가 실행은 하지 않고 저장된 결과를 읽는 마지막 테스트 포함):

```bash
DR_VELOCITY_RESULTS=artifacts/runs/dr-transfer-u_pc02_3 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
  python -m pytest -q tests/test_velocity_input.py tests/test_dr_transfer.py \
  tests/test_missions.py tests/test_mission_variants.py tests/test_plotting_comparison.py \
  -k 'not existing_checkpoint_same_model_all_cases and not circle_air_same_checkpoint_reproducible'
```

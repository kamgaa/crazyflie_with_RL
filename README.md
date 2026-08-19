# Crazyflie with RL

MuJoCo Crazyflie에서 cascade PID baseline, PID + PPO residual, PPO end-to-end(E2E)를 동일한 실험 설정과 산출물 규칙으로 학습·평가하는 프로젝트다. 이번 구조는 기존 제어·학습 수식을 유지하면서 설정, 실행 lifecycle, 파일 이름을 한곳에서 관리한다.

## 제어 mode

- `residual`: cascade PID가 500 Hz로 계산한 wrench에 policy action을 scale해 더한다.
- `e2e`: scaled policy wrench에 base-mass gravity compensation만 더한다. PID는 사용하지 않는다.

두 mode의 실제 observation은 기존 checkpoint 호환성을 위해 모두 15차원이다.

```text
[position error(3), world velocity(3), quaternion wxyz(4),
 body angular velocity(3), sin(yaw error), cos(yaw error)]
```

Action은 두 mode 모두 `(4,)`, 순서 `[tau_x, tau_y, tau_z, Fz]`다. 자세한 수식과 활성값은 [동작 기준선](docs/BEHAVIOR_BASELINE.md), 의도적으로 유지한 문제는 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)에 기록했다.
기존 명령과 산출물에서 옮기는 절차는 [migration guide](docs/MIGRATION.md)에 있다.

## 구조

```text
configs/                    공통값과 실험별 YAML override
crazyflie_rl/
  config.py                 frozen dataclass, merge, validation
  artifacts.py              run ID, 파일명, resolved config, manifest
  controllers.py            cascade PID와 allocation/rotation 순수 함수
  environment.py            15D Gymnasium MuJoCo 환경
  factories.py              train/eval 환경 생성
  evaluation.py             고정-seed policy 평가
  training.py               PPO 생성, callback, best/final 저장
  missions.py               Hover/Circle/Lissajous reference와 legacy circle
  eval_cli.py               viewer/headless 공통 rollout lifecycle
  plotting.py               artifact 경로에만 저장하는 plot helper
artifacts/runs/              새 실행의 격리된 산출물
model/, tb/, root PNG       이동하지 않은 legacy 산출물
tests/                      설정·naming·환경·학습·import 계약
```

루트 Python 파일은 기존 명령을 유지하는 얇은 CLI wrapper다. `train_ppo_02.py`가 공식 학습 진입점이며 `train_ppo.py`는 기존 residual 30k smoke 동작을 보존한 legacy 진입점이다.

## 설치

Python 3.10 환경에서 다음 의존성을 설치한다.

```bash
python -m pip install -r requirements.txt
```

Legacy ZIP이 Stable-Baselines3 2.9.0에서 생성되었으므로 reload와 학습 의미를 맞추기 위해 해당 버전을 고정한다.

MuJoCo XML 기본값과 필요한 STL mesh는 저장소에 포함되어 있다.

```text
resources/mujoco/cf21B_500.xml
resources/mujoco/assets/cf21B/*.stl
```

코드는 이 project-relative 경로를 resolved config의 project root에서 절대경로로 해석한다. 다른 XML/mesh tree를 자동 탐색하거나 대체하지 않는다. runtime package 또는 asset이 없는 머신에서는 config/artifact/unit test만 실행하고 MuJoCo 통합 검증은 skip한다.

## Config profile

| Profile | 목적과 보존된 기본값 |
|---|---|
| `base.yaml` | 공통 vehicle/PID/reward/termination, E2E 활성 기본값 |
| `e2e_train.yaml` | 기존 `train_ppo_02.py`: 1M step, nominal, 30 evaluation episodes |
| `residual_train.yaml` | 기존 `train_ppo.py`: 30k step, fixed 10 g, 5 evaluation episodes |
| `e2e_hover_eval.yaml` | 기존 `view_live.py`: E2E, seed 42, nominal hover |
| `residual_hover_eval.yaml` | residual 30 g / 30 mm integrator diagnostic, scale 0.006 |
| `residual_circle_eval.yaml` | 기존 `circle_traj.py`: 5 g / 100 mm / 180°, scale 0.022 |
| `residual_circle_legacy006_eval.yaml` | 기존 `view_live_hover.py`: nominal circle, scale 0.006 |
| `cf21b_actuator_eval.yaml` | 명시적인 CF2.1 BLDC hover evaluation alias |
| `cf21b_actuator_torque_poly_eval.yaml` | BLDC plant-side paper-candidate propeller torque polynomial 추가 profile |
| `view_live_hover_eval.yaml` | 통합 `view_live.py` hover 기본값 |
| `view_live_circle_eval.yaml` | `small`: center (0.5, 0), radius 0.5 m |
| `view_live_circle_wide_eval.yaml` | `wide`: center (0, 0), radius 0.8 m |
| `view_live_lissajous_eval.yaml` | `figure8`: amplitude (0.5, 0.5), ratio 1:2, phase 90° |
| `view_live_lissajous_clover_eval.yaml` | `clover`: amplitude (0.55, 0.4), ratio 3:2, phase 90° |

Profile은 같은 디렉터리의 부모 파일을 `extends: base.yaml`처럼 상속한다. Mapping은 재귀 merge되고 list는 전체 교체된다. 알 수 없는 key, 빠진 필수 key, 잘못된 mode, 4차원이 아닌 scale, 0/음수 frequency와 episode는 runtime 객체를 만들기 전에 오류가 난다.

새 profile은 가장 가까운 profile을 상속하고 바뀌는 조건만 override한다.

```yaml
extends: residual_train.yaml

environment:
  payload:
    randomize: false
    mass: 0.010
    offset: [0.030, 0.0]

experiment:
  condition: m10g-r30mm-th0deg-fixed
  description: fixed off-center payload experiment
```

Resolved configuration은 각 run의 `config/`에 저장된다. Frozen dataclass와 tuple을 사용하므로 실행 중 shared list/NumPy reference가 설정을 바꾸지 않는다.

## CF2.1 BLDC actuator

모든 기본 train/eval/view_live profile은 `enabled: true`, `model: cf21b_first_order`를 사용한다. allocator가 만든 clipped requested motor thrust `f_cmd`는 positive-thrust branch의 bounded inverse를 거쳐 rotor speed target이 되고, physics 500 Hz substep마다 exact first-order update 뒤 actual thrust와 torque가 MuJoCo에 적용된다. 기본값은 `T=0.050 s`, `K=2900 rad/s`이며, 기본 plant-side yaw torque는 기존 allocation ratio `q = direction × 0.00594 × f_actual`를 유지한다.

외부 custom YAML도 위 두 값을 유지해야 한다. `actuator.enabled: false` 또는 `actuator.model: instantaneous`는 더 이상 유효한 실행 설정이 아니며 config 로딩 단계에서 실패한다. `randomization.enabled`만은 모터 모델 자체와 별개인 episode별 파라미터 변동 옵션이므로 기본적으로 꺼져 있다.

candidate thrust/torque polynomial과 rotor parameters는 논문의 값에서 가져온 **paper-candidate / unverified** 설정이다. 이 저장소의 XML·기체 질량·실기체에 대해 검증된 calibration이라고 주장하지 않는다. [How to Model Your Crazyflie Brushless (arXiv:2603.05944)](https://arxiv.org/abs/2603.05944)를 원 출처로 기록한다. 특히 과거 instantaneous plant에서 학습된 PPO checkpoint는 새 BLDC plant에서 같은 비행 성능을 보장하지 않으므로 재평가·필요 시 재학습해야 한다.

```bash
python view_live.py --mode hover --model model/ppo_best.zip --headless

# plant-side torque polynomial까지 별도로 켜기
python view_live.py --config configs/cf21b_actuator_torque_poly_eval.yaml \
  --model model/ppo_best.zip --headless
```

두 번째 profile은 allocator의 legacy constant yaw ratio는 바꾸지 않고 plant torque만 polynomial으로 바꾼다. 따라서 controller–plant yaw mismatch가 의도적으로 발생할 수 있으며, nonlinear/QP allocator나 thrust-limit 재분배는 수행하지 않는다. `0.20 N/motor` 상한도 그대로다.

각 run의 resolved config와 manifest에는 nominal actuator 설정이, evaluation policy outcome에는 episode에서 실제 사용한 sampled `T/K`, motor command, desired/actual thrust, omega, reaction torque, commanded/achieved wrench summary가 남는다. 기존 floor/PPO 두 장의 16:9 plot은 유지되며 motor thrust panel은 MuJoCo에 실제 적용한 force(N)를 표시한다.

## 학습

먼저 runtime이나 산출물 디렉터리를 만들지 않는 dry run으로 설정을 확인할 수 있다.

```bash
python train_ppo_02.py --config configs/e2e_train.yaml --dry-run
python train_ppo.py --config configs/residual_train.yaml --dry-run
```

실제 학습:

```bash
python train_ppo_02.py --config configs/e2e_train.yaml
python train_ppo_02.py --config configs/residual_train.yaml
```

짧은 server smoke test에는 `--total-timesteps`를 사용할 수 있다.

```bash
python train_ppo_02.py --config configs/e2e_train.yaml --total-timesteps 4096
```

E2E best 조건은 기존과 동일하게 fixed-seed tail position error의 strict improvement와 tilt disqualification 0이다. Floor보다 좋아야 한다는 추가 조건은 없다. Residual smoke는 periodic callback이 없으므로 학습 후 checkpoint를 해당 run의 best와 final로 각각 기록한다.

## 평가

`view_live.py`는 Hover, Circle, Lissajous를 제공하는 통합 비행 테스트 진입점이다. 인자 없이 TTY에서 실행하면 Hover는 비행 시간만, Circle/Lissajous는 사전 정의 경로·한 바퀴(기본 주기)에 걸리는 시간·반복 횟수만 묻는다. 경로 geometry와 나머지 안전 설정은 versioned YAML profile에 고정되어 있다.

```bash
python view_live.py
```

```text
=== Crazyflie trajectory test ===
[1] Hover
[2] Circle trajectory
[3] Lissajous trajectory
Select mode [1-3]:
```

스크립트나 CI처럼 stdin이 TTY가 아닐 때는 `--mode`를 지정한다. 숫자 `1|2|3`과 이름 `hover|circle|lissajous`를 모두 허용한다. Circle path는 `small|wide`, Lissajous path는 `figure8|clover` 중 하나를 고른다.

```bash
python view_live.py --mode hover --duration 10 --headless

python view_live.py --mode circle \
  --path-preset small --period 10 --laps 2 --headless

python view_live.py --mode lissajous \
  --path-preset clover --period 10 --cycles 2 --headless
```

통합 `view_live.py`는 항상 floor와 PPO를 같은 mission/seed 조건에서 차례로 실행한다. floor는 `residual` 환경에서 zero policy action으로 cascade PID만 사용하고, PPO는 선택한 profile의 control mode를 사용한다. 기본 `view_live_*` profile은 E2E checkpoint용이므로 기본 비교는 PID floor 대 E2E PPO다. 명시적으로 residual profile을 넘기면 PPO도 residual(PID+RL) mode로 실행된다.

각 rollout은 별도의 16:9 PNG로 저장되어 한 번 실행할 때 `plots/floor.png`와 `plots/ppo.png` 두 장이 생성된다. 각 그림에는 position/reference, linear velocity, motor별·총 thrust(N), attitude, angular velocity, XY path overview, normalized control input `u`가 모두 들어간다. PWM calibration은 저장소에 없으므로 motor 값은 N만 제공한다. `--policy`는 legacy wrapper 호환을 위해 parser에 남지만 통합 진입점에서는 `both`로 고정된다.

Lissajous의 XY reference는 다음 식을 사용하며 `theta(t)`에는 Circle과 동일한 half-cosine 속도 ramp를 적용한다.

```text
x(t) = center_x + amplitude_x * sin(a * theta(t) + phase)
y(t) = center_y + amplitude_y * sin(b * theta(t))
z(t) = altitude
```

Circle/Lissajous의 GOTO endpoint는 실제 첫 reference에서 계산된다. 새 mission의 HOLD는 실제 마지막 reference를 유지하지만, 기존 `circle_traj.py`/`view_live_hover.py` profile은 호환성을 위해 기존 CIRCLE→HOLD jump를 보존한다. 고급 override(`--center`, `--radius`, amplitude/frequency/phase, `--seed`, perturbation, camera, phase timing 등)는 CLI 호환을 유지하지만 대화형 기본 질문에는 표시하지 않는다.

기존 명시적 hover 명령과 legacy circle wrapper도 계속 동작한다.

```bash
python view_live.py --config configs/e2e_hover_eval.yaml \
  --model model/ppo_best.zip --policy both

python view_live_hover.py --config configs/residual_circle_legacy006_eval.yaml \
  --preset 1 --model model/ppo_best.zip --policy both

python circle_traj.py --config configs/residual_circle_eval.yaml \
  --preset 1 --policy both --headless
```

공통 선택지는 `--model`, `--headless`, `--no-realtime`, `--no-camera`다. `--preset`은 기존 `circle_traj.py`/`view_live_hover.py`의 legacy circle 전용이고, 통합 진입점의 `--path-preset`과 다르다. 통합 `view_live.py`의 floor는 profile mode와 관계없이 항상 residual-mode cascade PID baseline이다. PPO rollout은 profile mode를 그대로 따르며, 실제 rollout별 control mode는 metrics와 manifest에 기록된다.

Legacy model은 자동으로 이름을 바꾸지 않는다. 기본 model은 `model/ppo_best.zip`이며 다른 checkpoint는 항상 `--model`로 명시한다. Shape `(15,) -> (4,)`만으로 residual/E2E provenance를 알 수 없으므로 사용자가 실제 학습 mode와 profile을 확인해야 한다. Legacy inventory와 hash는 [docs/LEGACY_ARTIFACTS.md](docs/LEGACY_ARTIFACTS.md)에 있다.

진단과 기존 learning curve도 같은 run naming을 사용한다.

```bash
python diag_entropy.py --config configs/residual_train.yaml
python diag_iterm_sat.py --config configs/residual_hover_eval.yaml \
  --model model/ppo_best.zip
python plot_curve.py --config configs/residual_train.yaml
```

기존 `tb/`에는 run별 control mode/condition/seed provenance가 없으므로 entropy 진단은 이를 residual이라고 추론하지 않는다. 생성된 metrics와 manifest의 input provenance는 `unverified`이며, top-level mode는 진단 실행 profile만 뜻한다.

## 산출물과 naming

한 번의 실행은 `Asia/Seoul`에서 timestamp 하나를 만들고 모든 파일에 재사용한다.

```text
artifacts/runs/
└── ppo_<mode>_<mission>_<condition>_seed<seed>_<YYYYMMDD-HHMMSS>/
    ├── models/
    ├── tensorboard/
    ├── plots/
    ├── metrics/
    ├── config/
    └── manifests/
```

예:

```text
ppo_residual_hover_m10g-r30mm-th0deg-fixed_seed42_best_20260807-153012.zip
ppo_residual_hover_m10g-r30mm-th0deg-fixed_seed42_final_20260807-153012.zip
ppo_residual_circle_rho0p5-T10-m5g-r100mm-th180deg-fixed_seed42_trajectory-residual_20260807-153012.png
ppo_e2e_hover_nominal_seed42_manifest_20260807-153012.json
ppo_e2e_hover_nominal_seed42_resolved-config_20260807-153012.yaml
plots/floor.png
plots/ppo.png
ppo_e2e_lissajous_l-clover-T10-C2_seed42_runtime-resolved_20260807-153012.yaml
```

같은 초에 같은 run이 생기거나 best가 여러 번 개선되면 `-01`, `-02` suffix로 이전 파일을 보존한다. `.zip.zip`은 만들지 않는다.

Run manifest에는 condition/timestamp/timezone, Git SHA/branch/dirty state, 실행 명령, mode와 observation/action shape, scale, payload, seed, PPO 값, 고정 XML 경로, resolved config, best/final 경로, Python/NumPy/MuJoCo/Gymnasium/SB3/PyTorch 버전이 들어간다. 통합 비행 테스트의 짧은 condition은 path key, period, laps/cycles만 파일명에 넣는다. 전체 geometry, phase timing, perturbation, floor-start 요청·적용 결과, control/motor 요약은 같은 timestamp의 `runtime-resolved.yaml`, metrics, manifest에 보존한다. 새 model은 이 provenance와 함께 해석해야 한다.

## 기존 명령 대응

| 기존 | 새 명령 |
|---|---|
| `python train_ppo.py` | `python train_ppo.py --config configs/residual_train.yaml` |
| `python train_ppo_02.py` | `python train_ppo_02.py --config configs/e2e_train.yaml` |
| `python view_live.py` | TTY 대화형 메뉴 또는 `python view_live.py --mode hover` |
| 기존 명시적 hover | `python view_live.py --config configs/e2e_hover_eval.yaml --model model/ppo_best.zip` |
| `python view_live_hover.py 1 both headless` | `python view_live_hover.py --config configs/residual_circle_legacy006_eval.yaml --preset 1 --policy both --headless` |
| `python circle_traj.py 1 both headless` | `python circle_traj.py --config configs/residual_circle_eval.yaml --preset 1 --policy both --headless` |

## 테스트

```bash
pytest -q
```

순수 함수·mock 기반 계약 테스트는 master의 정적 기준선과 비교한다. 실제 MuJoCo fixed-action trace, legacy checkpoint action, 짧은 PPO save/reload/headless smoke는 runtime 패키지와 project-local XML/STL asset이 모두 있는 환경에서만 실행되고, 없으면 명시적으로 skip된다.

실제 runtime dependency가 설치된 환경에서 추가로 실행할 smoke 검증:

```bash
test -f resources/mujoco/cf21B_500.xml
python train_ppo_02.py --config configs/e2e_train.yaml --total-timesteps 4096
python circle_traj.py --config configs/residual_circle_eval.yaml \
  --preset 1 --policy both --headless
python view_live.py --mode hover --duration 8 --headless
python view_live.py --mode circle --path-preset small \
  --period 10 --laps 1 --headless
python view_live.py --mode lissajous --path-preset figure8 \
  --period 10 --cycles 1 --headless
```

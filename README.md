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
  missions.py               기존 circle phase/reference
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

MuJoCo XML 기본값은 의도적으로 다음 서버 절대경로다.

```text
/home/mrl_6534/ros2_ws/src/mujoco_crazyflie/plant/data/cf21B_500.xml
```

코드는 다른 XML을 검색하거나, 저장소로 복사하거나, placeholder XML/mesh/texture를 만들지 않는다. XML과 종속 asset이 없는 머신에서는 config/artifact/unit test만 실행하고 MuJoCo 통합 검증은 skip한다.

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

```bash
python view_live.py --config configs/e2e_hover_eval.yaml \
  --model model/ppo_best.zip --policy both

python view_live_hover.py --config configs/residual_circle_legacy006_eval.yaml \
  --preset 1 --model model/ppo_best.zip --policy both

python circle_traj.py --config configs/residual_circle_eval.yaml \
  --preset 1 --policy both --headless
```

공통 선택지는 `--model`, `--headless`, `--policy floor|residual|both`, `--preset`, `--no-realtime`, `--no-camera`다. `view_live.py`의 E2E floor는 PID가 아니라 zero action + gravity compensation이다.

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
```

같은 초에 같은 run이 생기거나 best가 여러 번 개선되면 `-01`, `-02` suffix로 이전 파일을 보존한다. `.zip.zip`은 만들지 않는다.

Run manifest에는 condition/timestamp/timezone, Git SHA/branch/dirty state, 실행 명령, mode와 observation/action shape, scale, payload, seed, PPO 값, 고정 XML 경로, resolved config, best/final 경로, Python/NumPy/MuJoCo/Gymnasium/SB3/PyTorch 버전이 들어간다. 새 model은 이 manifest와 함께 해석해야 한다.

## 기존 명령 대응

| 기존 | 새 명령 |
|---|---|
| `python train_ppo.py` | `python train_ppo.py --config configs/residual_train.yaml` |
| `python train_ppo_02.py` | `python train_ppo_02.py --config configs/e2e_train.yaml` |
| `python view_live.py` | `python view_live.py --config configs/e2e_hover_eval.yaml --model model/ppo_best.zip` |
| `python view_live_hover.py 1 both headless` | `python view_live_hover.py --config configs/residual_circle_legacy006_eval.yaml --preset 1 --policy both --headless` |
| `python circle_traj.py 1 both headless` | `python circle_traj.py --config configs/residual_circle_eval.yaml --preset 1 --policy both --headless` |

## 테스트

```bash
pytest -q
```

순수 함수·mock 기반 계약 테스트는 master의 정적 기준선과 비교한다. 실제 MuJoCo fixed-action trace, legacy checkpoint action, 짧은 PPO save/reload/headless smoke는 runtime 패키지와 보호된 서버 XML 및 종속 asset이 있는 환경에서만 실행되고, 없으면 명시적으로 skip된다.

실제 server asset이 있는 환경에서 추가로 실행할 smoke 검증:

```bash
test -f /home/mrl_6534/ros2_ws/src/mujoco_crazyflie/plant/data/cf21B_500.xml
python train_ppo_02.py --config configs/e2e_train.yaml --total-timesteps 4096
python circle_traj.py --config configs/residual_circle_eval.yaml \
  --preset 1 --policy both --headless
```

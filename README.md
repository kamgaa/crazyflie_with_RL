# Crazyflie PPO control

MuJoCo Crazyflie 환경에서 cascade PID에 residual PPO 출력을 합성하거나, PPO가 전체 wrench를 내는 end-to-end(E2E) 제어를 학습·평가하는 프로젝트다.

현재 저장소에는 필수 MuJoCo XML과 종속 mesh·texture가 없다. 리소스를 제공하기 전에는 환경 생성, 학습, rollout을 실행할 수 없으며 가짜 XML이나 추측한 경로를 사용하지 않는다. 배치 위치와 검증 절차는 [`resources/README.md`](resources/README.md)를 따른다.

## Entrypoints

- `train_ppo_02.py`: 공식 주 학습 CLI 진입점
- `train_ppo.py`: 과거 smoke-test용 legacy 코드. 현재 설정이나 회귀 기준으로 사용하지 않는다.

기존 리팩터링 직전의 활성 기본 모드는 E2E였다. 그 사실은 보존하되, 이제는 실행할 때 반드시 residual 또는 E2E 설정 프로파일을 명시한다.

```bash
python train_ppo_02.py --config configs/residual_train.yaml
python train_ppo_02.py --config configs/e2e_train.yaml
```

모듈을 import하는 것만으로 환경·PPO 모델을 만들거나 평가·학습·저장·TensorBoard 기록을 시작해서는 안 된다.

## Configuration

공통 설정은 `configs/base.yaml`에 있고 각 프로파일은 이를 확장해 제어 모드와 observation 계약만 선택한다.

```text
configs/
├── base.yaml
├── residual_train.yaml  # residual / residual_v1 / 13
└── e2e_train.yaml       # e2e / e2e_v1 / 15
```

공통 리소스 경로는 다음 네 항목에서만 해석한다.

```yaml
paths:
  resource_root: resources
  mujoco_xml: resources/mujoco/cf21B_500.xml
  pretrained_model_root: resources/pretrained_models
  artifact_root: artifacts
```

현재 기본 action 물리 scale은 두 모드 모두 다음 4차원 벡터다.

```text
[tau_x, tau_y, tau_z, F_z] = [0.022, 0.022, 0.0001, 0.3]
```

`0.022`는 roll·pitch torque 두 축의 scale이다. 네 축 전체를 단일 scalar로 취급하거나 과거 실험값 `0.006`을 활성 기본값으로 되돌리지 않는다.

`base.yaml`에는 리팩터링 직전 `train_ppo_02.py`의 활성 학습 계약을 기록한다. 환경은 nominal bias 설정(`com_bias_randomize=false`, mass `0`, attitude perturbation `0°`, position perturbation `0.05 m`)이고, PPO는 `n_steps=2048`, `batch_size=256`, learning rate `3e-4`, `ent_coef=0.003`, `clip_range=0.1`, `target_kl=0.03`, `log_std_init=-1.5`, `[64, 64]` network를 사용한다. SB3 기본값 중 동작 계약에 포함되는 `gamma=0.99`, `gae_lambda=0.95`, `n_epochs=10`, `vf_coef=0.5`, `max_grad_norm=0.5`, advantage normalization도 명시되어 있다. 총 학습량은 1,000,000 timestep이다.

기존 활성 코드는 학습 seed를 명시하지 않았으므로 동작 보존을 위해 기본값은 `null`이다. 이 경우 run ID와 산출물 파일명에 `seed-unset`을 기록한다. 재현 가능한 새 실험에서는 비음수 정수 seed를 명시하며, resolved config, manifest, run ID와 개별 산출물 파일명에 그 값을 동일하게 기록한다.

## Observation contracts

두 모드의 첫 13개 원소는 순서와 의미가 완전히 같다.

```text
common_v1 = [
  position_error(3),
  linear_velocity_world(3),
  quaternion_wxyz(4),
  angular_velocity_body(3)
]
```

| Mode | Schema | Observation | Dimension |
|---|---|---|---:|
| residual | `residual_v1` | `common_v1` | 13 |
| e2e | `e2e_v1` | `common_v1 + [sin(yaw_error), cos(yaw_error)]` | 15 |

Action dimension은 항상 4다. 체크포인트를 읽을 때 모델의 observation/action shape와 manifest의 `control_mode`, `observation_schema`를 현재 프로파일과 비교해야 한다. 불일치 모델을 자동 변환하거나 억지로 실행하지 않는다.

새 모델 manifest의 최소 계약은 다음과 같다.

```json
{
  "control_mode": "residual",
  "observation_schema": "residual_v1",
  "observation_dim": 13,
  "action_dim": 4
}
```

## Evaluation contract

평가는 30 episode, 시작 seed 1000, episode 후반 30% 위치 오차, tilt 제한 30°, 20,000 timestep 간격을 사용한다. 학습·평가·callback의 다른 활성 동작은 `train_ppo.py`가 아니라 `train_ppo_02.py`를 기준으로 보존한다.

## Outputs and naming

정적 리소스와 새 실행 결과를 섞지 않는다. 새 실행은 다음 구조를 사용한다.

```text
artifacts/runs/<run-id>/
├── models/
├── tensorboard/
├── plots/
├── metrics/
├── config/
│   └── resolved.yaml
└── manifest.json
```

run ID는 `mode-<mode>__condition-<condition>__seed-<NNNN|unset>__date-<YYYYMMDD>__time-<HHMMSSZ>` 형식이다. 모델·그래프·metric 파일명에도 mode, 조건, seed와 생성 날짜 tag를 포함한다.

```text
ppo-best__mode-e2e__condition-nominal__seed-0000__date-20260806.zip
trajectory__mode-e2e__condition-nominal__seed-0000__date-20260806.png
evaluation__mode-e2e__condition-nominal__seed-0000__date-20260806.json
```

`resolved.yaml`과 `manifest.json`에는 실제 residual scale, 선택 모드/schema, observation/action 차원, seed, 생성 날짜와 해석된 경로를 기록한다.

## Legacy outputs

기존 `model/`, `tb/`, 루트 PNG는 자동 이동·삭제하지 않는다. 두 기존 모델은 내부 observation shape가 `(15,)`여서 새 residual 13차원 프로파일과 호환되지 않으며, 이 사실을 보존하는 legacy sidecar manifest가 함께 제공된다. 전체 이전 후보, 해시와 제약은 [`LEGACY_MIGRATION_INVENTORY.md`](LEGACY_MIGRATION_INVENTORY.md)에 기록되어 있다.

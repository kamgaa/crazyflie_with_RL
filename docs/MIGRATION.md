# Experiment pipeline migration

이 문서는 `master@6dd7ddb`의 고정 경로·전역 상수 기반 실행에서 config/run 기반 실행으로 옮기는 방법을 정리한다.

## 명령 대응

| 목적 | 이전 | 현재 |
|---|---|---|
| residual 30k smoke | `python train_ppo.py` | `python train_ppo.py --config configs/residual_train.yaml` |
| active E2E 1M | `python train_ppo_02.py` | `python train_ppo_02.py --config configs/e2e_train.yaml` |
| E2E hover viewer | `python view_live.py` | `python view_live.py --config configs/e2e_hover_eval.yaml --model model/ppo_best.zip` |
| legacy scale 0.006 circle | `python view_live_hover.py 1 both headless` | `python view_live_hover.py --config configs/residual_circle_legacy006_eval.yaml --preset 1 --policy both --headless` |
| payload circle | `python circle_traj.py 1 both headless` | `python circle_traj.py --config configs/residual_circle_eval.yaml --preset 1 --policy both --headless` |
| entropy diagnostic | `python diag_entropy.py` | `python diag_entropy.py --config configs/residual_train.yaml` |
| I-term diagnostic | `python diag_iterm_sat.py` | `python diag_iterm_sat.py --config configs/residual_hover_eval.yaml --model model/ppo_best.zip` |
| hand-recorded curve | `python plot_curve.py` | `python plot_curve.py --config configs/residual_train.yaml` |

기존 bare positional token `headless`, `norealtime`, `nocam`, `floor|residual|both`는 각각 `--headless`, `--no-realtime`, `--no-camera`, `--policy`로 바뀌었다.

## Profile 선택

- 기존 `train_ppo_02.py` 활성값은 `e2e_train.yaml`에 있다.
- 기존 `train_ppo.py`의 30k residual 값은 `residual_train.yaml`에 별도로 있다. E2E 설정을 상속해 임의 통일하지 않았다.
- scale 0.006을 사용하던 `view_live_hover.py`와 `diag_iterm_sat.py`도 전용 profile로 분리했다.
- 원 궤적 preset은 config에 있고 실행 시 `--preset 1|2|3`으로 선택한다.

실행 전 resolved 값을 확인하려면 다음처럼 dry run을 사용한다.

```bash
python train_ppo_02.py --config configs/e2e_train.yaml --dry-run
```

## 현재 필수 BLDC plant

모든 shipped profile은 `actuator.enabled: true`,
`actuator.model: cf21b_first_order`를 상속한다. 설정 검증도 actuator를
끄거나 instantaneous model을 선택하는 것을 허용하지 않는다. 기본 plant는
기존 allocator가 만든 `f_cmd`를 즉시 MuJoCo에 넣지 않고, 매 500 Hz physics
substep에서 목표 회전수와 1차 모터 상태를 거쳐 `f_actual`로 적용한다. 기본
reaction torque는 여전히 `direction * 0.00594 * f_actual`이며, paper torque
polynomial은 `cf21b_actuator_torque_poly_eval.yaml`에서만 명시적으로 선택한다.

`master@6dd7ddb` 기준선의 force/torque 즉시 적용 경로는 역사적 비교용으로만
남는다. PID, allocation, observation, action, reward는 그대로여도 motor lag로
plant state transition이 달라지므로 legacy checkpoint는 shape가 같아도 현재
plant에서 행동적으로 동등하거나 재현 가능하다고 볼 수 없다. 기존 ZIP에는
actuator provenance가 없으므로, 이를 사용하는 run은 resolved config와
manifest의 actuator 항목을 함께 보존하고 provenance를 `unverified`로 취급한다.
BLDC mapping 계수도 현재 XML/실기체에서 검증된 calibration이 아니라
paper-candidate 값이다.

## Model과 output

이전에는 `model/ppo_best.zip`, `model/ppo_residual_cf.zip`, root PNG, `tb/PPO_N`이 같은 이름을 재사용했다. 현재는 실행마다 다음 run directory를 만든다.

```text
artifacts/runs/ppo_<mode>_<mission>_<condition>_seed<seed>_<timestamp>/
```

Best/final model과 plot/metrics/config/manifest는 이 directory 아래에만 기록된다. 기존 `model/`, `tb/`, root PNG는 자동 이동·삭제·rename하지 않는다. 이전 checkpoint를 쓸 때는 `--model model/ppo_best.zip`처럼 명시한다.

새 checkpoint를 다른 실행에서 사용할 때는 model 단독이 아니라 같은 run의 manifest와 resolved config를 함께 보존한다. Manifest의 `control_mode`, shapes, scale, payload, actuator, seed, XML, dependency versions를 소비 profile과 비교해야 한다.

## 호환성 경계

- residual과 E2E observation은 모두 15차원으로 유지했다.
- XML은 project-local `resources/mujoco/cf21B_500.xml`을 명시적으로 사용하며 자동 탐색이나 fallback이 없다.
- PID/allocation/reward/randomization/termination/circle 식은 기준선 그대로다.
- historical instantaneous plant와 현재 required BLDC plant는 별개의 동역학 계약이다. 같은 shape의 legacy checkpoint를 동등한 행동 기준선으로 취급하지 않는다.
- 새 naming과 manifest는 output lifecycle 변경이며 학습 수식 변경이 아니다.
- 13차원 residual observation은 이번 migration 대상이 아니다. 필요하면 새 schema/checkpoint로 별도 도입한다.

의도적으로 수정하지 않은 legacy 문제는 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md), 정확한 수식은 [BEHAVIOR_BASELINE.md](BEHAVIOR_BASELINE.md)를 참고한다.

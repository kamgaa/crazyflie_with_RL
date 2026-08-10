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

## Model과 output

이전에는 `model/ppo_best.zip`, `model/ppo_residual_cf.zip`, root PNG, `tb/PPO_N`이 같은 이름을 재사용했다. 현재는 실행마다 다음 run directory를 만든다.

```text
artifacts/runs/ppo_<mode>_<mission>_<condition>_seed<seed>_<timestamp>/
```

Best/final model과 plot/metrics/config/manifest는 이 directory 아래에만 기록된다. 기존 `model/`, `tb/`, root PNG는 자동 이동·삭제·rename하지 않는다. 이전 checkpoint를 쓸 때는 `--model model/ppo_best.zip`처럼 명시한다.

새 checkpoint를 다른 실행에서 사용할 때는 model 단독이 아니라 같은 run의 manifest와 resolved config를 함께 보존한다. Manifest의 `control_mode`, shapes, scale, payload, seed, XML, dependency versions를 소비 profile과 비교해야 한다.

## 호환성 경계

- residual과 E2E observation은 모두 15차원으로 유지했다.
- XML은 기존 서버 절대경로를 유지하며 local fallback이 없다.
- PID/allocation/reward/randomization/termination/circle 식은 기준선 그대로다.
- 새 naming과 manifest는 output lifecycle 변경이며 학습 수식 변경이 아니다.
- 13차원 residual observation은 이번 migration 대상이 아니다. 필요하면 새 schema/checkpoint로 별도 도입한다.

의도적으로 수정하지 않은 legacy 문제는 [KNOWN_ISSUES.md](../KNOWN_ISSUES.md), 정확한 수식은 [BEHAVIOR_BASELINE.md](BEHAVIOR_BASELINE.md)를 참고한다.

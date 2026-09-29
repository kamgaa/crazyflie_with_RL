# E2E PPO 위치 비용 xy/z 분리

> 아래 설정값과 검증 결과는 2026-09-24 구현 당시 기록이다. 이후 비교 YAML이 튜닝되었으며, 2026-09-29 업로드 시점의 설정과 테스트 상태는 [업데이트 기록](GITHUB_UPDATE_20260929.md)을 참고한다.

현재 체크아웃의 `train_ppo_02.py` 기본값인 `configs/e2e_train.yaml`을 기준으로 구현했다. 작업 전 수정된 `base.yaml`의 위치 가중치는 4.0이다. 별도 실험용 `e2e_train_pose_dr_10cm_30deg_scale006.yaml`은 기본 진입점이 아니며, 그 설정값은 변경하지 않았다. 체크아웃에서 AGENTS.md는 발견되지 않았다. 기존 미커밋 변경 사항은 보존했다.

## 수식과 호환성

기존 위치 비용은 `position_weight * (ex² + ey² + ez²)`다. 오차 노름이 아니라 제곱합이며 위치 비용에 정규화·클리핑은 없다.

새 비용은 다음과 같다.

```text
position_cost_xy = position_xy_weight * (ex² + ey²)
position_cost_z  = position_z_weight * ez²
position_cost   = position_cost_xy + position_cost_z
reward = -(position_cost + 기존 velocity/tilt/angular_velocity/yaw/action/action_rate 비용)
         - (충돌한 경우 crash_penalty)
```

수평 항을 2로 나누지 않는다. 두 축 가중치가 같으면 수학적으로 같은 `weight * (e @ e)`를 사용해 기존 부동소수점 연산 순서와 총보상까지 정확히 보존한다. 분리 비용 합과 전체 위치 비용 사이에는 일반적인 부동소수점 반올림 차이만 가능하다. 기존 위치 비용을 별도로 중복 가산하지 않는다.

E2E에서 action/action_rate 비용을 0으로 만드는 기존 동작, residual의 해당 비용, 관측 15차원·행동 4차원·allocation·종료 조건은 유지한다.

## 설정과 파일

- `crazyflie_rl/config.py`: 선택적 축 가중치, 유한한 0 이상 숫자 검증, 실효 가중치 조회 및 resolved YAML/JSON 저장.
- `crazyflie_rl/environment.py`: 실제 비용 계산 및 info 진단.
- `crazyflie_rl/training.py`: rollout 평균 TensorBoard 태그, 평가 JSON 필드, 시작 시 실효 가중치/action scale 출력.
- `crazyflie_rl/evaluation.py`: 고정 시드 평가의 전체/tail RMSE. 기존 score·실격·에피소드 길이 유지.
- `crazyflie_rl/eval_cli.py`: 전체/tail/phase/trajectory RMSE, 시작 로그 및 runtime-resolved 기록.
- `crazyflie_rl/reward_balance.py`: 평가 후 비용 재구성도 분리 가중치를 사용. 전체·phase 진단 및 콘솔 표 추가.
- `configs/e2e_train_position_split_equal.yaml`: 기본 E2E 설정 상속, xy=4.0/z=4.0만 명시. 다른 설정은 전부 상속.
- `tests/test_position_split.py`: 신규 비학습 회귀 검증.
- `tests/test_reward_diagnostics.py`, `tests/test_training_contracts.py`: 진단 태그 및 로그에 필요한 테스트 설정 갱신.
- `docs/validation/position_split_checkpoint.json`: 실제 체크포인트 비교 결과와 SHA256.

YAML은 부모 `extends`를 먼저 읽고 자식 값을 재귀 병합한다. 병합 후 각 축에 명시된 값이 있으면 그 값(0 포함)을 사용한다. 없는 축만 최종 `position_weight`로 대체한다. 부모에 명시된 축 값도 상속된 명시값이므로 자식에서 legacy 가중치만 바꿔도 유지된다. 명시적 null, 음수, NaN, inf, bool, 문자열은 거부한다. Python 설정 객체에서는 None이 생략을 나타내며 `effective_position_*_weight`가 대체값을 반환한다. 최종 설정 파일에는 None 대신 실제 숫자를 저장한다.

현재 기본/비교 설정은 모두 xy=4.0, z=4.0이다. reward weight 전용 CLI 옵션은 없으며 YAML로 설정한다. 기존 CLI 실행 옵션은 로딩 후 적용된다.

## 로그와 집계 구간

기존 `reward_terms/position`(음수), `reward_terms/total`, `reward_raw/position_sq`, `reward_fraction/*`를 유지한다. 새 TensorBoard 태그는 다음과 같다.

| 태그 | 정의 |
| --- | --- |
| `reward_raw/position_sq_xy` | ex² + ey², m² |
| `reward_raw/position_sq_z` | ez², m² |
| `reward_costs/position_xy` | 양의 가중 수평 비용 |
| `reward_costs/position_z` | 양의 가중 수직 비용 |

기존 callback과 동일하게 rollout 동안 받은 info 표본의 산술평균을 기록한다. 비용 하위 항은 reward fraction 분모에 추가하지 않는다. 매 스텝 콘솔 출력은 없다.

`position_rmse_xy = sqrt(mean(ex² + ey²))`, `position_rmse_z = sqrt(mean(ez²))`이며 단위는 m다. xy는 두 축을 합친 수평 거리 기준이며, 두 축으로 나눈 평균이나 축별 RMSE 평균이 아니다.

- `view_live`: 전체 실측 표본, 기존 tail, 기존 모든 phase(TAKEOFF/SETTLE/GOTO/CIRCLE/LISSAJOUS/HOVER/HOLD 등 실제 존재하는 phase)에 적용한다. 전체 `position_rmse = sqrt(mean(ex² + ey² + ez²))`, phase별 기존 `rmse`, 기존 `trajectory_phase_rmse`는 유지한다. 새 키는 `position_rmse_xy/z`, `tail_position_rmse_xy/z`, phase 내부 `position_rmse_xy/z`, `trajectory_phase_rmse_xy/z`다. 위치와 해당 표본의 reference 차이를 사용한다.
- 학습 중 `PolicyEvaluator`: 전체 에피소드 표본과 각 에피소드의 기존 후반 `tail_fraction` 표본을 각각 모아서 RMSE를 구한다. E2E tail은 `max(1, floor(length * fraction))`, residual은 기존 `[-int(...):]` 동작을 그대로 쓴다. 기존 score는 **에피소드별 tail 평균 거리의 평균**이며 RMSE로 교체하지 않는다. 학습 평가 JSON은 `policy_`/`floor_` 접두사를 붙인다.
- 실패·실격·조기 종료·truncation·경계 이탈 정보를 그대로 남긴다. 조기 종료된 에피소드의 실제 관측 표본도 포함한다. 임의의 정착 시간을 만들지 않는다. 빈 trace의 새 RMSE는 null이다.

`reward_balance`는 float32 관측에서 재구성한 연속 상태 비용이며 action/action_rate/crash를 포함한 총 episode return이 아니다. 새 `position_sq_xy_mean`, `position_sq_z_mean`, `position_cost_xy_mean`, `position_cost_z_mean`을 전체/phase별로 제공한다.

## Action scale 확인

실제 환경에서 `residual_scale * clip(action, -1, 1)`을 적용하며 순서는 `[tau_x, tau_y, tau_z, delta_fz]`다. 현재 기본 학습·hover/circle 평가·새 비교 설정은 모두 `[0.0075, 0.0075, 0.001, 0.5]`이고, 평가 코드가 별도로 덮어쓰지 않는다.

별도 이력 차이: 비교에 쓴 2026-09-22 13:43:06 run의 저장된 학습 scale은 `[0.0065, 0.0065, 0.0001, 0.35]`다. 따라서 현재 설정으로 그 체크포인트를 평가하면 당시 학습 scale과 다르다. 이번 작업에서는 이 값을 조정하지 않았다. 아래 회귀 비교는 세 환경 모두 **현재 scale**로 통일한 보상 변경 동등성 검증이며, 과거 학습 조건의 재현 실험은 아니다. `model/ppo_best.zip`은 이 run과 같은 파일이라고 가정하지 않는다.

## 비학습 검증

- 최종 집중 회귀 테스트 **62개 통과**. 확장 실행은 **147개 통과 / 아래 기존 설정 불일치 4개 실패**였다(이후 상속 우선순위 테스트 1개 추가 포함한 집중 실행 완료). 체크포인트 로더의 NumPy deprecated namespace 경고 4건이 있었다.
- legacy 설정과 명시적 동일 가중치의 위치 비용/총보상 보존, xy/z 개별 변경, 한쪽 생략, 명시적 0, 상속 우선순위 및 잘못된 값 검증.
- 기존 pre-diagnostics 환경에 대한 E2E/residual × float32/float64 1,024-step 동등성 검사와 충돌 패널티 검사 통과.
- 실제 TensorBoard 이벤트를 생성해 rollout 평균, 새 태그, 기존 fraction 합, RNG 불변 검증. 학습은 실행하지 않는다.
- 알려진 오차 `[3,4,2], [0,0,-4], [6,8,0]`: 전체 xy RMSE=`sqrt(125/3)`, z RMSE=`sqrt(20/3)`, 기존 3D RMSE=`sqrt(145/3)`. phase/tail, 실패 정보, 빈 trace도 검증.
- `model/ppo_best.zip`을 기존 15/4 차원 환경에 로드하여 200스텝 동일 행동·궤적·보상 검증 및 파일 SHA256 불변 확인.
- 추가로 9월 22일 run의 final checkpoint와 seed 42로 **작업 전 환경 소스 / 변경 후 legacy 설정 / 변경 후 명시적 동일 가중치 설정**을 200스텝 비교했다. 행동·관측·보상·종료 플래그·qpos·qvel·ctrl 모두 비트 단위 일치, 조기 종료 없음. xy RMSE 0.0326441251 m, z RMSE 0.0160606500 m, 3D RMSE 0.0363810855 m. 체크포인트 해시 불변.
- 실제 `view_live.py` 2초 headless 평가도 성공. PPO/Floor 각각 200표본, HOVER 지표·runtime/resolved 가중치·action scale·종료 정보 저장을 확인했다. 정상 길이 truncation은 1.99초 표본에 기록되었다.
- 관련 확장 테스트 실행에서 기존 설정 기대값 불일치 4건 확인: `test_config.py` 2건은 기존 action scale, `test_view_live_cli.py` 2건은 기존 circle 10초 기대값과 현재 5초 설정의 차이. 이번 작업에서 해당 설정이나 기대값을 수정하지 않았다.
- PPO 학습, optimizer update, 기존 체크포인트 수정은 수행하지 않았다. 새 정책의 학습 성능 비교와 과거 action scale 재현 평가는 미수행이다.

검증 명령(ROS pytest 플러그인의 미설치 lark 의존성과 분리):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_position_split.py tests/test_reward_diagnostics.py \
  tests/test_reward_balance.py tests/test_training_contracts.py
```

## 실행 명령

저장소 루트 및 기존 crazyflie_rl Python 환경에서 실행한다. 아래 학습 명령은 이번 작업에서는 실행하지 않았다.

설정 확인과 새 정책 학습:

```bash
python train_ppo_02.py --config configs/e2e_train_position_split_equal.yaml --dry-run
python train_ppo_02.py --config configs/e2e_train_position_split_equal.yaml
```

**기존 체크포인트** 평가(실제로 검증한 명령):

```bash
python view_live.py --config configs/e2e_train_position_split_equal.yaml \
  --mode hover \
  --model artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260922-134306/models/ppo_e2e_hover_nominal_seedunset_final_20260922-134306.zip \
  --seed 42 --duration 2 --headless --no-realtime --no-camera
```

이는 기존 정책의 평가이며 보상 설정을 바꿨다고 기존 정책이 새로 학습되는 것은 아니다. 초기 상태 분포는 비교 설정 그대로 사용한다.

**위 새 설정으로 학습을 완료한 직후**, 그 학습이 가장 최근 E2E best checkpoint를 생성했다면:

```bash
python view_live.py --config configs/e2e_train_position_split_equal.yaml \
  --mode hover --model latest-best \
  --seed 42 --duration 8 --headless --no-realtime --no-camera
```

`latest-best`는 artifact manifest에서 E2E 모드의 최신 best를 선택하며 보상 가중치로 필터링하지 않는다. 다른 학습을 뒤이어 실행했다면 `--model`에 원하는 새 run의 실제 zip 경로를 지정한다. 이 작업에서는 새 학습을 하지 않았으므로 현재 `latest-best`는 기존 정책을 가리킨다.

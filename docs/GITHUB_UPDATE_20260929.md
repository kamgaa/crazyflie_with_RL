# 2026-09-29 GitHub 업데이트 기록

현재 작업 트리의 Python 코드, YAML 설정, 테스트와 관련 문서를 기존
`refactor-experiment-pipeline-v2` 브랜치에 반영하는 스냅샷이다.
업로드 과정에서 보상 가중치나 action scale 등 실험 설정값을 조정하지 않았다.

## 포함 내용

- 위치 비용 xy/z 분리, 축별 가중치 하위 호환, rollout 진단 및 평가 RMSE.
- 초기 위치·자세 랜덤화, 초기 자세 진단 CLI, 위치만 랜덤화하는 학습 설정.
- PID 모터 성능 저하 진단, 정착 hover 및 외란 pulse/staircase 실험 도구.
- 보상 균형, yaw 및 wrench authority 평가 리포트와 관련 테스트.
- 현재 base/hover/circle 설정 변경, 구현 설명 및 이전 비학습 검증 기록.

자동 생성된 `artifacts/view_live/`, Windows `.lnk`, `:Zone.Identifier`는
업로드 대상에서 제외하고 `.gitignore`에 추가했다. 기존 체크포인트와
TensorBoard 결과는 수정하거나 새로 업로드하지 않았다.
로컬 `artifacts/runs/.gitkeep` 삭제도 코드 업데이트에 포함하지 않았다.

## 현재 설정과 이전 기록의 차이

- `base.yaml`: position_weight=4.0, velocity_weight=0.005,
  action scale=[0.0075, 0.0075, 0.001, 0.5].
- `e2e_train_position_split_equal.yaml`: 파일명·주석과 달리 현재는
  xy=10.0, z=6.0이며, 초기 위치 랜덤화 max_norm_m=0.15,
  자세 랜덤화 disabled다. 따라서 현재 파일은 동등 가중치 비교 설정이 아니다.
- `e2e_train_position_only_dr.yaml`: 위 파일을 상속하므로 해당 가중치와
  0.15m 위치 범위를 사용한다.
- `e2e_train_pose_dr_10cm_30deg_scale006.yaml`: 이름과 달리 현재 값은
  위치 0.05m, 자세 5도, position_weight=3.0,
  action scale=[0.022, 0.022, 0.0001, 0.5]다.
- `view_live_circle_eval.yaml`: 원 궤적 주기는 5초다.

`POSITION_REWARD_SPLIT.md`와 `validation/position_split_checkpoint.json`의
동등 가중치 검증 결과는 9월 24일의 설정에 대한 기록이며 현재 튜닝 설정의
동등성을 의미하지 않는다.

## 업로드 전 검증

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/crazyflie-publish-mpl \
  python -m pytest -q --ignore=tests/test_runtime_smoke.py
```

결과: **334 passed, 16 failed**, NumPy 체크포인트 역직렬화 deprecation
warning 4건. 실제 PPO 학습과 optimizer update를 수행하는 runtime smoke는
제외했다. 학습 계약 테스트는 fake model을 사용한다.

실패 분류:

- `test_config.py` 2개: 과거 action scale 기대값과 현재 base 설정의 차이.
- `test_environment_contracts.py` 7개: 생성자를 우회한 테스트 환경 객체에
  `position_xy_weight`/`position_z_weight` 초기화가 없어 AttributeError 발생.
- `test_initial_pose.py` 2개: 과거 0.1m 범위/동일 보상 기대값과 현재 pose 설정의 차이.
- `test_mission_config.py` 1개 및 `test_view_live_cli.py` 2개: 과거 10초
  circle 주기 기대값과 현재 5초 설정의 차이.
- `test_position_split.py` 2개: 동일 가중치/동일 초기 상태를 가정하는 테스트가
  현재 xy=10/z=6 및 위치 랜덤화 설정과 불일치.

이 업데이트는 위 실패를 해결한 릴리스가 아니다. 현재 사용자 작업 상태를
보존해 업로드하며, 실패를 숨기기 위해 설정이나 assertion을 바꾸지 않았다.
`git diff --check`도 확인했다.

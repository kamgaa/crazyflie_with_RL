# Known issues intentionally preserved

다음 항목은 발견했지만 학습 결과·기존 모델 호환성·viewer trajectory를 바꿀 수 있어 이번 리팩터링에서 고치지 않았다.

1. `master`의 환경 주석은 residual observation을 13차원이라고 설명하지만 실행 코드는 yaw sine/cosine을 항상 붙여 두 mode 모두 15차원이다.
2. legacy `ppo_best.zip`과 `ppo_residual_cf.zip`에는 신뢰할 수 있는 control-mode provenance가 없다. 둘 다 shape `(15,) -> (4,)`이므로 shape만으로 residual/E2E를 구분할 수 없다.
3. E2E에서 zero action은 PID floor가 아니라 base-mass gravity compensation뿐이다. 과거 출력의 `PID-only` 또는 `residual` 라벨은 실제 의미와 다를 수 있다.
4. PID의 `kd_velocity`와 `kd_rate`는 선언되어 있지만 현재 계산식에서 사용되지 않는다. Rate integral은 clip되지 않지만 활성 `ki_rate`가 0이라 출력에 영향이 없다.
5. `round(physics_hz/policy_hz)` 방식 때문에 두 frequency가 나누어떨어지지 않으면 실제 policy interval이 양자화된다.
6. Circle ramp 누적 phase 때문에 CIRCLE 종료점과 HOLD target `(1,0,1)` 사이에 reference jump가 생길 수 있다.
7. Circle script는 바닥 위치를 강제로 설정한 직후 observation을 다시 만들지 않아 첫 action이 reset 직후의 stale observation을 사용한다. 이동 reference도 policy observation에 한 step 늦게 반영된다.
8. Legacy circle headless loop는 termination/truncation을 무시하고 NaN에서만 멈췄고 viewer loop는 termination을 기록만 했다. 이 stress-test 의미를 바꾸지 않는다.
9. Payload inertia는 대각 성분만 갱신하며 product-of-inertia는 반영하지 않는다.
10. E2E best selection은 floor 개선이나 survival length를 요구하지 않고, 1,000,000 callback 뒤 rollout 반올림으로 만들어진 final checkpoint를 다시 평가하지 않는다.
11. 기존 `tb/` event 파일에는 control mode, condition, seed를 신뢰할 수 있게 연결하는 metadata가 없다. `diag_entropy.py`는 이를 추론하지 않고 새 metrics/manifest의 input provenance를 `unverified`로 기록한다. 이때 manifest의 top-level `control_mode`는 진단 실행에 사용한 config profile이지 legacy event의 출처를 증명하지 않는다.
12. Residual evaluator에서 `tail_fraction=0`이면 legacy slice `[-int(length * fraction):]`가 `[-0:]`가 되어 한 sample이 아니라 episode 전체를 평가한다. 활성 profile은 0.3이지만 경계 동작도 회귀 호환성을 위해 유지했다. E2E evaluator는 기존처럼 최소 한 sample을 보장한다.

Residual 13차원 observation이 필요하다면 별도의 schema 이름, 새 profile, 새 checkpoint로 분리해야 한다. 기존 15차원 모델의 입력 계약을 변경해서는 안 된다.

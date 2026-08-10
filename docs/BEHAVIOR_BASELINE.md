# Master 동작 기준선

리팩터링 기준 커밋은 `master@6dd7ddb667e7483bb6dce6c81edf783ca0ffe133`이다. 이 문서는 주석이 아니라 해당 커밋에서 실제 실행되는 코드를 정적 분석한 결과다.

## 환경·제어 계약

- MuJoCo XML 기본 경로: `/home/mrl_6534/ros2_ws/src/mujoco_crazyflie/plant/data/cf21B_500.xml`
- physics/policy rate: 500 Hz / 100 Hz, 기본 5 substep
- episode: 8 s, 800 policy step
- action: `(4,)`, `[d_tau_x, d_tau_y, d_tau_z, d_Fz]`, 입력을 `[-1, 1]`로 clip한 뒤 scale 적용
- 기본 scale: `[0.022, 0.022, 0.0001, 0.3]`
- residual: `u_pid + scaled_action`
- E2E: `scaled_action + [0, 0, 0, 0.04338 * 9.81]`

상단 주석의 13차원 설명과 달리 residual/E2E 모두 실제 observation은 다음 15차원이다.

| Slice | 값 |
|---|---|
| `0:3` | `pos - pos_des` |
| `3:6` | world linear velocity |
| `6:10` | normalized quaternion `wxyz`, `qw >= 0` |
| `10:13` | body angular velocity (`imu_gyro`) |
| `13` | `sin(wrapped yaw error)` |
| `14` | `cos(wrapped yaw error)` |

PID 순서와 활성값:

1. position P (`kp=4`) → velocity target, norm limit `1.5 m/s`
2. velocity PI (`kp=4`, `ki=1`) → acceleration; integral component-wise clip `[-2,2]`
3. base mass와 gravity로 desired force 계산, body-z projection, force clip `[0,1] N`
4. desired attitude, tilt limit `35°`, attitude P `12`
5. rate PI (`kp=[0.0008,0.0008,0.0006]`, `ki=0`)와 gyroscopic feed-forward
6. body torque component-wise clip `[-0.02,0.02] N·m`

선언된 velocity/rate D gain은 모두 0이고 실제 식에서도 사용되지 않는다. Allocation은 motor 순서와 함께 그대로 유지한다.

```text
B rows = [tau_x, tau_y, tau_z, Fz]
motor directions = [+1, -1, +1, -1]
f = pinv(B) @ wrench
f_i = clip(f_i, 0, 0.20)
motor torque_i = direction_i * 0.00594 * f_i
```

## Reward와 종료

```text
cost = 3.0 * ||position error||²
     + 0.01 * ||world velocity||²
     + 3.0 * 2(qx² + qy²)
     + 0.001 * ||body angular velocity||²
     + 1.0 * wrapped yaw error²
     + mode action penalty
     + mode action-rate penalty
reward = -cost
```

- residual action penalty `0.001`; E2E는 0
- action-rate weight 기본 0; E2E는 항상 0
- crash 시 추가 `-10`
- terminated: `z < 0.2`, `z > 2.5`, tilt `> 60°`, 또는 position error `> 1.5 m`
- truncated: policy step 수가 `episode_sec * policy_hz`에 도달

## Reset과 payload

- explicit reset seed는 NumPy Generator를 재생성하며 같은 설정/seed에서 draw 순서가 동일하다.
- 위치는 목표 주위 각 축 uniform perturbation이다.
- 자세는 yaw를 제외한 임의 roll/pitch 축과 `[0, attitude_perturbation]` 각도로 만든 `wxyz` quaternion이다.
- 고정 payload는 body mass, `body_ipos`, 대각 inertia를 갱신한다.
- 편심 중력 torque는 매 physics substep `xfrc_applied`로 명시 주입한다.
- randomized payload는 `r~U(0.02,0.10)`, angle `U(0,2π)`, target torque `U(0,0.5*TAU_MAX_RP)`, mass upper bound 15 g를 사용한다.

## 학습 진입점

| 항목 | `train_ppo.py` | `train_ppo_02.py` |
|---|---:|---:|
| mode | residual | E2E |
| perturbation | position 0.15 m, attitude 5° | position 0.05 m, attitude 0° |
| payload | fixed 10 g at center | 0 g |
| timesteps | 30,000 | 1,000,000 |
| PPO entropy | 0 | 0.003 |
| clip | SB3 default 0.2 | 0.1 |
| target KL | `None` | 0.03 |
| log std | -2.0 | -1.5 |
| evaluation | 5 ep, seeds 100–104 | 30 ep, seeds 1000–1029 |

공통 PPO 값은 `n_steps=2048`, `batch_size=256`, learning rate `3e-4`, gamma `0.99`, GAE `0.95`, 10 epochs, network `[64,64]`이다.

E2E callback은 20,000 step마다 평가한다. best 조건은 `mean tail error < previous best`이고 tail tilt disqualification이 0인 경우다. PID/open-loop floor보다 좋아야 한다는 조건이나 survival gate는 없다.

## 평가·진단 진입점

- `view_live.py`: 실제 E2E hover, seed 42, 0 g, perturbation 0. `floor` 라벨은 실제로 zero-action + gravity compensation이다.
- `view_live_hover.py`: 파일명과 달리 residual circle, scale `[0.006,0.006,0.0001,0.3]`, 0 g.
- `circle_traj.py`: residual circle, scale `[0.022,0.022,0.0001,0.3]`, 활성 조건 5 g / radius 100 mm / angle 180°.
- `diag_iterm_sat.py`: residual, scale 0.006 계열, 30 g / radius 30 mm / angle 0°, 단일 8 s rollout, 정상상태 마지막 2 s.
- `diag_entropy.py`: TensorBoard scalar plot, rollout 없음.
- `plot_curve.py`: 20k–300k metric 15개가 소스에 내장되어 있으며 원래 import만으로 PNG를 생성했다.

Circle phase는 TAKEOFF 4 s, SETTLE1 2 s, GOTO 4 s, SETTLE2 2 s, angular-speed ramp 2 s, constant circle `2*period`, HOLD 2 s다. 원 중심은 `(goto_x-radius, goto_y)`이며 ramp phase 적분식도 원본과 동일하게 보존한다.

## 기존 refactor 브랜치에서 제외한 변경

`origin/refactor@4b13da7`은 참고만 했다. 다음 변경은 현재 계약과 충돌하므로 가져오지 않았다.

- residual observation을 15 → 13차원으로 변경
- XML 기본값을 `resources/mujoco/cf21B_500.xml`로 변경
- `.006` scale을 사용하는 legacy 평가/진단까지 `.022`로 통일
- observation shape만으로 legacy model을 E2E라고 단정
- residual 30k smoke profile을 E2E 1M hyperparameter로 대체

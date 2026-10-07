# A/B best payload·모터 효율 비교

`compare_ab_payload_motor.py`는 `completion.json`에 명시된 A best(980000),
B best(960000)를 SHA256 및 checkpoint 내부 timestep까지 확인해 사용하는 비학습 평가다.
두 모델 모두 native **actual world velocity** 관측 15차원과 wrench 행동 4차원을 유지한다.
B의 학습 reward 차이를 관측 차이로 해석하지 않는다. 기존 frozen loader/normalization 경로를 재사용한다.

## 공통 초기화와 물리

- 20초, 100Hz policy / 500Hz physics, seed42, deterministic inference.
- 초기 p=target=[0,0,1]m, level quaternion wxyz=[1,0,0,0], v/omega=0, yaw target=0.
- 기존 EvaluationAdapter가 정상 reset하여 payload 및 COM/관성을 적용하고 episode 내부 상태를 초기화한다.
- 기존 일반 actuator reset은 **payload 포함 질량**으로 평형을 잡으므로, 이번 평가 observer는 첫 추론 전에
  기존 `actuator_model.reset(airborne=True, episode_mass=env.mass, randomize=False)`를 호출해
  **nominal 질량 기준** 모터 상태로 통일한다. 물리 시간은 진행하지 않는다. payload를 trim하지 않는다.
- payload는 조건별 0/5g 및 body frame xy=[0,0]/[.03,0]m로 reset 시부터 적용한다.
  new initial-pose DR와 legacy perturbation, payload randomization, actuator parameter randomization을 끈다.
  고정 payload 값은 유지한다. 기체 물리/allocator/action scale/nominal hover bias/종료 한계는 변경하지 않는다.
- 현재 `_set_com_bias`의 합성 mass/COM/full inertia + `mj_setConst` 경로를 사용한다.
  rotor site 힘의 모멘트가 엔진 COM 기준으로 반영되며, 명시적 payload 중력 토크는 재추가하지 않는다.
  이번 실험의 별도 외란 토크는 0이다.
- 효율은 기존 InteractiveEnv → `apply_motor_effectiveness` 경로에서 actuator 출력에 한 번 적용한다.
  `f_actual=lambda*f_nominal`, `q_actual=lambda*q_nominal`. RPM 상태나 allocator를 보정하지 않는다.
- motor 1은 allocator index0이다. 실제 rotor site·회전 방향은 manifest의 `physical.rotors`에 기록한다.

## 기록과 시각

기존 `dr_transfer.run_case`에 선택적 observer/env_factory 인자를 추가했다. 기본 경로는 그대로다.
정책 로딩, 제어·물리 step, 기본 상태·행동·wrench·reward 로깅은 기존 루프를 재사용한다.

모터 고장 조건의 이벤트는 **step500의 t=5**에서 한 번 적용한다. 이벤트 직전/직후 snapshot을 대조해
물리·모터 내부 상태·목표·관측·step이 바뀌지 않았음을 검사한다. 처음 영향을 받은 post-state는 t=5.01이다.
모터2–4는 항상1.0이며 각 조건은 새 환경에서 독립 reset한다. 고장 없는 조건의 event CSV는 header만 있다.

- `<condition>-<model>.csv`: 정책 input/action은 `time`=t, 위치/속도/오차는 `time_post`=t+dt.
- `position_error_world_0/1/2` = world e_x/e_y/e_z. `velocity_0/1/2`는 실제 속도다.
- `desired_velocity`와 `internal_velocity_error`는 post-state에서 `norm_clip(-4e_p,1.5)`로 계산한다.
- `motor_thrust_command`: 기존 clipping 이후 command; `motor_thrust_nominal`: actuator 이후/효율 이전;
  `motor_thrust_actual` 및 기존 `motor_thrust`: 효율 이후 실제 출력.
- `*-physics.csv`: **모든 physics substep**의 command, ESC, RPM, nominal/actual force와 반력 토크,
  효율, 실제 MuJoCo ctrl, 상·하한 여유. 시간은 `[physics_time, physics_time_post]` 물리 구간이다.
- `*-events.csv`: 실제 simulation time, policy input time, control step, 효율 전후와 상태 불변 검증.

모터 신호는 매 physics substep에서 직접 대조하여 lambda가 추력·반력 토크에 한 번 적용됐고
MuJoCo ctrl에 같은 값이 쓰였음을 검사한다. 정책 action 경계는 public `predict`가 반환해 환경에
전달한 [-1,1] 행동 기준이며, clipping 이전 latent Gaussian 행동을 측정했다는 뜻은 아니다.

## 분석 정의

모든 구간은 post-state의 **(start,end]** 표본이다: (0,20], (3,5], (5,20], (18,20].
t=5 상태는 변경 이전 구간에 포함하며, 고장 뒤 구간은 처음 영향을 받은 t=5.01부터다.
각 구간의 표본 개수와 실제 첫/끝 시각도 기록한다.

- XY offset은 `||mean(e_xy)||`, 흔들림은 `sqrt(mean(||e_xy-mean(e_xy)||²))`다.
- Z offset은 `mean(e_z)`의 부호를 유지하고 절댓값도 별도 기록한다.
  흔들림은 `sqrt(mean((e_z-mean(e_z))²))`다. 모집단 평균(ddof0)을 사용한다.
- XY/Z RMSE 및 최대 오차, 실제 v와 내부 e_v의 XY/Z RMS를 각각 계산한다.
- `RMSE² = bias² + sway²` 잔차를 모든 사용 구간에 기록하고 수치 검증한다.
- mean-centered RMS는 drift/과도응답도 포함하므로 그 값만으로 주기적·지속적 진동을 단정하지 않는다.
- 조기 종료 시 full20/post15/tail18–20은 null이다. 실제 관측 부분은 `partial_observed_windows`로 분리한다.
  (3,5]를 끝까지 관측했으면 그 구간은 유효하다. 종료 직전 2초로 tail을 대체하지 않는다.

회복은 기존 Thresholds의 위치0.005m, 속도0.02m/s, 최소1초 유지 규약을 사용한다.
XY는 수평 위치/속도 norm, Z는 수직 위치/속도 절댓값으로 판정하며 기존 3D 정착도 유지한다.
후보 시각부터 t=20까지 조건을 계속 만족해야 한다. 모터 조건의 회복 시간은 후보시각−5이며,
목표 위치를 기준으로 한다. 고장 전 평균 위치를 새 목표로 사용하지 않는다. 미회복/조기 종료는 null.
`mean_shift_xy/z`는 (3,5] 평균과 (5,20] 평균의 차이다. 조기 종료이면 post 평균의 partial 범위를 명시한다.

포화 통계는 physics 표본의 총 지속시간과 최장 연속시간을 로터별로 저장한다.
allocator clipping, allocator command 경계, ESC 0/1 경계, 실제 effective 추력 상한을 구분한다.
정책 행동 경계는 control dt로 집계한다. `f_cmd - f_nominal`은 actuator 지연/추종 차이이며 포화로 부르지 않는다.
실효 최대 추력은 기존 forward map의 ESC1 정상 출력(기존 thrust cap 적용)에 lambda를 곱한다.
그 여유를 폐루프 안정성이나 제어 가능성의 보장으로 해석하지 않는다.

## 실행

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
python compare_ab_payload_motor.py --config configs/eval_velocity_ab.yaml
```

`--dry-run`은 모델 계약·경로·해시와 조건별 resolved 설정을 출력한다.
`--record`는 명시적 comparison completion.json 경로, `--output-dir`은 새 고유 디렉터리의 부모 경로다.
새 학습 YAML이나 모델 선택용 latest selector를 만들지 않는다.

## 결과

실행 결과: `artifacts/runs/ab-payload-motor-791j3j2g`. 14회 모두 **20초 완료**, `terminated=false`,
`truncated=true`(정해진 horizon)다. 초기 위치/속도/자세/모터 상태는 모든 조건에서 같았고,
각 조건 안의 A/B 초기 snapshot도 동일했다. 8개 모터 고장 rollout에만 t=5 이벤트가 하나씩 있었다.

A best SHA256: `a541595c4a59a989e486300f7adef2ddac6f145e9490dd0041300086da134cf4`

B best SHA256: `657559be961489528f0f70c70245f531ec72172e5e3a763fa898f5d21755584f`

두 모델의 저장 정규화 설정은 `enabled=false`다. 모델 로딩에서 원래 계약과 해시를 검증했고
새 참조 속도 차감을 입력에 추가하지 않았다.

### 조건 적용 확인

nominal 엔진 질량은 **0.04338kg**, payload 조건은 **0.04838kg**이었다.
centered payload의 COM은 [0,0,0], offset payload의 `body_ipos`는
**[0.0031004547333608927,0,0]m**였다. payload의 [0.03,0]은 body origin 기준 부착 위치이며
합성 COM 자체가 0.03m 이동한다는 뜻은 아니다.
모든 조건의 초기 모터 추력은 로터당 **0.10638945N**으로 동일했다.
정책 hover bias 질량도 0.04338kg으로 유지했다. payload 값이 난수화 해제 과정에서 지워지지 않았다.

### 주요 비교

거리 단위 mm, 마지막 구간은 (18,20]이다. `Z 평균`의 음수는 목표보다 낮음을 뜻한다.
`0.000`은 표시 정밀도에 따른 반올림이며 정확한 값은 JSON/CSV에 보존한다.
고장 없는 세 조건의 5–20초 값도 같은 구간 통계지만, 고장 사건이나 회복 시간은 없다.

| 조건 | 모델 | 완료/종료 | 5–20s 최대 XY/absZ (mm) | tail XY offset / 흔들림 (mm) | tail Z 평균 / 흔들림 (mm) | 회복 XY/Z/3D (s) | allocator clip (s) |
|---|---|---|---:|---:|---:|---|---:|
| nominal | A_best | 20.00s / horizon | 1.990 / 1.333 | 1.941 / 0.000 | 1.328 / 0.000 | 해당 없음 | 0.000 |
| nominal | B_best | 20.00s / horizon | 1.663 / 1.773 | 1.648 / 0.000 | 1.766 / 0.000 | 해당 없음 | 0.000 |
| centered_payload | A_best | 20.00s / horizon | 9.343 / 20.658 | 9.325 / 0.000 | -20.638 / 0.000 | 해당 없음 | 0.000 |
| centered_payload | B_best | 20.00s / horizon | 5.884 / 20.542 | 5.881 / 0.000 | -20.529 / 0.000 | 해당 없음 | 0.000 |
| offset_payload | A_best | 20.00s / horizon | 72.019 / 18.638 | 71.872 / 0.001 | -18.576 / 0.000 | 해당 없음 | 0.000 |
| offset_payload | B_best | 20.00s / horizon | 81.143 / 33.197 | 81.081 / 0.000 | -33.040 / 0.000 | 해당 없음 | 0.000 |
| motor_80 | A_best | 20.00s / horizon | 58.463 / 12.129 | 58.075 / 0.008 | 4.831 / 0.001 | null / 1.09 / null | 0.000 |
| motor_80 | B_best | 20.00s / horizon | 100.146 / 24.333 | 100.091 / 0.000 | -6.459 / 0.000 | null / null / null | 0.000 |
| motor_70 | A_best | 20.00s / horizon | 100.982 / 23.642 | 100.318 / 0.012 | 7.217 / 0.002 | null / null / null | 0.000 |
| motor_70 | B_best | 20.00s / horizon | 173.185 / 46.071 | 173.117 / 0.000 | -12.807 / 0.000 | null / null / null | 0.000 |
| combined_80 | A_best | 20.00s / horizon | 137.597 / 36.816 | 137.438 / 0.008 | -14.987 / 0.001 | null / null / null | 0.000 |
| combined_80 | B_best | 20.00s / horizon | 186.736 / 67.544 | 186.694 / 0.000 | -44.366 / 0.000 | null / null / null | 0.000 |
| combined_70 | A_best | 20.00s / horizon | 205.388 / 64.551 | 190.606 / 0.097 | -12.761 / 0.022 | null / null / null | 0.520 |
| combined_70 | B_best | 20.00s / horizon | 277.542 / 103.240 | 277.486 / 0.000 | -53.652 / 0.000 | null / null / null | 0.290 |

### 관측된 현상과 해석 한계

1. **포화 없는 잔류 offset:** centered/offset payload, motor_80/motor_70, combined_80에서는
   allocator clipping과 ESC/action 경계 도달이 없었다. 그럼에도 목표에서 벗어난 평균 위치가 남았다.
   예를 들어 motor_80의 tail XY offset은 A 58.075mm, B 100.091mm이며,
   흔들림은 각각 0.008mm와 0.001mm 미만이다. 목표 복귀와 일정 위치에서의 비행을 구분해야 한다.
2. **진동:** 고장 직후에는 감쇠하는 위치·자세·추력 진동이 보였다. 그러나 마지막 구간의
   XY 흔들림 최대는 A combined_70의 0.097mm, Z는 0.022mm로 잔류 offset에 비해 작다.
   이번 20초 결과에서 큰 지속 진동이 tail 오차를 지배하는 조건은 관측하지 못했다.
   mean-centered RMS에는 drift도 들어가므로 이 숫자만으로 limit cycle이나 장기 안정성을 판정하지 않는다.
3. **포화와 오차 증가:** combined_70에서 모터1 command가 0.2N 상한에 잘렸다.
   A는 [5.05,5.27], [5.44,5.74]초 합계 0.52초(최장0.30초),
   B는 [5.07,5.19], [5.49,5.66]초 합계 0.29초(최장0.17초)다.
   해당 과도 구간에 오차가 증가했지만 이후 무한히 증가하거나 물리 종료되지는 않았다.
   tail에는 A XY 190.606mm/Z −12.761mm, B XY 277.486mm/Z −53.652mm의 편향이 남았다.
   포화 지속시간이 짧은 모델이 반드시 위치 오차도 작지는 않았다.

모든 모터 고장 조건에서 목표 기준 XY·3D 회복은 null이었다. Z는 A motor_80만 고장 후
1.09초에 회복했다. 위치0.005m/속도0.02m/s band를 t=20까지 유지하는 기존 suffix 규약이다.
combined 조건은 고장 전부터 payload에 의한 큰 편향이 있었다. 아래 표는 그 편향과
고장 뒤 평균 변화량을 별도로 보여준다. pre/post 평균의 차이가 작더라도 목표에 복귀했다는 뜻은 아니다.

| 조건 | 모델 | pre XY 평균 [x,y] mm | pre Z 평균 mm | pre XY/Z 흔들림 mm | Δμ XY [x,y] mm | Δμ Z mm | post 최대 tilt deg | post 최대 각속도 rad/s |
|---|---|---|---:|---|---|---:|---:|---:|
| motor_80 | A_best | [-1.434, -1.299] | 1.325 | 0.074 / 0.008 | [41.420, -37.842] | 2.015 | 4.384 | 1.319 |
| motor_80 | B_best | [1.572, -0.616] | 1.796 | 0.036 / 0.019 | [41.153, -83.256] | -9.723 | 3.935 | 1.245 |
| motor_70 | A_best | [-1.434, -1.299] | 1.325 | 0.074 / 0.008 | [72.007, -64.969] | 3.295 | 8.368 | 2.158 |
| motor_70 | B_best | [1.572, -0.616] | 1.796 | 0.036 / 0.019 | [71.926, -144.872] | -17.189 | 7.199 | 2.013 |
| combined_80 | A_best | [71.499, -6.645] | -18.855 | 0.873 / 0.253 | [52.754, -45.826] | 2.025 | 5.282 | 1.490 |
| combined_80 | B_best | [78.207, -19.990] | -33.489 | 0.606 / 0.339 | [52.869, -103.503] | -12.682 | 4.738 | 1.414 |
| combined_70 | A_best | [71.499, -6.645] | -18.855 | 0.873 / 0.253 | [95.333, -80.986] | 2.687 | 11.834 | 2.599 |
| combined_70 | B_best | [78.207, -19.990] | -33.489 | 0.606 / 0.339 | [94.541, -182.744] | -23.546 | 9.141 | 2.289 |

### XY/Z RMSE와 속도

아래 full/pre/post/tail은 각각 (0,20]/(3,5]/(5,20]/(18,20]이다.
실제 v와 `e_v=v−norm_clip(-4e_p,1.5)`는 별도 지표다. 모든 구간의 μ_xy 벡터,
부호 있는 μ_z, RMS, 최대값과 표본 수는 `window_metrics.csv`에도 있다.

| 조건 | 모델 | full XY/Z RMSE mm | pre XY/Z RMSE mm | post XY/Z RMSE mm | tail XY/Z RMSE mm | tail 실제 속도 XY/Z RMS m/s | tail 속도 오차 XY/Z RMS m/s |
|---|---|---|---|---|---|---|---|
| nominal | A_best | 1.902 / 1.296 | 1.936 / 1.325 | 1.941 / 1.328 | 1.941 / 1.328 | 0.000001 / 0.000000 | 0.007763 / 0.005311 |
| nominal | B_best | 1.682 / 1.775 | 1.689 / 1.796 | 1.649 / 1.766 | 1.648 / 1.766 | 0.000000 / 0.000000 | 0.006593 / 0.007064 |
| centered_payload | A_best | 9.056 / 20.345 | 9.312 / 20.686 | 9.325 / 20.638 | 9.325 / 20.638 | 0.000002 / 0.000000 | 0.037301 / 0.082553 |
| centered_payload | B_best | 5.683 / 20.186 | 5.833 / 20.560 | 5.881 / 20.530 | 5.881 / 20.529 | 0.000000 / 0.000000 | 0.023522 / 0.082116 |
| offset_payload | A_best | 70.343 / 19.090 | 71.812 / 18.857 | 71.873 / 18.578 | 71.872 / 18.576 | 0.000009 / 0.000001 | 0.287490 / 0.074302 |
| offset_payload | B_best | 78.914 / 33.128 | 80.724 / 33.491 | 81.080 / 33.046 | 81.081 / 33.040 | 0.000000 / 0.000000 | 0.324323 / 0.132158 |
| motor_80 | A_best | 48.993 / 4.290 | 1.936 / 1.325 | 56.563 / 4.906 | 58.075 / 4.831 | 0.000052 / 0.000004 | 0.232295 / 0.019326 |
| motor_80 | B_best | 82.831 / 7.618 | 1.689 / 1.796 | 95.640 / 8.735 | 100.091 / 6.459 | 0.000002 / 0.000000 | 0.400363 / 0.025837 |
| motor_70 | A_best | 84.714 / 6.837 | 1.936 / 1.325 | 97.814 / 7.865 | 100.318 / 7.217 | 0.000081 / 0.000014 | 0.401268 / 0.028870 |
| motor_70 | B_best | 143.398 / 14.530 | 1.689 / 1.796 | 165.578 / 16.745 | 173.117 / 12.807 | 0.000001 / 0.000000 | 0.692468 / 0.051227 |
| combined_80 | A_best | 121.641 / 18.255 | 71.812 / 18.857 | 135.266 / 17.423 | 137.438 / 14.987 | 0.000052 / 0.000009 | 0.549749 / 0.059947 |
| combined_80 | B_best | 161.087 / 43.514 | 80.724 / 33.491 | 181.300 / 46.406 | 186.694 / 44.366 | 0.000001 / 0.000000 | 0.746776 / 0.177464 |
| combined_70 | A_best | 167.092 / 19.437 | 71.812 / 18.857 | 189.195 / 19.051 | 190.606 / 12.761 | 0.000657 / 0.000162 | 0.762418 / 0.051029 |
| combined_70 | B_best | 235.485 / 52.756 | 80.724 / 33.491 | 268.717 / 57.790 | 277.486 / 53.652 | 0.000001 / 0.000000 | 1.109942 / 0.214608 |

### 모터 여유

로터별 command 하한/상한 최소 여유(N), ESC 하한/상한 여유, 실제 유효 최대 추력과 여유,
포화 총/최장 연속 시간은 [motor_summary.csv](../artifacts/runs/ab-payload-motor-791j3j2g/motor_summary.csv)의 56개 행에 있다.
원본 JSON에는 정책 행동 채널별 경계 시간 및 고장 이후만의 모터 통계도 있다.

nominal 최대 추력은 모터당0.2N이며, 모터1의 효율 변경 후 실제 최대값은
80%에서0.16N, 70%에서0.14N이다. 모터2–4는 계속0.2N이다.
combined_70에서 최소 실제 상한 여유는 모터1 기준 A 약0.0000384N, B 약0.0001536N이었다.
최대 ESC 명령은 약0.78185로 1에 도달하지 않았다. allocator 0.2N cap이 먼저 작동하므로
ESC가 1이 아니더라도 allocator 포화는 가능하다. 1차 지연 때문에 실제 추력이 정확히
0.14N에 닿는지와 command clipping도 별개다. 관측된 여유는 안정성/제어 가능성 보장이 아니다.

### 검증

- 관련 범위 **93 passed, 1 skipped**. skip은 환경 변수로 켜는 기존 선택적 replay다.
  기존 interactive plot의 tight_layout 경고10개가 있었으며 새 비교 그래프를 직접 확인했다.
  초기 pytest 실행은 설치된 ROS 자동 플러그인의 lark 의존성 누락으로 collection 전에 실패했고,
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`로 해당 무관 플러그인을 비활성화해 실제 테스트를 완료했다.
- A/B nominal의 첫8초 행동·관측·물리 상태는 이전 A/B 공통 평가 CSV와 정확히 일치했다.
- 14개 rollout, control28000/physics140000 표본을 기록했다.
  매 physics substep의 force/torque 효율 적용과 실제 ctrl 일치를 확인했다.
- CSV에서 독립 재계산한 XY/Z 통계 차이는0, bias–variance 항등식 최대 잔차는
  **1.7824e-16m²**였다.
- 각 조건의 고정 payload, 합성 COM, 동일 nominal 초기 모터 상태, 이벤트 경계와 상태 불변을 검증했다.
- 조기 실패 사례는 이번 실측에 없었다. full/tail null 및 partial 분리는 합성 실패 표본의 테스트로 검증했다.
- 체크포인트·기존 A/B·nominal/D 결과·설정·물리 소스를 포함한 **451개 파일 해시**가 일치했다.
- PPO 학습, optimizer update, 학습 YAML/체크포인트 수정, gain/보상 튜닝, commit/push는 하지 않았다.

이번 한 번씩의 deterministic 실행에서 B의 offset이 여러 부하/고장 조건에서 A보다 컸다.
중앙 payload의 XY offset에서는 B가 더 작았다. 이를 일반적인 통계적 우월성이나 성공 확률로 표현하지 않는다.
적분기·history가 필요하다는 결론은 이 데이터만으로 확정하지 않았다. 편향 보상 기능의 필요성/효과는
명시적 후속 비교로 검증할 가설이며 이번에는 그런 제어 변경을 넣지 않았다.

### 산출물과 변경 파일

- 새 진입점 `compare_ab_payload_motor.py`, 새 모듈 `crazyflie_rl/payload_motor_eval.py`.
- 기존 `crazyflie_rl/dr_transfer.py`에 선택적 env_factory/observer hook만 추가.
- 새 `tests/test_payload_motor_eval.py` 및 이 문서. 기존 local 변경은 보존했다.
- [manifest](../artifacts/runs/ab-payload-motor-791j3j2g/manifest.json), [summary JSON](../artifacts/runs/ab-payload-motor-791j3j2g/summary.json), [summary CSV](../artifacts/runs/ab-payload-motor-791j3j2g/summary.csv).
- [구간별 전체 지표](../artifacts/runs/ab-payload-motor-791j3j2g/window_metrics.csv), [검증 결과](../artifacts/runs/ab-payload-motor-791j3j2g/verification.json), [테스트 로그](../artifacts/runs/ab-payload-motor-791j3j2g/tests.log).
- 각 조건의 `*-comparison.png`는 기존 위치/속도/자세/각속도 구조, `*-fault.png`는 XY/Z 오차·효율·추력을 비교한다.
- [motor_80 그래프](../artifacts/runs/ab-payload-motor-791j3j2g/motor_80-fault.png), [combined_70 그래프](../artifacts/runs/ab-payload-motor-791j3j2g/combined_70-fault.png).
- 원본 control/physics/event CSV는 같은 디렉터리의 조건·모델별 파일이다.

저장된 명령으로 14개 평가를 새 고유 디렉터리에 재실행할 수 있다.

```bash
bash artifacts/runs/ab-payload-motor-791j3j2g/rerun.sh
```

`analyze_outputs.py`는 저장 CSV로부터 부가 표·구간별 CSV·검증 결과를 생성한 후처리 스크립트이며
같은 결과 디렉터리에 보존했다. 학습은 호출하지 않는다.

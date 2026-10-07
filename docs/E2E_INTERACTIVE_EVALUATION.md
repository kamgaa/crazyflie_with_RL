# E2E 단독 대화형 평가

`view_e2e_interactive.py`는 학습된 E2E 정책 하나를 deterministic inference로 실행한다.
PID/residual/baseline 비교 rollout, 학습, optimizer update, checkpoint 저장은 없다.
현재 checkout과 적용 지침을 확인했으며 AGENTS.md는 발견되지 않았다.

## 실행

저장소 루트, 현재 MuJoCo/SB3 환경에서:

```bash
MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python view_e2e_interactive.py \
  --config configs/eval_dr_transfer.yaml \
  --model artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip
```

기본은 사용자 종료 또는 물리적 종료까지 실행한다. `--duration 60`은 simulation time
60초, `--duration 0`은 무제한(기본값)이다. 시간은 제어 스텝으로 올림하며 실제 적용
step 수와 시간을 기록한다. `--position-step 0.02`로 키당 목표 이동 거리(m)를 설정한다.
`--seed` 기본값은 42다. `--help`는 모델 없이도 볼 수 있다.

화면 없는 비학습 smoke 검사:

```bash
MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python view_e2e_interactive.py \
  --config configs/eval_dr_transfer.yaml \
  --model artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip \
  --headless --duration 2 --no-realtime
```

GUI는 항상 벽시계 pacing을 사용한다. simulation의 policy/physics rate는 100/500Hz를
유지하며 렌더링 갱신만 최대 30Hz다. 처리 지연이 생겨도 physics step을 건너뛰거나
중복해서 만회하지 않는다. 장비가 느리면 벽시계보다 simulation이 느려질 수 있다.
headless는 종료할 키 창이 없으므로 유한한 `--duration`이 필수다.

선택 checkpoint는 실제 존재하며 SHA256은
`c054ba76c7813256a84e038040b905d23c7962c1063de3f464cc27365169108b`다.
보관 학습 manifest의 E2E 모드, action scale `[.0075,.0075,.001,.5]`, 관측 15/행동 4가
선택 config 및 checkpoint와 호환된다. 유효 위치 가중치는 xy=10/z=6이다.
`load_frozen_policy`는 기존 모델 로더/shape·관측 의미·action bound·scale 검증을 재사용한다.
E2E임을 확인할 학습 metadata가 없으면 실행하지 않으며 `--manifest PATH`로 지정할 수 있다.
residual 설정·모델, 알려진 scale/shape 불일치는 명확한 오류다.

현재 모델은 raw 관측 경로다. 저장된 정규화가 필요한 모델은
`--normalization /absolute/path/to/vecnormalize.pkl`로 기존 통계를 제공한다.
고정 mean/variance와 기존 clipping을 재사용하고 통계를 갱신하지 않는다.

## 초기화와 입력 계약

공통 공중 평가 `EvaluationAdapter.reset_to_case_initial_state`를 재사용한다.
정상 reset → 위치 [0,0,1], identity wxyz, zero qvel → MuJoCo forward → airborne
hover actuator reset → PID 상태/previous action/step 초기화 → 새 관측 생성 순서다.
PID 객체의 초기화는 환경 공통 내부 상태 정리이며 PID 제어 실행은 아니다.

초기 목표 [0,0,1] m, yaw=0 rad, 효율=[1,1,1,1]. 위치·자세 perturbation은 0,
새 pose sampler 설정은 None이므로 legacy의 0 perturbation 경로를 사용한다.
payload mass/offset은 입력 YAML 값을 유지한다(기본 eval_dr_transfer.yaml은 0).
payload 및 actuator 파라미터 randomization off,
actuator reset mode=auto. 모터 1차 지연 모델과 기존 시상수는 유지한다.
변경값은 평가용 immutable config 사본과 runtime metadata에 기록하며 학습 YAML은 수정하지 않는다.

### 고정 무게추 + 모터 효율 저하

후속 수정 완료: 새 속도 규약 학습 작업에서 아래 감사의 문제를 수정했다.
현재는 COM/전체 관성과 엔진 파생 상수를 갱신하고 명시적 payload 중력 토크는 제거한다.
[현재 구현 설명](POSITION_VELOCITY_ERROR.md)을 참고한다. 아래 감사 설명은 수정 전 상태다.

**후속 감사에서 물리 반영의 불일치를 발견했다.** `body_ipos` 설정값은 변경되지만 현재
reset의 파생 상수 갱신 누락으로 `data.xipos`는 기체 원점에 남는다. 현재 payload 모멘트는
환경이 명시적으로 더하는 중력 토크로 발생한다. 따라서 아래 COM 기록은 model 설정값이며
정상적으로 반영된 실제 COM이라고 해석하면 안 된다. 상수 갱신만 추가하면 토크가 중복된다.
자세한 정적 검사와 실제 로그 비교는 [PAYLOAD_MOMENT_AUDIT.md](PAYLOAD_MOMENT_AUDIT.md)를 참고한다.

`configs/eval_e2e_interactive_payload.yaml`은 기존 평가 설정을 상속하는 예시다.
기존 실행 명령의 `--config`만 이 파일로 바꾸면 된다.

```yaml
extends: eval_dr_transfer.yaml
environment:
  payload:
    randomize: false
    mass: 0.005          # kg: 5 g
    offset: [0.02, 0.0] # m: 기체 원점에서 body +X 방향 2 cm
```

질량과 이격거리는 이 YAML에서 바꾼다. `offset`은 기체에 고정된 [x,y] 좌표이며
거리의 크기는 sqrt(x²+y²)다. 예를 들어 반경 r, 방향각 θ이면 [r*cosθ, r*sinθ]를
직접 지정한다. 월드 좌표가 아니며 현재 schema는 z 이격거리를 지원하지 않는다.
현재 기체의 기본 COM은 원점이지만, 이 설정의 기준은 항상 body 원점이다.

기존 환경 `_set_com_bias`를 그대로 재사용해 model의 질량, COM 위치, 대각 관성 배열을 바꾼다.
별도 추 형상/충돌체는 추가하지 않으며 비대각 관성을 포함한 완전한 부착물 모델은 아니다.
모터 고장 계수는 이 물성 변경과 독립적으로 기존 actuator 출력 이후 적용된다.
`0`은 모터 효율만 복원하고 무게추를 제거하지 않는다.

공중 시작 모터 상태는 기존 초기화 규약대로 **추를 포함한 총질량**의 균등 hover 추력으로
초기화한다. 이격된 COM의 자세 토크까지 상쇄하는 trim 해는 아니다.
반면 정책 action 변환의 hover 기준 질량 `env.mass`, allocator, action scale은 그대로다.
따라서 추가 질량/COM에 대응하는 제어는 학습된 정책의 출력에 달려 있다.

runtime-resolved YAML의 `payload`에는 지정 질량/offset, model의 질량/COM/대각 관성 배열 값,
정책 hover 기준 질량을 함께 기록한다. 기본 무게추 0 평가 경로는 유지한다.
이 기능은 대화형 평가에만 적용되며 DR 비교 CLI의 nominal-payload 계약은 바꾸지 않는다.

고정 payload 추가 검증: `tests/test_interactive_eval.py -k 'fixed_payload or nominal_real_checkpoint'`
3개 통과. YAML 고정값 보존/난수 비활성화, 실제 질량·COM·관성 및 hover 초기 추력,
모터 효율과 동시 적용, 실제 PPO/metadata 저장, 무게추 0 기존 궤적 일치를 확인했다.
5 g/+X 2 cm 설정으로 실제 baseline checkpoint의 headless 0.5초(50 step) 실행도 완료했다:
`artifacts/runs/ppo_e2e_hover_interactive-motor-effectiveness_seed42_20261001-140246/`.
실제 body 질량 0.04838 kg, COM x=0.00206697 m, 시간 한계로 종료, checkpoint 해시 불변.
이 짧은 실행은 장시간 호버 안정성 검증이 아니다. payload의 별도 GUI 형상은 추가하지 않았다.

정책 raw 입력은 `[p-p_ref, v_world, wxyz, omega_body, sin(yaw_error), cos(yaw_error)]`다.
키마다 고정 목표를 갱신하며 `v_ref=0`; 목표점 차분, 위치 P 루프, `vel_des`, 적분/history,
참조 가속도 feedforward를 추가하지 않았다. 효율 및 발생 추력은 정책에 입력하지 않는다.
목표 변경 후 첫 predict 직전에 관측을 새로 생성한다.

## 조작과 모터 대응

MuJoCo 창에 포커스를 둔 상태에서:

| 키 | 동작 |
|---|---|
| W / S | 월드 목표 X + / - |
| D / A | 월드 목표 Y + / - |
| R / F | 월드 목표 Z + / - |
| 1 / 2 / 3 / 4 | 해당 모터 효율 2%포인트 감소 |
| 0 | 효율만 모두 1.0 복원 |
| Space | 일시정지 / 재개 |
| Q | 종료 및 저장 |

위치 이동은 기본 키당 0.02m다. 기체 qpos/qvel, actuator 상태, 목표 이외의 물리
상태를 reset하지 않는다. `0`도 목표/물리 상태를 유지한다.
효율은 정수 감소 횟수 n=0…50으로 관리하여 `lambda=(50-n)/50`으로 계산한다.
10회=.80, 15회=.70, 50회 이후=0. 현재 효율에 .98을 곱하는 방식이 아니다.

실제 XML actuator의 site 위치를 초기 MuJoCo state에서 읽은 대응:

| 키/표시 번호 | allocator index | force / torque actuator | site | body xyz (m) | motor_direction |
|---|---:|---|---|---|---:|
| 1 | 0 | motor0_force / motor0_torque | motor0 | [.03536,-.03536,0] | +1 |
| 2 | 1 | motor1_force / motor1_torque | motor1 | [-.03536,-.03536,0] | -1 |
| 3 | 2 | motor2_force / motor2_torque | motor2 | [-.03536,.03536,0] | +1 |
| 4 | 3 | motor3_force / motor3_torque | motor3 | [.03536,.03536,0] | -1 |

부호는 저장소의 회전 방향/반력 yaw 토크 부호 계약이다. 시선 방향에 따른 CW/CCW나
앞왼쪽 등의 배치를 추측하지 않는다. 실제 실행마다 이 매핑과 actuator gear를 metadata에
기록하고 위치·방향을 콘솔에 출력한다. allocator 설정은 변경하지 않는다.

### PRESS/REPEAT, 큐와 표시

설치 버전은 MuJoCo 3.12.0. 공개 `launch_passive(key_callback=...)`는 keycode만
전달한다. 해당 버전의 [Python 콜백 브리지](https://github.com/google-deepmind/mujoco/blob/3.12.0/python/mujoco/simulate.cc#L57)와
[GLFW 판정](https://github.com/google-deepmind/mujoco/blob/3.12.0/simulate/glfw_adapter.cc#L222)을 확인했다:
PRESS만 Python에 전달되고 REPEAT/RELEASE는 제외된다. 없는 action/release 인자를 가정하거나
시간 debounce로 정상적인 빠른 연타를 제거하지 않는다. 다른 버전/백엔드의 이벤트 동작은 재확인이 필요하다.

콜백은 thread-safe queue에 키만 넣는다. 시뮬레이션 루프의 제어 경계에서 순서대로 적용한다.
일시정지 중 policy predict와 env.step은 호출하지 않는다. 위치/효율 이벤트는 FIFO에 대기하고
재개하는 경계에서 적용한다. Space/Q는 정지 중에도 처리한다. 적용 시각/step과 원래 수신
sequence를 기록한다. 정지 중 종료하여 적용되지 않은 이벤트는 `pending_unapplied`,
`applied=false`로 기록하며 적용된 것으로 간주하지 않는다.

공개 `Handle.set_texts`로 목표, 효율%, simulation time, RUNNING/PAUSED 및 키 도움말을
표시한다. `user_scn`의 초록색 구가 현재 목표다. 숫자키의 기본 geom-group 토글 때문에
기체가 사라지지 않도록 예약 키 0…4의 표시 그룹을 초기 설정으로 유지한다.

viewer는 **렌더링 전용 model/data 사본**을 사용한다. 각 갱신에서 실제 integration state를
복사해 표시하며 이 사본으로 physics step을 실행하지 않는다. native GUI의 reset/control/
mouse perturbation은 실제 기체에 전파되지 않는다. 표시용 model/data/user_scn 접근은
`viewer.lock()`으로 보호하고 `sync()`는 lock 밖에서 호출한다. 실제 정책·물리 갱신은
시뮬레이션 루프 한 곳에서만 이루어진다.

## 효율 적용과 물리적 가정

기존 `motor_degradation.DiagnosticEnv`의 계산을 `apply_motor_effectiveness()`로 추출했다.
`InteractiveEnv._apply_control()`은 생산 환경의 allocator → thrust clipping → actuator
동역학을 먼저 그대로 실행한 뒤 공용 함수를 한 번 호출한다.

```text
nominal_i = clipping과 actuator dynamics를 지난 새 출력
f_applied_i = lambda_i * nominal_i
q_applied_i = lambda_i * signed_nominal_reaction_torque_i
```

`data.ctrl[act_force]`와 `data.ctrl[act_torque]`에 위 실제 출력이 전달된다.
실제 wrench와 allocation error도 실제 출력 기준으로 갱신한다.
모터 내부 RPM/command/state, B/B_pinv, action scale, hover 기준값은 변경하지 않는다.
매 물리 substep마다 새 actuator 출력을 읽으므로 이전 감소 출력에 다시 곱하지 않는다.

이는 회전수 변화 모델이 아닌 **로터 출력 손실 모델**이다. 기존 고장 모델처럼 추력과
반력 토크에 같은 계수를 적용한다. 이번 profile은 legacy_ratio 토크 모델이다.
다른 토크 모델을 선택하면 actuator가 계산한 전체 signed rotor torque에 같은 계수가 적용된다.
별도의 고장 물리식을 새로 구현하지 않았고 공용 환경 기본 동작은 변경하지 않았다.

## 기록과 종료

기존 `view_live.py` 경로를 확인한 결과 그 진입점에는 CSV/event writer가 없다.
`RolloutTrace`, `ArtifactManager`, `trace_metrics`, `_save_policy_report` 및 reward/yaw/wrench
진단 함수를 재사용하고 CSV/이벤트 직렬화만 새 진입점에 추가했다.

새 실행마다 `artifacts/runs/ppo_e2e_hover_interactive-motor-effectiveness_seed42_<timestamp>/`를
생성한다. 충돌 시 기존 ArtifactManager의 suffix 처리를 사용하며 덮어쓰지 않는다.

- `config/`: 기존 resolved config, 실제 runtime overrides/모델 hash/초기 snapshot/rotor 매핑.
- `manifests/`: 기존 코드 버전·dirty 상태·환경 버전·결과 정보. runtime에 실행 소스 SHA256도 기록.
- `metrics/`: 기존 evaluation/reward_balance/yaw_authority/wrench_authority JSON.
- `plots/ppo.png`: 기존 단일 정책 16:9 위치·자세·선속도·각속도·제어·모터 보고서.
- `plots/`의 기존 yaw/wrench 진단 그래프 및 추가 `motor_effectiveness.png`.
- `rollout.csv`: 매 제어 스텝 즉시 flush하는 CSV.
- `events.csv`: 이벤트 즉시 flush, 적용 전후 목표/효율, 대상 모터, key, simulation time/control step.
- `trace.npz`: 기존 RolloutTrace 배열 이름으로 저장한 단일 E2E trace.

가짜 PID trace나 중복 baseline은 없다. 기존 진단 JSON의 미실행 `floor` 항목은 null이다.
원래 trace 필드 `position`, `reference_position`, `attitude_deg`, `linear_velocity`,
`angular_velocity`, `control_input`, `motor_thrust`, `motor_thrust_command`, `motor_command`,
`motor_omega_rad_s`, `reaction_torque_nm`, `wrench_command`, `wrench_actual`, `allocation_error`,
`position_error`와 평가 지표를 유지한다. 보고용 상태는 기존처럼 float32 관측에서 재구성한다.
CSV의 벡터 suffix는 1부터 시작하며 xyz, wxyz, 모터 1…4, wrench tau_x/tau_y/tau_z/Fz 순서다.

`motor_thrust`는 기존과 동일하게 **실제로 물리에 사용한 최종 모터 추력(N)**이다.
`motor_thrust_command`는 clipping된 명령 추력, `motor_command`는 정규화 actuator 입력이다.
다음 별도 열을 추가했다.

- `motor_effectiveness_1`…`_4`: 현재 로터 출력 효율.
- `motor_thrust_before_effectiveness_1`…`_4`: 동역학 이후, 효율 적용 직전 추력(N).
- `motor_thrust_applied_1`…`_4`: 물리에 사용한 효율 적용 후 추력(N), 기존 motor_thrust와 같음.
- `reaction_torque_before_effectiveness_nm_1`…`_4`: 효율 적용 전 반력 토크.
- `policy_action`, `raw_observation`, `position_before`, `velocity_before`, `quaternion`, `reward`, 종료 flags.
- `time_post`, `control_step`: 기존 `time_sec`의 의미를 바꾸지 않고 시각을 명확히 구분.

기존 `view_live`의 `time_sec=t`는 제어 구간 시작인데 상태는 step 이후였다. 이 계약을 유지하고
**post-state의 실제 시각 `time_post=t+dt`**를 추가했다. raw observation/action/pre-state는 t,
상태는 t+dt, 모터 명령/출력은 [t,t+dt]의 마지막 physics substep 값이다. 참조는 해당 구간 동안
고정되므로 상태 오차는 그 참조와 계산한다. `actual_duration_sec`가 실제 완료된 물리 구간 길이다.
기존 `trace_metrics.duration_sec`은 기존 time_sec 규약을 유지한다.
효율 그래프는 적용된 제어 경계에서 계단으로 표시하고 고장/복원 시각을 점선으로 표시한다.

평가 인스턴스의 `env.max_steps`만 유한 step 수 또는 infinity로 변경한다. 학습 YAML의 8초 제한은
실행을 자르지 않으며 물리 guard는 그대로 적용한다. Q/창 닫기/Ctrl+C/시간 종료/물리 종료는
공통 finally 저장·정리 경로를 사용한다. 예외도 수집된 CSV/trace/가능한 보고서를 남기고 원래
오류를 다시 보고한다. 표본 0이면 지표는 null이며 그래프를 만들지 않는다. 자동 reset/이어붙이기는 없다.

## 검증 (2026-10-01)

관련 비학습 테스트 **45개 통과 + 최종 render-copy/예약 키 검사 1개 통과**:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
  python -m pytest -q tests/test_interactive_eval.py tests/test_motor_degradation.py \
  tests/test_plotting_comparison.py
```

- 지정한 실제 PPO로 정상 환경과 InteractiveEnv의 동일 초기 snapshot, action, 관측, qpos/qvel/ctrl,
  actuator/wrench 및 reward/종료 값 정확히 일치. PPO.learn/train은 테스트에서 호출 금지.
- 목표 이벤트의 원본 물리/actuator 상태 불변, callback의 큐 전용 동작.
- 개별 모터 번호·감소 독립성, 10/15/50회 및 복원, pause/resume FIFO.
- 실제 적용 전후 신호와 MuJoCo ctrl을 반복 직접 대조. RPM/internal actuator 상태는 정상 모델과 동일.
- 기존 PID 고장 진단 회귀(공용 함수 추출 범위), 기존 plotting 회귀.
- Q/창 닫기/KeyboardInterrupt/예외/시간/물리 종료, 빈 기록과 한 표본 기록의 저장.
- 무제한 평가가 8초를 넘어 805 step(8.05초)까지 진행한 뒤 Q 종료.
- E2E가 아닌 모델/config, 입력·출력 shape 불일치 오류.
- 렌더링 사본/geometry 표시 변경이 실제 환경을 바꾸지 않는지 확인.
- baseline checkpoint SHA256 불변, Python 컴파일, git diff --check 확인.

실제 GUI 2초 무입력 실행:
`artifacts/runs/ppo_e2e_hover_interactive-motor-effectiveness_seed42_20261001-101346/`

최종 GUI 키 이벤트 실행:
`artifacts/runs/ppo_e2e_hover_interactive-motor-effectiveness_seed42_20261001-101956/`

최종 실행은 실제 창에 X11 XSendEvent로 keydown/release를 보내 GLFW와 공개 callback을 통과시켰다.
release 없이 1.5초 동안 보낸 30회 추가 keydown이 감산으로 처리되지 않았으며, 첫 PRESS와
추가 14번의 개별 PRESS로 정확히 15번 감산(.70)했다. 10번째 값은 .80이었다.
모터 2/3/4, 0 복원, W/S/D/A/R/F, Space, Q도 확인했다. pause/resume은 같은 simulation time
0.44초/step 44에서 처리됐고, 마지막 Q는 1.01초/step 101에서 저장됐다.
`gui_validation.json`에 증거를 기록했다. 처음 XTest 포커스 방식은 창 표시 타이밍/입력 전달
문제로 실패하여, 표시된 새 MuJoCo 창에만 직접 이벤트를 보내 검증했다.

실물 키보드의 장시간 수동 조작, OS별 반복 설정, 다른 MuJoCo 버전은 미검증이다.
기존 단일 trace plotting에서 tight_layout 경고가 나지만 파일 생성과 데이터 검증은 통과했다.
전체 저장소 테스트 통과나 모든 고장 효율에서 비행 가능함을 의미하지 않는다.
학습 설정·checkpoint 변경, commit/push는 수행하지 않았다.

## 변경 파일

- `view_e2e_interactive.py`: import-safe 진입점.
- `crazyflie_rl/interactive_eval.py`: E2E 초기화/큐/루프/GUI 사본/공통 보고서 연결.
- `crazyflie_rl/motor_degradation.py`: 기존 효율 적용을 공용 함수로 추출, 기존 진단에 재사용.
- `crazyflie_rl/plotting.py`: 효율 및 적용 전후 추력 보조 그래프.
- `tests/test_interactive_eval.py`: 집중 비학습 검증.
- `docs/E2E_INTERACTIVE_EVALUATION.md`: 이 문서.

# Crazyflie CF2.1 Brushless — Isaac Lab / PhysX

`isacc_sim` 브랜치는 MuJoCo 환경에서 이식한 Isaac Lab 코드입니다.
기존 MuJoCo 코드는 원래 브랜치에 보존됩니다. 이 브랜치의 파일 트리에는
학습·평가 결과, 그래프, 로그, TensorBoard 이벤트를 넣지 않습니다.

## 포함된 항목

- `src/crazyflie_isaaclab/`: task, 물리·모터·allocation·관측 구현
- `src/crazyflie_isaaclab/assets/cf21b.usda`: 현재 PhysX가 사용하는 Crazyflie 기체 모델
- `resources/mujoco/`: 원래 CF2.1 Brushless XML 및 STL 기체 모델
- `scripts/`: 기존 단계별 학습·평가·분석 코드와 외부 위치 적분 wrapper
- `models/phase14/best.pt`: phase14에서 선택한 **448회 best** 체크포인트 하나
- `models/phase14/{agent_config,reward_config,manifest}.json`: 모델 설정과 식별 정보

## 환경

Python 3.12, Isaac Lab v3.0.0-EA (배포 패키지 `isaaclab==17.0.2`),
Isaac Sim 6.1.0.0, PyTorch 2.11.0+cu128, RSL-RL 5.4.1, TensorDict 0.14.2.
기존 Isaac Lab 환경에 설치하며 다음 명령은 의존성을 업그레이드하지 않습니다.

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

Task: `MRL-Crazyflie-Hover-E2E-v0`, backend: `isaacsim_physx`.
Isaac Lab/Isaac Sim 설치 자체는 포함하지 않습니다.

## 저장된 phase14 정책 재생 (추가 학습 없음)

저장소 루트에서 실행합니다. `python`은 Isaac Lab 환경의 Python입니다.
실행마다 새 출력 경로를 사용하십시오. 결과는 Git에서 제외됩니다.

```bash
python scripts/evaluate_baseline.py --task hover --integral off --output runs/manual_hover --device cuda:0 --headless --physics isaacsim_physx
python scripts/evaluate_baseline.py --task T10 --integral off --output runs/manual_circle_T10 --device cuda:0 --headless --physics isaacsim_physx
python scripts/evaluate_baseline.py --task T5 --integral off --output runs/manual_circle_T5 --device cuda:0 --headless --physics isaacsim_physx
```

GUI는 그래픽 데스크톱 터미널에서 `--headless`를 `--visualizer kit`로 바꿉니다.
호버는 최대 30초, T10/T5 원형은 반지름 1m로 각각 두 바퀴입니다.
첫 실제 실패에서 종료합니다. Actor mean을 사용하고 optimizer update는 0회입니다.
GUI 실행 여부는 headless 검증과 별개입니다.

`--integral on`은 같은 동결 정책에 외부 위치 적분을 적용합니다. 기본값은 OFF입니다.
Ki=0.20/s, XY norm 한계 0.40m, Z 한계 ±0.15m이며 allocator clipping 시 적분을 멈춥니다.
실제 목표·reward·실패 판정은 유지합니다.

## 모델과 제어 계약

`models/phase14/best.pt`는 원본 체크포인트 그대로이며 actor/critic, 행동분포,
Adam 및 저장된 학습 상태를 포함합니다. 다른 학습 모델 또는 final은 포함하지 않습니다.
SHA256은 `models/phase14/manifest.json`에 있습니다.

물리 500Hz, 정책 100Hz, decimation 5. World Z-up/body FLU.
관측 15D: 위치 오차(현재−목표) 3, 절대 world 속도 3, quaternion wxyz 4,
body 각속도 3, yaw 오차 sin/cos 2. 관측 정규화 없음.
행동 4D 순서는 roll/pitch/yaw torque, collective입니다.
[-1,1] clipping 후 `[0.0075,0.0075,0.001,0.5]`를 곱하며 단위는 `[Nm,Nm,Nm,N]`.
Collective에 본체 질량 0.04338kg 기준 중력 보상을 더합니다.
물리 총질량은 0.043384kg입니다. 모터 추력 상한 0.289N, RPM 시상수 0.050초입니다.

Phase14 보상: 축별 exponential 위치, XY 복귀 속도 오차,
**Z 절대 속도 `-0.015*vz^2`**, roll/pitch `-0.075*angle_rad^2`,
yaw·body 각속도·실패 항의 합. dt 배율 없음.
정확한 설정은 `models/phase14/reward_config.json`, 구현은 `scripts/phase14z_reward.py`입니다.
Task의 기본 reward A는 유지하며 전용 실행기가 phase14 보상을 명시적으로 적용합니다.

## 기존 단계별 코드

단계별 스크립트는 연구 재현을 위해 보존했습니다. 일부 `prepare_*`, `report_*`,
`analyze_*`는 당시의 로컬 runs 및 과거 체크포인트를 요구합니다.
그 자료는 이번 저장소에 포함하지 않았으므로 새 clone에서 모든 과거 실험 명령이
즉시 동작하는 것은 아닙니다. 저장 모델의 이식 가능한 진입점은 `evaluate_baseline.py`입니다.
학습 스크립트를 실행하면 실제 학습이 시작되므로 재생에는 사용하지 마십시오.
기체 설명과 코드 보존 범위는 `docs/ASSETS.md`, `docs/SOURCE.md`를 참고하십시오.

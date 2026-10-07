# 고정 payload 모멘트 감사 — 2026-10-01

후속 상태: 이 문서는 **수정 전 실행의 감사 기록**이다. 이후 새 속도 규약 학습 작업에서
COM/전체 관성의 엔진 반영 및 명시적 중력 토크 제거를 함께 적용했다.
현재 구현은 [POSITION_VELOCITY_ERROR.md](POSITION_VELOCITY_ERROR.md)를 참고한다.

결론: 현재 checkout + MuJoCo 3.12.0 + cf21B_500.xml에서는 payload 중력 모멘트가
두 번 작용하지 않았다. 그러나 올바른 COM 이동 모델도 아니다. `_set_com_bias`가
`body_ipos`를 바꾸고 MuJoCo 파생 상수를 갱신하지 않아, 실제 `data.xipos`는 기체 원점에
남는다. 현재 pitch 부하는 `step()`에서 별도로 더하는 `r × mg` 토크로 발생한다.
`mj_setConst`만 추가하면 실제 COM 이동과 명시적 payload 토크가 중복된다.

이번 작업은 감사다. 환경 동역학, 설정 YAML, 체크포인트, 기존 결과는 수정하지 않았다.
감사 산출물과 문서만 추가/보완했다. 이전 설명에서 `body_ipos` 값을 실제 적용된 COM으로
표현한 것은 불충분했다. `data.xipos`, 중력 일반화 힘과 정적 평형까지 확인해야 한다.

## 확인한 실행과 reset

사용자 요청 직전의 최근 대화형 실행:
`artifacts/runs/ppo_e2e_hover_interactive-motor-effectiveness_seed42_20261001-141021`.
53.31초, max_tilt 종료. 학습이 아니라 실제 E2E 대화형 평가다.
보관 resolved config 및 runtime metadata의 값은 **5 g, [0.02,0] m**였다.
요청한 [0.03,0]과 다르므로 기존 로그를 3 cm 실행으로 해석하지 않았다.

| 항목 | 기존 실행 재구성: 2 cm | 독립 검사: 3 cm |
|---|---:|---:|
| payload.mass (kg) | .005 | .005 |
| payload.offset (m) | [.02,0] | [.03,0] |
| reset 후 body_mass[drone] (kg) | .04838 | .04838 |
| reset 후 body_ipos[drone] (m) | [.002066969822,0,0] | [.003100454733,0,0] |
| 실제 xipos−xpos, 수평 자세 (m) | [0,0,0] | [0,0,0] |
| body_sameframe | 1 | 1 |
| cached body_subtreemass (kg) | .043384 | .043384 |
| 실제 body 질량 합 (kg) | .048384 | .048384 |

기존 실행 당시 `xipos`는 CSV에 없었다. 표의 xipos 및 sameframe은 보관 설정을
복원한 현재 환경의 reset 결과다. 기존 runtime에 해시가 기록된 실행 모듈 4개는 모두
현재 파일과 일치했고, 기존 CSV의 첫 10개 action을 재생한 위치 기록 오차는 **0**이었다.
기존 body_mass/body_ipos는 원래 runtime metadata와도 일치한다.

XML 기본 drone body 질량은 .04338 kg이며, 별도 prop body 네 개가 각각 1e-6 kg이다.
초기 모터 추력은 기존 adapter 규약대로 drone body 질량만 사용하므로
각 .11865195 N이다. 자식 prop까지 포함한 정확한 정적 균등 추력은 .11866176 N이다.
이 9.81e-6 N/모터 차이는 payload 중력 토크 중복 문제와 별개다.

## 힘과 모멘트의 기준점

- `environment.py:_set_com_bias`는 질량, body_ipos, 대각 관성을 대입한다.
  `mj_setConst` 호출은 없고, 컴파일 당시 inertial/body frame이 같다는 최적화 상태가 남는다.
- 실제 로터 힘은 XML motor0…3 site의 body +Z 방향이다. 물리적 roll/pitch 모멘트는
  `Σ (r_i−r_C) × f_i`로 COM 기준으로 표현할 수 있다. 반력 yaw 토크는 순수 토크다.
- 로그 `wrench_actual`은 allocator B를 통한 **기체 원점 기준 로터 wrench**다.
  COM 기준 순 모멘트가 아니며, 중력이나 xfrc 추가 토크를 포함하지 않는다.
  allocator arm=.035355 m, XML site arm=.03536 m라서 정확한 물리 모멘트와 약 .0141% 차이가 있다.
- `step()`은 `cross(R @ offset, [0,0,-m_w*g])`를 계산하고
  `xfrc_applied[drone,3:6] = R @ dist_torque_body + torque_com_world`로 대입한다.
  adapter는 dist_torque_body를 0으로 초기화했다. 따라서 이 실행의 추가 토크는 payload 항뿐이다.
- `xfrc_applied`는 world frame이며 MuJoCo는 힘을 해당 body's xipos에 적용한다.
  여기서는 force 부분이 0이고 pure torque만 있으므로 기준점을 옮겨도 그 토크 벡터는 같다.
  이 토크의 **수식 자체는 기체 원점에서 payload 중력이 만드는 모멘트**다.

실제 전체 COM C로 계산할 때 균일 중력의 합 모멘트는 0이다. 대신 원점 O의 로터 모멘트를
`τ_rotor,C = τ_rotor,O − c × F_rotor`로 옮겨야 한다. 올바른 COM 이동 후의 이 효과와
원점 기준 `r_w × m_w*g`를 별도 토크로 동시에 넣으면 정적 호버에서 같은 부하를 중복한다.
MuJoCo의 일반화 힘은 Jacobian으로 합산되므로 COM 모멘트 로그와 혼용하면 안 된다.
아래 수평/정지 검사에서는 freejoint의 rotational generalized force를 원점 O 기준으로 비교했다.

MuJoCo 3.12.0의 공식 구현도 확인했다:
[관성 프레임 위치 생성과 body_sameframe](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_core_smooth.c),
[sameframe·subtree mass 재계산](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_setconst.c),
[xfrc의 적용점은 xipos](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_support.c).

## 5 g, +X 3 cm 독립 정적 계산

수평 자세, 모든 효율=1, 속도/각속도=0, 접촉 없음, 별도 교란 없음 기준이다.

```
m0 = .04338 kg, mw = .005 kg, d = .03 m, g = 9.81 m/s²
body COM x = mw*d/(m0+mw) = .003100454733 m
T_total = (.04338 + .005 + 4e-6)*9.81 = .47464704 N
payload 중력 모멘트, 원점 기준: τg,y = mw*g*d = +.0014715 N·m
필요 로터 모멘트: τrotor,y = −.0014715 N·m
```

XML의 x arm a=.03536 m를 사용하면
`τrotor,y = −a(f1−f2−f3+f4)`이다. roll과 yaw를 0으로 두는 해는:

| 로터 | 정상 물리 모델의 정적 추력 (N) |
|---|---:|
| 1 (+X) | .129065464751 |
| 2 (−X) | .108258055249 |
| 3 (−X) | .108258055249 |
| 4 (+X) | .129065464751 |

같은 두 +X/−X 쌍 안에서 반력 토크 방향이 반대이므로 yaw 토크는 상쇄된다.
이것은 정책 추론 결과가 아니라 독립 계산한 정적 평형이다. actuator dynamics를 우회해
효율 적용 후 실제 힘/반력 torque ctrl에 직접 입력하고 `mj_forward`로 검증했다.

## 격리 모델 검사: 상수 갱신과 추가 토크를 분리

검사 모델만 수정했으며 실제 환경 코드에는 변경하지 않았다.
`qvel=0`, 수평 자세에서 `qfrc_bias`, `qfrc_actuator`, `qfrc_smooth`, `qacc`를 비교했다.

| 조건 | 엔진 중력 pitch 모멘트 | 명시적 추가 토크 | 위 정적 추력에서 pitch 각가속도 |
|---|---:|---:|---:|
| 현재 상수, 추가 토크 없음 | 0 | 0 | −52.5620 rad/s² |
| **현재 실행 경로** | 0 | +.0014715 N·m | 약 0 |
| mj_setConst 후, 추가 토크 없음 | +.0014715 N·m | 0 | 약 0 |
| mj_setConst 후, 추가 토크 유지 | +.0014715 N·m | +.0014715 N·m | +52.5619 rad/s² |

상수 갱신 후에는 sameframe=3, xipos−xpos=[.003100454733,0,0], subtree mass=.048384가 됐다.
상수 갱신 + 추가 토크 유지 조건은 로터 모멘트를 −.002943 N·m로 **두 배**로 해야 평형이다.
그때 추력은 [.139469169502,.097854350498,.097854350498,.139469169502] N이다.
10개 정적 조합에서 예상 힘·모멘트 잔차 및 평형 시 각가속도를 assertion으로 확인했다.

## 실제 CSV와 비교

기존 2 cm 로그와 새 3 cm 실제 PPO 평가에 같은 near-hover 필터를 적용했다:
speed norm<.02 m/s, angular speed norm<.1 rad/s, |roll,pitch|<2°, 모든 효율=1.
연속 정착 구간을 뜻하지 않으며 3 cm 실행은 특히 짧은 과도응답이다.

| 항목 | 기존 2 cm | 새 3 cm, 2초 평가 |
|---|---:|---:|
| 선택 표본 수 | 1347 | 54 |
| 선택 표본의 시간 범위 | .80…14.72 s | .72…2.00 s |
| 정적 예상 로터 pitch 모멘트 (N·m) | −.000981 | −.0014715 |
| 실제 추력×XML arm의 평균 모멘트 (N·m) | −.000981386472 | −.001480343597 |
| 기존 wrench_actual의 평균 pitch (N·m) | −.000981247702 | −.001480134273 |
| 실제 총추력 평균 (N) | .474698955008 | .475605429624 |

3 cm의 평균 추력은 [.129232636203,.108532205221,.108338049223,.129502538977] N이다.
정적 해와 작은 차이는 남아 있으며, 2초 PPO 과도응답을 완전 정적 평형이라고 주장하지 않는다.
현재 경로의 한 번짜리 모멘트 규모와 일치하고 두 배 모멘트 규모와는 다르다.
CSV에는 xfrc 자체가 없으므로 이를 실측 저장값이라고 하지 않았다. 대신 기존 첫 10개
action의 정확한 재생에서 xfrc가 별도로 발생함을 확인했다. 첫 제어 구간 마지막 physics step의
추가 pitch 토크는 .000980998864 N·m(거의 수평인 2 cm 조건)였다.
CSV의 추력은 마지막 physics substep, 자세는 post-state여서 완전한 동시각 동적 토크 수지는
기존 CSV만으로 복원할 수 없다. 정적 프로브는 이 시간 혼합 없이 독립 검증했다.

새 실제 PPO 결과:
`artifacts/runs/ppo_e2e_hover_interactive-motor-effectiveness_seed42_20261001-142844`.
200 step/2초 완료, duration에 의한 truncated=true, terminated=false. 학습/optimizer update 없음.

## 산출물과 수정 시 주의점

감사 결과 디렉터리: `artifacts/runs/payload-moment-audit-zlb8dx47`.
`audit.json`, `static_probes.csv`, 보관 설정에서 복원한 YAML 2개, 재현 스크립트 `reproduce.py`.
실행하려면 저장소 루트에서:

```bash
MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python artifacts/runs/payload-moment-audit-zlb8dx47/reproduce.py
```

새 결과 디렉터리를 생성하고 별도 2초 PPO 평가를 실행한다. 기존 실행 14개 파일,
checkpoint, 입력 YAML, environment.py의 총 17개 파일 해시 불변을 확인했다.

물리 모델 수정 시에는 **질량/COM 변경 후 파생 상수 갱신**과 **명시적 payload 중력 토크 제거**를
함께 다뤄야 한다. 별도 dist_torque_body까지 제거하면 안 된다.
현재 `_set_com_bias`는 대각 관성만 갱신하는 근사도 포함하므로, 임의 xy offset의 비대각
관성까지 정확히 모델링하는 문제는 별도다. 이번 감사에서 정책 성능 원인을 단정하지 않았고,
과거 학습/평가 결과를 수정된 물리 모델의 결과로 바꾸지 않았다.

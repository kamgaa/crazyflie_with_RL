# Legacy artifact inventory

기존 산출물은 읽기 전용 legacy 자료로 취급하며 이번 작업에서 삭제·이동·이름 변경하지 않는다. 아래 hash는 작업 전 기준선이다.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `model/ppo_best.zip` | 157,102 | `9e66b29545b18495288078d0512efff1b82187b9dcdfee302c4c179430be61d9` |
| `model/ppo_residual_cf.zip` | 157,106 | `9e1601eca3179ae6a1e3a904a3fd94ecb8a7a8bfb5e0fd94240a31d17ee0cc01` |
| `traj_my-case-1_floor.png` | 63,539 | `1cc221038c9c668f81bf1d4103b3808c2c8c19152ec01b10ef071c30286ce38e` |
| `traj_my-case-1_residual.png` | 68,298 | `b26cc04e51d0fd93f866fb80b486cc070bc01e14d8e146cb5b5517205669f070` |

`tb/`에는 19개 event 파일, 총 3,619,077 bytes가 있다. 신뢰할 수 있는 run별 condition/seed 매핑이 없으므로 자동 이동이나 병합을 하지 않는다.

두 ZIP의 archive metadata:

- observation shape `(15,)`, action shape `(4,)`
- Stable-Baselines3 `2.9.0`, Gymnasium `1.3.0`, NumPy `1.26.4`, Python `3.10.20`
- `ppo_best.zip`: 820,000 timesteps
- `ppo_residual_cf.zip`: 1,001,472 timesteps
- 저장된 seed는 `null`

두 mode가 모두 15차원 observation을 사용하므로 shape만 보고 control mode를 추론할 수 없다. 파일명도 충분한 provenance가 아니므로 새 manifest에 사실처럼 복사하지 않는다.

## 현재 BLDC plant와의 경계

위 산출물은 `master@6dd7ddb`의 instantaneous force/torque plant에서 만들어진
역사적 자료다. 현재 모든 shipped profile은 500 Hz에서 동작하는 required
`cf21b_first_order` BLDC model을 사용하므로, ZIP의 observation/action shape가
맞더라도 현재 plant에서의 행동은 legacy 실행과 동등하지 않다. ZIP 안에는
actuator model·계수·검증 상태 provenance가 없으므로, 이를 현재 평가에 쓸 때는
새 run의 resolved config/manifest를 별도로 남기고 legacy 입력 provenance를
`unverified`로 기록한다.

# Legacy migration inventory

이 문서는 새 리소스·산출물 구조로 옮길 수 있는 기존 파일의 읽기 전용 목록이다. 아래 파일은 사용자 승인 전 이동·이름 변경·삭제하지 않는다. SHA-256은 현재 파일을 보존·검증하기 위한 값이다.

## Checkpoints

두 ZIP 모두 Stable-Baselines3 2.9.0 모델이며 action shape는 `(4,)`, observation shape는 `(15,)`다. 현재 저장소의 E2E 활성 학습 코드와 ZIP 내부 메타데이터를 함께 보면 E2E 후보이며, residual 13차원 프로파일에서는 사용할 수 없다. 원본 ZIP에는 제어 모드 manifest가 내장되어 있지 않았지만, 현재는 검증된 shape와 provenance를 기록한 `model/*.manifest.json` sidecar가 함께 있다. ZIP 내부에는 과거 TensorBoard 절대경로 `/home/mrl_6534/gwpark/crazyflie_RL/tb`가 메타데이터로 남아 있다.

| Current path | Bytes | Timesteps | SHA-256 | Proposed root after approval |
|---|---:|---:|---|---|
| `model/ppo_best.zip` | 157102 | 820000 | `9e66b29545b18495288078d0512efff1b82187b9dcdfee302c4c179430be61d9` | `resources/pretrained_models/e2e/` |
| `model/ppo_residual_cf.zip` | 157106 | 1001472 | `9e1601eca3179ae6a1e3a904a3fd94ecb8a7a8bfb5e0fd94240a31d17ee0cc01` | `resources/pretrained_models/e2e/` |

이전 승인 후에도 과거 이름만 보고 제어 모드를 추정해서는 안 된다. 함께 제공된 sidecar와 archive shape를 모두 검증해야 하며, provenance를 확정할 수 없다면 legacy/unverified 상태로 유지해야 한다. 과거 seed와 정확한 실험 조건은 저장되지 않았으므로 새 명명 규칙에 맞추기 위해 값을 추측하지 않는다.

## Root plots

| Current path | Bytes | SHA-256 | Proposed location after approval |
|---|---:|---|---|
| `traj_my-case-1_floor.png` | 63539 | `1cc221038c9c668f81bf1d4103b3808c2c8c19152ec01b10ef071c30286ce38e` | `artifacts/runs/<verified-legacy-run-id>/plots/` |
| `traj_my-case-1_residual.png` | 68298 | `b26cc04e51d0fd93f866fb80b486cc070bc01e14d8e146cb5b5517205669f070` | `artifacts/runs/<verified-legacy-run-id>/plots/` |

## TensorBoard event files

19개 파일, 총 3,619,077 bytes다. 실행 조건과 seed의 신뢰 가능한 매핑이 없으므로 승인 후에도 임의로 하나의 run으로 합치지 않는다.

| Current path | Bytes | SHA-256 |
|---|---:|---|
| `tb/PPO_1/events.out.tfevents.1783584217.MRL-server.109104.0` | 176149 | `00e581a7009d94b081bf1a6c88aa7ee6c92c513be289086153a7a0e9edc1f2e3` |
| `tb/PPO_2/events.out.tfevents.1783585643.MRL-server.473039.0` | 302629 | `5e8d2963af91b2832a67b2d6b5d7230561dd4bba19a6bd78ff806eac24c29dac` |
| `tb/PPO_3/events.out.tfevents.1783587887.MRL-server.1100361.0` | 302629 | `aaec745fdb13971c6fb664f70c5222726cc6bc903ee34148ba4528b7d6af2d38` |
| `tb/PPO_4/events.out.tfevents.1783601938.MRL-server.1727176.0` | 302629 | `ac86b1281d1220323dbf8420d4ff28c36c2b3b8d8727344d7fe7d8231273543a` |
| `tb/PPO_5/events.out.tfevents.1783646842.MRL-server.2374538.0` | 81909 | `f897f3bb97bd693f33a73375b421701395c40fd612abd000ae8d92387bf68d21` |
| `tb/PPO_6/events.out.tfevents.1783647321.MRL-server.2544341.0` | 186689 | `da080e01ebff5f71f9c015d8da4d808291fa7d91bb822638dc97043d1f3ab551` |
| `tb/PPO_7/events.out.tfevents.1783648410.MRL-server.2931981.0` | 302629 | `90c70137838082fc59f70b9c23a8a862ef6f72182605c7af0f6c4e7f55dd8198` |
| `tb/PPO_8/events.out.tfevents.1783905772.MRL-server.3635028.0` | 9369 | `030b90e54aface5fa596047ddc07112a6a5f9f9c6dd18a033e8b08f6f57e1822` |
| `tb/PPO_9/events.out.tfevents.1783931145.MRL-server.3663062.0` | 135 | `4909bb714c777c1a604ded762b67e2a384509c93b5f0117987d6dbe0ee774d3e` |
| `tb/PPO_10/events.out.tfevents.1783931200.MRL-server.3664431.0` | 74469 | `1dcd8f6095c316999d09e6ffcf45f71e95c9f64404d1fa4e4ed187d23acc0766` |
| `tb/PPO_11/events.out.tfevents.1783931853.MRL-server.3817814.0` | 302629 | `11a50effdbec28951a06156c77a0fb9a1deb393c309773965f356a4d1755aba9` |
| `tb/PPO_12/events.out.tfevents.1783939501.MRL-server.253118.0` | 26109 | `e09721e06d4cbd5f9e269caf07bb593e12d4bc3ee1169f1ffa92486ca753adec` |
| `tb/PPO_13/events.out.tfevents.1783939722.MRL-server.308374.0` | 302629 | `c2a5c43fbce43227031886b56938ecb0e4179fd8a33d4c28881ddd5b3bbe9448` |
| `tb/PPO_14/events.out.tfevents.1783942133.MRL-server.930544.0` | 302629 | `76a85d538c764dc138551ff6440beea7dfeba237efb7b28b9cd110d55ed65d2a` |
| `tb/PPO_15/events.out.tfevents.1783946514.MRL-server.1559345.0` | 12469 | `583a918e689625833fd39472d38c96de080779582b2786dae0b37fd042c13ff7` |
| `tb/PPO_16/events.out.tfevents.1783946790.MRL-server.1586438.0` | 25489 | `a272e092406ed9eb41893e59ba143329e824ba0bf64eaf622abeb1ea1df6e705` |
| `tb/PPO_17/events.out.tfevents.1783947085.MRL-server.1640415.0` | 302629 | `435218bef418981147d21b57c6f62eca9ac2943fa8ef77a4397894422fbe3051` |
| `tb/PPO_18/events.out.tfevents.1785896591.MRL-server.110808.0` | 302629 | `ef6ce524a9f6ece28ffba8b73dd6c08af922ed29d5e529b49e2ff4e64c33118e` |
| `tb/PPO_19/events.out.tfevents.1785917046.MRL-server.748927.0` | 302629 | `40441f544c21f1f9a0a72b15a29e820671039515d34681513883dc746e4ce6eb` |

## Approval checklist

이전하기 전에 다음을 사용자와 확정한다.

1. 각 파일의 실제 control mode, 조건, seed와 생성 날짜
2. 체크포인트의 manifest 내용과 최종 이름
3. TensorBoard 파일별 run 경계
4. 복사 또는 이동 여부와 원본 보존 기간
5. 이전 전후 SHA-256 일치 여부

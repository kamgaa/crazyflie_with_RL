# A/B 신규 학습과 공통 평가

이 작업은 [비학습 진단](VELOCITY_AB_DIAGNOSIS.md) 이후 사용자가 승인한 A/B 학습이다.
기존 정상 nominal 정책과 D의 체크포인트/결과는 보존한다. C/D 재학습, 추가 seed,
게인·가중치·학습률 튜닝, commit/push는 포함하지 않는다.

## 실행 설정

| 항목 | A | B |
|---|---|---|
| 프로파일 | `configs/e2e_train_velocity_ab_a.yaml` | `configs/e2e_train_velocity_ab_b.yaml` |
| 관측 속도 | actual world velocity | actual world velocity |
| 속도 reward | `-.005 * ||v||²` | `-.005 * ||v-v_des||²` |
| 초기화 | fresh PPO, seed 42 | fresh PPO, seed 42 |
| 요청 스텝 | 1,000,000 | 1,000,000 |

공통 v_des는 `norm_clip(-4*position_error,1.5)`다. 위치 XY/Z=10/6,
action scale=[.0075,.0075,.001,.5], 100/500 Hz, payload=0, 모터 효율=1,
초기 위치 각 축 Uniform(-.05,.05), 자세 perturbation 0, 추가 동역학 DR 없음.
관측 15/행동 4차원이며 적분기·history·GRU를 추가하지 않았다.
네트워크 [64,64], n_steps=2048, batch_size=256, n_epochs=10,
learning_rate=.0003 및 나머지 PPO 설정은 기존 값을 유지했다.

실행 직전 resolved 차이는 `environment.e2e_velocity.reward_mode`, `source_path`,
`experiment.condition`, `experiment.description` 네 개뿐이다. 메타데이터를 제외하면
reward_mode 하나만 다르다. 두 실험에 공통으로 episode 기록과 checkpoint_interval=100000을 켰다.
각각 독립 프로세스, OMP/MKL 1 thread로 실행한다. 기존 체크포인트를 로드하거나 resume하지 않는다.

두 초기 정책 가중치 SHA256은 동일하다:
`c1b5166c1350eb7ceb7b6bdfb8f07cc5f4b7d6ab559fbd71d528de475a3eb97c`.
첫 optimizer update 전 2048 step 안에 완료된 45개 episode도 대조했다.
속도 reward 합계와 return을 제외한 종료 시각·사유·최종 상태 지표·다른 reward 항은 정확히 일치했다.
이는 모든 step의 상태 로그를 대조했다는 뜻은 아니며, 비교 범위는 저장된 episode 기록이다.

## 추가된 학습 관측 기록

`training_observers.EpisodeCSVRecorder`는 학습 환경에만 적용한다.
관측·행동·reward·RNG를 바꾸지 않으며 DummyVecEnv가 자동 reset하기 **전** 데이터를 기록한다.

- `metrics/*training-episodes*.csv`: episode id, 시작/종료 global timestep, 길이/시간,
  terminated/truncated, physical_termination/time_limit_reached, 실제 종료 원인,
  최종 위치 오차/tilt, return 및 기존 reward 항별 합계.
- 물리 종료는 min_altitude/max_altitude/max_tilt/max_position_error로 구분하며,
  시간 제한도 함께 걸렸으면 두 플래그와 두 사유를 모두 보존한다.
- 학습 종료 시 미완료 episode는 `episode_ended=false, end_reason=collector_closed`로 저장한다.
  이 행은 물리 실패나 정상 시간 제한 종료로 세지 않는다.
- SB3 표준 `info['episode']`를 제공하여 TensorBoard `rollout/ep_len_mean`, `rollout/ep_rew_mean`도 기록한다.
- 기존 reward_terms/reward_raw/reward_costs/reward_fraction TensorBoard 집계를 유지한다.

`training.checkpoint_interval`은 기본 0이며 이번 A/B에서만 100000이다.
중간 모델은 직전 optimizer update가 끝난 rollout 시작 경계에서 저장한다.
예를 들어 첫 100000 임계점은 100352 스텝에 저장된다. 마지막 학습 update 이후에도
임계점을 확인하므로 끝의 약 100만 스텝 snapshot도 남는다. 실제 timestep을 파일명/manifest에 기록한다.
중간 모델은 `model_history`에 `kind=intermediate`로 추가하며 best/final 선택을 대체하지 않는다.

매 모델에 SHA256과 정규화 상태를 기록한다. 이번 PPO는 VecNormalize를 사용하지 않아
`normalization.enabled=false`다. 정규화가 있는 경우에는 해당 시점 통계를 별도 pkl로 저장하고,
로더가 선택한 checkpoint의 통계 경로를 사용하도록 보완했다.
공통 resolved config와 핵심 소스 해시도 run에 보존한다.

기존 best 조건은 `score < best_score AND tail_tilt_disqualifications == 0` 그대로다.
tail은 실제 episode 마지막 30%, tilt 탈락 기준 30°이며 환경 종료 기준 60°와 다르다.
중간 checkpoint 저장은 이 조건과 무관하다.

## 공통 최종 평가

`configs/eval_velocity_ab.yaml`과 기존 `compare_dr_policies.py`를 사용한다.
각 실험의 final과 생성된 best(최종 선정본), 그리고 `historical_nominal` 참고 모델을 비교한다.
이전 정상 정책을 새 A라고 표시하지 않는다. 각 케이스는 deterministic=True, seed=42로 독립 초기화한다.

- 케이스: nominal hover, step-005; 8초 horizon, p0=[0,0,1], level/yaw0, v/omega0.
- 같은 공중 hover motor equilibrium, 위치/자세/payload/actuator parameter 난수화 off.
- 초기 snapshot과 reference sequence 동일성 확인, 종료 플래그 존중, 자동 reset 없음.
- 모든 모델에 같은 absolute velocity 관측과 A 정의의 **평가 reward**를 적용한다.
  B의 학습 reward 정의는 모델의 학습 manifest에 별도로 남는다. 평가 return으로 순위를 매기지 않는다.
- 내부 속도 오차 지표는 같은 post-state 위치에서 생성한 v_des에 대한 `v-v_des`다.
  고정 미션 reference의 속도 0에 대한 기존 `velocity_rmse`와 구분한다.
- 전체 완료와 partial을 구분하고, 마지막 2초 위치/속도/내부 속도 오차 지표는 완료한 경우에만 계산한다.
  정착 미달/조기 실패는 null로 기록한다.
- 한 모델·한 케이스의 완료율은 1회의 고정 초기 상태 시험이다. 여러 학습 seed에 대한 성공 확률로 해석하지 않는다.

자세는 roll/pitch RMS·최대 절대각과 최대 tilt를 기록한다. 모터는 마지막 physics substep의
unclipped allocation `B_pinv @ wrench`, clipped thrust command, actual thrust, ESC command를 기록한다.
allocator clipping 비율과 ESC 0/1 경계 도달 비율, 실제 추력 하한/상한 도달 비율을 각각
전체/모터별로 요약한다. actuator 지연에 따른 command-actual 차이는 포화로 부르지 않는다.
현재 E2E의 요청 wrench/allocator command는 한 제어 구간 내 동일하며, actual thrust는 substep에 따라 변한다.

## 실행 및 산출물

학습 실행은 기존 trainer/ArtifactManager를 호출하는 orchestration으로 수행하며, 아래 CLI와 같은 설정이다.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
python train_ppo_02.py --config configs/e2e_train_velocity_ab_a.yaml

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
python train_ppo_02.py --config configs/e2e_train_velocity_ab_b.yaml
```

비교 기록: `artifacts/runs/velocity-ab-training-comparison-isygddsb`.

- A: `artifacts/runs/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_20261002-140542`
- B: `artifacts/runs/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_20261002-140542`

학습과 공통 비교 평가를 완료했다. 두 실험 모두 요청 1,000,000스텝,
실제 **1,001,472스텝**이며 n_steps=2048의 마지막 rollout을 마친 결과다.
각각 주기 평가 50회와 best 여부와 무관한 중간 체크포인트 10개를 기록했다.
중간 저장 timestep은 100352, 200704, 301056, 401408, 501760,
600064, 700416, 800768, 901120, 1001472다.

| 모델 | 최종 선정 best timestep | final timestep |
|---|---:|---:|
| A | 980000 | 1001472 |
| B | 960000 | 1001472 |

각 모델 디렉터리에 이전 best 선정본도 보존돼 있다. 이번 고정 시험은 manifest의
마지막 선정 best와 final을 비교한다. 모델 경로와 SHA256은
[completion.json](../artifacts/runs/velocity-ab-training-comparison-isygddsb/completion.json)에 모두 기록했다.

- [A best](../artifacts/runs/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_best_20261002-140542-10.zip) — `artifacts/runs/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_best_20261002-140542-10.zip`
- [A final](../artifacts/runs/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_final_20261002-140542.zip) — `artifacts/runs/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_final_20261002-140542.zip`
- [B best](../artifacts/runs/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_best_20261002-140542-10.zip) — `artifacts/runs/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_best_20261002-140542-10.zip`
- [B final](../artifacts/runs/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_final_20261002-140542.zip) — `artifacts/runs/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_final_20261002-140542.zip`

## 완료된 공통 평가

결과 디렉터리는 `artifacts/runs/dr-transfer-a7_icqln`다.
다섯 모델 × 두 케이스, 총 10회 모두 8초 horizon을 완료했다.
각 행의 완료율은 **1/1 (100%)**, `terminated=false, truncated=true, partial=false`다.
시간 제한 종료이며 물리 종료는 없다. 이는 모델당 케이스별 한 번의 결정론적 시험이다.
정착은 위치 norm≤5mm, 속도 norm≤0.02m/s를 그 시점부터 종료까지 계속 만족하고
그 구간이 최소 1초일 때만 유효하다. 호버의 정착 0초는 초기 상태부터 조건을 유지했다는 뜻이다.

각 모델·케이스는 고정 초기 상태 1회 평가이며, partial은 실제 관측된 구간만의 지표다. 속도 오차는 v - norm_clip(-4 e_p,1.5)다.

| 모델 | 케이스 | 완료 | 구간 | 시간 s | 종료 | XY RMSE m | Z RMSE m | 3D RMSE m | 정착 s | 속도 RMS m/s | 속도 오차 RMS m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_final | hover | True | full | 8.000000 | horizon | 0.002822 | 0.002596 | 0.003834 | 0.000000 | 0.001767 | 0.015707 |
| A_best | hover | True | full | 8.000000 | horizon | 0.001841 | 0.001246 | 0.002223 | 0.000000 | 0.001139 | 0.009117 |
| B_final | hover | True | full | 8.000000 | horizon | 0.005182 | 0.004228 | 0.006688 | null | 0.003284 | 0.027411 |
| B_best | hover | True | full | 8.000000 | horizon | 0.001731 | 0.001788 | 0.002489 | 0.000000 | 0.001780 | 0.010258 |
| historical_nominal | hover | True | full | 8.000000 | horizon | 0.001910 | 0.000267 | 0.001928 | 0.000000 | 0.000585 | 0.007878 |
| A_final | step-005 | True | full | 8.000000 | horizon | 0.011152 | 0.003064 | 0.011565 | 3.060000 | 0.022800 | 0.037709 |
| A_best | step-005 | True | full | 8.000000 | horizon | 0.011111 | 0.002382 | 0.011364 | 1.460000 | 0.021521 | 0.035861 |
| B_final | step-005 | True | full | 8.000000 | horizon | 0.011496 | 0.004055 | 0.012190 | null | 0.024200 | 0.041744 |
| B_best | step-005 | True | full | 8.000000 | horizon | 0.011324 | 0.001629 | 0.011440 | 1.520000 | 0.021454 | 0.036206 |
| historical_nominal | step-005 | True | full | 8.000000 | horizon | 0.010982 | 0.001119 | 0.011038 | 1.440000 | 0.020399 | 0.033485 |

자세 단위는 도(deg), 모터 지표는 샘플 비율(0–1)이다.

| 모델 | 케이스 | roll RMS deg | pitch RMS deg | 최대 tilt deg | allocator clip | ESC lower | ESC upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_final | hover | 0.040452 | 0.027047 | 0.166341 | 0.000000 | 0.000000 | 0.000000 |
| A_best | hover | 0.027662 | 0.018695 | 0.115009 | 0.000000 | 0.000000 | 0.000000 |
| B_final | hover | 0.019760 | 0.077218 | 0.326278 | 0.000000 | 0.000000 | 0.000000 |
| B_best | hover | 0.004202 | 0.044871 | 0.177716 | 0.000000 | 0.000000 | 0.000000 |
| historical_nominal | hover | 0.003676 | 0.008399 | 0.048763 | 0.000000 | 0.000000 | 0.000000 |
| A_final | step-005 | 0.190397 | 0.620522 | 2.609696 | 0.000000 | 0.000000 | 0.000000 |
| A_best | step-005 | 0.206520 | 0.567746 | 2.532797 | 0.000000 | 0.000000 | 0.000000 |
| B_final | step-005 | 0.192140 | 0.604036 | 2.794228 | 0.000000 | 0.000000 | 0.000000 |
| B_best | step-005 | 0.144056 | 0.550548 | 2.500187 | 0.000000 | 0.000000 | 0.000000 |
| historical_nominal | step-005 | 0.057602 | 0.528473 | 2.383515 | 0.000000 | 0.000000 | 0.000000 |

마지막 2초 지표는 8초를 완료한 경우에만 계산한다.

| 모델 | 케이스 | 위치 RMSE m | 실제 속도 RMS m/s | 속도 오차 RMS m/s |
| --- | --- | --- | --- | --- |
| A_final | hover | 0.004082 | 0.000194 | 0.016342 |
| A_best | hover | 0.002351 | 0.000142 | 0.009417 |
| B_final | hover | 0.007041 | 0.000032 | 0.028163 |
| B_best | hover | 0.002417 | 0.000012 | 0.009666 |
| historical_nominal | hover | 0.002110 | 0.000001 | 0.008439 |
| A_final | step-005 | 0.004068 | 0.001385 | 0.016253 |
| A_best | step-005 | 0.002360 | 0.001370 | 0.009418 |
| B_final | step-005 | 0.007042 | 0.000358 | 0.028175 |
| B_best | step-005 | 0.002412 | 0.000219 | 0.009650 |
| historical_nominal | step-005 | 0.002110 | 0.000051 | 0.008439 |

A/B best는 이 두 시험을 모두 완주하고 정착했다. 5cm에서 A best의 전체 RMSE는
0.011364m, B best는 0.011440m로 가까웠고, 이번 한 seed·고정 초기 조건만으로
보상 정의의 일반적인 우열을 확정하지 않는다. 과거 nominal은 별도 참고이며 새 A가 아니다.

B final은 완주했지만 호버·5cm 모두 정착 미달이다. 5cm 마지막 2초 RMSE는
0.007042m이며 평균 오차 xyz=[0.004312,-0.003305,0.004480]m,
표준편차 xyz=[0.000026,0.000058,0.000007]m로, 이 tail의 오차는 주로 평균 offset이다.
B best의 같은 RMSE는 0.002412m다. final이 best보다 항상 낫지는 않았다.
원인을 추가로 확정하거나 가중치를 조정하지 않았다.

모든 고정 시험에서 allocator clipping, ESC 하한/상한, 실제 추력 하한/상한 도달 비율은 0이다.
이는 기록한 제어 구간 마지막 physics substep 신호 기준이며 장시간·큰 목표 이동의 포화 없음까지 뜻하지 않는다.
조기 종료가 없어서 이번 10행은 모두 full 지표다. 조기 종료의 tail/settling null 처리는 단위 검증으로 확인했다.

- [전체 summary JSON](../artifacts/runs/dr-transfer-a7_icqln/summary.json), [summary CSV](../artifacts/runs/dr-transfer-a7_icqln/summary.csv), [manifest](../artifacts/runs/dr-transfer-a7_icqln/manifest.json)
- [호버 그래프](../artifacts/runs/dr-transfer-a7_icqln/hover-comparison.png), [5cm 그래프](../artifacts/runs/dr-transfer-a7_icqln/step-005-comparison.png)
- [학습 episode·주기 평가 이력](../artifacts/runs/velocity-ab-training-comparison-isygddsb/training_episode_history.png)
- [학습 구간별 CSV](../artifacts/runs/velocity-ab-training-comparison-isygddsb/training_episode_windows.csv), [상세 학습 기록 요약](../artifacts/runs/velocity-ab-training-comparison-isygddsb/training_episode_summary.json)

## 학습 기록에서 확인한 사항

| 기록 | A | B |
|---|---:|---:|
| 완료된 학습 episode | 9972 | 11961 |
| max_tilt 종료 | 9263 | 11327 |
| max_position_error 종료 | 3 | 1 |
| 시간 제한 종료 | 706 | 633 |
| 학습 종료 시 미완료 collector_closed 기록 | 1 | 1 |

이 숫자는 탐색 행동을 사용하는 **학습 전체 기간**의 집계이며 최종 결정론적 정책 완료율이 아니다.
처음 30만 스텝 동안 두 실행의 완료 episode는 모두 물리 종료였다.
60만 이후 종료 timestep으로 분류한 10만 스텝 구간에서는 두 실행 모두 완료 episode가
8초 시간 제한에 도달했다. 각 1개의 마지막 미완료 episode를 포함한 기록 길이 합계가
각각 실제 학습 timestep 1,001,472와 정확히 일치한다.

기존 best 조건 첫 통과는 A 36만, B 44만 스텝이었다.
B는 48만 평가에서 tail tilt 탈락 17/30회로 다시 나빠졌다가 50만부터 탈락 0회를 유지했다.
주기 평가 로그·TensorBoard·항별 reward 합계·중간 모델로 해당 변화를 추적할 수 있다.
학습 return 합계를 서로 다른 reward 정의 사이의 성능 순위로 사용하지 않았다.

## 검증 및 재실행

관련 범위 **98 passed, 1 skipped**. 전체 저장소 테스트 통과 주장과는 다르다.
미실행 1개는 별도 환경 변수로 활성화하는 기존 선택적 replay 테스트다.
실제 체크포인트 평가 10회 및 첫 중간 체크포인트 두 개의 재로딩 검증은 별도로 완료했다.

- observer wrapper 전후 관측·reward·RNG/물리 진행의 회귀, 실제 종료/시간 제한/미완료 기록 구분.
- intermediate 저장 시점, best/final 분리, normalization checkpoint별 저장·로딩 계약.
- A/B 의미 있는 설정 차이가 reward_mode 하나임을 확인; 동일 초기 정책 해시 확인.
- 첫 optimizer update 전 완료 45개 episode의 기록은 속도 reward와 return을 제외하고 동일.
- 최종 다섯 정책의 케이스별 초기 snapshot과 참조 sequence 동일; native absolute 관측 유지.
- CSV에서 위치/속도/내부 속도 오차 RMS를 독립 재계산: 최대 차이 5.21e-18.
- 학습 중 고정한 소스/설정 해시 일치, 새 모델 전체 SHA256 일치, 과거 nominal·D 관련 169개 파일 해시 불변.

검증 산출물은 [evaluation_verification.json](../artifacts/runs/velocity-ab-training-comparison-isygddsb/evaluation_verification.json),
[final_integrity_check.json](../artifacts/runs/velocity-ab-training-comparison-isygddsb/final_integrity_check.json) 및 같은 디렉터리의 테스트 로그에 있다.

동일한 저장 모델로 평가만 다시 실행하려면 저장된 정확한 명령을 사용한다.
새 고유 결과 디렉터리가 생성되며 학습은 실행하지 않는다.

```bash
bash artifacts/runs/velocity-ab-training-comparison-isygddsb/rerun_evaluation.sh
```

실제 호출은 다음과 같다.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
/home/kenneth/miniconda3/envs/crazyflie_rl/bin/python compare_dr_policies.py \
  --config configs/eval_velocity_ab.yaml \
  --model A_final=/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_final_20261002-140542.zip \
  --model A_best=/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_best_20261002-140542-10.zip \
  --model B_final=/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_final_20261002-140542.zip \
  --model B_best=/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_best_20261002-140542-10.zip \
  --model historical_nominal=/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip \
  --cases hover step-005 --seed 42
```

이번에는 추가 seed, C/D 재학습, 0.5m·원 궤적 추가 평가, gain/가중치/학습률 튜닝,
commit/push를 수행하지 않았다. 보상 변경의 일반화 효과나 D 실패의 단일 원인을 확정하지 않는다.


수정 경로: config/training/artifacts/dr_policy의 학습 기록·checkpoint 지원,
새 `training_observers.py`, A 공통 로깅 설정(B 상속), 새 공통 평가 YAML,
dr_transfer의 자세/모터/완료율/마지막 2초 지표, 관련 테스트 및 이 문서.
기존 environment/actuator 물리 함수는 이번 작업에서 변경하지 않았다.

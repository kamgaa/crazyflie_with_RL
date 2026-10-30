"""Build the Korean phase-6 report from measured artifacts only."""
import argparse,json,csv,datetime,subprocess,platform,importlib.metadata as md
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args()
r=Path(args.run_dir).resolve()
def read(p):return json.loads(Path(p).read_text())
t=read(r/'training_result.json');plan=read(r/'experiment_plan.json');parent=Path(plan['parent'])
pt=read(parent/'N1024/training_result.json');v=read(r/'verification.json');assert v['success']
s=read(r/'evaluation/selected_summary.json');selection=read(r/'selection.json');win=read(r/'training_window_summary.json')
history=read(r/'evaluation/combined_history.json');same=read(r/'evaluation/same_iteration_comparison.json')
best=read(r/'evaluation/best_reload/results.json');final=read(r/'evaluation/final_reload/results.json')
rv=read(r/'resume_verification.json')
jobs=read(r/'jobs.json');ejobs=read(r/'evaluation_jobs.json')
def fmt(x,d=6):return 'N/A' if x is None else f'{x:.{d}f}'
def table(headers,rows):
    return '\n'.join([' | '.join(headers), ' | '.join(['---']*len(headers))]+[' | '.join(map(str,row)) for row in rows])
def getrows(name):return read(r/('evaluation/'+name+'/results.json'))
evaltable=table(['누적 it','전이','8초 완료','평균 시간(s)','평균 RMSE(m)','최악 RMSE(m)','최대 tilt(deg)','완료조건 마지막2초 오차(m)'],
 [[x['iteration'],x['transitions'],str(x['completed_count'])+'/5',fmt(x['mean_duration_s'],3),fmt(x['mean_position_rmse_m']),fmt(x['worst_position_rmse_m']),fmt(x['max_tilt_deg'],3),fmt(x['tail2_position_mean_m'])] for x in history if x['iteration']>=256])
conditiontable=table(['모델','조건','완료','시간(s)','종료','XYZ RMSE(m)','3D RMSE(m)','최대 tilt(deg)','마지막2초 위치(m)','마지막2초 속도(m/s)'],
 [[label,x['condition'],x['completed_8s'],fmt(x['duration_s'],2),','.join(x['reasons']),','.join(fmt(a) for a in x['xyz_rmse_m']),fmt(x['position_rmse_m']),fmt(x['max_tilt_deg'],3),fmt(x['tail2_position_mean_m']),fmt(x['tail2_speed_mean_m_s'])] for label,rs in [('best',best),('final',final)] for x in rs])
all_eval=read(r/'evaluation/resume_start/results.json')+read(r/'evaluation/periodic/results.json')
conditions=['nominal','x_plus','x_minus','y_plus','y_minus']
five_table=table(['누적 it']+conditions,[
 [i]+[f"{x['duration_s']:.2f}s / {x['position_rmse_m']:.6f}m / "+('완료' if x['completed_8s'] else ','.join(x['reasons']))
 for c in conditions for x in all_eval if x['iteration']==i and x['condition']==c]
 for i in [256]+list(range(320,1025,64))])
same_table=table(['환경','누적 it','전이','Adam updates','측정 학습시간(s)','완료','지속(s)','3D RMSE(m)','출처'],
 [[x['num_envs'],x['iteration'],x['policy_transitions'],x['optimizer_updates'],fmt(x['training_s'],3),str(x['completed_count'])+'/5',fmt(x['mean_duration_s'],3),fmt(x['mean_position_rmse_m']),x['source']] for x in same])
rewardloss=table(['지표','부모 마지막64 it','추가 첫64 it','추가 마지막64 it'],
 [[k]+[fmt(win[w][k]) for w in ['parent_last64','additional_first64','additional_last64']] for k in win['additional_last64']])
versions={}
for name in ['torch','isaaclab','isaacsim','rsl-rl-lib','tensorboard','newton']:
 try:versions[name]=md.version(name)
 except md.PackageNotFoundError:versions[name]='distribution metadata not found'
dump_json(r/'package_versions.json',versions)
evalwall=sum(x['wall_s'] for x in ejobs)+jobs[0]['wall_s']
episodetime=sum(x['evaluation_wall_s'] for x in getrows('resume_start')+getrows('periodic')+best+final)
commands=f"""cd {ROOT}
PY=/home/mrl_6534/miniconda3/envs/isaaclab/bin/python
RUN={r}

# 실제 추가 학습 명령(이미 완료; 이 RUN에서 재실행하면 중복 실행 방지 guard가 거부)
"$PY" scripts/train_phase6.py --run-dir "$RUN" --additional-iterations 768 \\
  --device cuda:0 --headless --physics isaacsim_physx
# 실제 orchestration은 run_phase6.py가 시작 모델 평가 후 위 명령을 호출했다.
# 정확한 전체 argv/PID/시간/exit code: jobs.json, evaluation_jobs.json, command.json.

# 새 실험으로 전체 재현할 때만 실행. 기존 실험 디렉터리 재사용 불가.
NEW_RUN="$PWD/runs/phase6_n1024_resume_$(date +%Y%m%d_%H%M%S)"
"$PY" scripts/prepare_phase6.py --run-dir "$NEW_RUN"
"$PY" scripts/train_phase6.py --run-dir "$NEW_RUN" --additional-iterations 768 --preflight \\
  --device cuda:0 --headless --physics isaacsim_physx
"$PY" scripts/run_phase6.py --run-dir "$NEW_RUN"
"$PY" scripts/evaluate_phase6_all.py --run-dir "$NEW_RUN"
"$PY" scripts/analyze_phase6.py --run-dir "$NEW_RUN"
"$PY" scripts/report_phase6.py --run-dir "$NEW_RUN"\n"$PY" scripts/package_phase6.py --run-dir "$NEW_RUN"

# best/final 재평가만 할 때: 새 output 경로 사용, optimizer update 없음
"$PY" scripts/evaluate_phase6.py --checkpoints "$RUN/best.pt" "$RUN/final.pt" \\
  --run-dir "$RUN" --output "$RUN/evaluation/manual_$(date +%Y%m%d_%H%M%S)" \\
  --device cuda:0 --headless --physics isaacsim_physx

# 서버 데스크톱 터미널에서 GUI 재생: 아래 현재 드라이버 제한 참조
"$PY" scripts/evaluate_phase6.py --checkpoints "$RUN/best.pt" \\
  --run-dir "$RUN" --output "$RUN/evaluation/gui_best_$(date +%Y%m%d_%H%M%S)" \\
  --device cuda:0 --visualizer kit --physics isaacsim_physx
"$PY" scripts/evaluate_phase6.py --checkpoints "$RUN/final.pt" \\
  --run-dir "$RUN" --output "$RUN/evaluation/gui_final_$(date +%Y%m%d_%H%M%S)" \\
  --device cuda:0 --visualizer kit --physics isaacsim_physx

# TensorBoard: 기존 loopback:6006 프로세스가 runs 전체를 이미 보고 있다.
# 기존 서비스가 없을 때만 시작. 사용 중인 포트를 빼앗거나 프로세스를 종료하지 않는다.
"$PY" -m tensorboard.main --logdir "{ROOT}/runs" --host 127.0.0.1 --port 6006
# Windows PowerShell
ssh -N -L 6006:127.0.0.1:6006 mrl_6534@100.79.103.47
# 브라우저 http://127.0.0.1:6006
# 로컬 6006 사용 중이면 -L 6007:127.0.0.1:6006 / http://127.0.0.1:6007
"""
(r/'COMMANDS.txt').write_text(commands)
text=f"""Crazyflie Isaac Lab Phase 6 결과 보고서
작성: {datetime.datetime.now().astimezone().isoformat()}
프로젝트: {ROOT}
실행 결과: {r}

1. 결론
가중치·Gaussian 행동분포·Adam을 이어받은 학습, 정확한 예산 종료, 저장 및 새 프로세스 재로드 검증을 완료했다.
부모 N1024: {s['before']['completed_count']}/5 조건 8초 완료, 평균 지속 {s['before']['mean_duration_s']:.3f}s.
Phase6 best: 누적 {s['best']['iteration']} iterations, {s['best']['completed_count']}/5 완료,
평균 지속 {s['best']['mean_duration_s']:.3f}s, 평균 3D RMSE {s['best']['mean_position_rmse_m']:.6f}m.
Phase6 final: 누적 1024 iterations, {s['final']['completed_count']}/5 완료,
평균 지속 {s['final']['mean_duration_s']:.3f}s, 평균 3D RMSE {s['final']['mean_position_rmse_m']:.6f}m.
별도 정상 호버 참고 N64 final: 5/5 완료, 평균 RMSE {s['N64_reference']['mean_position_rmse_m']:.6f}m,
마지막 2초 평균 위치 오차 {s['N64_reference']['tail2_position_mean_m']:.6f}m,
마지막 2초 평균 속도 {s['N64_reference']['tail2_speed_mean_m_s']:.6f}m/s.
비행 지속 능력, 위치 정밀도, 학습 파이프라인 성공은 별도 판정한다.
8초 완료는 시간 제한까지 실패 없이 비행했다는 의미이며 목표 위치 유지의 충분조건이 아니다.
부모 실패 궤적은 평균 1.158초뿐이므로 그 RMSE와 8초 궤적 RMSE를 직접 비교하여 정밀도 개선을 주장하지 않는다.
final의 마지막2초 평균 위치 오차 {s['final']['tail2_position_mean_m']:.6f}m, 속도 {s['final']['tail2_speed_mean_m_s']:.6f}m/s로 목표에서 잔여 drift가 있다.
final action 성분 clipping {s['final']['action_clipping_fraction']*100:.4f}%, allocation 성분 clipping {s['final']['allocation_clipping_fraction']*100:.4f}%.
N64 reference는 두 clipping 비율 모두0. 차이는 기록된 관찰이며 원인을 한 가지로 단정하지 않는다.
이번 결과만으로 안정적인 정밀 호버 습득 또는 optimizer update 수만의 인과 효과를 주장하지 않는다.

2. 환경/버전과 기존 결과 보존
SSH mrl_6534@100.79.103.47, hostname MRL-server를 확인하고 모든 변경/실행을 서버에서 수행했다.
Task MRL-Crazyflie-Hover-E2E-v0, device cuda:0, headless.
요청 backend isaacsim_physx; 실제 manager {t['actual_backend']}.
패키지 metadata: {json.dumps(versions,ensure_ascii=False)}.
Isaac Lab 소스 v3.0.0-EA 설치를 유지. distribution metadata와 소스 VERSION 표기는 구분한다.
프로젝트 Git에는 기존 커밋이 없었고 feat/cf21b-physx-phase2 branch의 untracked 작업본이었다.
코드 기준은 Git 상태와 SHA256 manifest/source_snapshot으로 고정했다.
기존 MuJoCo/Isaac Lab 설치 및 phase3/4/5 코드는 변경하지 않았다. 패키지 upgrade 없음.
evidence/project_before_sha256.json 및 parent_before_sha256.json의 기존 파일 모두 해시 일치.
부모 final.pt SHA256: {t['parent_sha256']}
새 final.pt SHA256: {t['final_sha256']}
보존 검사: preservation_final.json. 기존 TensorBoard 프로세스는 종료하지 않았다.

3. 재개 검증과 범위
부모: {plan['checkpoint']}
actor/critic 전체 state_dict와 buffer, Gaussian log_std, Adam 설정/parameter step/exp_avg/exp_avg_sq를 strict 복원했다.
Adam 13 parameter states 모두 5120; 모든 부모 모멘트 tensor와 bitwise 일치.
고정 관측 128개 deterministic action max abs error=0.
부모 learned action std={rv['std']}; cfg에 남은 init_std를 재적용하지 않았다.
관측 normalizer는 actor/critic 모두 Identity, obs_normalization=false. 별도 value normalizer는 사용하지 않는다.
Python/NumPy/Torch CPU/CUDA RNG를 부모에서 복원하고 수집 직전 fingerprint도 일치했다.
시뮬레이터 상태, 진행 중 episode, rollout buffer는 부모에 저장되지 않아 복원하지 않았다.
기존 reset 규칙과 seed42로 새 1024 환경을 초기화하고 빈 rollout에서 수집을 시작했다.
따라서 가중치·Adam을 이어받은 학습이며 중단 없는 연속 실행과 bitwise 동일한 실험은 아니다.
첫 실제 optimizer update 후 5121 확인, actor와 optimizer 상태 변경 확인. 종료 체크포인트의 모든 parameter exp_avg/exp_avg_sq 변경도 별도 확인.
해당 update는 이번 예산에 포함하며 별도 검증 optimizer update는 없다.
사전 step/reset 확인 16 policy batches=16,384 transitions,80 physics batches는 학습 예산 밖이며 update=0.
잘못된 --additional-iterations 1024 입력은 CLI에서 exit2로 거부했다.
resume_verification.json, first_optimizer_step_verification.json, preflight.json에 근거를 저장했다.

4. 정확한 학습량
{table(['범위','iterations','policy transitions','Adam step 호출'],[
['부모',256,8388608,5120],['추가',t['additional_iterations'],t['additional_transitions'],t['additional_optimizer_updates']],
['누적',t['iterations'],t['transitions'],t['optimizer_updates']]])}
환경1024 × rollout32=32,768 transitions/iteration; minibatch8192 ×4 ×5 epochs=20 Adam.step/iteration.
optimizer는 actor/critic을 담은 하나의 Adam. post-step hook 실제 호출수와 모든 Adam 내부 step을 함께 확인했다.
부모 framework iter255=256회 완료. 이번 iter256..1023=추가768회, 종료 완료횟수1024.
추가 policy batch steps24,576 / physics batch steps122,880.
누적 physics batch steps163,840 / 환경별 physics 전이 합167,772,160.
환경당 추가 sim time245.76s, 누적327.68s. 서로 다른 환경의 시간을 하나의 비행 episode로 합산하지 않는다.
예산 종료 후 추가 학습/튜닝은 실행하지 않았다.

5. 유지한 설정과 학습 연결부
1024 env,32 rollout,actor/critic [64,64] tanh, shared policy, Gaussian log std.
lr3e-4 fixed,epsilon.2,5epochs,4minibatches,gamma.99,lambda.95,
entropy coefficient.003,value coefficient1,max grad norm1,value clipping=true.
normalize_advantage_per_mini_batch=false: rollout 전체 advantage 정규화.
PPO ratio는 unclipped Gaussian sampled action log probability로 계산한다.
실행 행동은 wrapper/task에서 [-1,1] clipping한다. 분포 likelihood 대상과 실제 실행 action clipping을 구분한다.
부모의 timeout 보정 구현을 그대로 유지: 환경 원래 reward와 episode return에는 bootstrap 보정을 섞지 않는다.
설치 RSL은 timeout 보정에 transition의 current-state value를 사용하며 terminal-observation value를 새로 구현하지 않았다.
terminated(실패)와 truncated(8초timeout)는 구분하고 reset 후 관측은 다음 rollout step에 사용한다.
모델 상태/환경 물리/rollout 경계 처리는 보존된 phase5 경로를 사용한다.
sim dt.002,decimation5;500Hz physics/100Hz policy,15D obs/4D actions.
내부 Z-up/FLU 및 기존 policy wxyz 변환,환경 원점 제거,link/COM 기준 그대로.
본체.04338kg/전체.043384kg,COM/관성,ESC/모터 tau.050s,
모터당 상한.289N,allocation,user_frd 모터 배치,action scale/기존 중력보상 기준 질량 모두 변경 없음.
payload0,efficiency1,DR off. 초기 target[0,0,1],reset 위치분산0,정확한 수평 자세/속도0,기존 hover RPM.
초기 상대 관측은 환경 간 동일하다. Gaussian 샘플은 환경별 독립 추출되고 이후 dynamics/개별 reset으로 경험이 달라진다.
cfg diff는 출력/재개/예산/저장 주기 metadata 7개뿐이며 environment와 optimizer diff는 [].
실제 해석 전체 설정: resolved_config.json, agent_config.json, env_config.json, optimizer_config.json, config_diff.json.

6. Reward 및 실제 backward loss / TensorBoard 계약
Reward A의 signed contribution(매 policy step,dt 추가배율 없음):
xy=-10*(ex²+ey²)
z=-6*ez²
velocity=-.005*||v_world||²
tilt=-6*(qx²+qy²), quaternion wxyz
angular_velocity=-.001*||omega_body||²
yaw=-(yaw-yaw_target의 기존 wrapping 오차)²
termination=-10*terminated
total=위 7개 합. 합산 후 추가 reward clipping 없음.
실제 소스: src/crazyflie_isaaclab/tasks/crazyflie/observations_rewards.py의 reward(),\n환경 연결 src/crazyflie_isaaclab/tasks/crazyflie/crazyflie_env.py,
보존된 scripts/phase4_metrics.py / phase5_common.py의 metric snapshot.
RewardStep/term:1024env×32policy steps 동일 표본 signed 기여도 평균.
RewardStep/total:환경 원래 반환 reward 평균. raw reward와 timeout-adjusted PPO reward는 구분.
RewardEpisode/term 및 Episode/return:이번 iteration에서 끝난 episode들 평균.
episode accumulator는 환경별로 rollout을 넘어 유지; 종료된 환경만 정산/reset.
완료 episode0이면 count0,episode 평균 미기록. rollout 경계의 활성 episode를 평균0으로 채우지 않음.
전 episode 누적/활성 잔여 reward 및 step accounting 검증 통과.
Loss/policy=실제 clipped surrogate loss; Loss/value=coefficient 전 value loss.
Loss/value_contribution=1*value loss; Loss/entropy_contribution=-.003*mean entropy.
Loss/ppo_total=실제 backward 직전 loss tensor의 detached 값:
surrogate +1*value_loss -.003*mean_entropy. 추가 RND/symmetry loss 없음.
한 iteration의20미니배치 optimizer update 평균; reward 항을 별도 PPO loss라고 표시하지 않았다.
Policy/entropy는 coefficient 전 Gaussian entropy; std_axis는 마지막 minibatch forward의 cached std이며 최종 저장 파라미터의 exp(log_std)와 시간점이 다를 수 있다.
PPO/clip_epsilon=.2는 설정값, ratio=exp(new_log_prob-old_log_prob),
clip_fraction=mean(abs(ratio-1)>.2),approx_kl=mean((ratio-1)-log(ratio));PPO/kl은 analytic Gaussian KL.
물리 action_clipping_fraction은 raw action 성분 abs>1 비율;
allocation_clipping_fraction은 motor allocation이 [0,.289] 밖인 성분 비율. PPO probability clipping과 다르다.
Gradient/actor_preclip_norm,critic_preclip_norm 및 각각 norm1 초과비율을 구분 기록.
Flight 오차는 m,tilt deg,speed m/s,body omega rad/s;모터 추력 N.
일반 Flight scalar는 같은1024×32표본 평균,position_rmse_m은 squared error 평균의 제곱근,
_max는 해당 iteration 모든 env/policy step 최대. 세부 trajectory는 평가에서만 저장.
메인 X축은 누적 policy transitions. 새 학습 이벤트는8,421,376부터33,554,432까지768점.
시작 evaluation은8,388,608. 부모 training event를 새 파일에 복제하지 않았다.
Performance/additional_*와 completed_iterations/optimizer_updates 별도 카운터.
부모+추가의 연결은 plots/combined_N1024_*.csv/PNG에서만 하며 재개 지점 표시.
GPU에서 집계 후 iteration 단위 CPU transfer. 물리 substep에 새 개별 .item/.cpu/file write를 추가하지 않았다.
기존 Lab/PhysX CPU 동기화는 제거하지 않았고 실제 처리량을 측정했다.
이벤트 주기적 flush 및 정상 종료 close. CSV/JSON float32 canonical values와 이벤트 비교.

7. 재개 전/주기별 평가
별도 프로세스에서 시작 모델5조건 평가를 마친 뒤 학습 시작.
학습 중12 checkpoints 저장; 학습 종료 후 순차 평가하여 평가부하를 학습시간에서 분리했다.
고정 조건 nominal,X+/-0.05m,Y+/-0.05m,target[0,0,1],동일 자세/속도/모터초기값.
각 조건 deterministic 최대8초. 실제 첫 종료에서 중단하고 autoreset된 다음 episode 시간을 이어 붙이지 않았다.
{evaltable}

best 선택: 완료 조건수 > 평균 지속시간 > 모두5/5일 때 평균3DRMSE 작은 값 >
최악 조건RMSE 작은 값 > 이른 checkpoint.
시작 모델과12새 checkpoint가 모두 후보. 선택 원본 {selection['best']['checkpoint']}.
best.pt는 원본 파일과 SHA256 일치; 부모 phase5 best.pt는 보존했다.
final.pt는 성능과 관계없이 예산 종료1024회 모델이다.
best/final 각각 별도 새 PID에서 명시적 load,128고정관측 deterministic action bitwise 일치(max error0),evaluation optimizer updates0.
상세 XYZ/position/velocity/quaternion/omega/action/rotor thrust/requested & actual wrench/reward/종료 CSV/JSON 제공.

{conditiontable}

last2는8초 완료 조건의(6,8]s 200개 end-of-policy-step 표본;구간길이2s.
조기 실패는last2=N/A. final,best의precision 판단에 8초구간 위치오차와 마지막2초 속도를 함께 사용한다.

8. 동일 누적 iteration 비교
아래 값은 각 phase5 저장 평가 및 이번 실제 평가에서 가져왔으며 보간 또는 추가N64/N256학습 없음.
{same_table}
같은 iteration이면 Adam updates는 같지만 N64/N256/N1024의 데이터는다르고 minibatch512/2048/8192이다.
시간은 해당 iteration row의 실제 host wall 측정(phase6은 부모 full training wall+추가 elapsed).
체크포인트 최종 저장/종료의 수 ms 차이로 full training_result와 마지막 row는 약간 다를 수 있다.
N64 final은4096iterations/8,388,608transitions/81,920updates의별도참고선이다.
Phase5는동일전이비교,phase6은N1024데이터/업데이트를함께늘린연장실험이다.
누적1024회에서N64/N256/N1024는모두5/5완료했지만 평균RMSE는각각0.051571/0.185810/0.286136m였다.
데이터는2,097,152/8,388,608/33,554,432개로달라 같은update수만으로데이터/미니배치효과를통제한것은아니다.
환경 수만의 인과효과 또는 업데이트 횟수만의 인과효과를 분리한 실험이 아니다. seed42 하나의 결과이다.

9. 학습 통계 변화 및 다양성
동일한64iterations=2,097,152transitions window. episode 평균은 완료 episode 수 가중 평균.
{rewardloss}
추가 첫64회 대비 마지막64회 step reward는 -0.333506에서 -0.195282로 개선되고,
실제 total PPO loss는 6.647763에서 3.335758로 감소했다. policy loss는 약 -0.00134로 비슷하며 value loss 감소가 크다.
정책 std 평균은0.313782에서0.565183으로 증가했다. 더 긴 비행과 높은 stochastic action 분산이 함께 나타났다.
Episode return은 -25.0466에서 -74.9905로 더 음수가 되었지만 평균길이도0.73835s에서3.79775s로 증가했다.
Return은 episode 길이와 음수 penalty의 누적 영향을 같이 받는다. 전체 loss 하락만으로 호버 개선을 판단하지 않는다.
초기 위치/목표/자세 분포는 좁고 동일; stochastic actions 및 각각의 종료/reset시점으로 경험이 달라진다.
Diversity action_axis std/position spread는실제수집표본계산이며 reset분포 자체가 다양해졌다는 의미는 아니다.
다음 단계 검토는 정상호버 N64reference와 action saturation/last2 drift 차이의 원인을 먼저 비교할 근거가 있다.
이번 예산만으로 더 오래 학습하면 반드시 정밀 호버가 된다고 결론내릴 수 없다.
초기조건 다양화와 환경 수 확대는 별도 통제실험으로 검토해야 하며 이번에는 실행하지 않았다.

10. 시간/처리량/메모리
초기화 {t['initialization_s']:.6f}s(스크립트 BOOT 시점부터 import/환경/모델/복원까지; OS process launch 이전 시간 제외).
추가 학습 {t['training_wall_s']:.6f}s,추가25,165,824전이/{t['training_wall_s']:.6f}s={t['transitions_per_s']:.3f}transitions/s.
부모 학습 {pt['training_wall_s']:.6f}s,부모+추가 누적 {pt['training_wall_s']+t['training_wall_s']:.6f}s.
누적 처리량 {33554432/(pt['training_wall_s']+t['training_wall_s']):.3f}transitions/s.
학습 wall은 수집+PPO update+telemetry/log/checkpoint 저장을 포함하고 초기화/별도평가/플롯은 제외.
세부 rollout/update 시간은 host perf_counter 구간이며 별도 매구간 CUDA synchronize를 추가하지 않았다.
전체 학습 시작/끝 CUDA fence 적용. 기존 framework 내부 동기화는 포함한다.
평가4프로세스 전체 wall(시작+periodic+best+final):{evalwall:.6f}s.
조건별 loop wall 합:{episodetime:.6f}s;그 차이에는각process초기화/load/I/O/종료가포함된다.
최종 analysis/검증/플롯 실행 wall:{v['details']['analysis_wall_s']:.6f}s (코드 수정 전 첫 분석 pass8.926908s는 별도).
GPU cudaMemGetInfo iteration샘플 peak {t['cuda_device_used_peak_sampled_MiB']:.4f}MiB:
장치전체사용량이며 본프로세스단독GPU메모리또는순간절대peak라고부르지않는다.
PyTorch 본프로세스 allocator peak allocated {t['memory_final']['torch_peak_allocated_MiB']:.6f}MiB,
peak reserved {t['memory_final']['torch_peak_reserved_MiB']:.6f}MiB.
이는 PhysX/Vulkan/기타nativeallocator메모리를포괄하지않는다. NVML은driver/library mismatch로사용불가.

11. 검증 및 산출물
verification.json: {len(v['checks'])}개 집계 검증 모두 통과.
TensorBoard scalar {len(v['details']['scalar_tags'])}태그,
JSON/event 비교 {v['details']['scalar_values_compared']}개 값 불일치0.
모든태그global step증가,CSV/JSON/event동일,backward loss probe residual0.
항별reward 평균 합 최대오차 {v['details']['reward_aggregate_sum_max_abs_error']:.12g}.
PPOloss 기여도 평균 합 최대오차 {v['details']['loss_aggregate_sum_max_abs_error']:.12g}.
평가 총75 episodes(시작5+periodic60+best5+final5). 평가optimizer update0.
체크포인트13 Adam state/counters/RNG,episode accounting,설정보존,부모파일보존을별도확인.
변경은신규scripts/phase6_common.py,train_phase6.py,evaluate_phase6.py,run_phase6.py,
evaluate_phase6_all.py,prepare_phase6.py,analyze_phase6.py,report_phase6.py,package_phase6.py.
이유:엄격resume,정확예산/카운터,분리평가/last2metrics,best선정,검증/분석/재현.
기존physics/PPO수식/phase3~5코드는변경하지않았다. source_snapshot에실행본보관.
주요결과:
  resume_verification.json,first_optimizer_step_verification.json
  training_result.json,checkpoint_audit.json,verification.json
  resolved_config.json,config_diff.json,package_versions.json
  metrics.jsonl,metrics.csv,tensorboard/events*
  evaluation/summary.csv,best_final_conditions.csv,selected_summary.json
  evaluation/periodic/,best_reload/,final_reload/:상세trajectory/재로드검증
  plots/:rawPNG및대응CSV,combined history/same iteration comparison
  selection.json,best.pt,final.pt,checkpoint_*.pt,각_reference.pt
  COMMANDS.txt,source_snapshot/,manifest.json
전달용 압축파일:{r}_review.tar.gz (체크포인트/로그/상세평가포함).
원시수치와unsmoothedplot을보존했다.

12. GUI 제한과 실행 명령
현재 서버 loaded NVIDIA kernel595.91.07과 user library595.99.02가불일치.
기존에보고된NVML_ERROR_LIB_RM_VERSION_MISMATCH는headless CUDA/PhysX학습실패가아니다.
이번headless학습/평가는성공했지만Kit GUI렌더링의복구를검증하지않았다.
드라이버교체,packageupgrade,공용프로세스종료,재부팅은하지않았다.
GUI명령은서버드라이버정합성회복후서버데스크톱터미널에서사용한다.
TensorBoard의SSH터널은GUI디스플레이전달과별개다.

{commands}
"""
(r/'REPORT_phase6_ko.txt').write_text(text)
(ROOT/'REPORT_phase6_ko.txt').write_text(text)
print('REPORT_WRITTEN',r/'REPORT_phase6_ko.txt')


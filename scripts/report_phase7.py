"""Phase7 Korean report from actual completed experiment artifacts."""
import argparse,json,csv,datetime
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
def read(p):return json.loads(Path(p).read_text())
def fmt(v,d=6):return 'N/A' if v is None else f'{v:.{d}f}'
def tab(headers,rows):return '\n'.join([' | '.join(headers),' | '.join(['---']*len(headers))]+[' | '.join(map(str,x)) for x in rows])
plan=read(r/'experiment_plan.json');v=read(r/'verification.json');assert v['success']
selected=read(r/'selected_summary.json');hist=read(r/'evaluation_history.json');diag=read(r/'diagnosis/diagnosis.json');win=read(r/'training_window_summary.json')
tr={n:read(r/n/'training_result.json') for n in plan['order']}
finals={x['branch']:x for x in selected if x['kind']=='final'};bests={x['branch']:x for x in selected if x['kind']=='best'}
c0=finals['C0_control'];m=finals['M64_minibatch'];rr=finals['R2_position'];parent=read(r/'common/evaluation_initial/results.json')
parentrmse=sum(x['position_rmse_m'] for x in parent)/5
def precision_compare(x,y):
    if x['completed_count']==y['completed_count']==5:
        delta=x['mean_position_rmse_m']-y['mean_position_rmse_m']
        return f"동일한 다섯 8초 조건 평균 RMSE 차이 {delta:+.6f}m ({delta/y['mean_position_rmse_m']*100:+.2f}%). 음수이면 개선."
    return "완료 개수/지속 시간이 다르므로 전체 궤적 RMSE만으로 정밀도 우열을 판단하지 않는다."
train_table=tab(['조건','추가 it','추가 전이','추가 Adam','부모포함 Adam','학습초','transitions/s','초기화초','GPU장치샘플MiB','TorchpeakMiB'],
 [[n,t['additional_iterations'],t['additional_transitions'],t['additional_optimizer_updates'],t['optimizer_updates'],fmt(t['training_wall_s'],3),fmt(t['transitions_per_s'],1),fmt(t['initialization_s'],3),fmt(t['cuda_device_used_peak_sampled_MiB'],2),fmt(t['memory_final']['torch_peak_allocated_MiB'],3)] for n,t in tr.items()])
eval_table=tab(['조건','모델','추가 it','부모선택','완료','평균지속s','RMSEm','최악RMSEm','최대tiltdeg','last2위치m','last2속력m/s','actionclip','allocationclip'],
 [[x['branch'],x['kind'],x['additional_iterations'],x['best_is_parent'],str(x['completed_count'])+'/5',fmt(x['mean_duration_s'],3),fmt(x['mean_position_rmse_m']),fmt(x['worst_position_rmse_m']),fmt(x['max_tilt_deg'],3),fmt(x['tail2_position_mean_m']),fmt(x['tail2_speed_mean_m_s']),fmt(x['action_clipping_fraction']),fmt(x['allocation_clipping_fraction'])] for x in selected])
periodic_table=tab(['조건','추가it','추가전이','추가Adam','학습초','완료','평균지속s','RMSEm','last2위치m','last2속력m/s'],
 [[x['branch'],x['additional_iterations'],x['additional_transitions'],x['additional_optimizer_updates'],fmt(x['additional_training_s'],3),str(x['completed_count'])+'/5',fmt(x['mean_duration_s'],3),fmt(x['mean_position_rmse_m']),fmt(x['tail2_position_mean_m']),fmt(x['tail2_speed_mean_m_s'])] for x in hist])
conditions=[]
for n in plan['order']:
    for label in ['best','final']:
        for x in read(r/n/'evaluation'/(label+'_reload')/'results.json'):
            conditions.append([n,label,x['condition'],x['completed_8s'],fmt(x['duration_s'],2),','.join(x['reasons']),
                fmt(x['position_rmse_m']),str([round(v,6) for v in x['xyz_rmse_m']]),str([round(v,6) for v in x['tail2_error_vector_m']]) if x['tail2_error_vector_m'] is not None else 'N/A',
                fmt(x['tail2_position_mean_m']),fmt(x['tail2_speed_mean_m_s']),fmt(x['max_tilt_deg'],3)])
condition_table=tab(['조건','모델','시작위치','8s','기간s','종료','RMSEm','XYZ RMSEm','last2오차벡터m','last2오차m','last2속력m/s','max tiltdeg'],conditions)
window_table=tab(['조건','지표','첫64회','마지막64회'],
 [[n,k,fmt(w['first64'][k]),fmt(w['last64'][k])] for n,w in win.items() for k in w['first64']])
ejobs=read(r/'evaluation_jobs.json');tj=read(r/'training_jobs.json')
evaltimes={n:sum(x['wall_s'] for x in ejobs if x['branch']==n) for n in plan['order']}
def outcome(x,y,label):
    if x['completed_count']!=y['completed_count']:
        return f"{label}: 완료 조건수 {x['completed_count']}/5 대 {y['completed_count']}/5로 차이가 있으며 정밀도보다 완료율을 우선 해석한다."
    if x['completed_count']==5:
        improved=x['mean_position_rmse_m']<y['mean_position_rmse_m']
        return f"{label}: 이 부모와 예산에서 평균 위치 정밀도 "+('개선이 관찰됐다.' if improved else '개선은 관찰되지 않았다.')+" 단일 seed 결과이며 분리되지 않은 효과를 독립 원인으로 단정하지 않는다."
    return f"{label}: 모든 조건의 8초 완료를 확보하지 못해 짧은 궤적 RMSE로 우열을 확정하지 않는다."
parent_tilt=max(x['max_tilt_deg'] for x in parent)
parent_speed=sum(x['tail2_speed_mean_m_s'] for x in parent)/5
parent_action_clip=sum(x['action_clipping_fraction'] for x in parent)/5
parent_alloc_clip=sum(x['allocation_clipping_fraction'] for x in parent)/5
risk_keys=['max_tilt_deg','tail2_speed_mean_m_s','action_clipping_fraction','allocation_clipping_fraction']
r2_worse=[k for k in risk_keys if rr[k] is not None and c0[k] is not None and rr[k]>c0[k]]
r2_risk_text='C0 대비 값이 증가한 지표: '+', '.join(r2_worse) if r2_worse else 'C0 대비 최대 기울기, 마지막2초 속력, action/allocator clipping에서는 불리한 증가가 관찰되지 않았다.'
m_conclusion=outcome(m,c0,'M64 대 C0')
r_conclusion=outcome(rr,c0,'R2 대 C0')
align=[]
for name,w in win.items():
    reward_improved=w['last64']['Monitor/common_reward_A']>w['first64']['Monitor/common_reward_A']
    f=finals[name]
    precision_improved=f['completed_count']==5 and f['mean_position_rmse_m']<parentrmse
    align.append(f"{name}: common_reward_A step 평균 개선={reward_improved}, 부모 대비 동일8초 위치 RMSE 개선={precision_improved}. 두 판정은 별도이다.")
alignment='\n'.join(align)
commands=f"""cd {ROOT}
PY=/home/mrl_6534/miniconda3/envs/isaaclab/bin/python
RUN={r}
# 실제 학습: run_phase7.py가 C0 -> M64 -> R2를 순차 실행했다.
# 각 명령의 --additional-iterations는 추가 256회이며 최종 누적 1280회이다.
"$PY" scripts/train_phase7.py --run-dir "$RUN/C0_control" --additional-iterations 256 --device cuda:0 --headless --physics isaacsim_physx
"$PY" scripts/train_phase7.py --run-dir "$RUN/M64_minibatch" --additional-iterations 256 --device cuda:0 --headless --physics isaacsim_physx
"$PY" scripts/train_phase7.py --run-dir "$RUN/R2_position" --additional-iterations 256 --device cuda:0 --headless --physics isaacsim_physx
# 위 명령은 완료된 RUN의 재실행을 guard로 거부한다. 추가 학습을 자동 시작하지 않는다.

# 전체 재현은 사용자가 새로 실행할 때만:
NEW_RUN="$PWD/runs/phase7_diagnosis_ablation_$(date +%Y%m%d_%H%M%S)"
"$PY" scripts/prepare_phase7.py --run-dir "$NEW_RUN"
"$PY" scripts/diagnose_phase7.py --run-dir "$NEW_RUN"
"$PY" scripts/preflight_phase7.py --run-dir "$NEW_RUN"
"$PY" scripts/run_phase7.py --run-dir "$NEW_RUN"
"$PY" scripts/evaluate_phase7_all.py --run-dir "$NEW_RUN"
"$PY" scripts/analyze_phase7.py --run-dir "$NEW_RUN"
"$PY" scripts/report_phase7.py --run-dir "$NEW_RUN"
"$PY" scripts/package_phase7.py --run-dir "$NEW_RUN"

# 단일 조건 headless best/final 재평가(학습 update0)
BRANCH=C0_control  # 또는 M64_minibatch / R2_position
"$PY" scripts/evaluate_phase7.py --checkpoints "$RUN/$BRANCH/best.pt" "$RUN/$BRANCH/final.pt" \\
  --run-dir "$RUN/$BRANCH" --output "$RUN/$BRANCH/evaluation/manual_$(date +%Y%m%d_%H%M%S)" \\
  --device cuda:0 --headless --physics isaacsim_physx

# GUI: 서버 데스크톱에서 사용. 이번 단계에서는 실행/검증하지 않았다.
"$PY" scripts/evaluate_phase7.py --checkpoints "$RUN/$BRANCH/best.pt" "$RUN/$BRANCH/final.pt" \\
  --run-dir "$RUN/$BRANCH" --output "$RUN/$BRANCH/evaluation/gui_$(date +%Y%m%d_%H%M%S)" \\
  --device cuda:0 --visualizer kit --physics isaacsim_physx

# 기존 TensorBoard 서비스와 이벤트는 보존. 새 서버는 기존 서비스가 없을 때만 시작.
"$PY" -m tensorboard.main --logdir "{ROOT}/runs" --host 127.0.0.1 --port 6006
# Windows PowerShell 터널:
ssh -N -L 6006:127.0.0.1:6006 mrl_6534@100.79.103.47
# http://127.0.0.1:6006
# 로컬6006이 사용 중이면 -L 6007:127.0.0.1:6006 / http://127.0.0.1:6007
"""
(r/'COMMANDS.txt').write_text(commands)
text=f"""Crazyflie Isaac Lab Phase7: 진단 및 공통 부모 후속 비교
작성 {datetime.datetime.now().astimezone().isoformat()}
프로젝트 {ROOT}
실험 {r}

1. 결론과 직접 비교
세 분기 모두 동일한 부모 actor/critic/Gaussian/Adam/RNG에서 독립적으로 시작하여 정해진 추가256회를 완료했다.
C0 final: {c0['completed_count']}/5 완료, RMSE {c0['mean_position_rmse_m']:.6f}m,
부모는5/5 완료/RMSE {parentrmse:.6f}m였다.
C0가5/5를유지한경우의부모대비RMSE차이는 {fmt(c0['mean_position_rmse_m']-parentrmse)}m이다.
다만 C0는 부모 대비 최대기울기 {parent_tilt:.3f}→{c0['max_tilt_deg']:.3f}deg,
마지막2초 평균속력 {parent_speed:.6f}→{c0['tail2_speed_mean_m_s']:.6f}m/s,
action clipping {parent_action_clip:.6f}→{c0['action_clipping_fraction']:.6f},
allocation clipping {parent_alloc_clip:.6f}→{c0['allocation_clipping_fraction']:.6f}.
위치 RMSE 감소를 비행 품질 전반의 개선으로 해석하지 않는다.
M64 final vs C0: {precision_compare(m,c0)}
M64의 추가 Adam update는81,920회로 C0 5,120회의16배.
실제 학습시간은 {tr['M64_minibatch']['training_wall_s']:.3f}s vs {tr['C0_control']['training_wall_s']:.3f}s,
비율 {tr['M64_minibatch']['training_wall_s']/tr['C0_control']['training_wall_s']:.3f}배이다.
R2 final vs C0: {precision_compare(rr,c0)}
R2 최대 tilt {rr['max_tilt_deg']:.3f}deg vs C0 {c0['max_tilt_deg']:.3f}deg;
last2 평균속력 {fmt(rr['tail2_speed_mean_m_s'])} vs {fmt(c0['tail2_speed_mean_m_s'])}m/s;
action clip {rr['action_clipping_fraction']:.6f} vs {c0['action_clipping_fraction']:.6f};
allocation clip {rr['allocation_clipping_fraction']:.6f} vs {c0['allocation_clipping_fraction']:.6f}.
{r2_risk_text}
완료 개수·기간이 다른 경우 조기실패 RMSE를 정밀도 개선으로 해석하지 않는다.
학습 성공,8초 완료,위치 정밀도는 아래 표에서 구분한다. 8초만으로 무한시간 안정성/정상상태를 주장하지 않는다.
N64 final RMSE 0.014310829m는 다른 학습 이력의 참고선이다.

{eval_table}

2. 학습 전 진단
학습 전 diagnosis_before_training_ko.txt 및 diagnosis/ PNG3개,CSV,JSON을 저장했다.
진단 optimizer update0. phase6의 완전한 기존 trajectory를 사용하여 불필요한 부모 재시뮬레이션은 하지 않았다.
오차 정의 e = p_link_world - env_origin - target_local. target_local = [0,0,1].
p_link_world는 기체 link 원점의 world 위치이며 별도의 origin을 두 번 빼지 않는다.
모든 조건에서 마지막2초에 대체로-X,+Y,+Z 편향. 200개 표본(6,8]s의 평균벡터/크기/RMSE/속력/축별선형회귀 slope를 계산했다.
nominal 마지막2초 평균벡터 {diag['trajectory'][0]['tail2_error_vector_m']}m,
축별 error slope {diag['trajectory'][0]['tail2_error_slope_m_s']}m/s.
부모 roll/pitch raw action clipping은 각각 약78~80%였고 0.18~0.35s부터 발생했다.
평가는 deterministic이므로 이는 Gaussian noise 표본이 아니라 학습된 actor mean 자체의 bounds 초과이다.
학습 후 해석 보충은 diagnosis/interpretation_addendum_ko.txt에 분리했고 학습 전 진단 파일/해시는 유지했다.
1초구간별 clipping과 위치오차,오차와 clipping의 상관계수를 저장했다. 시간/상태의 공동 변화이므로 인과로 해석하지 않는다.
phase6 처음/마지막64회의 XY 기여도는 {diag['windows']['first64']['RewardStep/xy']:.6f} -> {diag['windows']['last64']['RewardStep/xy']:.6f},
z는 {diag['windows']['first64']['RewardStep/z']:.6f} -> {diag['windows']['last64']['RewardStep/z']:.6f},
tilt는 {diag['windows']['first64']['RewardStep/tilt']:.6f} -> {diag['windows']['last64']['RewardStep/tilt']:.6f},
termination은 {diag['windows']['first64']['RewardStep/termination']:.6f} -> {diag['windows']['last64']['RewardStep/termination']:.6f}.
즉 자세/종료와Z는개선됐지만XY는악화되었다. 다른 물리 단위의 가중치 숫자만으로 항의 지배력을 판단하지 않았다.
Policy std/entropy 증가,gradient norm,KL/clip fraction을 실제 로그로 분석했다.
과거 매iteration log_std는 없으므로 보간하지 않았다. 저장 checkpoint의 log_std만 직접 읽었다.
RewardRaw는 기존 signed contribution/(-계수)로 산출한 cost임을 명시했다.
실제 관측/reward 함수로N1/64/256/1024 origin 제거·환경별 reward 동일·batch/개별 연산 일치를 검사했다.
잘못된 좌표/환경축 reward 합산/Adam 누락은 발견되지 않아 지정 실험을 진행했다.
5·6단계 보고서의 주요 수치와 실제 checkpoint/config/trajectory 간 불일치는 발견하지 못했다.
기존 RSL timeout current-state V 보정은 알려진 구현 제한으로 보존; 이번 비교 중 수정하지 않았다.

3. 사전 지정 조건과 예산
공통 부모 {plan['checkpoint']}
SHA256 {plan['parent_sha256']}
공통 부모 완료1024회 /33,554,432전이 /Adam20480,framework iter1023 확인.
N1024,rollout32,epochs5 유지. C0 minibatche4,size8192;M64 minibatches64,size512;R2 minibatches4,size8192.
각 분기 추가256회=8,388,608전이;총25,165,824전이.
실제 추가 Adam 총92,160=5,120+81,920+5,120.
공통 부모를 각분기총합에 세 번 더하지 않는다. 분기마다 종료누적전이41,943,040;
실험트리에서부모1회+이번추가전체=58,720,256전이,112,640updates(부모20480+추가92160).
이 수치는 분기별 단일 정책의 학습 이력과 다르다.
하나의 Adam이 actor+critic13parameter states를 업데이트. post-step hook actual count와 내부step일치.
C0/R2 final 내부step25,600;M64 final102,400.
저장/평가추가64/128/192/256회;누적1088/1152/1216/1280회.
종료 framework iter1279는 보고서의 완료1280회와 구분한다.
CLI --additional-iterations는추가256회만허용. 성능에따른early stop/추가예산/튜닝없음.

{train_table}

4. 동일 출발 및 R2 구현
사전세프로세스에서actor/critic 모든tensor와buffer,Gaussian log_std,Adam의step/exp_avg/exp_avg_sq/paramgroups를strict비교했다.
고정128관측의actor mean/std/critic출력sha256이세분기동일하고부모action오차0.
각실제학습프로세스도처음부터같은부모를다시로드했다. 이전분기의final을이어받지않았다.
Python/NumPy/Torch CPU/CUDA RNG를저장값으로복원하고수집직전fingerprint일치.
simulator/진행episode/rollout상태는저장되지않았으므로동일기존reset규칙/seed42/빈rollout사용.
따라서중단없는물리궤적의bitwise재개라고표현하지않는다.
분기간첫rollout Flight/Actuator/common reward 통계 최대 차이 {v['details']['initial_rollout_max_abs_error']:.12g}.
부모Identity obsnormalizer 전체state_dict유지;value normalizer와동적scheduler없음.
분기별 첫 optimizer 호출 뒤20481확인;별도시험optimizerupdate없음.
사전step/reset각16policy batches=16,384전이는학습밖,모든optimizerupdate0.

C0는기존reward함수반환을그대로사용.
M64는PPO num_mini_batches만4→64.
R2는원래reward A에기존xy,z항을한번더더함:
r_R2=r_A-10||e_xy||²-6ez²
따라서xy=-20||e_xy||²,z=-12ez².그외항은아래처럼유지.
R2도동일critic/Adam을복원했으며새rollout에서return/advantage를계산했다.
reward target 변경에따른value loss과도응답가능성이있으며R2만critic/Adam을초기화하지않았다.
공통reward_A는학습환경상태에서기존함수로계산하여monitor로만기록한다.
R2의학습reward를이값으로대체하거나다시더하지않는다. 상태분포가다르므로제어성능평가를대신하지않는다.
기존src/task/phase3~6코드를수정하지않고새phase7 run-local reward hook만사용.
runtime reward_config.json과resolved_config.reward에유효가중치표시;task cfg reward_variant는A로유지.
condition_diff.json:의도한algorithm.num_mini_batches 또는reward.position_scale/xy/z와run metadata만다름.
env물리/obs/action/reset/종료/optimizer설정diff는[].

5. 보존된 설정과 미니배치 의미
물리 isaacsim_physx 실제PhysxManager,cuda:0,headless.
dt.002,decimation5,500Hz/100Hz,관측15D/행동4D/좌표/scale/clipping모두부모유지.
전체질량.043384kg,COM/inertia,모터tau.050s,max .289N,allocation/ESC/중력보상 변경없음.
payload0,motor efficiency1,DR off,목표[0,0,1],reset분산0,수평/속도0/기존hover RPM,8s종료유지.
actor/critic[64,64]tanh,lr3e-4 fixed,epsilon.2,gamma.99,lambda.95,entropy_coef.003,value_coef1.
rollout 전체advantage정규화,minibatch별정규화false,value clippingtrue,max_grad_norm1각actor/critic유지.
Adam betas(.9,.999),eps1e-8,weight_decay0.행동std는부모학습값복원.
설치generator는32768표본을한번shuffle한뒤동일partition을5epochs순회한다.
실제generator로모든표본이총5번사용되는지검증했다.4분할은20updates,64분할은320updates/iteration.
M64는미니배치크기와optimizer연산횟수를함께바꾼다.N64학습과동일하지않고크기만의독립효과가아니다.

6. Reward/loss/통계 계약
RewardStep/<term>은1024환경×32policy steps 동일표본의signed contribution평균:
xy=-10or20||e_xy||²,z=-6or12ez²,velocity=-.005||v_world||²,
tilt=-6(qx²+qy²),angular_velocity=-.001||omega_body||²,yaw=-yaw_error²,termination=-10*failed.
RewardStep/total은그항들의실제환경reward합.추가dt배율/합산후clipping없음.
원래reward와PPOtimeout bootstrap을혼합하지않는다.
Monitor/common_reward_A는모든분기에원래가중치로계산한모니터값.
RewardEpisode는환경별episode누적후종료때만정산;rollout경계에서미완료episode리셋없음.
완료0개면count0,episode평균태그생략.종료원인이동시에참일수있어원인별비율합>1가능.
Loss/policy=실제clipped surrogate;Loss/value=coefficient전clipped value loss;
Loss/value_contribution=1*value;Loss/entropy_contribution=-.003*entropy;
Loss/ppo_total=실제backward직전tensor.각iteration의20또는320update평균.
Policy/entropy는Gaussian합entropy(nats);reward항은PPOloss가아니다.
PPO/clip_epsilon=.2;ratio=exp(new_logp-old_logp),clip_fraction=mean(|ratio-1|>.2),
approx_kl=mean((ratio-1)-logratio).PPO/kl은analytic Gaussian KL.
Policy/std_axis는마지막minibatch forward cached std.
이번새Policy/log_std_axis는iteration마지막optimizer후실제parameter이며시간점차이를명시한다.
Gradient norm은원래clip_grad_norm의반환값,threshold1 초과비율별도.
환경actionclip은raw성분|a|>1,allocatorclip은motor분배값이[0,.289]N밖인성분비율.
std와환경별action분산은다르다.실제실행action과PPOlogprob대상unclipped표본을구분한다.
TensorBoard주축부모포함누적전이33,554,432~41,943,040.
Init/Eval시작점33,554,432,첫학습행33,587,200.과거training event복제없음.
원시CSV/JSON과event의float32값을동일하게저장;시간과iteration/추가Adam은별도카운터.
학습상세trajectory전량저장없음.통계GPU배치집계후iteration단위CPU,평가에서만상세trajectory저장.

7. 고정 평가와 best
시작부모평가는기존phase6 final의5조건8초자료를공통기준으로재사용했다.
학습종료후평가별도프로세스순차실행;학습환경/RNG/rollout과평가부하분리.
각checkpoint별nominal,X±.05,Y±.05,target[0,0,1],기존자세/속도/모터상태,deterministic최대8초.
실제첫종료에서중단,autoreset후다음episode시간을합산하지않음.
reward hook에서autoreset전상태를clone하며terminal상태/종료조건/last2벡터를저장CSV로재검증했다.
last2는완료조건만(6,8]s200표본.조기실패는N/A이며0으로채우지않음.
best기준:완료수,평균지속시간,모두5/5이면평균RMSE,최악RMSE,이른모델.
부모도후보;best_is_parent가true이면추가학습모델보다부모가선정된것이다.
final은항상추가256회모델로별도보존.
각분기의best와final을각각새PID에서명시적load,고정128관측action error0,evaluation optimizer step0.

{periodic_table}

{condition_table}

8. 학습 지표와 평가의 관계
동일전이규모64iterations window의비교.episode평균은완료수로가중.
{window_table}
R2는reward정의와critic target이다르므로total reward/total loss절대크기를C0와직접우열비교하지않는다.
같은정의의common_reward_A도학습상태분포가다르므로보조지표이며위deterministic평가가주판정이다.
진단에서관찰한큰actionclip/편향과개선조건의평가clip/위치오차변화를함께비교해야한다.
{m_conclusion}
{r_conclusion}
M64 결과는 이 부모에서 미니배치 구성과 더 많은 optimizer 연산을 함께 바꾼 처치의 효과를 지지한다.
R2 결과는 이 부모와 예산에서 위치 가중치 2배 후보가 C0보다 위치 오차를 줄였음을 보여 준다. 최적 가중치의 증명은 아니다.
{alignment}
C0의 episode return이 덜 음수가 된 현상에는 episode 단축이 함께 있어 reward step 평균과 구분한다.
M64와 R2는 common_reward_A와 최종 평가 정밀도가 함께 개선됐으나, C0의 common_reward_A는 악화됐다.
학습 분포의 첫/마지막 64회 평균과 부모/최종 deterministic 평가를 비교한 것이며 동일 표본의 상관분석은 아니다.
어느결과든상관만으로독립원인을확정하거나단일seed결과를일반적우월성으로확대하지않는다.
세분기는같은학습된부모에서시작한후속실험이며처음부터학습한우열을말하지않는다.
관측분포다양화/다른seed/결합튜닝/reward탐색/추가학습은실행하지않았다.

9. 시간/메모리/검증
학습 wall은CUDA시작/끝동기화,수집+returns/PPO+로깅+checkpoint저장포함.
초기화BOOT부터import/환경/모델/복원은별도.평가/플롯시간제외.
rollout_s/update_s는host perf_counter구간;추가구간별CUDA fence를넣지않았으며GPU kernel단독시간이아니다.
평가 플롯의 시간점은 checkpoint 직전 iteration row의 실제 경과시간이며 full training wall에는 마지막 저장/종료 비용도 포함된다.
플롯의 선은 실측 checkpoint 연결용이고 중간 평가값을 보간하여 기록하지 않았다.
평가프로세스전체wall초: {evaltimes}
부모평가는기존자료재사용하므로새평가실행시간0;각best/finalfresh load평가시간은위값에포함.
전체추가학습초 {sum(x['training_wall_s'] for x in tr.values()):.6f},
초기화합 {sum(x['initialization_s'] for x in tr.values()):.6f},평가합 {sum(evaltimes.values()):.6f},
최종분석/검증/플롯pass {v['details']['analysis_wall_s']:.6f}s.
GPU device sampled peak는cudaMemGetInfo의전체장치사용량(다른process/nativeallocator포함),
Torch peak allocated/reserved는본process PyTorch allocator만이며PhysX전체VRAM이아니다.
기존Lab/PhysX CPU동기화는그대로두고실제처리량계측;핵심packageupgrade/driver작업없음.

총 {len(v['checks'])}개 집계검증 통과.
각branch event/CSV/JSON 첫/중간/마지막뿐아니라모든기록을대조했다.
추가optimizer합92,160,추가전이합25,165,824,Adam내부step/모멘트변경확인.
reward항합/backwardloss/episodeaccounting/common_reward_A식/설정차이/부모보존검사통과.
상세check keys 및최대오차는verification.json.
기존phase3~6파일은evidence/prior_runs_sha256.json과project_before_sha256.json으로검사했다.
source_snapshot/source_sha256.json은학습실행본,후처리추가코드는final_source_sha256.json에포함.
Git은기존no commits branch상태를보존하고임의commit하지않음.

10. 파일과 실행
신규scripts: prepare_phase7.py,diagnose_phase7.py,phase7_common.py,phase7_metrics.py,
preflight_phase7.py,train_phase7.py,run_phase7.py,evaluate_phase7.py,evaluate_phase7_all.py,
analyze_phase7.py,report_phase7.py,finalize_phase7.py,package_phase7.py.
이전physics/task/phase3~6코드와설치/체크포인트를덮어쓰지않음.
run/diagnosis_before_training_ko.txt,diagnosis/diagnosis.json,진단PNG/CSV;
분기별resolved_config/reward_config/config_diff/resume_verification/preflight/first_optimizer_step;
분기별tensorboard/,metrics.csv/jsonl,checkpoint_*/best/final과reference;
분기별evaluation/periodic,best_reload,final_reload의상세trajectory/재로드검증;
plots/ 비교PNG와대응원시CSV;condition_diff,selected_summary,evaluation_history,checkpoint_audit,verification.
전달용 {r}/review.tar.gz 및manifest/archive verification/hash.
GUI와driver복구는범위밖이며headless만실행검증했다.기존TensorBoard와공용프로세스보존.

{commands}
"""
(r/'REPORT_phase7_ko.txt').write_text(text);(ROOT/'REPORT_phase7_ko.txt').write_text(text)
print('REPORT_WRITTEN',r/'REPORT_phase7_ko.txt',flush=True)


"""Korean phase8 report using measured results, without retrospective tuning."""
import argparse,json,datetime
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
def read(p):return json.loads(Path(p).read_text())
def fmt(v,d=6):return 'N/A' if v is None else f'{v:.{d}f}'
def table(headers,rows):return '\n'.join([' | '.join(headers),' | '.join(['---']*len(headers))]+[' | '.join(map(str,row)) for row in rows])
measured_findings=(r/'measured_findings_ko.txt').read_text() if (r/'measured_findings_ko.txt').exists() else '이 실행의 수치 표는 자동 생성되었다. 해석 문장은 해당 실행의 평가 결과를 검토하여 별도로 작성해야 한다.'
plan=read(r/'experiment_plan.json');verify=read(r/'verification.json');assert verify['success']
selected=read(r/'selected_summary.json');conditions=read(r/'evaluation_conditions.json');history=read(r/'evaluation_history.json')
tr=read(r/'training_results.json');win=read(r/'training_window_summary.json');jobs=read(r/'evaluation_jobs.json')
def sel(branch,kind,h):return next(x for x in selected if x['branch']==branch and x['kind']==kind and x['horizon_s']==h)
def compare(a,b,metric,label):
    if a['completed_count']!=5 or b['completed_count']!=5:return f'{label}: 완료 조건 수가 {a["completed_count"]}/5와 {b["completed_count"]}/5로, 짧은 실패 궤적의 정밀도 우열은 판정하지 않는다.'
    av,bv=a[metric],b[metric]
    if av is None or bv is None:return label+': 해당 지표 미확인.'
    delta=av-bv;pct=100*delta/bv if bv else None
    return f'{label}: {fmt(bv)} → {fmt(av)}, 차이 {delta:+.6f}'+(f' ({pct:+.2f}%)' if pct is not None else '')+'.'
interpret=[]
for h in [8,30]:
    q=sel('Q_base','final',h);e=sel('E_position','final',h);w=sel('E_omega','final',h);parent=sel('Parent','initial',h)
    interpret+= [f'\n{h}초 final 평가의 직접 비교:',
        compare(q,parent,'position_rmse_m','Q_base 대 부모: 3D 위치 RMSE [m]'),
        compare(e,q,'position_rmse_m','E_position 대 Q_base: 3D 위치 RMSE [m]'),
        compare(e,q,'tail_z_peak_to_peak_m','E_position 대 Q_base: 마지막 구간 Z peak-to-peak [m]'),
        compare(e,q,'initial_2s_position_rmse_m','E_position 대 Q_base: 초기 2초 복귀 구간 RMSE [m]'),
        compare(w,e,'tail_roll_rad_band_10_20_power','E_omega 대 E_position: 마지막 구간 roll 10~20Hz 파워 [rad²]'),
        compare(w,e,'tail_roll_rms_rad','E_omega 대 E_position: 마지막 구간 실제 roll RMS [rad]'),
        compare(w,e,'tail_body_omega_rms_rad_s','E_omega 대 E_position: 마지막 구간 body omega RMS [rad/s]'),
        compare(w,e,'tail_position_mean_m','E_omega 대 E_position: 마지막 구간 평균 위치 오차 [m]'),
        compare(w,e,'initial_2s_position_rmse_m','E_omega 대 E_position: 초기 2초 복귀 구간 RMSE [m]')]
eval_table=table(['조건','모델','horizon s','추가it','완료/5','평균기간s','위치RMSEm','last 위치m','last 속력m/s','last Z p-p m','last omega RMS rad/s','last roll RMS rad','roll10~20Hz rad²'],
    [[x['branch'],x['kind'],x['horizon_s'],x['additional_iterations'],x['completed_count'],fmt(x['mean_duration_s'],2),fmt(x['position_rmse_m']),fmt(x['tail_position_mean_m']),fmt(x['tail_speed_mean_m_s']),fmt(x['tail_z_peak_to_peak_m']),fmt(x['tail_body_omega_rms_rad_s']),fmt(x['tail_roll_rms_rad']),fmt(x['tail_roll_rad_band_10_20_power'],9)] for x in selected])
budget=table(['조건','추가it','추가전이','실제Adam호출','최종Adamstep','학습초','전이/s','초기화초','GPU장치peak MiB','Torchpeak MiB'],
    [[n,t['additional_iterations'],t['additional_transitions'],t['additional_optimizer_updates'],t['adam_steps'],fmt(t['training_wall_s'],3),fmt(t['transitions_per_s'],1),fmt(t['initialization_s'],3),fmt(t['cuda_device_used_peak_sampled_MiB'],2),fmt(t['memory_final']['torch_peak_allocated_MiB'],3)] for n,t in tr.items()])
ctable=table(['조건','모델','horizon','초기조건','완료','지속s','종료사유','3DRMSEm','XYZ RMSEm','last 편향m','last Zstd m','last Zptp m','last omega RMS','rollband rad²'],
    [[x['branch'],x['kind'],x['horizon_s'],x['condition'],x['completed'],fmt(x['duration_s'],2),','.join(x['reasons']),fmt(x['position_rmse_m']),str([round(v,6) for v in x['xyz_rmse_m']]),str(x['tail_error_vector_m']),fmt(x['tail_z_std_m']),fmt(x['tail_z_peak_to_peak_m']),fmt(x['tail_body_omega_rms_rad_s']),fmt(x.get('tail_roll_rad_band_10_20_power'),9)] for x in conditions])
htable=table(['조건','추가it','추가전이','완료','평균기간s','3DRMSEm','last omega RMS','last Zptp m'],
    [[x['branch'],x['additional_iterations'],x['additional_transitions'],x['completed_count'],fmt(x['mean_duration_s']),fmt(x['position_rmse_m']),fmt(x['tail_body_omega_rms_rad_s']),fmt(x['tail_z_peak_to_peak_m'])] for x in history])
wkeys=['RewardStep/xy','RewardStep/z','RewardStep/angular_velocity','RewardStep/total','Monitor/common_reward_A','Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total','PPO/approx_kl','PPO/clip_fraction','Policy/entropy','Policy/std_mean']
wtable=table(['조건','지표','첫64회','마지막64회'],[[n,k,fmt(x['first64'][k]),fmt(x['last64'][k])] for n,x in win.items() for k in wkeys])
pairs=read(r/'paired_completed_comparison.json')
pair_table=table(['조건','대조','horizon','동시완료수','같은초기조건','위치RMSE차m','tailomega차rad/s','roll대역파워차rad²','Zptp차m','초기2초RMSE차m'],
    [[x['branch'],x['baseline'],x['horizon_s'],x['paired_completed_count'],','.join(x['conditions']),
      fmt(x['position_rmse_m_difference']),fmt(x['tail_body_omega_rms_rad_s_difference']),fmt(x['tail_roll_rad_band_10_20_power_difference'],9),
      fmt(x['tail_z_peak_to_peak_m_difference']),fmt(x['initial_2s_position_rmse_m_difference'])] for x in pairs])
spectral_table=table(['조건','모델','horizon','last rollpeakHz','last Z peakHz','last Z 1~20Hz peak','Z2~5Hz power m²','XYkernel<.05율','Zkernel<.05율'],
    [[x['branch'],x['kind'],x['horizon_s'],fmt(x['tail_roll_rad_peak_hz'],3),fmt(x['tail_ez_peak_hz'],3),fmt(x['tail_ez_peak_1_20_hz'],3),fmt(x['tail_ez_2_5_power'],10),fmt(x['xy_kernel_below_005_fraction']),fmt(x['z_kernel_below_005_fraction'])] for x in selected])
selection=read(r/'checkpoint_selection.json')
selection_lines='\n'.join(f"{n}: best 추가 {v['best']['additional_iterations']}회, 부모 선택={v['best_is_parent']}; best={r/n/'best.pt'}; final={r/n/'final.pt'}" for n,v in selection.items())
commands=f"""cd {ROOT}
PY=/home/mrl_6534/miniconda3/envs/isaaclab/bin/python
RUN={r}

# 실제 학습 명령: 각 분기는 동일 부모에서 추가256회, 완료 RUN 재실행은 거부됨.
""" + '\n'.join(f'"$PY" scripts/train_phase8.py --run-dir "$RUN/{n}" --additional-iterations 256 --device cuda:0 --headless --physics isaacsim_physx' for n in plan['order']) + """
# 명시적 새 프로세스 재로드 + 8초 또는30초 다섯 조건 평가. 학습은 수행하지 않음.
BRANCH=E_omega  # Q_base / E_position / E_omega 중 선택
"$PY" scripts/evaluate_phase8.py --checkpoints "$RUN/$BRANCH/best.pt" "$RUN/$BRANCH/final.pt" \\
  --run-dir "$RUN/$BRANCH" --output "$RUN/$BRANCH/evaluation/manual_$(date +%Y%m%d_%H%M%S)" \\
  --horizon 30 --device cuda:0 --headless --physics isaacsim_physx

# 기존 TensorBoard가 실행 중이면 그대로 사용. 아래는 서비스가 없을 때의 시작 명령.
"$PY" -m tensorboard.main --logdir "/home/mrl_6534/rl_crazyflie/crazyflie_isaaclab/runs" --host 127.0.0.1 --port 6006
# Windows PowerShell:
ssh -N -L 6006:127.0.0.1:6006 mrl_6534@100.79.103.47
# http://127.0.0.1:6006 ; 로컬6006 사용중이면 -L 6007:127.0.0.1:6006

# 전체 재현: 새 RUN에서만, 사용자가 별도로 실행할 때. 이번 작업에서 추가 실행하지 않음.
NEW_RUN="$PWD/runs/phase8_exponential_$(date +%Y%m%d_%H%M%S)"
"$PY" scripts/prepare_phase8.py --run-dir "$NEW_RUN"
"$PY" scripts/validate_phase8.py --run-dir "$NEW_RUN"
"$PY" scripts/evaluate_phase8.py --checkpoints "$NEW_RUN/common/initial.pt" --run-dir "$NEW_RUN/common" --output "$NEW_RUN/common/evaluation_8s" --horizon 8 --device cuda:0 --headless --physics isaacsim_physx
"$PY" scripts/preflight_phase8.py --run-dir "$NEW_RUN"
"$PY" scripts/run_phase8.py --run-dir "$NEW_RUN"
"$PY" scripts/evaluate_phase8_all.py --run-dir "$NEW_RUN"
"$PY" scripts/analyze_phase8.py --run-dir "$NEW_RUN"
"$PY" scripts/plot_phase8.py --run-dir "$NEW_RUN"
"$PY" scripts/report_phase8.py --run-dir "$NEW_RUN"
"$PY" scripts/package_phase8.py --run-dir "$NEW_RUN"
"""
(r/'COMMANDS.txt').write_text(commands)
failed_jobs_path=r/'evidence/failed_evaluation_attempt1/evaluation_jobs.json'
failed_eval_s=sum(x['wall_s'] for x in read(failed_jobs_path)) if failed_jobs_path.exists() else 0.
time_eval=sum(x['wall_s'] for x in jobs)+read(r/'common/evaluation_8s/execution.json')['initialization_and_evaluation_wall_s']
report=f"""Crazyflie IsaacLab Phase8: quadratic / exponential / angular-velocity ablation
작성: {datetime.datetime.now().astimezone().isoformat()}
서버 MRL-server / 프로젝트 {ROOT}
실험 {r}

1. 실행 완료 및 해석
동일 phase7 M64 final에서 독립적으로 세 조건을 시작해 각각 256회 추가 학습했다.
학습·저장·새 프로세스 재로드·8초와30초 평가가 완료되었다. 예산 추가·계수 튜닝·네 번째 조건은 실행하지 않았다.
Q_base도 minibatches64/size512의 M64 설정이다. phase7 C0_control(minibatches4)과 다르다.
동일 추가 학습 예산의 효과 비교는 final 대 final을 기준으로 한다. best는 서로 다른 학습 시점이나 부모가 선택될 수 있는 별도 산출물이다.
아래 평균은 다섯 episode별 지표의 산술평균이다. tail 평균에는 완료 episode만 포함하며 tail_count를 별도 JSON에 기록했다.
실패한 짧은 궤적의 낮은 RMSE는 정밀도 개선 근거로 삼지 않는다.

{eval_table}

{chr(10).join(interpret)}

두 모델이 모두 완료한 동일 초기조건만 짝지은 보조 비교(차이=해당조건−대조,음수이면해당값감소):
{pair_table}
이는 생존한 조건에 한정한 비교다. 일부 조건의 진동 감소로 전체5조건의 완료율 저하를 상쇄해 해석하지 않는다.
{measured_findings}
위 비교는 위치 정확도와 진동을 별도로 평가한다. 위치 RMSE가 작아도 진동이 없는 호버라고 표현하지 않는다.
8초와30초의 전체 RMSE는 서로 다른 시간 구간을 포함한다. 30초 마지막10초 편향·속력·진동과10초 블록 플롯을 함께 확인해야 한다.
초기 복귀 성능은 같은 ±0.05m 시작 조건의 첫2초 RMSE와 원본 파형으로 비교했다. 별도의 큰 오차 복귀 실험은 실행하지 않았다.
단일 부모·seed42의 후속 비교로, 다중seed 일반성이나 처음부터 학습하는 조건의 우월성을 주장하지 않는다.
E_position이 개선되지 않더라도 모든 exponential reward가 부적절하다는 결론은 낼 수 없다.
E_omega의 계수0.010도 사전 지정 후보이며 최적값의 증거가 아니다.

2. 부모 확인과 기존 자료 재현
부모: {plan['checkpoint']}
SHA256: {plan['parent_sha256']}
실제 부모 counters: 완료1280회,41,943,040전이,Adam102,400; framework index1279.
13개 Adam parameter state의 step·exp_avg·exp_avg_sq와 param groups를 모두 strict 비교했다.
부모를 새 프로세스에서 명시적으로 로드해 다섯8초 평가를 먼저 실행했다.
기존 phase7 궤적과 위치·속도·quaternion·body angular velocity의 비교 최대 차이는 모든 조건0이었다.
고정128관측 deterministic action도 일치했다. parent_reproduction.json에 허용 기준과 실제 차이를 저장했다.
이 기록에서 약14Hz roll과약3.5Hz Z 진동을 확인했다. 실제 전체/구간별 peak와 RMS는 아래 표와 원본 스펙트럼에 있다.
phase7 보고서·설정·체크포인트의 부모 학습량은 요청 예상값과 일치했다.
기존 phase7 체크포인트의 누적 학습시간 metadata 계산은 phase6 구간을 누락하는 구조였음을 발견했다.
phase8 시간표는 이번 실제 측정 시간을 사용하며, 부모 포함 시간은 phase5 N1024+phase6+phase7 M64의 training_result를 합산했다. 학습 설정/학습량에는 영향이 없다.

3. reward 구현
scripts/phase8_reward.py가 명시적인 reward_config.json 옵션을 읽는다. 기존 src/task와phase3~7 기본 동작은 수정하지 않았다.
e = p_link_world - env_origin - target_local, target_local=[0,0,1].
Q_base: xy=-10(ex²+ey²),z=-6ez².
E_position/E_omega:
  xy=0.100*torch.expm1(-(ex²+ey²)/0.10²)
  z =0.015*torch.expm1(-ez²/0.05²)
목표에서0; Hessian[-20,-20,-12]로 기존 quadratic의 국소2차계수와 일치.
멀리서는xy→-0.100,z→-0.015. sigma는 길이 척도이며 성공 허용오차가 아니다.
같은 국소 민감도가 전체 상태 공간의 reward 규모 일치를 뜻하지 않는다. 멀리서 벌점이 포화해 복귀가 불리할 가능성이 있다.
velocity=-.005||v_world||²
tilt=-6(qx²+qy²), policy quaternion wxyz
yaw=-wrapped_yaw_error²
termination=-10*failure
angular_velocity=-c||omega_body_FLU||² [rad/s], c=.001(Q/E_position),.010(E_omega).
추가dt배율,action-rate/Z속도 벌점,PID,필터,추가관측 없음.
Q_base는 기존 reward 함수를 그대로 반환해 bitwise 일치. E조건은 지정된 signed 항 합계를 반환한다.
E_position과E_omega의 reward 설정 차이는 weights.angular_velocity 하나뿐이다.
common_reward_A는 동일 상태에서 기존 전체reward를 계산한 모니터 값으로 실제 E조건 reward에 더하지 않는다.

4. 학습 설정과 재개
모든 조건:1024환경,rollout32,batch32768,minibatches64,size512,epochs5.
actor/critic[64,64]tanh,lr3e-4 fixed,gamma.99,lambda.95,epsilon.2,entropy_coef.003,value_coef1.
기존 clipped value loss,max_grad_norm1,rollout 전체 advantage 정규화,Identity observation normalizer 유지. reward 정규화 추가 없음.
500Hz physics/100Hz policy,dt.002,decimation5,15D/4D,기존 mass/COM/inertia/모터tau.050s/allocation/clipping/좌표계 유지.
payload0,efficiency1,DR off,학습 episode8s,기존 실패조건 및 reset_range0 유지.
Actor/critic/Gaussian/Adam과Python/NumPy/TorchCPU/CUDA RNG를 세 조건 모두 부모에서 복원했다.
환경/모델 초기화 및 검증 뒤 첫 rollout 직전에 저장RNG를 다시 복원했다.
첫 rollout의 물리·행동·kernel·common_reward_A 집계는 세 조건 간 최대 차이 {verify['first_rollout_max_error']}.
환경 내부 물리 상태·진행 episode·rollout buffer는 복원되지 않는다. 공통reset(seed42),빈rollout으로 시작했다.
따라서 물리 궤적까지 중단 없이 정확히 이어진 재개라고 표현하지 않는다.
변경된 reward 조건에서도 critic/Adam을 초기화하지 않아 value target 적응 과도기가 있을 수 있다.
RSL의 기존 timeout current-state V bootstrap 구현은 그대로 유지했다. 이번 단계에서 framework를 수정하지 않았다.

5. 실제 예산과 시간
{budget}
추가 총전이25,165,824,실제 Adam 호출245,760. 각분기 종료 완료1536회/50,331,648전이/184,320Adam.
framework 마지막 index1535와 완료횟수1536을 구분했다.
한 iteration의320 Adam step을 post-step hook으로 실제 계측하고 내부13parameter steps 및 checkpoint counters와 대조했다.
사전 step/reset은 조건별16policy batches=16,384전이,optimizer0으로 본 학습 예산 밖이다.
학습 순서 Q_base→E_position→E_omega, GPU 학습 동시 실행 없음.
주기별 checkpoint는 추가64/128/192/256회에 저장했다. 처리량 측정과 평가를 분리하기 위해 전체 학습 종료 뒤 별도 프로세스로 순차 평가했다.
학습wall 합계 {sum(x['training_wall_s'] for x in tr.values()):.3f}s, 학습초기화 합계 {sum(x['initialization_s'] for x in tr.values()):.3f}s.
정상 완료한 평가 프로세스 wall(부모 사전8초 포함) {time_eval:.3f}s. 평가 파일 선택 오류로 중단된 별도 시도 {failed_eval_s:.3f}s는 여기에 섞지 않았다. 평가 초기화/episode wall은 execution.json/개별result에 따로 있다.
학습wall은CUDA 시작/끝 synchronize,수집+PPO+로그+checkpoint 포함;초기화·평가·플롯 제외.
rollout/update 구간은 host perf_counter이며 구간마다 별도CUDA fence를 넣지 않았다.
GPU장치peak는 cudaMemGetInfo의 장치 전체 사용량 표본 최댓값으로 다른 프로세스/native allocator 포함.
Torchpeak는 해당프로세스 PyTorch allocator만이며 전체PhysX/프로세스 VRAM이 아니다.
기존Lab/PhysX CPU동기화는 유지했다. 설치·드라이버·GUI환경 변경 없음.

6. checkpoint 선택과 재로드
8초평가만 사용:완료수→평균기간→모두5/5인 후보의평균3DRMSE→last2s body omega RMS→이른checkpoint.
부모도 후보이며, 위치 우선best가 최소진동 모델과 같다는 보장은 없다.
{selection_lines}
30초결과는모든best선정이끝난후실행했으며선택에사용하지않았다.
best/final을새평가프로세스에서명시적으로로드했고고정128관측action 최대오차0을확인했다.
동일best/final의30초trajectory공유여부는evaluation_reuse.json 및selected_summary.json에명시했다.
고정모델 {verify['horizon_prefix_count']}개 조건쌍의 8초/30초 공통 시간 구간 상태와 raw action의 최대 차이는 {verify['horizon_prefix_max_error']}였다(horizon_prefix_comparison.json).\n네트워크재로드일치와PhysX궤적재현성은별도검증이다. 부모재현을확인했어도다른GPU/환경의bitwise보장은하지않는다.

{htable}

7. 8초 및30초 평가 계약/단위
각평가nominal,X±.05m,Y±.05m,목표[0,0,1],기존수평자세/속도0/hover모터초기화.
deterministic actor mean→기존action clipping→기존motor path.
30초는episode_length_s만8→30으로변경하고물리/실패조건은동일;cfg diff로검증했다.
실제첫failure/timeout에서중단. auto-reset 전reward hook에서상태를clone하여terminal이reset상태와섞이지않도록했다.
8초완료의tail=(6,8]s200표본,30초완료의tail=(20,30]s1000표본. 실패trajectorytail은N/A.
상태는정책step끝100Hz,명령은해당step시작에서계산된action. actual wrench/모터는5번째물리substep끝값이며policy구간평균이아니다.
실제Euler ZYX roll/pitch는rad와deg,body roll rate는rad/s,raw/applied action은정규화무차원,
request/actual wrench는[roll,pitch,yaw]Nm와collective N,모터omega rad/s 및rpm,추력N.
연속action 차이RMS는episode내 a[t]-a[t-1],정책100Hz;reset 경계와첫표본은제외.
XYZ/3DRMSE,원본Z peak-to-peak,속도,각속도축별RMS,각축actionclipping,모터/힘/토크 원시trajectory를저장했다.

{ctable}

8. 주파수 및 포화 해석
분석구간:초기(0,2]s,전체,완료비행tail. 구간별샘플링100Hz,Nyquist50Hz.
각구간평균을제거(constant detrend),symmetric Hann window,np.fft.rfft로one-sided PSD.
정규화 |FFT|²/(fs*sum(window²)),DC/Nyquist외2배. 대역파워는bin PSD 합*df.
8초tail주파수분해능.5Hz,30초tail.1Hz. 원본RMS/표준편차/peak-to-peak는detrending없이계산.
roll10~20Hz파워와roll/bodyomegaRMS를함께비교한다. Z는원본p-p/편향과2~5Hz파워를추가로기록했다.
최대peak만으로개선을판정하지않고,낮은주파수의복귀/드리프트가peak를차지할수있음을구분했다.
plot에서는초기복귀구간과마지막구간을나눴으며명시된범위와단위를사용한다.

{spectral_table}

kernel<.05율은exponential위치항이95%이상포화된상태의평가표본비율이다. Q/부모에도동일kernel을진단용으로계산했다.
해당비율,원본복귀파형,위치편향과완료율을함께보아포화가불리한상황의징후를판단해야한다.
고정XY±.05m는sigma_xy의절반이며먼거리복귀를포괄하지않는다. 이번자료만으로포화의인과효과나모든exponential설계의적합성을단정할수없다.

9. 학습 reward와 PPO loss
{wtable}
RewardStep는계수/부호적용후1024환경×32steps평균,RewardEpisode는환경별완료episode누적값의평균.
종료 원인 bool은 동시에 참일 수 있으므로 원인별 비율 합이1을 넘을 수 있다.
rollout경계에서미완료episode누적을초기화하지않는다. 완료episode가0이면평균을0으로만들지않고생략한다.
common_reward_A는모니터전용;timeout bootstrap으로수정된학습reward와원래환경reward/episode return을혼합하지않는다.
Loss/policy는실제clipped surrogate,Loss/value는value 계수 적용 전,Loss/value_contribution=1*value,
Loss/entropy_contribution=-.003*entropy,Loss/ppo_total은실제backward직전tensor를관찰한값이다.
reward항을독립PPOloss로표기하지않았다.
PPO epsilon=.2는설정,ratio=exp(newlogp-oldlogp),clip_fraction=mean(|ratio-1|>.2),
approx_KL=mean((ratio-1)-logratio). Gaussian analytic KL은별도PPO/kl.
Policy/std_axis는마지막minibatchforward cached값,log_std_axis는마지막update후parameter여서시점차이가있다.
보상정의가다른조건의total reward/total PPOloss크기만으로비행성능우열을정하지않는다.
TensorBoard 기본global_step은이번추가전이0→8,388,608. 과거누적값41,943,040→50,331,648은별도Performance태그.
0은Eval 부모점이며새학습첫행은32,768. 과거학습이벤트를복제하지않았다.
Physical/*는단위가명시된원래cost/물리량,Kernel/*는exp(-e²/sigma²)평균.
RewardConfig의sigma/weight는Q에서도비교기준으로기록하고exponential_active=0으로구분한다.
gradient·std4축·clipping4축·episode종료원인·실제카운터·처리량/메모리도CSV/JSON과동일수치로기록했다.

10. 검증/보존/산출물
검증 {verify['check_count']}개 통과.
Q bitwise동일,exponential0/곡률/유한포화,origin/batch독립,E조건간계수한개차이,
reward합/common_A/실제backwardloss/초기model·Adam·RNG/첫rollout일치/실제예산검증통과.
TensorBoard·CSV·JSON은첫/중간/마지막뿐아니라전체학습scalar를대조했다.
부모와phase3~7소스/산출물SHA256보존,실행source_snapshot와해시기록.
reload_count={verify['reload_count']},원시trajectory검증={verify['trajectory_count']},
평가reward합최대오차={verify['max_eval_reward_sum_error']:.9g}.
report/설정/차이/검증/CSV/JSON/TensorBoard/체크포인트/원시trajectory/PNG/source를review.tar.gz에포함한다.
manifest.json,files.sha256,review.tar.gz.sha256,archive_verification.json으로검증가능.
평가 첫 시도에서 checkpoint_*.pt 목록에 *_reference.pt(고정 관측 자료)가 포함되어 KeyError: actor_state_dict로 중단됐다.\n실패 로그와 이미 수집된5개 평가를 evidence/failed_evaluation_attempt1에 보존했다. 목록에서 reference 파일을 제외한 뒤 평가만 다시 실행했다.\n학습 코드·reward·가중치·Adam·학습 예산은 바뀌지 않았고 이 시도에서도 optimizer update는0이었다.\nperiodic과fresh reload의실제상태/action궤적차이는trajectory_replay_comparison.json에네트워크검증과별도로기록했다.\nGUI실행/드라이버복구는수행하지않았다. 기존TensorBoard서비스보존.
기존핵심코드수정없음. 신규phase8_reward/metrics/common/train/evaluate/evalmetrics,
prepare/validate/preflight/run/evaluate_all/analyze/plot/report/package/finalize scripts가phase8작업을구성한다.

{commands}
"""
(r/'REPORT_phase8_ko.txt').write_text(report);(ROOT/'REPORT_phase8_ko.txt').write_text(report)
(r/'TENSORBOARD_TAGS_ko.txt').write_text(report[report.index('9. 학습 reward와 PPO loss'):report.index('10. 검증/보존/산출물')])
print('REPORT_WRITTEN',flush=True)


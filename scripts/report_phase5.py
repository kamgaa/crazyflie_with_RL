"""Assemble the phase-5 Korean report solely from recorded measurements."""
import argparse,json,datetime,csv,hashlib
from pathlib import Path
import numpy as np
from ppo_common import ROOT,dump_json
from phase5_common import COUNTS,BUDGET,TERMS
p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args();r=Path(args.experiment).resolve()
load=lambda name:json.loads((r/name).read_text())
comparison=load('comparison.json');verification=load('verification.json');common=load('common/initialization.json');jobs=load('jobs.json');evaljobs=load('evaluation_jobs.json')
lines=[]
def add(x=''):lines.append(str(x))
def fmt(x,places=3):return 'N/A' if x is None else f'{x:.{places}f}'
add('Crazyflie Isaac Lab 5단계: 동일 전이 예산에서 64/256/1024 병렬 PPO 비교')
add(datetime.datetime.now().astimezone().isoformat()+' / MRL-server')
add('\n결론과 범위\n===========')
add('각 구성은 하나의 actor/critic 정책을 여러 환경이 공유한다. 공통 초기 가중치에서 각기 새 optimizer로 학습했으며 phase4 checkpoint에서 resume하지 않았다.')
add('이 비교는 동일한 데이터 예산에서의 병렬 PPO 구성 비교다. 환경 수와 함께 minibatch 크기, iteration 수, optimizer update 수 및 환경당 경험 길이가 달라진다.')
add('seed42 한 번의 비교이므로 환경 수만의 효과나 통계적 우월성을 주장하지 않는다. 자동 추가 학습/튜닝 없음.')
add(f'프로젝트: {ROOT}\n실험: {r}\nTask: MRL-Crazyflie-Hover-E2E-v0\nBackend: isaacsim_physx / device cuda:0 / headless')
add('\n주요 결과\n=========')
add('구성 | 성공 | 실제 전이 | iterations | optimizer updates | 학습초 | transitions/s | CUDA device peak MiB | Torch peak MiB | best 전이/8s완료/평균초 | final 8s완료/평균초/RMSE[m]')
for x in comparison:
    add(f'{x["config"]} | {x["success"]} | {x["transitions"]} | {x["iterations"]} | {x["optimizer_updates"]} | {fmt(x["training_wall_s"])} | {fmt(x["transitions_per_s"])} | {fmt(x["cuda_device_used_peak_MiB"])} | {fmt(x["torch_peak_allocated_MiB"])} | {x.get("best_transitions","N/A")}/{x.get("best_completed_count","N/A")}/5/{fmt(x.get("best_mean_duration_s"))} | {x.get("final_completed_count","N/A")}/5/{fmt(x.get("final_mean_duration_s"))}/{fmt(x.get("final_mean_position_rmse_m"),6)}')
add('실제 학습 전이 합계='+str(sum(x['transitions'] for x in comparison))+' / 계획25,165,824.')
add('CUDA device peak는 해당 프로세스 단독 사용량이 아니라 cudaMemGetInfo로 측정한 장치 전체 표본 최대다. NVML 오류 때문에 nvidia-smi process VRAM으로 표시하지 않았다.')
add('학습 실행 성공, 8초 비행 완료, 정밀 호버는 별도다. 아래 고정 평가 결과와 RMSE/tilt를 함께 해석한다.')
add('\n서버/기존 결과 보존\n==================')
add('착수 시 hostname, cwd, 프로세스 목록, 기존4단계 보고서/agent/env/optimizer config, train/eval/telemetry 코드를 확인했다.')
add('기존 TensorBoard PID430240, loopback6006을 유지. 기존 공용 프로세스를 종료하지 않았다. 단계5 전용 GPU 작업은 순차 실행했다.')
add('기존 phase3 산출물49개 및 phase4 산출물198개 hash 검증, 기존 프로젝트 before/after hash, MuJoCo/Lab git 상태를 evidence에 기록했다.')
add('프로젝트 Git은 기존 feat/cf21b-physx-phase2의 no commits 상태; 임의 commit/push 없음. source hash와 source_snapshot으로 실행 버전을 고정했다.')
add('기존 phase3/4 스크립트와 물리/asset/task 소스는 수정하지 않고 phase5 파일만 추가했다.')
add('추가 코드: phase5_common.py, train_phase5.py, run_phase5.py, evaluate_phase5.py, evaluate_phase5_all.py, analyze_phase5.py, report_phase5.py, prepare_phase5.py, finalize_phase5.py. 준비/실행/평가/후처리 자동화이며 기존 phase4 소스는 보존.')
add('phase5_common은 기존 phase4 Telemetry 재사용 + approximate KL/ratio/gradient norm observer. optimizer와 reward 식은 변경하지 않는다.')
add('실행 전 kernel NVIDIA595.91.07 / user library595.99.02 불일치로 NVML 실패를 확인했다. CUDA tensor 연산 및 모든 규모의 실제 PhysX preflight는 정상.')
add('패키지/드라이버 업그레이드, reboot, 공용작업 종료 없이 실험을 수행했다. NVML 복구 작업은 범위 밖이며 메모리 계측 한계를 공개했다.')
add('설치 버전: Lab 소스 VERSION=3.0.0 (3.0.0-EA 경로), importlib.metadata isaaclab=17.0.2를 별도로 기록. PyTorch2.11.0+cu128, IsaacSim6.1.0.0, RSL-RL5.4.1, TensorBoard2.21.0, Newton1.5.2 유지.')
add('\n공통 초기화와 사전 검사\n======================')
add('common/initial.pt는 N64 사전 검사 프로세스에서 환경 생성 후 네트워크 생성 직전 seed42를 다시 설정하여 한 번만 생성했다.')
add('N64/N256/N1024 모두 이 actor/critic state_dict를 strict load하고 모든 텐서를 공통값과 비교했다. 행동분포 log_std도 actor state에 포함된다.')
add('각 실행의 optimizer는 새 OnPolicyRunner가 만든 빈 Adam이며 공통 checkpoint의 optimizer를 load하지 않았다.')
add('공통 actor SHA256='+common['actor_sha256']+'\n공통 critic SHA256='+common['critic_sha256'])
add('초기 log_std='+str(common['actor_std_parameter'])+' (std=exp(-1.5)=.22313016014843).')
add('환경 reset 후 Python/NumPy/Torch CPU/CUDA seed를42로 다시 설정. 환경 생성의 난수 소비가 초기 network 또는 학습 시작 RNG를 바꾸지 않도록 했다.')
add('환경 수가 달라지면 이후 Gaussian 표본 수와 minibatch permutation의 RNG 소비 순서는 달라진다. 모든 rollout을 동일한 random trajectory라고 주장하지 않는다.')
add('고정 action 비교 관측128개는 독립 CPU generator seed20261010으로 생성, 학습 RNG를 소비하지 않는다.')
add('각 preflight는16 policy batch steps /80 physics batch steps, 부분 reset 후 나머지 환경의 상태/RPM 정확 보존, shape/finite 및 메모리 확인. optimizer update0.')
for n in COUNTS:
    pp=r/f'N{n}/preflight.json'
    if pp.exists():
        z=json.loads(pp.read_text());add(f'N{n}: preflight success={z["success"]}, 학습 밖 전이={z["transitions_outside_training"]}, CUDA device used={fmt(z["memory"]["cuda_device_used_MiB"])}MiB')
    else:add(f'N{n}: preflight failure; error log 참조')
add('본 학습은 각각 별도 새 프로세스/새 환경/공통 초기 가중치/빈 optimizer/새 seed42 상태에서 시작. preflight trajectory를 rollout storage에 넣지 않았다.')
add('공통 초기 정책 평가: 5조건 모두0.33s에 종료, 8초 완료0/5. 평가 결과는 common/evaluation_initial/.')
add('학습 순서: N64 -> N256 -> N1024. 실제 시작/종료 시각, PID, return code는 jobs.json 및 각 training_result에 기록했다.')
add('\n설정과 예산\n===========')
add('기준 phase4: actor/critic [64,64] tanh, lr3e-4 fixed, epochs5/minibatches4, gamma.99/lambda.95, entropy_coef.003/value_coef1.')
add('PPO epsilon=.2, value clipping=true, max_grad_norm1(각 actor/critic), rollout-wide advantage normalization, obs normalization=false, 초기 std exp(-1.5).')
add('Optimizer Adam beta(.9,.999), eps1e-8, weight_decay0. 학습률 자동 배율조정 없음. RND/symmetry/adaptive KL schedule 없음.')
add('물리 .002s/decimation5, 관측15D/행동4D, action scale [.0075,.0075,.001,.5], FLU/world Z-up, quaternion변환, origins 제거 그대로.')
add('본체 질량 .04338kg/전체 .043384kg, 기존 COM/inertia, motor cap .289N, RPM lag .050s, allocation/ESC/thrust/reaction 식 그대로.')
add('Reward A 각항/계수/합산 및 종료 penalty 그대로, dt 배율 없음. 목표/초기위치[0,0,1], 수평/정지/기존 hover RPM, reset dispersion0.')
add('payload0, efficiency1, DR없음. 종료 min_z .2/max_z2.5/tilt60deg/position1.5m/timeout8s 유지.')
add('초기 환경들은 같은 상대상태를 가지지만 독립 Gaussian action과 서로 다른 실패/reset 시점으로 경험이 달라진다.')
add('설정 diff는 config_diff.json. 알고리즘/네트워크는 동일, env config 차이는 scene.num_envs만 허용. budget/로그 이름/저장 간격 metadata 차이를 별도 기록.')
for n,it in COUNTS.items():add(f'N{n}: batch={n*32}, minibatch={n*8}, iterations={it}, 실제 목표 Adam updates={it*20}, 환경당 policy steps={BUDGET//n}, 환경당 sim time={BUDGET/n*.01:.2f}s')
add('각 구성의 물리 env-step 총계=41,943,040. physics batch 호출 횟수는 N에 따라 달라진다. actor/critic용 optimizer 두 개로 이중 계수하지 않는다.')
add('\n지표 계약\n=========')
add((r/'METRIC_CONTRACT_ko.txt').read_text())
add('\n학습 변화: 동일 전이 구간 비교\n=============================')
add('각 구성의 첫/마지막1,048,576 transitions 구간을 비교. Step/loss는 iteration 평균(각 구성 내 batch 크기 일정), episode는 완료개수 가중.')
trends={}
for n in COUNTS:
    d=r/f'N{n}';mp=d/'metrics.jsonl'
    if not mp.exists():continue
    rows=[json.loads(x) for x in mp.read_text().splitlines()];first=[x for x in rows if x['global_step']<=1048576];last=[x for x in rows if x['global_step']>BUDGET-1048576]
    if not first or not last:continue
    info={}
    for k in ['RewardStep/total',*[f'RewardStep/{t}' for t in TERMS],'Loss/policy','Loss/value','Loss/entropy_contribution','Loss/ppo_total','PPO/clip_fraction','PPO/approx_kl','PPO/kl','Policy/std_mean','Flight/position_rmse_m','Flight/tilt_deg','Actuator/action_clipping_fraction','Actuator/allocation_clipping_fraction','Diversity/position_spread_m']:
        info[k]=[float(np.mean([x[k] for x in block])) for block in [first,last]]
    for k in ['Episode/return','Episode/length_s']:
        info[k]=[sum(x.get(k,0)*x['Episode/completed_count'] for x in block)/sum(x['Episode/completed_count'] for x in block) for block in [first,last]]
    trends[f'N{n}']=info;add(f'\nN{n}:')
    for k,(a,b) in info.items():add(f'  {k}: {a:.8g} -> {b:.8g}')
add('Loss는 이동하는 rollout/critic target에 대한 값이며 세 구성은 update 개수도 다르다. 전체 loss 감소만으로 비행 성능을 결론 내리지 않는다.')
dump_json(r/'trends.json',trends)
add('\n공통 고정 평가 전체 표\n=====================')
add('전이 | 구성 | 완료/5 | 평균지속[s] | 전체기록 RMSE[m] | 완료조건 RMSE[m] | 평균최대tilt[deg] | 공통구간 RMSE[m]')
for n in COUNTS:
    sf=r/f'N{n}/evaluation/checkpoint_summary.json'
    if not sf.exists():continue
    for x in json.loads(sf.read_text()):add(f'{x["transitions"]} | N{n} | {x["completed_count"]}/5 | {fmt(x["mean_duration_s"])} | {fmt(x["mean_position_rmse_m"],6)} | {fmt(x["completed_position_rmse_m"],6)} | {fmt(x["mean_max_tilt_deg"])} | {fmt(x["common_prefix_rmse_m"],6)}')
add('동일 구간 비교 길이='+str(verification['common_prefix_s'])+'s (전체 실제 trajectory의 최소 지속 시간). 짧은 구간 지표는8초 정밀도를 대체하지 않는다.')
add('각 checkpoint의5조건별 XYZ/전체 RMSE, max error/tilt, speed/body omega, clipping 및 trajectory는 각 evaluation/periodic/에 저장.')
add('best 선택은 완료조건 수→평균지속 시간, 둘 다 같으면 먼저 나온 모델. 이번 기준에는 RMSE를 동률 해소로 쓰지 않았다.')
add('final은 예산 종료 모델이며 best와 같다고 가정하지 않는다. best/final 모두 별도 새 프로세스에서 명시적으로 load하고 다시5조건 평가했다.')
for n in COUNTS:
    rp=r/f'N{n}/evaluation/best_final_reload/reload_checks.json'
    if rp.exists():
        add(f'N{n} 재로드:')
        for x in json.loads(rp.read_text()):add(f'  {Path(x["checkpoint"]).name}: PID{x["pid"]}, fixed obs{x["reference_count"]}, max action error={x["max_action_error"]}, bitwise={x["bitwise_equal"]}, optimizer updates={x["optimizer_updates_in_process"]}')
add('\n새 프로세스 best/final 평가 상세 (주요 결과 표도 이 재로드 측정값)')
add('구성 | model | 조건 | 지속[s] | 8s완료 | XYZ RMSE[m] | 3D RMSE[m] | 최대오차[m] | 최대tilt[deg] | 종료원인')
for n in COUNTS:
    fp=r/f'N{n}/evaluation/best_final_reload/results.json'
    if fp.exists():
        for v in json.loads(fp.read_text()):
            add(f'N{n} | {v["checkpoint_name"]} | {v["condition"]} | {v["duration_s"]:.2f} | {v["completed_8s"]} | '+','.join(f'{a:.6f}' for a in v['xyz_rmse_m'])+f' | {v["position_rmse_m"]:.6f} | {v["max_position_error_m"]:.6f} | {v["max_tilt_deg"]:.4f} | '+','.join(v['reasons']))
add('\n시간/메모리/오류\n===============')
for n in COUNTS:
    pp=r/f'N{n}/training_result.json'
    if pp.exists():
        v=json.loads(pp.read_text());add(f'N{n}: init={v["initialization_s"]:.6f}s, train={v["training_wall_s"]:.6f}s, {v["transitions_per_s"]:.3f} transitions/s; CUDAdevice peak={v["cuda_device_used_peak_sampled_MiB"]:.3f}MiB, Torchpeak={v["memory_final"]["torch_peak_allocated_MiB"]:.3f}MiB.')
        add('  '+json.dumps(v['episode_accounting']))
        timing_rows=[json.loads(t) for t in (r/f'N{n}/metrics.jsonl').read_text().splitlines()]
        add('  누적 수집 host wall='+fmt(sum(t['Performance/rollout_s'] for t in timing_rows))+'s, returns+update host wall='+fmt(sum(t['Performance/update_s'] for t in timing_rows))+'s; 별도 CUDA kernel 시간이 아닌 실제 호출 구간 wall.')
    else:
        for name in ['preflight_failure.json','failure.json']:
            if (r/f'N{n}'/name).exists():add(f'N{n}: '+(r/f'N{n}'/name).read_text())
for job in evaljobs:add(f'평가 N{job["num_envs"]}: 프로세스 wall {job["wall_s"]:.3f}s, returncode={job["returncode"]}, output={job["command"][job["command"].index("--output")+1]}')
add('Lab/PhysX CPU 동기화는 유지. GPU kernel task 배치 계산 유지. CUDA device 메모리는 다른 프로세스 영향을 포함하므로 순수 모델VRAM 비교로 해석하지 않는다.')
add('\n검증\n====')
add('전체 성공='+str(verification['success'])+', event/CSV/JSON 비교 scalar수='+str(verification['comparison_count']))
for name,v in verification['runs'].items():
    add(name+': '+json.dumps({k:x for k,x in v.items() if k!='scalar_tags'},ensure_ascii=False))
add('원본 update AST와 instrumented update의 optimizer 산술 동일성, reward 합, 실제 Adam 카운터, 초기 tensor 동일성, config diff, 보호 파일 hash를 검사했다.')
add('검증용 별도 optimizer updates=0. 사전 데이터는 본 학습 예산과 분리. 정상 완료한 구성은 지정 예산을 초과하지 않았다.')
add('OOM/수치 오류가 발생한 구성은 failure.json 및 마지막 checkpoint를 유지하고 나머지 독립 구성을 계속 실행하도록 구현했다.')
add('\n판정/해석\n=========')
for x in comparison:
    add(f'{x["config"]}: 학습 실행={x["success"]}; final8초완료={x.get("final_completed_count","미검증")}/5; final 실제기록 평균RMSE={fmt(x.get("final_mean_position_rmse_m"),6)}m; best8초완료={x.get("best_completed_count","미검증")}/5.')
by_name={x['config']:x for x in comparison}
if all(x['success'] for x in comparison):
    add('세 구성 모두 계획 예산의 PPO 학습을 완료했다. 처리량 배율(N64 기준): N256='+fmt(by_name['N256']['transitions_per_s']/by_name['N64']['transitions_per_s'])+', N1024='+fmt(by_name['N1024']['transitions_per_s']/by_name['N64']['transitions_per_s'])+'.')
if by_name['N64'].get('final_completed_count')==5 and by_name['N256'].get('final_completed_count')==5:
    add('동일8초를 모두 완료한 N64와 N256의 final 평균 위치 RMSE 비율(N256/N64)='+fmt(by_name['N256']['final_mean_position_rmse_m']/by_name['N64']['final_mean_position_rmse_m'])+'. 이번 seed/예산에서는 N64의 위치 오차가 더 작았다. 그 차이를 환경 수만의 효과로 분리할 수는 없다.')
if by_name['N1024'].get('final_completed_count')==0:
    add('N1024는 학습/재로드 실행에 성공했지만 final은8초 완료0/5다. 높은 처리량이 이 예산에서의 안정적 호버를 보장하지 않았다. 조기종료 RMSE를8초 완료 정책과 직접 순위화하지 않는다.')
add('다음 실험을 설계한다면 복수 seed와 optimizer update/미니배치 효과를 분리할 조건이 필요하다. 이번에는 그 실험이나 추가 학습을 실행하지 않았다.')
add('정밀 호버의 별도 사전 수치 합격선은 지정되지 않았다. 완료율·실제 지속시간·XYZ/RMSE·tilt·속도를 함께 제공하며, 단순 생존을 정밀 호버 성공으로 대체하지 않는다.')
add('Seed42 단일 실행, 순서 N64→N256→N1024, 공유 하드웨어/드라이버 상태와 큰 batch의 더 적은 optimizer updates라는 한계가 있다.')
add('어느 구성의 결과가 좋더라도 환경 수 하나만의 통제된 인과 효과로 해석할 수 없다. 과거phase4는 초기화·학습이력이 달라 이번 대조군에 섞지 않았다.')
add('이후 학습 확대/환경수/rollout/reward 튜닝은 수행하지 않았다.')
add('\n산출물과 명령\n=============')
add('common/: initial.pt, 초기 hash/agent config, initial_reference.pt, 학습 전5조건평가.')
add('각 N*/: resolved_config.json, env/agent/optimizer config, initialization/preflight, metrics.csv/jsonl, tensorboard/, checkpoint_*.pt, best.pt, final.pt, evaluation/, verification.json.')
add('plots/: 전이/경과시간 기준 평가, XYZ/tilt, reward total/각항, loss, clip/KL/std, 처리량/시간/메모리, 포화/gradient PNG+CSV.')
add('comparison.json/csv, config_diff.json, METRIC_CONTRACT_ko.txt, REPORT_phase5_ko.txt, source_snapshot 및 코드/산출물 hash.')
add('전달용 전체 실험 압축파일: '+str(r)+'_review.tar.gz (보고서/설정/CSV/PNG와 체크포인트/trajectory 포함, SHA256 별도).')
add('학습 실제 명령(이미 실행 완료; 동일 run 재시작 guard 적용):')
for n in COUNTS:
    cp=r/f'N{n}/command.json'
    if cp.exists():add(json.loads(cp.read_text())['command'])
add('새 실험 재현 전체 명령은 REPRODUCE.sh에 저장. prepare_phase5.py로 새 timestamp 경로/config 준비→create-common preflight→common initial evaluation→run_phase5→evaluate_phase5_all→analyze/report 순서. 이 재현 스크립트를 추가 실행하지 않았다.')
add('상세 실제 순차 명령/return code는 jobs.json, evaluation_jobs.json에 전부 저장했다. GUI replay(서버 그래픽 데스크톱 terminal):')
add('cd '+str(ROOT)+'\nPY=/home/mrl_6534/miniconda3/envs/isaaclab/bin/python\nRUN='+str(r))
add('"$PY" scripts/evaluate_phase5.py --checkpoints "$RUN/N64/best.pt" --run-dir "$RUN/N64" --device cuda:0 --visualizer kit --physics isaacsim_physx')
add('N64를 N256/N1024로, best.pt를 final.pt로 바꾸어 재생. output 미지정 시 새 timestamp directory. GUI 화면 자체는 이번 범위에서 실행하지 않았다.')
add('기존 TensorBoard가 실행 중이므로 중복 시작하지 않고 현재 loopback6006에 접속. 재기동이 필요할 때 기존 사용 여부 확인 후:')
add('/home/mrl_6534/miniconda3/envs/isaaclab/bin/python -m tensorboard.main --logdir '+str(ROOT/'runs')+' --host 127.0.0.1 --port 6006')
add('Windows PowerShell:\nssh -N -L 6006:127.0.0.1:6006 mrl_6534@100.79.103.47\nhttp://127.0.0.1:6006')
add('로컬6006 충돌 시 -L 6007:127.0.0.1:6006 및 http://127.0.0.1:6007. 기존 프로세스는 종료하지 않는다.')
add('TensorBoard Runs에서 이 phase5의 N64/N256/N1024/tensorboard만 선택. 다른 단계 legacy iteration축과 혼합하지 않는다.')
text='\n'.join(lines)+'\n';(r/'REPORT_phase5_ko.txt').write_text(text);(ROOT/'REPORT_phase5_ko.txt').write_text(text)
print('REPORT',str(r/'REPORT_phase5_ko.txt'),len(text))

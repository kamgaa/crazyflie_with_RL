"""Summarize a completed run without simulating or optimizing anything."""
from pathlib import Path
import argparse,json,hashlib,datetime
parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);args=parser.parse_args()
run=Path(args.run_dir).resolve();root=Path(__file__).resolve().parents[1]
t=json.loads((run/'training_result.json').read_text())
r=json.loads((run/'evaluation/reload_result.json').read_text())
e=json.loads((run/'evaluation/evaluation_result.json').read_text())
v=json.loads((run/'verification.json').read_text())
agent=json.loads((run/'agent_config.json').read_text());opt=json.loads((run/'optimizer_config.json').read_text())
commands=dict(train=json.loads((run/'command.json').read_text())['shell_command'],
    evaluation=json.loads((run/'evaluation/command.json').read_text())['shell_command'])
commands['gui']=f'/home/mrl_6534/miniconda3/envs/isaaclab/bin/python scripts/evaluate_ppo.py --checkpoint {run}/final.pt --device cuda:0 --visualizer kit --physics isaacsim_physx'
summary=dict(completed_at=datetime.datetime.now().astimezone().isoformat(),run=str(run),
    pipeline_success=t['success'] and r['success'] and e['pipeline_success'],
    flight_success=e['hover_8s_survived'],training=t,reload=r,evaluation=e,verification=v,commands=commands,
    no_extra_training=True,no_reward_tuning=True,cpu_sync_zero_required=False)
(run/'phase3_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
text=f'''Crazyflie 3단계: RSL-RL PPO 학습 파이프라인 검증
{summary['completed_at']} / MRL-server

결과
====
학습·저장·새 프로세스 재로드·동일 관측 action 비교: 성공.
비행 성능: 8초 호버 실패. 74 policy steps / 0.74초에서 기울기 61.421289도 > 60도로 종료.
종료 당시 z=.928954 m이므로 지면 충돌을 관측했다는 뜻은 아니다. 자세 한계에 의한 조기 실패다.
이번 성공 기준은 PPO 파이프라인 검증이며 안정 비행과 구분한다.
정확히 50 iterations에서 끝냈다. 추가 학습, reward 튜닝, 물리/종료 조건 변경 없음.
CPU 동기화 0은 사용자 지시대로 완료 조건에서 제외했다.

경로
====
프로젝트: {root}
Run: {run}
Task: MRL-Crazyflie-Hover-E2E-v0
Final checkpoint: {t['checkpoint']}
Checkpoint SHA256: {t['checkpoint_sha256']}
Python: /home/mrl_6534/miniconda3/envs/isaaclab/bin/python
실제 manager: {t['actual_backend']}
선택값: isaacsim_physx, cuda:0, headless.
RSL-RL 5.4.1 / Lab 17.0.2(릴리스 v3.0.0-EA) / Isaac Sim 6.1.0.0 /
Torch 2.11.0+cu128 / Newton 1.5.2 유지. 패키지 설치·업그레이드·다운그레이드 없음.

보존한 2단계 계약
=================
REPORT_phase2_ko.txt, results/contract.json 및 실제 환경/등록/설정을 읽었다.
본체 .04338 kg, 물리 총질량 .043384 kg, 고정 자식 합성 COM/관성, 단일 강체 근사 유지.
world Z-up/body FLU, user_frd 모터 번호/반토크, quaternion xyzw<->wxyz 유지.
관측 15D, 행동 4D, action scale [.0075,.0075,.001,.5], clip[-1,1] 유지.
중력 보상 및 공중 reset RPM은 기존 본체 질량 기준을 유지했다.
모터 상한 .289 N, omega 시상수 .050 s, 다항식·allocation·ESC·반토크 유지.
physics dt=.002 s, decimation=5, policy dt=.01 s, episode 8초/800 policy steps.
기본 nominal reset=(0,0,1), 수평 자세, 위치 분산 0. 임의 episode 시작 길이 randomization도 껐다.
payload=0, 효율 모두 1, 물성 randomization 없음.

사용 reward A (정책 step당, dt 배율 없음):
r=-(10*(ex^2+ey^2)+6*ez^2+.005*||v_world||^2
    +6*(qx^2+qy^2)+.001*||omega_body||^2+yaw_error^2)-10*failure.
A의 absolute velocity 비용을 사용했다. B의 v-v_des 비용으로 바꾸지 않았다.
Failure: z<.2, z>2.5, tilt>60도, 위치오차 norm>1.5 중 하나.
Timeout: >=800 steps, failure와 별도 bool.
관측/보상 정규화나 reward scaling을 추가하지 않았다.

등록·진입점 변경
================
추가:
- src/crazyflie_isaaclab/tasks/crazyflie/agents/__init__.py
- src/crazyflie_isaaclab/tasks/crazyflie/agents/rsl_rl_ppo_cfg.py
- scripts/train_ppo.py, scripts/evaluate_ppo.py, scripts/ppo_common.py
- scripts/report_phase3.py
변경:
- src/crazyflie_isaaclab/tasks/__init__.py: rsl_rl_cfg_entry_point, default_agent 등록.
- README.rst: PPO 학습/평가/GUI 명령 및 설정 설명.
- .gitignore: runs/ 추가.
물리·환경·관측·행동·reward·모터 소스 및 USD는 2단계 SHA256과 모두 일치한다.
run/source_snapshot 및 run_source_sha256.json에 실제 실행 코드를 고정했다.
기존 Lab와 MuJoCo 작업본의 HEAD/status도 verification.json에 기록했다.

PPO 설정
========
병렬 환경 64, 환경당 rollout 32, 50 iterations, seed 42.
매 iteration 수집 2048, 총 계획/실측 전이 모두 102400.
Actor: 15 -> 64 -> 64 -> 4, hidden tanh, Gaussian policy.
Critic: 15 -> 64 -> 64 -> 1, hidden tanh. Actor와 별도 network.
Actor/critic observation normalization=false (Identity).
Gaussian 초기 std=exp(-1.5)=.22313016014842982, 학습 가능한 log std, 상태 독립.
훈련은 Gaussian sampling, 평가는 deterministic mean. 환경 전달 행동은 [-1,1] clip.
매 rollout 5 learning epochs, 4 minibatches, minibatch당 512 transitions.
Adam LR=3e-4 fixed, betas=(.9,.999), eps=1e-8, weight_decay=0, amsgrad=false.
PPO clip=.2, gamma=.99, GAE lambda=.95, value loss coefficient=1,
entropy coefficient=.003, clipped value loss=true, max gradient norm=1.
Advantage는 rollout 전체에서 정규화하며 minibatch별 정규화는 하지 않는다.
Adaptive LR/desired KL/early stopping 없음. RND/symmetry augmentation 없음.
Torch compile 사용 안 함, CUDA matmul/cudnn TF32=false.
Logger=로컬 TensorBoard. 외부 logging service에 전송하지 않았다.
NaN 검사는 켰고 실제 학습 loss 및 최종 actor 가중치 finite를 확인했다.
RSL-RL 기본 timeout bootstrap 경로를 사용했다. 이번 훈련에서 timeout은 0회였다.
전체 해석값: agent_config.json, optimizer_config.json, env_config.json.

실제 학습/optimizer 증거
========================
초기화만 하는 사전 검사: optimizer update 0, 관측 shape=(64,15).
유일한 실제 학습 프로세스 PID={t['pid']}.
학습 iterations={t['completed_iterations']}, policy batch steps={t['policy_batch_steps']},
수집 transitions={t['collected_transitions']}, motor/physics batch updates={t['motor_batch_updates']}.
실제 Adam step post-hook 호출={t['optimizer_updates']}.
예상 50*5*4=1000과 일치. 각 iteration당 20회, iteration 로그 1..50 연속.
저장된 optimizer state의 13개 파라미터 step도 모두 1000이다.
마지막 RSL-RL iteration index=49는 0-based 표기이며 완료 횟수는 50이다.
final.pt의 infos.completed_iterations=50, infos.optimizer_updates=1000으로 명시했다.
Actor 전체 파라미터 delta L2={t['actor_delta_l2']:.12f}.
Actor MLP만의 delta L2={t['actor_mlp_delta_l2']:.12f}: noise std만 변한 것이 아니다.
초기 actor SHA256={t['actor_initial_sha256']}
최종 actor SHA256={t['actor_final_sha256']}
원본 초기 가중치=initial_actor.pt, 파라미터별 변화=training_result.json.
훈련 중 failure 종료={t['training_terminations']}, timeout={t['training_timeouts']}.
이 episode 집계는 비행 성공을 뜻하지 않는다.

시간·처리량·메모리
==================
초기화: {t['initialization_s']:.6f} s (import/config {t['imports_config_s']:.6f},
  simulator/env {t['simulator_env_s']:.6f}, runner {t['runner_setup_s']:.6f}).
초기화는 프로세스 진입부터 학습 준비 완료까지이며 사전 초기화 검사 시간과 별개다.
학습 wall: {t['training_wall_s']:.6f} s, 시작/끝 CUDA synchronize 후 측정.
이 학습 wall에는 50 rollout/update, 로그와 runner checkpoint 저장이 포함된다.
별도 최종 검증/reference/final.pt 저장 시간은 이 값 이후다.
GPU 메모리 snapshot 준비 등 별도 계측 준비 시간은 초기화/학습 수치에 합산하지 않았다.
Policy transitions/s={t['policy_transitions_per_s']:.6f}=102400/학습 wall.
Runner 집계 rollout={t['rollout_time_sum_s']:.6f} s,
  update/return 계산={t['update_time_sum_s']:.6f} s.
NVIDIA 프로세스 VRAM: 초기화 후 {t['gpu_after_initialization']['process_mib']:.0f} MiB,
  학습 중 1초 간격 표본 최대 {t['gpu_process_peak_sampled_mib']:.0f} MiB.
GPU0 전체 사용량 표본 최대 {t['gpu_device_peak_sampled_mib']:.0f} MiB는 다른 프로세스도 포함한다.
PyTorch peak allocated={t['torch_peak_allocated_bytes']} bytes
  ({t['torch_peak_allocated_bytes']/1048576:.3f} MiB), reserved={t['torch_peak_reserved_bytes']} bytes
  ({t['torch_peak_reserved_bytes']/1048576:.3f} MiB).
PyTorch allocator 통계는 PhysX/Kit/context VRAM 전체를 포함하지 않는다.
NVIDIA 표본 최대는 순간 최대를 보장하지 않는다. 원본 표본은 gpu_samples.json.

저장·새 프로세스 재로드
======================
final.pt에는 actor, critic, optimizer state, iteration 및 검증 metadata를 저장했다.
새 PID={r['pid']}에서 명시적 checkpoint 경로로 strict=True load를 수행했다.
기존 메모리에 있던 actor를 평가한 것이 아니다. 두 PID가 다름을 검사했다.
저장 전 고정한 128개 관측(초기 64개 + 학습 종료 관측 64개), shape=(128,15)를 사용했다.
재로드 raw deterministic action 최대 절대 오차={r['max_abs_action_error']}.
clip[-1,1] 적용 action 최대 절대 오차={r['max_abs_clipped_action_error']}.
허용 오차=1e-6; 실제로 bitwise_equal=true. Actor state hash도 일치.
새 프로세스 optimizer updates=0. 재로드/평가 중 추가 학습 없음.
증거: deterministic_reference.pt, evaluation/reload_result.json.

Nominal hover 평가
==================
1개 환경, seed42, 목표/시작 (0,0,1), 수평 quaternion(1,0,0,0),
선속도/각속도 0, 기존 본체 질량 평형 RPM reset, deterministic policy.
최대 8초 중 첫 종료에서 중단했다. 실패 후 reset된 다른 episode를 이어 붙이지 않았다.
실측={e['policy_steps']} policy steps, {e['duration_s']:.2f}초;
PhysX post-step events={e['actual_physx_step_events']}, dt 합={e['actual_physx_elapsed_s']:.12f}초.
terminated=true, truncated=false, 원인=max_tilt.
종료 위치=({e['terminal_state']['x']:.9f},{e['terminal_state']['y']:.9f},{e['terminal_state']['z']:.9f}) m.
종료/최대 tilt={e['max_tilt_deg']:.9f}도, 한계 60도.
최대 위치 오차={e['max_position_error_m']:.9f} m.
위치 오차 RMSE={e['position_rmse_m']:.9f} m (초기 관측 포함, 종료까지).
Episode return={e['episode_return']:.9f} (dt 배율 없음).
평가 wall={e['evaluation_wall_s']:.6f} s, 초기화/재로드={e['initialization_and_reload_s']:.6f} s.
compute_final_obs=True는 종료 전 관측을 저장하는 평가 진단 옵션으로만 사용했다.
원래 DirectRLEnv의 물리·reward·종료/자동 reset 방식은 그대로이며,
trajectory 마지막 행은 자동 reset 이전 terminal 관측이다.
단일 seed, 단일 episode 결과이므로 성공률의 통계 추정이 아니다.
GUI 화면 자체는 이 SSH 세션에서 실행하지 않았고 아래 재생 명령을 제공한다.

실제 실행 명령 (서버 프로젝트 디렉터리에서)
===========================================
학습:
{commands['train']}

새 프로세스 평가:
{commands['evaluation']}

GUI 재생 (서버 그래픽 데스크톱 터미널에서):
{commands['gui']}
GUI도 동일 checkpoint의 deterministic policy로 최대 8초/첫 종료까지 실행한다.
평가 출력 경로를 생략하면 새 timestamp 디렉터리를 사용한다.
학습 진입점은 이미 학습한 run 재사용을 차단한다. 이번 작업에서는 추가 run을 학습하지 않았다.

산출물 (run 상대 경로)
======================
agent_config.json / optimizer_config.json / env_config.json / phase2_contract.json
training_started.json / training_result.json / iterations.jsonl / iterations.csv
initial_actor.pt / deterministic_reference.pt / final.pt
training TensorBoard: tensorboard/ (model_0.pt, model_49.pt 및 events 포함)
gpu_samples.json / initialization_check.json / initialization.log / train.log / eval.log
command.json / evaluation/command.json
phase2_protected_sha256.json / run_source_sha256.json / source_snapshot/
evaluation/reload_result.json / evaluation/evaluation_result.json
evaluation/trajectory.csv / evaluation/trajectory.json / evaluation/env_config.json
verification.json / phase3_summary.json / REPORT_phase3_ko.txt

완료 판정
=========
통과: 지정 예산 50 iterations/102400 transitions, 실제 1000 optimizer updates,
actor 변화, final 저장, 새 프로세스 strict load, 동일 관측 action 일치, 실제 PhysX 평가,
2단계 물리/관측/행동/reward 소스 보존, 기존 설치/작업본 유지.
비행 실패: 8초 호버 미달, .74초 자세 한계 종료.
미검증: 장기 비행/여러 seed의 안정성, GUI 화면, 추가 학습의 성능.
추가 학습·reward 튜닝 없이 요청된 50 iterations에서 종료했다.
'''
(root/'REPORT_phase3_ko.txt').write_text(text)
(run/'REPORT_phase3_ko.txt').write_text(text)
manifest={str(p.relative_to(run)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in run.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
(run/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(report=str(run/'REPORT_phase3_ko.txt'),pipeline_success=summary['pipeline_success'],flight_success=summary['flight_success'],artifact_count=len(manifest)),indent=2))

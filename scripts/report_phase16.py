"""Report and source snapshot for PID-only rigid-payload evaluation."""
import csv,json,sys,shutil,subprocess,hashlib
from pathlib import Path
import numpy as np
from ppo_common import ROOT,sha256,dump_json
r=Path(sys.argv[1]);p=json.loads((r/'experiment_plan.json').read_text());v=json.loads((r/'verification.json').read_text());prior=Path(p['rl_reference_run']);old=Path(p['historical_pid_run']);suites=['hover','circle_T10','circle_T5'];pid={s:json.loads((r/'evaluation'/s/'results.json').read_text()) for s in suites};rl={s:json.loads((prior/'evaluation'/s/'results.json').read_text()) for s in suites}
def arr(path,keys):
 with path.open() as f:rows=list(csv.DictReader(f))
 return np.array([[float(x[k]) for k in keys] for x in rows])
history={}
for s,T in [('circle_T10',10),('circle_T5',5)]:
 src=old/f'cases/PID_T{T}/trajectory.csv';dest=r/f'evaluation/{s}/nominal/trajectory.csv';keys=['ex','ey','ez','vx','vy','vz','wx','wy','wz'];a=arr(src,keys);b=arr(dest,keys);assert a.shape==b.shape
 error=float(np.max(np.abs(a[:,:3]-b[:,:3])));we=float(np.max(np.abs(a[:,6:]-b[:,6:])));history[s]=dict(source=str(src),sha256=sha256(src),samples=len(a),position_max_abs_error_m=error,body_omega_max_abs_error_rad_s=we,position_tolerance_m=1e-4,omega_tolerance_rad_s=.01,passed=error<1e-4 and we<.01);assert history[s]['passed'],history[s]
v['historical_PID_comparison']=history;dump_json(r/'verification.json',v)
for folder in ['src','scripts']:shutil.copytree(ROOT/folder,r/'source_snapshot'/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(ROOT/'pyproject.toml',r/'source_snapshot/pyproject.toml')
added=['scripts/evaluate_phase16.py','scripts/validate_phase16_pid.py','scripts/analyze_phase16.py','scripts/report_phase16.py','scripts/package_phase16.py'];dump_json(r/'source_changes.json',dict(added={s:sha256(ROOT/s) for s in added},existing_modified=[],physics_implementation='unchanged scripts/phase15_payload.py',physics_sha256=sha256(ROOT/'scripts/phase15_payload.py')))
git={}
for name,cmd in [('branch',['git','branch','--show-current']),('head',['git','rev-parse','HEAD']),('status',['git','status','--short'])]:
 x=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);git[name]=dict(code=x.returncode,stdout=x.stdout,stderr=x.stderr)
dump_json(r/'evidence/git_state.json',git)
cmd=f'''# Execute on MRL-server; no learning or tuning.
cd {ROOT}
PY=/home/mrl_6534/miniconda3/envs/isaaclab/bin/python
RUN={r}

# Actual preflight and scored commands, GPU suites run sequentially:
"$PY" scripts/validate_phase16_pid.py "$RUN"
'''
for s in suites:cmd+=f'"$PY" scripts/evaluate_phase16.py --run-dir "$RUN" --suite {s} --device cuda:0 --headless --physics isaacsim_physx\n'
cmd+='''
# Fresh output path required; scored results are never overwritten by evaluator.
"$PY" scripts/evaluate_phase16.py --run-dir "$RUN" --suite hover --case m04g_x10cm --output "$RUN/evaluation/manual_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --headless --physics isaacsim_physx

# GUI command only: not executed/verified in this task.
"$PY" scripts/evaluate_phase16.py --run-dir "$RUN" --suite circle_T10 --case m02g_x10cm --output "$RUN/evaluation/gui_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --visualizer kit --physics isaacsim_physx
# --suite hover | circle_T10 | circle_T5; --case nominal | m02g_x10cm ... m10g_x30cm

"$PY" scripts/analyze_phase16.py --run-dir "$RUN"
"$PY" scripts/report_phase16.py "$RUN"
"$PY" scripts/package_phase16.py "$RUN"

# Existing TensorBoard services on6006/6007 preserved. Check6008 again before use.
ss -ltn 'sport = :6008'
"$PY" -m tensorboard.main --logdir "$RUN/evaluation" --host 127.0.0.1 --port 6008
# Windows PowerShell tunnel:
ssh -N -L 6008:127.0.0.1:6008 mrl_6534@100.79.103.47
# Browser http://127.0.0.1:6008 ; X axis=sample step, time=step/100s
'''
cmd+=f'''\n# Windows PowerShell download:
$R = '{r}'
scp "mrl_6534@100.79.103.47:${{R}}/review.tar.gz" .
scp "mrl_6534@100.79.103.47:${{R}}/review.tar.gz.sha256" .
Get-FileHash .\\review.tar.gz -Algorithm SHA256
''';(r/'COMMANDS.txt').write_text(cmd)
lines=['PID 단독 고정 추 평가 (phase16)',f'실행: {r}',f'RL 대조 자료: {prior}',
'실제 서버 hostname MRL-server, cuda:0, isaacsim_physx. Headless. PID만 힘/토크를 결정. Actor/critic/optimizer 생성 없음, checkpoint 로드 없음, 학습·자동 튜닝 없음.',
'', '1. 동일 조건의 의미',
'추 질량2~10g, 1g 간격. body +X10/20/30cm, Y=Z=0. 27조건과 무부하1조건. 막대 질량0, 강체 고정 질점. 공기저항/막대굽힘/추 자체 회전관성/추 충돌 제외.',
'실제 total mass/COM/full inertia/새 COM 기준 rotor arm은 phase15의 검증된 구현을 그대로 재사용. 모든 case/suite에서 엔진 properties JSON이 RL 실행과 완전히 같은지 확인.',
'최초 위치·수평 yaw0·선/각속도0·기존 무부하 hover RPM도 동일. payload는 시작부터 부착, 중간 부착 아님. 학습이나 payload 평형 warm start 없음.',
'호버: local[0,0,1],30s. 원: 반지름1m, yaw0, z1m, [1,0,1]에서 속도0으로 시작. 주기10s 두바퀴20s, 주기5s 두바퀴10s.',
'원 목표 위치만 제공. 목표 속도/가속도 feedforward, 추 질량·COM에 맞춘 PID 모델 보정 추가 없음.',
'외부 출력 경로는 기존 normalized action clip[-1,1], scales(.0075,.0075,.001,.5), gravity compensation .04338*9.81N, nominal allocator, rotor0~.289N clip, ESC 역변환, 50ms RPM 지연 그대로.',
'500Hz physics/100Hz command, policy/control step당 5motor/physics update. 총 torque는 새 COM 기준으로 합산하며 gravity torque를 중복 추가하지 않음.',
'고도 .2~2.5m, tilt60deg, 목표오차1.5m 실패 기준 그대로. 최초 실패 이전 최종상태를 기록하며 autoreset 이후를 이어붙이지 않음. timeout은 정상 완료.',
'관측의 위치/속도는 COM이 아닌 link origin. world Z-up/body FLU, quaternion wxyz 유지.',
'', '2. 사용 PID와 제어기 고유 설정',
f"원본: {old/'source_snapshot/original_controllers.py'}",f"SHA256: {p['pid']['sha256']}",
'기존 CascadePID: 위치 P → 속도 PI → 자세 P → 각속도 PI. 저장된 D계수는 원래 코드에서 계산하지 않으며 이번에도 추가하지 않음.',
'kp_position=4s^-1, velocity_limit=1.5m/s(norm clip), kp_velocity=4, ki_velocity=1, kp_attitude=12.',
'kp_rate=[.0008,.0008,.0006], ki_rate=[0,0,0], kd_rate=[0,0,0]. 속도 적분기 축별[-2,2] 제한. 각속도 적분기는 원본처럼 clipping 없음(현재 ki_rate0).',
'PID 내부 max_tilt35deg, max_torque.02Nm, max_force1N 유지. 이후 공통 action 경로의 더 좁은 torque clip도 그대로 적용됨.',
'PID 내부 nominal mass=.04338kg, gravity9.81, inertia=[2.3951e-5,2.3951e-5,3.2347e-5]kgm². 실제 payload 물성으로 갱신하지 않았음.',
'PID에는 적분 상태·내부 목표자세35deg 제한이 있고 RL에는 같은 내부 구조가 없으므로, 동일 물리/외부 제한 비교이지 두 제어기 내부가 같다는 뜻은 아님.',
'28개의 원본 NumPy PID 인스턴스와 독립 적분 상태 사용. 물리는 GPU배치 처리. 정확한 기존 구현 보존을 위해 PID는 CPU에서 계산하고 policy step마다 배치 전송. 이를 GPU-only PID 또는 RL 대비 처리량 비교라고 하지 않음.',
'Baseline reward는 동일 상태 진단으로만 기록되며 PID의 출력 계산/계수/적분기에 사용되지 않음.',
'', '3. 실측 결과',
'RMS=제곱평균제곱근, RMSE=오차 제곱평균제곱근. 실패 시 RMSE는 짧은 궤적 구간값으로 전체 완료와 직접 순위 비교 불가.',
'조건 | 호버 완료/시간/RMSE(cm)/종료원인 | T10 동일 | T5 동일']
for c in p['cases']:
 parts=[]
 for s in suites:
  x=next(z for z in pid[s] if z['name']==c['name']);parts.append(f"{'완료' if x['completed'] else '실패'}/{x['duration_s']:.2f}s/{x['position_rmse_m']*100:.3f}/"+','.join(x['reasons']))
 lines.append(c['name']+' | '+' | '.join(parts))
lines+=['', '4. 보존된 RL 자료와 비교 (RL 재실행/재학습 없음)', '프로토콜 | PID 부하완료/27 | RL 부하완료/27 | PID 완료 조건']
for s in suites:lines.append(s+' | '+str(sum(x['completed'] for x in pid[s] if x['mass_g']))+' | '+str(sum(x['completed'] for x in rl[s] if x['mass_g']))+' | '+','.join(x['name'] for x in pid[s] if x['mass_g'] and x['completed']))
lines+=['동일 조건에서 두 제어기가 모두 완료한 결과:', '프로토콜/조건 | PID RMSE(cm) | RL RMSE(cm) | PID/RL body omega RMS(rad/s)']
for s in suites:
 for x,z in zip(pid[s],rl[s]):
  assert x['name']==z['name']
  if x['completed'] and z['completed']:lines.append(f"{s}/{x['name']} | {x['position_rmse_m']*100:.3f} | {z['position_rmse_m']*100:.3f} | {x['body_omega_rms_rad_s']:.4f}/{z['body_omega_rms_rad_s']:.4f}")
lines+=['ホバー'.replace('ホバー','호버')+' 후반20~30초: 완료 조건만 평균 XYZ 편향/위치RMSE/속력RMS/각속도RMS 기록. 실패 조건 N/A.']
for x in pid['hover']:
 if x['completed']:lines.append(f"{x['name']}: bias_m={x['tail_error_vector_m']}, RMSE_m={x['tail_position_rmse_m']:.6f}, speed_RMS_m_s={x['tail_speed_rms_m_s']:.6f}, omega_RMS_rad_s={x['tail_body_omega_rms_rad_s']:.6f}")
lines+=['', '결과 요약:',
'호버 및 T10에서 PID는 10cm 2/3/4g와20cm2g을 완료했다. T5에서는10cm2/3g만 완료했다. RL은 세 protocol에서10cm2/3/4g을 완료했다.',
'10cm2g 호버는 PID의 전체 RMSE3.82cm가 RL3.03cm보다 크지만, 후반20~30초 PID RMSE는0.89cm로 줄었다. 초기 과도구간과 후반 성능을 구분해야 한다.',
'10cm3g 및4g 호버의 PID 전체 RMSE는10.83cm 및47.20cm로, RL4.52cm 및6.06cm보다 컸다.',
'10cm4g 및20cm2g은 호버를 끝까지 유지했으나 후반 속력RMS≈0.811/0.863m/s, body 각속도RMS≈1.237/0.950rad/s로 지속 진동이 크다. 이를 안정적 정밀 호버 성공으로 부르지 않는다.',
'T10 원 추종은10cm2/3g에서 PID가 RL보다 정확하지만4g에서는 반대다. T5에서는10cm2g PID RMSE44.71cm vs RL46.56cm,3g PID64.96cm vs RL46.71cm이며4g PID는 실패했다.',
'PID 적분 상태도 trajectory의 pid_integrator_0..5에 저장: 앞3축은 velocity_error 적분, 뒤3축은 rate_error 적분. 서로 다른 case의 적분값을 공유하지 않는다. PID는 원래 ki_rate=0이며 새로운 적분기를 추가하지 않음.',
'', '5. 해석 범위',
'PID의 완료 개수가 더 많아도 큰 위치 오차나 진동을 허용한 비행일 수 있다. 완료율·지속시간과 동일한 완료 조건의 위치오차·자세/각속도를 함께 해석해야 한다.',
'물리적으로 수평 평형 불가6조건, 추력은 가능하나 기존 action 경로에서 평형 불가12조건은 phase15와 동일. static feasibility는 PID/RL의 동적 안정성을 보장하지 않는다.',
'PID의 nominal model/고정 gain/내부 제한 그대로인 결과이며, 모든 PID 설계 또는 payload-aware/tuned PID의 최선 성능을 대표하지 않는다.',
'단일 기존 RL모델과 기존 PID, 결정된 질량/거리·초기상태의 시뮬레이션 비교. 실제 기체 성능이나 모든 OOD에서 일반적 우월성을 주장하지 않는다.',
'위치 원자료는 float32 world 좌표에서 env_origin을 뺀 값. 미세한 수치잔차를 실제 기체 정밀도로 일반화하지 않음.',
'', '6. 검증',
'기존 PID source/설정과 초기 wrench, 적분상태 독립·reset, nominal model 유지, 동일200입력 시퀀스에서 원본과 wrench 오차0 검증.',
'phase15 부하 물리 검증을 보존·재사용. 현 실행의 질량/COM/관성 readback, initial_state, env_config는 각 protocol에서 RL과 완전히 일치. 이미 검증된 전체 물리 검사들을 반복하지 않음.',
'각 실행에서 5substeps, 실제 dt, command→wrench, target/관측오차, finite trajectory, reward 합, 최초 종료 시각을 검증.',
'원본 과거 무부하 PID 원 추종과 새 무부하 결과의 비교: '+json.dumps(history,ensure_ascii=False),
f"보호 파일 {v['protected_files_checked']}개 SHA256 확인, 변경0. 기존 RL 결과/체크포인트/PID source/물리 코드 보존.",
'TensorBoard 이벤트와 JSON scalar 전수 일치: '+json.dumps(v['scalar_checks'],ensure_ascii=False),
'초기 사전검사 JSON에서 NumPy bool 직렬화 문제를 Python bool 변환으로 수정. 평가 시작 전 guard에서 종료된 로그는 evidence/startup_attempt1_hover.log에 보존. 물리/계수 변경이나 학습은 발생하지 않았음.',
'GUI는 실제 실행하지 않았으며 재생 명령만 제공. 기존 설치·드라이버·서비스 유지.',
'', '7. 시간과 파일']
for s in suites:
 d=v['executions'][s];lines.append(f"{s}: init={d['initialization_s']:.3f}s, simulation_wall={d['simulation_wall_s']:.3f}s, PID_compute_transfer={d['controller_compute_and_transfer_wall_s']:.3f}s (simulation에 포함), total={d['total_wall_s']:.3f}s; scored samples={d['scored_policy_samples']}, optimizer0")
lines+=['시간 측정은 Python import 후 simulation launch 직전부터. 전체 개발/검증/패키지시간은 포함하지 않음. CPU PID와 GPU 물리를 합친 wall time으로 GPU-only 처리량을 주장하지 않는다.',
'학습 없음이므로 PPO loss/optimizer update 그래프 없음. 각 평가 tensorboard_time의 가로축은 sample step, 시각=step/100s.',
'핵심 비교: plots/08_PID_vs_RL_duration.png, 09_PID_vs_RL_common_completed_rmse.png, 10_PID_vs_RL_hover_transient.png.',
'PID 전체 결과: summary.csv, PID_vs_RL.csv, evaluation/{hover,circle_T10,circle_T5}/<case>/trajectory.csv 및 result.json.',
'01~07 그래프: 지속시간, 완료 RMSE, 정적 평형 가능성, 초기 호버, 원 경로, 행동/torque/수직추력, 완료 호버 후반.',
'원본 PID는 source_snapshot/original_controllers.py에 고정. 현재 repo는 아직 commit이 없어 source_snapshot 및 SHA256으로 재현.',
'COMMANDS.txt에 실행·headless 재평가·GUI·TensorBoard·Windows scp 명령. review.tar.gz에 보고서/설정/검증/CSV/plots/원시궤적/source snapshot 및 파일목록/SHA256 포함.',
'학습·PID gain 조정·payload-aware compensation·새 조건은 추가하지 않고 종료한다.']
(r/'REPORT_phase16_ko.txt').write_text('\n'.join(lines)+'\n');print('REPORT_READY',r,flush=True)

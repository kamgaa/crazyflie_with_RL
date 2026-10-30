"""Finalize motor-effectiveness comparison report and reproducible source snapshot."""
import csv,json,sys,shutil,subprocess
from pathlib import Path
import numpy as np
from ppo_common import ROOT,sha256,dump_json
r=Path(sys.argv[1]);p=json.loads((r/'experiment_plan.json').read_text());v=json.loads((r/'verification.json').read_text());overview=json.loads((r/'overview.json').read_text());ctrls=['RL','PID'];res={k:json.loads((r/'evaluation'/k/'results.json').read_text()) for k in ctrls}
def values(path,keys):
 with path.open() as f:rows=list(csv.DictReader(f))
 return np.array([[float(x[k]) for k in keys] for x in rows])
healthy={}
for k,key in [('RL','RL_reference_run'),('PID','PID_reference_run')]:
 old=Path(p[key])/'evaluation/hover/nominal/trajectory.csv';new=r/'evaluation'/k/'healthy/trajectory.csv';a=values(old,['ex','ey','ez','wx','wy','wz']);b=values(new,['ex','ey','ez','wx','wy','wz']);assert a.shape==b.shape;ep=float(np.abs(a[:,:3]-b[:,:3]).max());ew=float(np.abs(a[:,3:]-b[:,3:]).max());healthy[k]=dict(previous=str(old),sha256=sha256(old),position_max_abs_difference_m=ep,omega_max_abs_difference_rad_s=ew,position_tolerance_m=1e-4,omega_tolerance_rad_s=.01,passed=ep<1e-4 and ew<.01);assert healthy[k]['passed'],healthy[k]
v['healthy_reference_comparison']=healthy
single_dir=r/'evaluation/independence_RL_M2_eta070'
main=next(x for x in res['RL'] if x['name']=='M2_eta070')
single=json.loads((single_dir/'results.json').read_text())[0]
se=json.loads((single_dir/'execution.json').read_text());sr=json.loads((single_dir/'reload.json').read_text())
repeat_checks=dict(execution_success=se['success'],no_optimizer_updates=se['optimizer_updates']==0,
 checkpoint_identical=sr['sha256']==p['checkpoint_sha256'],action_reload_exact=sr['max_action_error']==0,
 single_environment=se['env_count']==1,fault_onset_substep=single['first_fault_substep_index']==2500,
 completion_matches=single['completed']==main['completed'],duration_matches=single['duration_s']==main['duration_s'])
repeat=dict(case=single['name'],main_path=str(r/'evaluation/RL'),single_path=str(single_dir),
 checks=repeat_checks,execution=se,metrics={},trajectory_bitwise_identity_claimed=False,
 interpretation='A single-environment repeat reproduces sustained oscillation and clipping. Metric differences are reported descriptively; no post-hoc trajectory identity tolerance is used.')
for segment in ['postfault','tail20_30']:
 repeat['metrics'][segment]={}
 for metric in ['position_rmse_m','omega_rms_rad_s','max_tilt_deg','action_clipping_fraction']:
  a=main['response'][segment][metric];b=single['response'][segment][metric]
  repeat['metrics'][segment][metric]=dict(batch13=a,single1=b,difference=b-a,relative_difference=(b-a)/a if a else None)
repeat['success']=all(repeat_checks.values());assert repeat['success'],repeat_checks
dump_json(r/'independence_verification.json',repeat)
v['single_environment_repeat']=dict(success=repeat['success'],checks=repeat_checks,metrics=repeat['metrics'],details='independence_verification.json')
protected=json.loads((r/'evidence/protected_sha256.json').read_text())
changed=[name for name,expected in protected.items() if not (ROOT/name).is_file() or sha256(ROOT/name)!=expected]
assert not changed,changed
v['protected_files_checked']=len(protected);v['protected_files_changed']=changed
v['success']=v['success'] and repeat['success'] and not changed
dump_json(r/'verification.json',v)
for folder in ['src','scripts']:shutil.copytree(ROOT/folder,r/'source_snapshot'/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(ROOT/'pyproject.toml',r/'source_snapshot/pyproject.toml');shutil.copy2(Path(p['baseline'])/'best_reference.pt',r/'reference_observations.pt')
added=['scripts/phase17_motor_fault.py','scripts/phase17_metrics.py','scripts/validate_phase17.py','scripts/evaluate_phase17.py','scripts/analyze_phase17.py','scripts/report_phase17.py','scripts/package_phase17.py'];dump_json(r/'source_changes.json',dict(added={x:sha256(ROOT/x) for x in added},existing_modified=[],baseline_checkpoint=sha256(Path(p['checkpoint']))))
git={}
for name,args in [('branch',['git','branch','--show-current']),('head',['git','rev-parse','HEAD']),('status',['git','status','--short'])]:
 x=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);git[name]=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr)
dump_json(r/'evidence/git_state.json',git)
commands=f'''# MRL-server, no training or gain changes.
cd {ROOT}
PY=/home/mrl_6534/miniconda3/envs/isaaclab/bin/python
RUN={r}

# Actual preflight commands (preserve existing verification when rerunning):
"$PY" scripts/validate_phase16_pid.py "$RUN"
"$PY" scripts/validate_phase17.py --run-dir "$RUN" --device cuda:0 --headless --physics isaacsim_physx

# Actual scored commands, sequential:
"$PY" scripts/evaluate_phase17.py --run-dir "$RUN" --controller RL --device cuda:0 --headless --physics isaacsim_physx
"$PY" scripts/evaluate_phase17.py --run-dir "$RUN" --controller PID --device cuda:0 --headless --physics isaacsim_physx

# Actual single-environment repeat (completed separately, optimizer0):
"$PY" scripts/evaluate_phase17.py --run-dir "$RUN" --controller RL --case M2_eta070 --output "$RUN/evaluation/independence_RL_M2_eta070" --device cuda:0 --headless --physics isaacsim_physx

# Reevaluate one case without overwriting scored output. M2_eta070 = user motor2 retains70%.
"$PY" scripts/evaluate_phase17.py --run-dir "$RUN" --controller RL --case M2_eta070 --output "$RUN/evaluation/manual_RL_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --headless --physics isaacsim_physx

# GUI command only: not executed/verified. No driver or GUI changes were made.
"$PY" scripts/evaluate_phase17.py --run-dir "$RUN" --controller PID --case M2_eta070 --output "$RUN/evaluation/gui_PID_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --visualizer kit --physics isaacsim_physx
# --controller RL or PID; --case healthy or M1_eta090 ... M4_eta070

"$PY" scripts/analyze_phase17.py --run-dir "$RUN"
"$PY" scripts/report_phase17.py "$RUN"
"$PY" scripts/package_phase17.py "$RUN"

# Existing TensorBoard6006/6007 untouched. Check6008 before launching; do not kill another service.
ss -ltn 'sport = :6008'
"$PY" -m tensorboard.main --logdir "$RUN/evaluation" --host 127.0.0.1 --port 6008
# Windows PowerShell SSH tunnel; browse http://127.0.0.1:6008
ssh -N -L 6008:127.0.0.1:6008 mrl_6534@100.79.103.47
# TensorBoard X axis is control sample index; t=step/100 seconds. Fault starts at500.

# Download from Windows PowerShell:
$R = '{r}'
scp "mrl_6534@100.79.103.47:${{R}}/review.tar.gz" .
scp "mrl_6534@100.79.103.47:${{R}}/review.tar.gz.sha256" .
Get-FileHash .\\review.tar.gz -Algorithm SHA256
''';(r/'COMMANDS.txt').write_text(commands)
lines=['단일 모터 부분 출력 저하: frozen RL baseline vs 기존 CascadePID (phase17)',f'실행 폴더: {r}',
'MRL-server / Isaac Lab isaacsim_physx / cuda:0 / headless. 추가 학습0, optimizer.step0, PID gain 튜닝0.',
'', '1. 확정한 요청과 조건',
'사용자 확인: 정상 추력의90/80/70%를 유지한다(각각10/20/30% 감소). 추력90/80/70% 감소 실험이 아니다.',
'정상 호버5초 후 한 번에 한 모터에만 고장 적용. user motor1~4 각각3수준=12조건/제어기, healthy 기준1조건/제어기. 합계26평가.',
'전체30초=정상5초+고장후25초. 모든 초기값은 수평 yaw0, zero link velocity/body rates, 기존 healthy hover 모터각속도≈1581.52rad/s(약15102rpm), 위치local[0,0,1]. 목표는 고정.',
'추를 제거하고 정상 질량/COM/관성으로 평가. 추가 질량0, 위치 분포/외란/센서잡음/동시다중고장/완전모터정지 없음.',
'phys500Hz, command100Hz, dt.002, decimation5. 관측15D, 행동4D, scale(.0075,.0075,.001,.5), action[-1,1], nominal allocator, .289N/rotor command cap, 50ms RPM 지연 유지.',
'실패 기준 고도.2~2.5m, tilt60deg, 목표3D오차1.5m 그대로. 최초 물리실패에서 기록 종료, timeout은정상완료. autoreset 상태를 궤적에 붙이지 않음.',
'', '2. 모델·PID·번호',f"RL checkpoint: {p['checkpoint']}",f"SHA256: {p['checkpoint_sha256']}",
'baseline은 phase14 최종선정 best/checkpoint448, vz 절대속도 벌점.015, roll/pitch.075 정책. Actor mean deterministic, 학습 없음. 128개 유효 고정 관측으로 로드 출력 비교.',
f"PID source: {r/'source_snapshot/original_controllers.py'}",f"PID SHA256: {p['pid']['sha256']}",
'PID는 이전 원 추종/추 평가의 CascadePID 그대로. 위치P4, 속도PI(4,1), 자세P12, rateP(.0008,.0008,.0006), rateI0. 속도norm limit1.5m/s, 내부자세35deg, velocity integrator 축별[-2,2].',
'PID nominal mass.04338kg, nominal inertia[2.3951e-5,2.3951e-5,3.2347e-5] 그대로. 출력wrench를 공통 normalized action으로 변환한 뒤 같은 clip/allocation/motor 경로 사용.',
'RL/PID 모두 고장 번호·효율·발생 시각을 입력받지 않는다. 고장 추정기/효율보정 allocator/추가 feedforward 없음. PID의 기존 적분기만 계속 누적.',
'body FLU:+X앞,+Y왼,+Z위. 정책 quaternion wxyz. user_frd 모터 번호만 적용하며 정책좌표를 FRD로 바꾸지 않음.',
'M1 앞왼쪽(+X,+Y)→native[3], body+Z 반토크부호+; M2 뒤왼쪽(-X,+Y)→native[2], 부호-;',
'M3 뒤오른쪽(-X,-Y)→native[1], 부호+; M4 앞오른쪽(+X,-Y)→native[0], 부호-. 반토크부호는 프로펠러의 회전방향 표기가 아님.',
'', '3. 고장 모델과 시간 경계',
'기존 RPM state를50ms 시상수로 갱신하고 nominal thrust polynomial을 계산한 뒤 plant에서만 actual_thrust_i=eta_i*nominal_thrust_i를 적용.',
'기존 반토크식 Q_i=direction_i*.00594*f_i와 일관되도록 actual_reaction_i=eta_i*nominal_reaction_i. 정상 motor는eta1.',
'RPM/ESC 명령을eta배로 바꾸지 않고, 출력 저하에 별도 지연을 추가하지 않는다. 매 substep 원래 nominal 값을 새로 계산하므로 eta가 반복해서 누적 곱해지지 않는다.',
'실제 최대 추력은 고장 모터에서.2601/.2312/.2023N. 기존 nominal 명령cap.289N과 allocation은 그대로여서 제어기가 고장을 미리 보상하지 않는다.',
'고장은0-based physics substep2500의 시작 t=5.000s부터 적용, 첫 고장 physics 결과는5.002s. policy sample t=5.000은 아직 healthy 결과이고 첫 고장 policy 기록은5.010s.',
'5.000s 제어 명령은 직전 healthy 관측으로 계산된다. 고장 알림을 선행 제공하지 않는다. 5.010s 다음 명령부터 측정 상태 변화에 반응할 수 있다.',
'이 모델은 thrust effectiveness step이며, 전기적 고장에 따른 RPM rundown/프로펠러 파편/진동/항력비 변화/rotor angular momentum 변화를 새로 모사하지 않는다.',
'', '4. 결과: 전체 완료율과 고장후 오차',
'RMS=신호 제곱평균제곱근, RMSE=오차 제곱평균제곱근. 고장후 지표는(5,30]s, 후반은(20,30]s. 전체0~30s와 구분.',
'정상 healthy2조건과 고장24조건 모두30초 완료(각제어기12/12). 30초 완료는 위치정밀도·진동 없는 호버와 동일한 의미가 아니다.',
'남는추력 | RL/PID 완료개수 | 4모터평균 고장후위치RMSE(cm) RL/PID | 후반위치RMSE(cm) RL/PID | 후반각속도RMS(rad/s) RL/PID']
for eta in ['0.9','0.8','0.7']:
 x,y=overview['RL'][eta],overview['PID'][eta];lines.append(f"{eta} | {x['completed']}/4,{y['completed']}/4 | {x['mean_post_position_rmse_cm']:.5f}/{y['mean_post_position_rmse_cm']:.5f} | {x['mean_tail_position_rmse_cm']:.5f}/{y['mean_tail_position_rmse_cm']:.5f} | {x['mean_tail_omega_rms_rad_s']:.6f}/{y['mean_tail_omega_rms_rad_s']:.6f}")
lines+=['모터별: controller/case | 완료 | 고장후RMSE(cm) | 후반RMSE(cm) | 후반omegaRMS | 고장후최대tilt(deg) | action clip fraction | 1cm 정착시간(고장후s)']
for k in ctrls:
 for x in res[k]:
  d=x['response'];post=d['postfault'];tail=d['tail20_30'];lines.append(f"{k}/{x['name']} | {x['completed']} | {post['position_rmse_m']*100:.5f} | {tail['position_rmse_m']*100:.5f} | {tail['omega_rms_rad_s']:.6f} | {post['max_tilt_deg']:.5f} | {post['action_clipping_fraction']:.6f} | {d['settling_1cm_for_final1s_after_fault_s']}")
lines+=['정착시간은 고장후 최초로3D오차<=1cm에 들어와 최소1초 이상 남고 종료까지 벗어나지 않는 시점. 정착하지 않거나 실패하면None/N/A.',
'', '5. 해석',
'이번12조건의 고장후25초 평균 위치RMSE는 RL이 평균적으로 작았다. PID는 초기 과도응답이 커서 그 구간의 오차가 전체RMSE를 높였다.',
'후반20~30초에서는 PID의 위치오차와 각속도 진동이 더 작다. 따라서 전체 고장후RMSE 하나만으로 RL이 더 잘 회복했다고 결론 내리지 않는다.',
'70% 유지에서 PID는 네 모터 모두 후반 위치RMSE≈.196cm, 각속도RMS≈.0024~.0025rad/s. 1cm 정착까지 고장후≈11.65s가 걸렸다.',
'RL은 같은70%에서 모터별 후반 위치RMSE≈3.30,7.58,3.66,3.96cm로 차이가 있고 모두1cm 정착 조건을 충족하지 않았다.',
'특히 RL M2_eta070은 후반 body omegaRMS≈4.94rad/s, 고장후4축 평균 action범위초과비율≈.2338. 고장후 위치RMSE≈7.18cm로 같은 조건PID≈5.54cm보다도 크다. 29~30초 raw/applied 명령·실제토크 파형을 별도 제공.',
'일부 RL 모터조건의 지속진동·포화는 관측된 현상이다. 이 실행만으로 high gain, 학습량, 보상 가중치 중 하나를 원인으로 확정하지 않는다.',
'PID도70% 조건의 후반 yaw오차 약3.03deg가 남는다. 위치오차와 진동이 작다는 사실을 모든 자세오차가0이라는 뜻으로 해석하지 않는다.',
'PID에는 기존 속도 적분 상태가 있고 RL은 기존15D 관측 기반 정책이다. 추가 적분기 또는 fault-aware 보정으로 결과를 개선시키지 않았다.',
'이12조건 모두 수평·yaw고정 정적 평형을 만들 rotor command와 normalized action이 기존 한계 안에 있다(physics_verification.json cases). 부분 출력저하가 곧 물리적 호버불가능을 뜻하지 않는다.',
'단일 frozen baseline·기존 고정 PID, 정상 질량, 한 모터10~30% 손실, 수평 명목초기조건의 시뮬레이션 비교이다. 완전모터정지/동시고장/추+고장/모든OOD에 일반화하지 않는다.',
'M2/70%를 별도 단일환경에서도 재평가했다. 30초 완료와 지속 진동/포화가 재현되며, 병렬환경에서만 발생하는 현상은 아니었다.',
f"후반 위치RMSE: 13환경배치 {main['response']['tail20_30']['position_rmse_m']*100:.5f}cm / 단일환경 {single['response']['tail20_30']['position_rmse_m']*100:.5f}cm. 후반각속도RMS: {main['response']['tail20_30']['omega_rms_rad_s']:.6f} / {single['response']['tail20_30']['omega_rms_rad_s']:.6f}rad/s.",
'이 반복은 궤적의 비트단위 동일성을 입증하지 않는다. 환경 배치/원점/수치 연산에 따른 차이의 원인을 분리한 실험도 아니다. 정량 차이는 independence_verification.json에 그대로 저장.',
'', '6. 검증',
f"실제 PhysX 고장검증 {len(v['physics']['checks'])}개 통과. 정상mass/COM/inertia, user/native mapping, RPM미변경, 추력/반토크단일효율, 실제 COM가속도/각가속도, 정확한2500substep 경계, 5physics/motor업데이트, 부분reset 보존/타이머 재시작 검증.",
'평가각행에서 실제force=eta*nominalforce 및 반토크를 확인하고, t<=5에서는eta1, t>5에서는지정eta인지 전수검사. 첫고장index는12조건모두2500, healthy는-1.',
'두제어기의 물성readback·초기상태·env config가 동일. 같은제어기의 고장전5초와 healthy기준도float32 허용오차 내 일치.',
'RL 128고정관측 deterministic action 최대오차: '+str(v['reload']['max_action_error']),
'새healthy와 이전nominal궤적 비교: '+json.dumps(healthy,ensure_ascii=False),
f"기존파일 {v['protected_files_checked']}개 SHA256 확인, 변경0. Baseline 및 과거실험/코드 보존.",
'TensorBoard/JSON scalar 전수일치: '+json.dumps(v['scalar_checks'],ensure_ascii=False),
'위치와 상태 source는GPU float32. CSV는Python float round-trip이며 임의 소수자리절삭 없음. 극소수치잔차를 실제기체 정밀도로 해석하지 않음.',
'PID는 원본 NumPy코드/독립적분기, GPU PhysX batch. CPU↔GPU 전송을 포함하므로 GPU-only 제어성능 benchmark가 아니다. RL은 inference만 수행.',
'GUI는 실제 실행하지 않음. 설치·드라이버·TensorBoard기존서비스 변경 없음.',
'', '7. 시간 및 산출물']
for k in ctrls:
 e=v['executions'][k];lines.append(f"{k}: init {e['initialization_s']:.3f}s, simulation wall {e['simulation_wall_s']:.3f}s, total {e['total_wall_s']:.3f}s, scored samples {e['scored_policy_samples']}, batch physics steps {e['physics_batch_steps']}, optimizer0")
lines.append(f"단일환경 추가검증: init {se['initialization_s']:.3f}s, simulation wall {se['simulation_wall_s']:.3f}s, total {se['total_wall_s']:.3f}s; 본 비교26조건과 분리, optimizer0.")
lines+=['시계는Python imports 후 simulation launch직전부터. 초기화/실제simulation/총시간 구분. 학습없음. 두 GPU 실행은 순차.',
'보고서 REPORT_phase17_ko.txt; experiment_plan.json, pid_config.json, baseline/reward/agent 설정; summary.csv/json, overview.json, verification.json.',
'evaluation/{RL,PID}/{healthy,M1_eta090,...,M4_eta070}/: result.json, trajectory.csv, spectra.json. native arrays0..3와 user_*1..4를 명시적으로 구분.',
'TensorBoard: evaluation/{RL,PID}/tensorboard_time. 가로축control step, time=step/100s. 이벤트파일과같은값 tensorboard_scalars.json에 보존.',
'plots/01_motor_numbering.png: 모터 번호/반토크/기하',
'02_postfault_duration.png: 고장후 지속시간',
'03_postfault_position_rmse.png: 완료조건 고장후 위치오차',
'04_postfault_body_omega.png: 고장후 각속도RMS',
'05_position_error_all_conditions.png: 12조건 PID/RL 위치오차시간이력',
'06_M1_eta070_states.png: 대표M1/70% 위치·자세·각속도',
'07_M1_eta070_motor_response.png: 실제/nominal추력·RPM·효율',
'08_M1_eta070_commands_wrench.png: raw/applied명령·요청/실제wrench',
'09_worst_RL_tail_29_30s.png: RL고장후RMSE최대조건(M2/70%) 후반확대. 결과를 숨기지 않기 위한 진단이며 추가계수선택/튜닝에 사용하지 않음.',
'각PNG 대응CSV, 전체원시궤적, source_snapshot, source_changes, COMMANDS 제공. 현재Git은commit없는작업본이므로source/hash로재현.',
'review.tar.gz에는보고서/설정/검증/원시자료/plots/source/filelist/SHA256 포함. checkpoint는기존baseline경로에보존하고중복포함하지않음. reference_observations.pt는네트워크checkpoint가아님.',
'압축파일자체SHA256은review.tar.gz.sha256, 내용전수검사는archive_verification.json. 추가학습/튜닝/고장조건확대 없이 종료.']
(r/'REPORT_phase17_ko.txt').write_text('\n'.join(lines)+'\n');print('REPORT_READY',r,flush=True)

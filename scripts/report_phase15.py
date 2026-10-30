"""Finalize evidence/report for the fixed-policy rigid-payload evaluation."""
import csv,json,sys,time,shutil,subprocess,tarfile,hashlib
from pathlib import Path
import numpy as np
from ppo_common import ROOT,sha256,dump_json
r=Path(sys.argv[1]);plan=json.loads((r/'experiment_plan.json').read_text());v=json.loads((r/'verification.json').read_text());suites=['hover','circle_T10','circle_T5'];results={s:json.loads((r/'evaluation'/s/'results.json').read_text()) for s in suites};base=Path(plan['baseline']);source=Path(json.loads((base/'BASELINE_INFO.json').read_text())['source_run'])
def arr(path,keys):
 with Path(path).open() as f:rows=list(csv.DictReader(f))
 return np.array([[float(x[k]) for k in keys] for x in rows])
# Compare fresh zero-payload behavior with frozen historical evidence.
history={};keys=['ex','ey','ez','vx','vy','vz','wx','wy','wz']
for s in suites:
 old=source/'evaluation/best_fixed_30s/best/nominal.csv' if s=='hover' else source/f'circle/cases/phase14z_best_T{10 if s=="circle_T10" else 5}/trajectory.csv'
 new=r/'evaluation'/s/'nominal/trajectory.csv';a=arr(old,keys);b=arr(new,keys);assert a.shape==b.shape
 poserr=float(np.max(np.abs(a[:,:3]-b[:,:3])));omegaerr=float(np.max(np.abs(a[:,6:]-b[:,6:])));assert poserr<1e-4 and omegaerr<.01
 history[s]=dict(source=str(old),source_sha256=sha256(old),samples=len(a),position_max_abs_difference_m=poserr,omega_max_abs_difference_rad_s=omegaerr,position_tolerance_m=1e-4,omega_tolerance_rad_s=.01,passed=True)
# One representative loaded case is rerun alone, in its own fresh process.
idir=r/'evaluation/independence_m04g_x10cm';alone=json.loads((idir/'results.json').read_text())[0];batch=next(x for x in results['hover'] if x['name']=='m04g_x10cm');a=arr(idir/'m04g_x10cm/trajectory.csv',keys);b=arr(r/'evaluation/hover/m04g_x10cm/trajectory.csv',keys);assert a.shape==b.shape
poserr=float(np.max(np.abs(a[:,:3]-b[:,:3])));omegaerr=float(np.max(np.abs(a[:,6:]-b[:,6:])));ind=dict(case='m04g_x10cm',separate_process=True,envs_compared=[1,28],completed_same=alone['completed']==batch['completed'],duration_difference_s=abs(alone['duration_s']-batch['duration_s']),position_max_abs_difference_m=poserr,omega_max_abs_difference_rad_s=omegaerr,position_tolerance_m=1e-4,omega_tolerance_rad_s=.01)
ind['passed']=ind['completed_same'] and ind['duration_difference_s']==0 and poserr<1e-4 and omegaerr<.01;assert ind['passed'],ind
v.update(unloaded_historical_comparison=history,representative_batch_independence=ind);v['independence_execution']=json.loads((idir/'execution.json').read_text());dump_json(r/'verification.json',v)
# Reproducible source snapshot including old modules imported by the new scripts.
for folder in ['src','scripts']:
 shutil.copytree(ROOT/folder,r/'source_snapshot'/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['pyproject.toml']:
 if (ROOT/name).exists():shutil.copy2(ROOT/name,r/'source_snapshot'/name)
shutil.copy2(base/'BASELINE_INFO.json',r/'baseline_info.json');shutil.copy2(base/'best_reference.pt',r/'reference_observations.pt')
git={}
for name,args in [('status',['git','status','--short']),('head',['git','rev-parse','HEAD']),('branch',['git','branch','--show-current'])]:
 p=subprocess.run(args,cwd=ROOT,text=True,capture_output=True);git[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
dump_json(r/'evidence/git_state.json',git)
changed=['scripts/phase15_payload.py','scripts/validate_phase15.py','scripts/evaluate_phase15.py','scripts/analyze_phase15.py','scripts/report_phase15.py','scripts/package_phase15.py']
dump_json(r/'source_changes.json',dict(added_files={p:sha256(ROOT/p) for p in changed},existing_files_modified=[],protected_files_checked=v['protected_files_checked'],git_note='Current repository has no commits; source_snapshot and SHA256 are the reproducibility baseline.'))
PYEXE='/home/mrl_6534/miniconda3/envs/isaaclab/bin/python'
commands=f'''# SSH on MRL-server. No training, optimizer updates, driver changes.
cd {ROOT}
PY={PYEXE}
RUN={r}

# Focused physics checks (rerun evidence would overwrite verification, use a new run copy to preserve it):
"$PY" scripts/validate_phase15.py --run-dir "$RUN" --device cuda:0 --headless --physics isaacsim_physx

# Actual scored commands used, sequentially:
'''
for s in suites:commands+=f'"$PY" scripts/evaluate_phase15.py --run-dir "$RUN" --suite {s} --device cuda:0 --headless --physics isaacsim_physx\n'
commands+=f'''
# Reevaluate without overwriting scored outputs:
"$PY" scripts/evaluate_phase15.py --run-dir "$RUN" --suite hover --case m04g_x10cm --output "$RUN/evaluation/manual_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --headless --physics isaacsim_physx

# GUI replay COMMAND ONLY, GUI not tested. Rod/sphere are visual-only; rigid point mass physics is unchanged.
"$PY" scripts/evaluate_phase15.py --run-dir "$RUN" --suite circle_T10 --case m04g_x10cm --output "$RUN/evaluation/gui_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --visualizer kit --physics isaacsim_physx
# --suite hover | circle_T10 | circle_T5; --case nominal | m02g_x10cm ... m10g_x30cm

# Offline plots and verification:
"$PY" scripts/analyze_phase15.py --run-dir "$RUN"
"$PY" scripts/report_phase15.py "$RUN"
"$PY" scripts/package_phase15.py "$RUN"

# TensorBoard: 6006/6007 were occupied and left untouched; 6008 was free at check time.
# Check again before launching; do not kill an existing service.
ss -ltn 'sport = :6008'
"$PY" -m tensorboard.main --logdir "$RUN/evaluation" --host 127.0.0.1 --port 6008
# Windows PowerShell, separate terminal:
ssh -N -L 6008:127.0.0.1:6008 mrl_6534@100.79.103.47
# Browser: http://127.0.0.1:6008
# TensorBoard X axis in this evaluation is policy sample index; time = index / 100 seconds.

# Download (Windows PowerShell):
$R = '{r}'
scp "mrl_6534@100.79.103.47:${{R}}/review.tar.gz" .
scp "mrl_6534@100.79.103.47:${{R}}/review.tar.gz.sha256" .
Get-FileHash .\\review.tar.gz -Algorithm SHA256
'''
(r/'COMMANDS.txt').write_text(commands)
lines=['고정 baseline의 전방 고정 추 평가 보고서 (phase15)',f'실행 폴더: {r}','서버: MRL-server / cuda:0 / isaacsim_physx / headless',
'추가 학습 없음. optimizer.step 호출 0회. 기존 baseline·설치·드라이버·기존 TensorBoard 서비스 유지.',
'요청 해석: 질량 2~10g을 1g 간격으로 평가. +X(기체 앞쪽) 10/20/30cm, Y=Z=0. 27 부하 조건 + 무부하 기준.',
'각 질량·위치는 결정된 격자 조건이다. 2~10g의 연속확률분포나 학습용 DR로 해석하지 않는다.',
'', '1. 기준 모델',f"{plan['checkpoint']}",f"SHA256: {plan['checkpoint_sha256']}",
'phase14_vz_weight3_fresh_20261019_120809의 최종 선정 best: checkpoint_448. 저장된 계보 14,680,064 전이 / 143,360 Adam 호출. 새로 학습하지 않음.',
'현재 평가에서 체크포인트·128개 고정 관측을 새 프로세스에 명시적으로 로드. reference_observations.pt는 체크포인트가 아님.',
'15D 관측: link 위치오차3 + 절대 world 속도3 + quaternion(wxyz)4 + body FLU 각속도3 + yaw sin/cos2.',
'4D 행동: roll/pitch/yaw 토크, 수직 총추력. clip [-1,1] 후 scale=(.0075,.0075,.001,.5), 중력 보상 .04338*9.81N 그대로.',
'기존 allocation → rotor thrust clip [0,.289]N → ESC 역변환 → 50ms RPM 지연 → 실제 추력/반토크 그대로. 물리500Hz/정책100Hz.',
'보상은 baseline 정의 그대로(축별 exponential 위치, XY 복귀속도, -.015*vz^2, roll/pitch -.075*angle^2 등). 보상은 평가 기록용이고 최적화에 사용하지 않음.',
'', '2. 부하 물리 구현과 모델링 가정',
'추는 질점, 막대는 질량0·완전 강체. 흔들리는 진자 또는 유연한 막대가 아님. 막대/추의 공기저항·자체 회전관성·충돌형상은 모델링하지 않음.',
'원래 CF 단일 강체 총질량 m0=.043384kg과 실제 엔진 COM/inertia에서 시작. nominal USD의 float32 반올림을 그대로 이어받음.',
'부착 위치 rp=(distance,0,0)는 기체 link 원점 기준 body FLU 좌표. 원래 COM은 z≈1.1064e-6m이므로 z=0 부착 시 새 COM z도 반영.',
'M=m0+mp; c=(m0*c0+mp*rp)/M.',
'I_new=I0 + m0*(||c0-c||²*I-(c0-c)(c0-c)^T) + mp*(||rp-c||²*I-(rp-c)(rp-c)^T).',
'전체 3x3 관성 텐서를 엔진에 적용. rotor arm은 r_rotor-c로 갱신. 각 rotor의 힘과 반토크를 합하여 새 COM 기준 torque를 한 번만 적용.',
'중력은 엔진에서 새 COM에 적용된다. 별도 mp*g 지레 토크를 중복 추가하지 않았다.',
'제어 allocator는 기존 nominal geometry를 유지한다. 물리 모멘트 암만 바뀐다. 요청 wrench의 토크 기준(기존 allocator)과 실제 wrench 기준(새 COM)이 달라 차이를 단순 모터 오차로 부르면 안 된다.',
'기체 link 위치/속도를 계속 관측. COM 속도를 정책 속도로 바꾸지 않음.',
'각 조건은 시작부터 추를 부착하고 수평 yaw0·선/각속도0·기존 무부하 hover RPM≈1581.5217rad/s로 시작. 추에 맞춘 RPM feedforward나 평형 초기화 없음.',
'따라서 이 결과는 시작 시 추가 중량/모멘트 불균형을 포함한다. 이미 부하 평형에 도달한 상태에서의 안정성만을 시험한 결과가 아님.',
'', '3. 프로토콜',
'호버: 목표 local[0,0,1] 고정, 최대30s. 원: R=1m, z=1m, yaw=0; [1,0,1]에서 정지 상태로 시작; T=10s 두 바퀴20s, T=5s 두 바퀴10s.',
'원에서는 이전과 동일하게 움직이는 목표 위치만 정책 오차에 반영. 분석용 원 목표 속도는 저장하지만 정책에 속도 feedforward를 추가하지 않음.',
'추는 body +X에 고정된다. yaw=0이므로 원의 접선/방사방향으로 부착 방향을 회전시키지 않는다.',
'실패: 고도 .2~2.5m, tilt60deg, 목표 위치오차1.5m. 실제 첫 실패에서 기록 종료. timeout은 정상 완료. autoreset 이후 기록은 궤적에 포함하지 않음.',
'28환경 GPU 배치 평가. 각 case는 서로 다른 물성을 갖고 동일 frozen actor를 사용. 이후 독립 1환경 재실행으로 대표 부하 조건 일치 확인.',
'', '4. 실제 결과 (RMSE=오차 제곱평균제곱근; RMS=신호 제곱평균제곱근)',
'조건 | hover 완료/시간/RMSE(cm) | T10 완료/시간/RMSE(cm) | T5 완료/시간/RMSE(cm)']
for name in ['nominal']+[x['name'] for x in plan['cases'][1:]]:
 vals=[]
 for s in suites:
  z=next(x for x in results[s] if x['name']==name);vals.append(f"{'완료' if z['completed'] else '실패'}/{z['duration_s']:.2f}s/{z['position_rmse_m']*100:.3f}")
 lines.append(name+' | '+' | '.join(vals))
lines+=['실패 조건의 RMSE는 짧은 관측 구간의 수치이며, 완료 조건과 성능 순위를 비교할 수 없다.',
'각 프로토콜에서 부하 조건 3/27 완료: +X10cm의 2/3/4g. 무부하 기준은 세 프로토콜 모두 완료. 부하 실패는 모두 max_tilt.',
'30초 호버 완료는 정밀한 위치 유지와 다르다. 2/3/4g에서 전방 X 편향이 커지며, 목표 오차가 누적 토크 요구를 보상하는 상태와 양립하지만 별도 원인 식별 실험은 하지 않았다.',
'호버 후반 20~30초 값: 조건 | 평균 XYZ 오차(m) | 각속도 벡터 RMS(rad/s)']
for z in results['hover']:
 if z['completed']:lines.append(f"{z['name']} | {z['tail_error_vector_m']} | {z['tail_body_omega_rms_rad_s']:.6g}")
lines+=['', '5. 물리/출력 한계와 정책 실패를 구분',
'정적 수평 호버에서 실제 COM 기준 allocation으로 [0,0,0,Mg]를 만드는 rotor force를 풀었다. 모든 force가 [0,.289]N이어야 물리적으로 가능하다.',
'이 force를 기존 controller allocation으로 역매핑한 normalized action도 [-1,1]이어야 현행 출력 경로에서 해당 평형을 만들 수 있다. 본 격자에서 feasible 해는 clipping 경계가 아닌 내부 해이다.',
'10cm: 물리 추력은 2~10g 모두 가능, 기존 action 범위에서는 2~7g 가능.',
'20cm: 물리 추력은 2~9g 가능, 기존 action 범위에서는 2~3g 가능.',
'30cm: 물리 추력은 2~5g 가능, 기존 action 범위에서는 2g만 가능.',
'27조건 중 물리 추력만으로 수평 평형 가능한 조건21개, 기존 출력 경로까지 가능한 조건9개. 관측된 호버 성공3개.',
'따라서 6조건은 이상적 rotor force를 쓰더라도 수평 평형 불가이고, 12조건은 물리 추력은 가능하지만 현재 action/allocator로 평형을 만들 수 없다. 나머지6개의 실패는 평형 가능성만으로 배제되지 않는다.',
'10g/30cm의 COMx≈5.62cm는 rotor 전방 x≈3.536cm 밖이며, 평형 rotor thrust에 음수와 .289N 초과가 필요하다. 이 경우 단순히 학습을 늘려 해결된다고 해석하면 안 된다.',
'4g/10cm는 완료했지만 같은 mp*d의 2g/20cm는 실패했다. 동일한 중력 모멘트만 맞춰도 관성·총질량·과도응답이 달라지므로 이번 결과를 토크 크기 하나로 설명할 수 없다.',
'정적 평형 가능성은 원 추종 성공을 보장하지 않는다. 원 추종에는 추가 가속도/토크 여유가 필요하다.',
'', '6. 검증과 한계',
f"실제 PhysX focused tests {len(v['physics']['checks'])}개 통과. 상세 오차·허용치는 physics_verification.json.",
'질량·COM·전체 관성 readback, 중력 COM 가속도, 부하 torque 부호/각가속도, yaw90에서 wrench 변환, 물리적으로 가능한 평형 force, link/COM 속도 구분, 정책당 5물리/모터 갱신, 선택적 reset 보존 검증.',
'검증용 직접 평형 rotor force 주입은 사전 물리 검사에서만 사용했고, baseline 평가에는 사용하지 않음.',
'설치 버전 get_inertias는 actor/body axes의 full tensor를 반환한다. 초기 검증에서 principal-frame rotation을 한 번 더 적용한 오류를 수정했다. off-diagonal은 실제 readback과 가속도 검사로 확인했다.',
'실패했던 사전 검사와 첫 evaluation dtype 오류는 evidence/attempt* 및 eval_attempt1_*에 보존. 최초 목표 tuple의 NumPy float64 때문에 관측 double이 된 부분을 Python float로 고쳐 원래 float32 계약을 유지. 해당 실패 시 scored step0, optimizer0.',
'첫 mass reset 비교는 float64 이론값 대신 reset 직전 엔진값과 정확 일치하는지 검증하도록 변경. 이론값의 float32 반올림과 상태 보존은 별개.',
f"기존 보호 파일 {v['protected_files_checked']}개 SHA256 불변. 체크포인트 불변. 모든 성공 평가에서 actor hash 불변·optimizer 호출0·reward 합·500/100Hz 검사 통과.",
'128개 reference action의 최대 오차: '+str({s:v['checkpoint_reloads'][s]['max_action_error'] for s in suites}),
'무부하 과거 궤적과 fresh 비교: '+json.dumps(history,ensure_ascii=False),
'1환경 vs 28환경 대표 부하 비교: '+json.dumps(ind,ensure_ascii=False),
'TensorBoard JSON scalar 전수 비교: '+json.dumps(v['scalar_checks'],ensure_ascii=False),
'position은 PhysX float32 world 좌표에서 env_origin을 빼며 CSV는 float round-trip precision. 매우 작은 잔류 진동을 실제 기체 정밀도로 일반화하지 않음.',
'별도 모델링하지 않은 추 자체 형상/회전관성/막대 공기저항/굽힘/추 충돌 영향, 추가 방향(Y/-X), 비행 중 부착은 미검증.',
'단일 frozen 모델·초기 수평 상태·정해진 질량/거리 격자의 시뮬레이션 결과다. 모든 OOD에 대한 강건성이나 실제 기체 안전성을 입증하지 않는다.',
'GUI는 실행하지 않았다. 재생 명령만 COMMANDS.txt에 제공. 드라이버나 GUI 설정은 변경하지 않음.',
'', '7. 시간·아티팩트',
'초기화/실제 simulation wall/후처리 포함 elapsed는 아래 값. 초기화 시계는 모듈 import 후 launch 직전부터 시작. 프로세스 import/SSH 전송/개발 디버깅 시간은 제외. 세 GPU suite와 단독 검사는 순차 실행.']
for s in suites:
 d=v['executions'][s];lines.append(f"{s}: initialization={d['initialization_s']:.3f}s, simulation={d['simulation_wall_s']:.3f}s, total={d['total_wall_s']:.3f}s; scored policy samples={d['scored_policy_samples']}, actual batch physics steps={d['physics_batch_steps']}; optimizer=0")
lines+=['종료 환경의 추가 batch 연산은 성능평가 궤적·scored samples에 포함하지 않는다. 이번에는 학습 처리량/GPU 메모리 benchmark를 주장하지 않는다.',
'주요 결과 summary.csv/summary.json; 모든 84개 scored case의 trajectory/results는 evaluation/{hover,circle_T10,circle_T5}/에 저장.',
'plots/01_duration_completion.png: 완료/실패 지속 시간',
'plots/02_completed_position_rmse.png: 끝까지 완료한 조건의 위치 RMSE',
'plots/03_static_hover_feasibility.png: COM, 평형 pitch 요구, 물리/출력 가능성',
'plots/04_hover_first5s_states.png: 호버 초기 위치·Z·pitch',
'plots/05_circle_paths.png: 두 원 주기의 XY 경로',
'plots/06_hover_action_torque_vertical_thrust.png: raw/applied pitch 행동, 요청/실제 토크, 수직 추력',
'plots/07_completed_hover_tail.png: 완료 조건 후반 위치 편향/진동',
'모든 PNG 대응 CSV 및 전체 원시 평가 trajectory 보존. TensorBoard time 폴더의 X축=100Hz policy sample index, t=index/100초.',
'보고서·설정·snapshot·원시자료·플롯·검증·로그는 review.tar.gz에 포함. baseline checkpoint 자체는 기존 경로에 보존하고 archive에 중복하지 않음. 파일목록/전체 SHA256을 archive 내부에 포함, archive 자체 SHA256은 외부 companion 파일.',
'Git 저장소에 아직 commit이 없어 commit 해시 대신 source_snapshot과 SHA256 사용. 원래 파일을 수정하지 않고 scripts/phase15_* 등 새 파일만 추가.',
'결론: 무부하에서 학습한 baseline은 제한된 전방 부하(이번에는10cm/2~4g)의 비행을 유지하지만, 호버 위치 편향이 3~6cm로 증가한다. 더 큰 거리/질량에 대한 대응은 이번 실험에서 확인되지 않았고 대부분 실패했다. 추가 학습/정책 gain/보상 조정은 시행하지 않았다.']
(r/'REPORT_phase15_ko.txt').write_text('\n'.join(lines)+'\n')
print('REPORT_READY',str(r),flush=True)

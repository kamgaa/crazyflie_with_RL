"""Finalize measured OFF/ON report; no fitting, tuning, simulation or optimization."""
import csv,json,sys,shutil,datetime
from pathlib import Path
import numpy as np
from ppo_common import ROOT,sha256,dump_json
r=Path(sys.argv[1]);p=json.loads((r/'experiment_plan.json').read_text());v=json.loads((r/'verification.json').read_text());f=json.loads((r/'functional_verification.json').read_text());assert v['success'] and f['success']
labels={'off':'PPO 단독','on':'PPO + 외부 적분'};tasks=['hover','T10','T5'];results={(mode,t):json.loads((r/'evaluation'/mode/t/'result.json').read_text()) for mode in ['off','on'] for t in tasks}
for folder in ['src','scripts']:shutil.copytree(ROOT/folder,r/'source_snapshot'/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(ROOT/'pyproject.toml',r/'source_snapshot/pyproject.toml')
model=r/'frozen_model';model.mkdir(exist_ok=True)
shutil.copy2(p['checkpoint'],model/'phase14_best_448.pt');shutil.copy2(p['reference'],model/'reference_observations_128.pt')
assert sha256(model/'phase14_best_448.pt')==p['checkpoint_sha256']
dump_json(model/'manifest.json',dict(source_checkpoint=p['checkpoint'],copied_checkpoint='phase14_best_448.pt',sha256=p['checkpoint_sha256'],selected_iteration=448,role='unchanged copy of existing phase14 policy; no newly trained/selected best',reference_role='128 observations and deterministic actions, not a network checkpoint'))
added=['prepare_external_integral.py','external_position_integral.py','validate_external_integral.py','external_integral_metrics.py','evaluate_external_integral.py','run_external_integral.py','analyze_external_integral.py','report_external_integral.py','package_external_integral.py']
dump_json(r/'source_changes.json',dict(new_files={str(Path('scripts')/x):sha256(ROOT/'scripts'/x) for x in added if (ROOT/'scripts'/x).exists()},existing_files_modified=[],protected_files=v['protected_count']))
commands=f'''# Run on MRL-server. Same existing frozen phase14 best448; optimizer0.
cd {ROOT}
PY=/home/mrl_6534/miniconda3/envs/isaaclab/bin/python
RUN={r}

# Actual commands (already completed):
"$PY" scripts/prepare_external_integral.py
"$PY" scripts/validate_external_integral.py --run-dir "$RUN" --device cuda:0 --headless --physics isaacsim_physx
"$PY" scripts/evaluate_external_integral.py --run-dir "$RUN" --integral off --task hover --device cuda:0 --headless --physics isaacsim_physx
"$PY" scripts/run_external_integral.py "$RUN"
# Orchestrator preserves completed OFF hover; remaining five tasks run sequentially.
# Exact subprocess argument lists and timings: execution_commands.json.

# Headless replay to a NEW output directory (no overwrite, no learning):
"$PY" scripts/evaluate_external_integral.py --run-dir "$RUN" --integral on --task T10 --output "$RUN/evaluation/manual_on_T10_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --headless --physics isaacsim_physx
# --integral off|on; --task hover|T10|T5. Horizons fixed at30/20/10seconds.

# GUI command only, NOT actually tested. Use a server desktop with DISPLAY set.
"$PY" scripts/evaluate_external_integral.py --run-dir "$RUN" --integral on --task T10 --output "$RUN/evaluation/gui_on_T10_$(date +%Y%m%d_%H%M%S)" --device cuda:0 --visualizer kit --physics isaacsim_physx

# Offline post-processing:
"$PY" scripts/analyze_external_integral.py "$RUN"
"$PY" scripts/report_external_integral.py "$RUN"
"$PY" scripts/package_external_integral.py "$RUN"

# Existing6006/6007 services untouched. Check6008 before starting; do not kill another service.
ss -ltn 'sport = :6008'
"$PY" -m tensorboard.main --logdir "$RUN/evaluation" --host 127.0.0.1 --port 6008
# Windows tunnel: ssh -N -L 6008:127.0.0.1:6008 mrl_6534@100.79.103.47
# Browse http://127.0.0.1:6008. Scalar X=policy sample index; seconds=X/100.

# Windows PowerShell download:
$R = '{r}'
scp "mrl_6534@100.79.103.47:${{R}}/review.tar.gz" .
scp "mrl_6534@100.79.103.47:${{R}}/review.tar.gz.sha256" .
Get-FileHash .\\review.tar.gz -Algorithm SHA256
'''
(r/'COMMANDS.txt').write_text(commands)
lines=['Phase14 best 동결 정책 + 외부 위치 적분 OFF/ON 평가',str(r),'',
'1. 범위와 모델',
f"MRL-server / cuda:0 / isaacsim_physx / headless. 실제 checkpoint: {p['checkpoint']}",f"SHA256: {p['checkpoint_sha256']}",
'phase14 best_selection manifest 및 checkpoint448과 SHA256 일치. final512/다른단계 모델 사용 없음. 같은 정책과 Gaussian 분포 상태 동결; 각 실행 고정128관측 재로드 action 최대오차0.',
'학습0, optimizer.step0, gain 탐색0, 새로운 best 선정0. frozen_model/phase14_best_448.pt는 기존 모델의 동일해시 복사본이다. reference_observations_128.pt는 관측/출력 검증자료이며 checkpoint가 아니다.',
'총6조건: PPO 단독(OFF), PPO + 외부 적분(ON) 각각 hover30s/T10두바퀴분20s/T5두바퀴분10s. 별도독립프로세스, num_envs1, seed42, GPU순차실행. payload0, motor efficiency1, sensor noise/외란 없음.',
'', '2. 기존 계약과 이번 wrapper',
'15D 관측: link 원점 위치오차3, 절대world 속도3, quaternion wxyz4, body FLU 각속도3, yaw오차sin/cos2. 위치오차=world link position-env_origin-실제local target.',
'world Z-up/body FLU. link 원점과 COM을 혼동하지 않음. 원래 관측을 복사해 앞3개만 e_policy=e_actual-xi로 바꾸고 기존 actor 전처리 앞에 전달. 현재 obs normalization=false; 다른12채널/정책구조는 변경하지 않음.',
'가상목표=실제목표+xi. raw.target은 오직 실제 궤적용이며 wrapper가 변경하지 않는다. reward와 내부복귀속도v_des=norm_clip(-4*e_actual,1.5), 실패/평가 역시 실제목표 기준이다.',
'현재13항 phase14 reward 그대로: X/Y각.100expm1(-e²/.10²), Z .015expm1(-ez²/.05²), X/Y -.005(v-vdes)², Z -.015vz², roll/pitch각-.075angle_rad², yaw-error², bodyomega각-.001omega², failure-10. 추가dt/보너스없음.',
'행동/wrench 순서 [tau_x,tau_y,tau_z,Fz], 단위 [Nm,Nm,Nm,N]. normalized action[-1,1] 후 scale[.0075,.0075,.001,.5], Fz에.04338*9.81N 중력보상. allocation→rotor[0,.289]N→ESC역변환→RPM50ms지연→actual COM wrench. 실제힘/토크는 마지막500Hzsubstep buffer, 100Hz저장; 구간평균힘으로 해석하지 않음.',
'정상 총질량.043384kg, 기존COM/관성/로터기하/반토크 그대로. physics500Hz/policy100Hz/decimation5. 실패: 고도[.2,2.5]m, tilt60deg, 실제목표3D오차1.5m. 정상timeout과실패분리.',
'', '3. 적분 정의·순서·reset',
'Ki=diag(.20,.20,.20)s^-1, dt=.01s. xi는[N,3] GPU float32 world축 가상목표변위[m]. candidate=xi_k-.01*Ki*e_actual_k. XY벡터norm<=.40m로방향유지투영; Z±.15m clamp.',
'각t_k: e_actual와xi_k보존→복사관측e_policy로actor mean→기존한policy구간실행→제한전allocator rotor명령확인→clipping없으면보존한e_actual로한번적분. 어떤rotor든<0또는>.289N이면그환경3축모두동결.',
'정규화action clipping/모터실제추력의한계접근/ESC경계는동결조건이아니다. 실제예측전allocator가한구간held이므로500Hz마다같은판정을중복적분하지않음.',
'새episode/부분reset시선택환경xi=0. reset generation으로env.step중autoreset된환경에stale적분값덮어쓰기방지. 목표변경/향후고장event에는reset하지않음. 오차0이면기존xi유지. GUI일시정지중policy/적분step발행안함.',
'xi_used=이번행동에사용한값, xi_interval_end=그구간적분/포화처리후episode reset 직전의논리적값, xi_next=실제다음episode상태(종료/reset이면0). 최종적분값은이세역할을구분하여저장.',
f"MuJoCo 비교원본: {p['old_mujoco']['path']}",f"원본SHA256: {p['old_mujoco']['sha256']}",
'발견한과거소스는단일환경NumPy float64이고allocator OR ESC boundary OR action boundary에서동결한다. 이번명세는allocator clipping만동결하며GPU batch/부분reset보호를추가했다. 따라서과거구현과완전히동일하다고표현하지않음.',
'', '4. 시간정합과 자료 의미',
'hover: local[0,0,1]에서수평yaw0/속도0/기존healthy hover motor상태로시작. 원: p_ref(t)=[cos(2pi*t/T),sin(2pi*t/T),1], CCW,yaw0,initial[1,0,1],속도0. ramp/정착시간/추가feedforward없음.',
'기존원형평가그대로: t_k관측과target(t_k)로행동을계산, _pre_physics_step에서실제목표를target(t_k+1)로갱신, 물리step종료후reward/failure/성능은p(t_k+1)-target(t_k+1). 적분만저장한e_actual(t_k)사용.',
'CSV time_s는종료상태시각t_k+1, control_time_s는행동계산t_k. e_actual_*와e_policy_*·xi_used·virtual_target은control시각; ex/ey/ez와position/target은종료시각. 두오차를서로다른시각임을무시하고비교하지않음.',
'원시trajectory의최초기록은.01s; t0초기상태는initial_state.json에별도보존. 최초실제종료상태를reward telemetry에서autoreset전캡처하며그뒤표본연결없음.',
'RMS=신호제곱평균제곱근, RMSE=오차제곱평균제곱근. 모든오차는실제목표기준. full=(0,horizon] 표본. tail=호버(20,30]s/T10(10,20]s/T5(5,10]s. 실패로완전한후반구간이없으면tail=None/N/A.',
'phase lag=wrap(reference phase-atan2(y,x))의degree, 양수=뒤처짐. 반경오차=실제XY반경-1m, 음수=안쪽. radius정확도가시간에맞춘목표점정확도를의미하지않음. 목표두바퀴분시간완료와기체의정확한기하학적두바퀴도구분.',
'clipping fraction은해당구간모든100Hz표본×4축/로터분모. action raw |u|>1, applied saturation |u|>=1-1e-6, allocator command<0 or>.289N. integral freeze/limit fraction은policy구간수분모. limit판정허용오차1e-6m, 실제투영경계는.40/.15.',
'', '5. 실제 결과',
'과제 | 제어기 | 완료/기간 | full위치RMSEcm | tail위치RMSEcm | full/tail각속도RMSrad/s | tail모터추력std 4로터평균N | action/allocator clip | integral정지/XY한계/Z한계']
for task in tasks:
 for mode in ['off','on']:
  m=results[mode,task]['metrics'];a=m['full'];b=m['tail'];lines.append(f"{task} | {labels[mode]} | {m['completed']}/{m['duration_s']:.2f}s | {a['position_rmse_m']*100:.6f} | {b['position_rmse_m']*100:.6f} | {a['body_omega_rms_rad_s']:.7f}/{b['body_omega_rms_rad_s']:.7f} | {np.mean(b['thrust_std_N']):.8g} | {a['action_clipping_fraction']:.6f}/{a['allocation_clipping_fraction']:.6f} | {a['integral_frozen_fraction']:.6f}/{a['integral_xy_limit_fraction']:.6f}/{a['integral_z_limit_fraction']:.6f}")
lines+=['', '원형 세부: 과제/제어기 | full/tail 반경RMSEcm | full/tail 평균위상지연deg | full/tail Z RMSEcm']
for task in ['T10','T5']:
 for mode in ['off','on']:
  m=results[mode,task]['metrics'];a=m['full'];b=m['tail'];lines.append(f"{task}/{labels[mode]} | {100*a['radial_rmse_m']:.5f}/{100*b['radial_rmse_m']:.5f} | {a['phase_lag_mean_deg']:.5f}/{b['phase_lag_mean_deg']:.5f} | {100*a['xyz_rmse_m'][2]:.5f}/{100*b['xyz_rmse_m'][2]:.5f}")
lines+=['', '後반 XYZ 편향m / 평균제거std m / 최종적분(reset전)m']
for task in tasks:
 for mode in ['off','on']:
  m=results[mode,task]['metrics'];lines.append(f"{task}/{labels[mode]}: {m['tail']['bias_xyz_m']} / {m['tail']['std_xyz_m']} / {m['final_xi_before_episode_reset_m']}")
lines+=['', '6. 해석']
for task in tasks:
 off=results['off',task]['metrics'];on=results['on',task]['metrics'];change=lambda seg,key:100*(on[seg][key]/off[seg][key]-1)
 motor=100*(np.mean(on['tail']['thrust_std_N'])/np.mean(off['tail']['thrust_std_N'])-1)
 lines.append(f"{task}: ON-OFF 변화 full위치RMSE {change('full','position_rmse_m'):+.3f}%, tail위치RMSE {change('tail','position_rmse_m'):+.3f}%, tail각속도RMS {change('tail','body_omega_rms_rad_s'):+.3f}%, tail로터추력std평균 {motor:+.3f}%.")
lines+=['호버: 고정된작은XYZ편향을적분이보상하면서full/tail위치오차감소. 후반각속도RMS는거의같고, XY평균제거std는소폭증가했다. 위치편향제거와진동제거는같지않다.',
'T10: 반경RMSE는감소했지만full/tail실제목표점위치RMSE는증가. 전체평균위상지연은소폭감소해도마지막한바퀴에서는거의같거나소폭증가. 안쪽반경편향이바깥쪽편향으로변했다. 반경지표하나만으로추종개선이라결론내리지않음.',
'T5: 반경RMSE는감소했지만위상지연과full/tail위치RMSE는증가. 후반각속도와모터추력변동도증가했다. 이번고정게인은원형추종정확도개선에성공하지못했다.',
'6조건모두정상완료, action/allocator clipping0, 적분정지0, 적분한계도달0. 따라서이번성능차이를한계포화나anti-windup동결때문이라고설명할근거없음.',
'마지막1초자세/명령/실제토크파형을직접확인했다. 원형각속도RMS에는경로를도는데필요한저주파운동이포함된다. 전체RMS/모터std증가만으로새로운불안정고주파진동이발생했다고단정하지않는다.',
'호버의아주작은잔차는무잡음float32 PhysX 시뮬레이션값이다. 실제비행체의마이크로미터정밀도로확대해석하지않음. CSV는float round-trip 저장, 후처리float64가엔진정밀도를높이지않음. 기존tilt acos 계산은작은자세에서float32반올림으로0이될수있어Euler파형/각속도도함께제공.',
'이번결과는동일phase14정책+고정Ki=.20/고정한계의외부적분만비교한것이다. 고장/추가질량에대한성능은이번에시험하지않았다. 게인최적성/일반적적분우월성을주장하지않음.',
'', '7. 검증과 보존',
f"사전 CUDA/PhysX 기능검증 {len(f['checks'])}개 통과. 사후일치검사 {len(v['checks'])}개 통과. 실제optimizer.step0.",
'사전검증은OFF/zero-gain 관측·정책출력정확일치, 부호/시간적분/zeroerror유지, XY방향보존norm/Z제한, allocator별정지, action clipping만으로정지하지않음, 부분reset, 실제timeout/실패autoreset순서, 실제목표/reward/failure불변, 매movingtarget에서xi보존을포함.',
'초기검증프로세스는Isaac앱context종료뒤JSON저장코드에도달하지않아요약파일이생성되지않았다. 저장시점을앱종료전으로옮겨재검증했고초기로그도보존. 본6조건과별도이며두검증모두optimizer0.',
'OFF와과거phase14평가의신경망재로드오차0, 별도물리궤적은별개기준으로검사. 허용최대위치차1e-4m/각속도차.02rad/s를실행전plan에저장. 결과:',json.dumps(v['historical_comparison'],ensure_ascii=False),
'T10/T5 OFF와과거저장된위치/각속도표본차0. hover는기존7환경평가와이번1환경의차이를포함하여허용범위내. 이관측을모든PhysX실행의비트단위결정성보장으로확대하지않음.',
'적분recurrence/전후시각관계/실제보상수식은원시CSV를독립NumPy식으로사후확인. TensorBoard는Eval/태그만저장; PPO loss생성없음. 이벤트/JSON/CSV동일float32 scalar값전수일치.',json.dumps(v['scalars'],ensure_ascii=False),
f"기존프로젝트코드·baseline·phase14자료 {v['protected_count']}개 SHA256 확인: 변경 {len(v['protected_changed'])}개. 과거MuJoCo 원본도변경없음. 기존6006/6007서비스유지.",
'', '8. 실행시간과 산출물']
for task in tasks:
 for mode in ['off','on']:
  j=results[mode,task];lines.append(f"{task}/{mode}: init={j['initialization_s']:.3f}s, simulation={j['simulation_wall_s']:.3f}s, export전total={j['total_wall_s']:.3f}s, physics steps={j['physics_steps']}, optimizer0")
lines+=['시간은프로세스import부터초기화/실제step루프를구분. 총시간측정지점은CSV/TensorBoard export직전이므로압축/플롯시간포함안함. process wall은execution_commands.json에별도(최초OFFhover제외). GPU학습없음/평가순차.',
'결과: summary.csv/json, evaluation/{off,on}/{hover,T10,T5}/trajectory.csv/json,result.json,initial_state.json,reload.json,resolved_config.json,engine_properties.json,tensorboard_time.',
'검증: functional_verification.json,verification.json,off_historical_comparison.json,common_interval_comparison.json. 모두완료하여공통구간은각task전체와같음. 실패가있으면짧은공통기간을분리하는후처리도포함.',
'그래프: 01_hover_error_integral,02/03_T10/T5_xy_path,04/05_xyz_reference,06/07_errors_phase,08/09_integral_virtual_policy,10_hover_dynamics,11_T10_dynamics,12_T5_dynamics,13_T10_last1s_dynamics,14_T5_last1s_dynamics. PNG마다대응CSV포함.',
'재현: source_snapshot에관련프로젝트코드/이번wrapper/과거MuJoCo소스, frozen_model에동일checkpoint와128reference. 기존평가기본동작은수정하지않음. 별도entrypoint기본적분OFF, --integral on일때만활성화.',
'GUI는실제실행하지않았음. COMMANDS.txt에headless재평가/GUI재생/TensorBoard/PowerShell scp명령제공. 설치/드라이버/GUI변경없음.',
'review.tar.gz,review.tar.gz.sha256,FILELIST.txt,SHA256SUMS.txt,archive_verification.json 제공. 정해진6조건후종료. 추가학습/게인탐색/조건추가없음.']
(r/'REPORT_external_integral_ko.txt').write_text('\n'.join(lines)+'\n')
assert sha256(p['old_mujoco']['path'])==p['old_mujoco']['sha256']
print('REPORT_READY',r,flush=True)

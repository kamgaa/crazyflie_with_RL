"""Freeze a read-only PID versus phase8 best-policy circle evaluation."""
import argparse,datetime,json,shutil,socket,subprocess,sys
from pathlib import Path
from ppo_common import ROOT,sha256,dump_json,protected_hashes
p=argparse.ArgumentParser();p.add_argument('--run-dir');a=p.parse_args()
r=Path(a.run_dir).resolve() if a.run_dir else ROOT/'runs'/('circle_pid_qe_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
r.mkdir(parents=True,exist_ok=False)
for d in ['evidence','source_snapshot','models','cases','plots','tensorboard']:(r/d).mkdir()
source=ROOT.parent/'migration_phase1_20261008/source'
resolved=ROOT.parent/'migration_phase1_20261008/evidence/eval_circle_fault_0289.resolved.json'
pid_cfg=json.loads(resolved.read_text())
shutil.copy2(source/'crazyflie_rl/controllers.py',r/'source_snapshot/original_controllers.py')
shutil.copy2(resolved,r/'evidence/original_resolved_config.json')
for base in [ROOT/'scripts',ROOT/'src']:
 for f in base.rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts:
   dest=r/'source_snapshot'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
phase8=ROOT/'runs/phase8_exponential_20261012_164652'
models={}
for name in ['Q_base','E_position']:
 src=phase8/name;dst=r/'models'/name;dst.mkdir()
 for fn in ['best.pt','best_reference.pt','agent_config.json','reward_config.json']:shutil.copy2(src/fn,dst/fn)
 models[name]=dict(source=str(src/'best.pt'),sha256=sha256(src/'best.pt'),copy=str(dst/'best.pt'),reference_sha256=sha256(src/'best_reference.pt'))
for fn in ['REPORT_phase8_ko.txt','checkpoint_selection.json']:
 shutil.copy2(phase8/fn,r/'evidence'/fn)
old={str(f.relative_to(ROOT)):sha256(f) for base in [ROOT/'scripts',ROOT/'src'] for f in base.rglob('*') if f.is_file() and '__pycache__' not in f.parts and 'circle_eval' not in f.name and f.name!='evaluate_circle.py'}
dump_json(r/'evidence/existing_sources_sha256.json',old)
dump_json(r/'evidence/protected_physics_sha256.json',protected_hashes())
(r/'evidence/git_status.txt').write_text(subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True).stdout)
(r/'evidence/processes_before.txt').write_text(subprocess.check_output(['ps','-eo','pid,args'],text=True))
plan=dict(experiment='circle tracking evaluation only',created=datetime.datetime.now().isoformat(),hostname=socket.gethostname(),project=str(ROOT),run_dir=str(r),
 task='MRL-Crazyflie-Hover-E2E-v0',physics='isaacsim_physx',device='cuda:0',seed=42,models=models,
 controllers=['PID','Q_base','E_position'],periods_s=[10.,5.],radius_m=1.,laps=2,center_local_m=[0.,0.,1.],direction='ccw viewed from +Z',start_angle_rad=0.,yaw_target_rad=0.,
 duration_by_period_s={'10':20.,'5':10.},initial_position_local_m=[1.,0.,1.],initial_velocity_world_m_s=[0.,0.,0.],initial_quaternion_wxyz=[1.,0.,0.,0.],initial_motor_state='unchanged airborne hover equilibrium using body mass 0.04338 kg',
 reference='p(t)=[cos(2*pi*t/T),sin(2*pi*t/T),1]; no ramp/settle/takeoff; start from rest at p(0)',
 control_dt_s=.01,physics_dt_s=.002,decimation=5,num_envs=1,
 observation='unchanged 15D: p_link_local-p_ref(t), ABSOLUTE world velocity, wxyz, body omega, sin/cos yaw error',
 target_timing='actor/PID sees reference at t; post-step observations/reward/failure/metrics use reference at t+0.01, updated in pre-physics hook after command construction',
 pid=dict(implementation=str(source/'crazyflie_rl/controllers.py'),sha256=sha256(source/'crazyflie_rl/controllers.py'),settings=pid_cfg['controller']['pid'],vehicle=pid_cfg['vehicle'],dt_s=.01,
  native_legacy_control_hz=500,comparison_control_hz=100,feedforward=False,description='unchanged CascadePID gains; dt=0.01 for equal command rate. Position P, velocity PI, attitude P, rate PI; configured D terms are not evaluated by original code.'),
 actuator_boundary='PID wrench converted to the same normalized action, common +/-1 action clip, allocation, rotor clipping, ESC inversion and 50ms RPM delay; no direct-force shortcut',
 failure='unchanged altitude [0.2,2.5], tilt 60 deg, 3D error >1.5m relative to MOVING target; first failure ends trial',
 horizon='only time limit changes to exactly 2*T; two reference laps are not necessarily two completed physical laps',
 policy_mode='deterministic actor mean; zero-shot moving-position-reference test of hover-trained policies; no target velocity/acceleration inputs or feedforward',
 training=False,optimizer_updates=0,gain_tuning=False,additional_cases=False,
 outcome='report completion, time-aligned position RMSE, geometric radial error and phase lag separately, per-lap metrics, saturation, full trajectories')
dump_json(r/'experiment_plan.json',plan)
Path('/tmp/crazyflie_circle_current_run').write_text(str(r))
print(r)

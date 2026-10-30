"""Aggregate executed evidence and preserve source/install provenance."""
from pathlib import Path
import json,csv,hashlib,subprocess,socket,sys,datetime,importlib.metadata as md
ROOT=Path(__file__).resolve().parents[1]
rows=[];suites={}
for name in ['math','physx_1','physx_16','independence']:
    d=json.loads((ROOT/f'results/{name}/results.json').read_text())
    suites[name]=dict(passed=d['passed'],count=len(d['checks']),metadata=d['metadata'])
    rows.extend(dict(suite=name,**r) for r in d['checks'])
versions={x:md.version(x) for x in ['isaaclab','isaacsim','torch','newton','mujoco','numpy','gymnasium','crazyflie-isaaclab']}
repos={}
for name,p in [('isaaclab',ROOT.parent/'IsaacLab-3.0.0-EA'),('reference',ROOT.parent/'migration_phase1_20261008/source'),('old_master',Path('/home/mrl_6534/gwpark/crazyflie_RL')),('old_refactor',Path('/home/mrl_6534/gwpark/crazyflie_RL_refactor_v2'))]:
    repos[name]=dict(path=str(p),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),status=subprocess.check_output(['git','status','--short','--branch'],cwd=p,text=True).strip())
files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','scripts','reference'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p) and '.egg-info' not in str(p)}
math_profile=json.loads((ROOT/'results/profile/task_math.json').read_text())
data=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),hostname=socket.gethostname(),python=sys.executable,
    task='MRL-Crazyflie-Hover-E2E-v0',selected_backend='isaacsim_physx',optimizer_updates=0,
    physical_and_math_checks_passed=all(r['passed'] for r in rows),check_count=len(rows),suites=suites,
    acceptance=dict(task_math_batched_on_cuda=True,task_math_host_transfer_free=math_profile['host_device_copy_free'],
        whole_framework_step_host_transfer_free=False),
    remaining=['Installed DirectRLEnv mask compaction synchronizes every policy step.',
        'Installed PhysX COM parameter view uses CPU; framework/engine step is not transfer-free.',
        'Single rigid body omits original free propeller hinges.',
        'Contact dynamics/long-horizon trajectories and GUI display are not validated in this phase.',
        'Motor polynomial remains source paper_candidate/unverified; no hardware calibration.'],
    versions=versions,repositories=repos,source_sha256=files,checks=rows)
(ROOT/'results/validation_summary.json').write_text(json.dumps(data,indent=2)+'\n')
with (ROOT/'results/validation_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['suite','name','expected','measured','max_abs_error','tolerance','unit','passed','note']);w.writeheader();w.writerows(rows)
from crazyflie_isaaclab.tasks.crazyflie import constants as k
contract={name:getattr(k,name) for name in dir(k) if name.isupper()}
contract.update(dict(observation_order=['position_error_world_xyz_m','link_velocity_world_xyz_m_s','quaternion_wxyz','body_omega_flu_xyz_rad_s','sin_yaw_error','cos_yaw_error'],observation_dimension=15,observation_dtype='float32',normalization='unit quaternion and canonical w>=0 only',action_order=['roll','pitch','yaw','collective'],action_range=[-1,1],action_dimension=4,reward_default='A',reward_variants=['A','B'],reward_dt_multiplier=False,payload_mass=0,motor_efficiency=[1,1,1,1],physical_randomization=False,policy_decimation=5,policy_dt=.01,episode_length_seconds=8,reset_position_range=0,reset_rpm_rule='body mass equilibrium when z>0.1, otherwise zero',body_simplification='locked-child composite single rigid body',physics_backend='isaacsim_physx'))
(ROOT/'results/contract.json').write_text(json.dumps(contract,indent=2)+'\n')
print('SUMMARY',data['physical_and_math_checks_passed'],len(rows),data['acceptance'])
for name in ['engine_mass','engine_com','engine_inertia','gravity_only_initial_acceleration','zero_action_acceleration_100ms','elapsed_simulation_time','total_mass_hover_velocity_1s','task_rpm_response_50ms']:
    r=next(x for x in rows if x['suite']=='physx_1' and x['name']==name)
    print(name,'MEASURED',r['measured'],'EXPECTED',r['expected'],'ERROR',r['max_abs_error'])

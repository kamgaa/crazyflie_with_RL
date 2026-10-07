"""Read-only production audit; static probes change isolated models only.

Run from repository root. Outputs a new directory and a separate real PPO run.
No training, parameter updates, source config edits, or checkpoint writes.
"""
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import yaml
from crazyflie_rl.config import load_config
from crazyflie_rl.dr_policy import sha256, load_frozen_policy
from crazyflie_rl.dr_transfer import Case, EvaluationAdapter
from crazyflie_rl.interactive_eval import InteractiveEnv, evaluation_config, run

archived_run = ROOT / 'artifacts/runs/ppo_e2e_hover_interactive-motor-effectiveness_seed42_20261001-141021'
output = Path(tempfile.mkdtemp(prefix='payload-moment-audit-', dir=ROOT/'artifacts/runs'))
(output/'reproduce.py').write_text(Path(__file__).read_text())
runtime = yaml.safe_load(next((archived_run/'config').glob('*runtime-resolved*.yaml')).read_text())
checkpoint = Path(runtime['model']['path'])
protected = [p for p in archived_run.rglob('*') if p.is_file()] + [checkpoint,
    ROOT/'configs/eval_e2e_interactive_payload.yaml', ROOT/'crazyflie_rl/environment.py']
hashes = {str(p): sha256(p) for p in protected}
raw = yaml.safe_load(next((archived_run/'config').glob('*resolved-config*.yaml')).read_text())
raw.pop('source_path', None)
actuator=raw['actuator']
actuator['thrust_polynomial']={key.removeprefix('thrust_polynomial_'):actuator.pop(key)
    for key in list(actuator) if key.startswith('thrust_polynomial_')}
saved_config = output/'archived_evaluation.yaml'
saved_config.write_text(yaml.safe_dump(raw, sort_keys=False))
config = evaluation_config(load_config(saved_config))
config03 = replace(config, environment=replace(config.environment,
    payload=replace(config.environment.payload, mass=.005, offset=(.03, 0.))))

def create(config):
    env = InteractiveEnv(config=config)
    EvaluationAdapter(env).reset_to_case_initial_state(Case('interactive',8.,goal=(0.,0.,1.)),42)
    return env

def snapshot(env):
    m,d,b = env.model,env.data,env.drone_bid
    return dict(body_mass=float(m.body_mass[b]), body_ipos=m.body_ipos[b].tolist(),
        world_inertial_minus_body_origin=(d.xipos[b]-d.xpos[b]).tolist(),
        body_sameframe=int(m.body_sameframe[b]),body_subtreemass=float(m.body_subtreemass[b]),
        sum_body_masses=float(m.body_mass.sum()),body_inertia=m.body_inertia[b].tolist(),
        dist_torque_body=env.dist_torque_body.tolist(),initial_motor_thrust=env._last_f.tolist())

env=create(config)
try:
    reset02=snapshot(env)
    # Reuse the recorded action sequence, not a fake policy. Check original trajectory
    # and capture the actual extra torque assigned by environment.step.
    rows=list(csv.DictReader((archived_run/'rollout.csv').open()))
    replay_errors=[]; applied=[]
    for row in rows[:10]:
        env.step(np.array([float(row[f'policy_action_{i+1}']) for i in range(4)]))
        replay_errors.append(float(np.max(np.abs(env.data.qpos[:3]-
            [float(row[f'position_{i+1}']) for i in range(3)]))))
        applied.append(env.data.xfrc_applied[env.drone_bid].tolist())
    assert max(replay_errors)<1e-12
finally: env.close()

def log_comparison(path, offset):
    rows=list(csv.DictReader((path/'rollout.csv').open()))
    def vec(name,n):
        return np.array([[float(r[f'{name}_{i+1}']) for i in range(n)] for r in rows])
    t=np.array([float(r['time_post']) for r in rows]); f=vec('motor_thrust_applied',4)
    v=vec('linear_velocity',3); w=vec('angular_velocity',3); att=vec('attitude_deg',3)
    efficiency=vec('motor_effectiveness',4)
    # A declared near-hover sample filter, not a claim that every sample is equilibrium.
    selected=(np.linalg.norm(v,axis=1)<.02)&(np.linalg.norm(w,axis=1)<.1)&(
        np.max(np.abs(att[:,:2]),axis=1)<2)&np.all(efficiency==1,axis=1)
    pitch=.03536*(-f[:,0]+f[:,1]+f[:,2]-f[:,3])
    return dict(run=str(path),sample_filter='speed<.02m/s, angular speed<.1rad/s, |roll,pitch|<2deg, all effectiveness=1',
        selected_samples=int(selected.sum()),time_extent_s=[float(t[selected].min()),float(t[selected].max())] if selected.any() else None,
        expected_physical_pitch_moment_origin_nm=-.005*9.81*offset,
        mean_rotor_forces_n=f[selected].mean(axis=0).tolist() if selected.any() else None,
        mean_pitch_moment_xml_origin_nm=float(pitch[selected].mean()) if selected.any() else None,
        mean_pitch_moment_logged_allocator_origin_nm=float(vec('wrench_actual',4)[selected,1].mean()) if selected.any() else None,
        mean_total_thrust_n=float(f[selected].sum(axis=1).mean()) if selected.any() else None,
        mean_rpy_deg=att[selected].mean(axis=0).tolist() if selected.any() else None,
        limitation='CSV thrust is last substep, attitude is post-step; filter is near-static only; xfrc is absent in original CSV')

probes=[]; snapshots={}
for rebuild in (False,True):
    env=create(config03)
    try:
        m,d=env.model,env.data
        if rebuild:
            # Counterfactual on this isolated model only. Never a production patch.
            mujoco.mj_setConst(m,d)
        d.qpos[:3]=[0,0,1];d.qpos[3:7]=[1,0,0,0];d.qvel[:]=0
        mujoco.mj_forward(m,d)
        snapshots[str(rebuild)]=snapshot(env)
        total=float(m.body_mass.sum()); g=env.gravity; tau=.005*.03*g
        for label,trim_scale,external in [('equal_no_extra',0,0),('equal_with_extra',0,tau),
            ('physical_trim_no_extra',1,0),('physical_trim_with_extra',1,tau),('double_trim_with_extra',2,tau)]:
            forces=np.full(4,total*g/4)+np.array([1,-1,-1,1])*trim_scale*tau/(4*.03536)
            d.ctrl[env.act_force]=forces
            d.ctrl[env.act_torque]=env.torque_coefficient*np.array([1,-1,1,-1])*forces
            d.xfrc_applied[:]=0;d.xfrc_applied[env.drone_bid,4]=external
            mujoco.mj_forward(m,d)
            expected_gravity=tau if rebuild else 0.
            expected_residual=-trim_scale*tau+expected_gravity+external
            assert abs(-d.qfrc_bias[4]-expected_gravity)<1e-12
            assert abs(d.qfrc_actuator[4]+trim_scale*tau)<1e-12
            assert abs(d.qfrc_smooth[4]-expected_residual)<1e-12
            if abs(expected_residual)<1e-12: assert np.max(np.abs(d.qacc[:6]))<1e-9
            probes.append(dict(rebuilt_constants=rebuild,case=label,
                rotor_forces_n=forces.tolist(),gravity_pitch_origin_nm=float(-d.qfrc_bias[4]),
                rotor_pitch_origin_nm=float(d.qfrc_actuator[4]),external_pitch_nm=external,
                residual_pitch_origin_nm=float(d.qfrc_smooth[4]),pitch_accel_rad_s2=float(d.qacc[4]),
                contact_count=int(d.ncon)))
    finally: env.close()

config03path=output/'evaluation_5g_3cm.yaml'
raw['environment']['payload'].update(mass=.005,offset=[.03,0.],randomize=False)
config03path.write_text(yaml.safe_dump(raw,sort_keys=False))
config03=load_config(config03path)
policy=load_frozen_policy('e2e',str(checkpoint),config03)
rollout,trace,end=run(config03,policy,duration=2.,headless=True,realtime=False,
    command=['payload_moment_audit','5g_3cm','2sec','no_learning'])

assert all(sha256(Path(p))==h for p,h in hashes.items())
report=dict(mujoco_version=mujoco.__version__,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    git_status=subprocess.check_output(['git','status','--short'],text=True),
    source_sha256={str(p):sha256(ROOT/p) for p in ['crazyflie_rl/environment.py','crazyflie_rl/interactive_eval.py','resources/mujoco/cf21B_500.xml']},
    original_run=str(archived_run),original_resolved_payload=config.environment.payload.__dict__,
    archived_runtime_payload=runtime['payload'],reset_2cm=reset02,
    action_replay_first_10_max_position_error_m=max(replay_errors),replayed_xfrc=applied,
    original_log_comparison=log_comparison(archived_run,.02),
    probe_3cm_resets=snapshots,static_probes=probes,
    new_3cm_log_comparison=log_comparison(rollout,.03),new_3cm_outcome=end,
    protected_hashes=hashes,protected_files_unchanged=True,
    conclusions=['Current constants leave xipos at body origin despite nonzero body_ipos.',
      'Current runtime explicit m*g*offset torque acts once; not twice in this checkout/model.',
      'Rebuilding constants AND retaining explicit torque doubles the static gravity moment.',
      'No production dynamics changes made; physical correction must refresh constants and remove duplicate explicit payload gravity torque together.'])
# nested dataclass randomization limits isn't needed for this fixed-payload audit
report['original_resolved_payload']={k:v for k,v in report['original_resolved_payload'].items() if k!='randomization_limits'}
(output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
with (output/'static_probes.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=probes[0].keys());writer.writeheader();writer.writerows(probes)
print('AUDIT RESULTS:',output)
print(json.dumps(report['original_log_comparison'],indent=2))
print(json.dumps(report['new_3cm_log_comparison'],indent=2))

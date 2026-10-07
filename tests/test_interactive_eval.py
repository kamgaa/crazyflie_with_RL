"""No learning: E2E identity, rotor output signals, event ordering and saved exits."""
from dataclasses import replace
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.dr_policy import load_frozen_policy, sha256
from crazyflie_rl.dr_transfer import EvaluationAdapter, Case
from crazyflie_rl.environment import CrazyflieResidualEnv
from crazyflie_rl.interactive_eval import (ROOT, InteractiveEnv, KeyEvents, Controls, evaluation_config,
                                         rotor_mapping, run, main)

CHECKPOINT = ROOT/'artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip'


@pytest.fixture
def config(tmp_path):
    config = evaluation_config(load_config(ROOT/'configs/eval_dr_transfer.yaml'))
    return replace(config,paths=replace(config.paths,artifact_root=tmp_path))


@pytest.fixture
def policy(config,monkeypatch):
    if not CHECKPOINT.is_file():
        pytest.skip('explicit user checkpoint unavailable')
    from stable_baselines3 import PPO
    def forbidden(*args,**kwargs):
        raise AssertionError('PPO optimization forbidden')
    monkeypatch.setattr(PPO,'learn',forbidden)
    monkeypatch.setattr(PPO,'train',forbidden)
    return load_frozen_policy('e2e',str(CHECKPOINT),config)


def initialize(env):
    adapter = EvaluationAdapter(env)
    adapter.reset_to_case_initial_state(Case('interactive',8,goal=(0,0,1)),42)
    return adapter


def test_fixed_payload_yaml_preserved_and_applied(config):
    source = load_config(ROOT/'configs/eval_e2e_interactive_payload.yaml')
    # Even randomize=true is disabled without deleting the fixed payload.
    source = replace(source, environment=replace(source.environment,
        payload=replace(source.environment.payload, randomize=True)))
    fixed = evaluation_config(source)
    assert source.environment.payload.randomize
    assert not fixed.environment.payload.randomize
    assert fixed.environment.payload.mass == .005
    assert fixed.environment.payload.offset == (.02, 0.)
    assert fixed.environment.position_perturbation == fixed.environment.attitude_perturbation_deg == 0
    assert fixed.environment.initial_pose_randomization is None
    assert not fixed.actuator.randomization.enabled
    env = InteractiveEnv(config=fixed)
    try:
        initialize(env)
        total = env._m0 + .005
        assert env.model.body_mass[env.drone_bid] == pytest.approx(total)
        np.testing.assert_allclose(env.model.body_ipos[env.drone_bid],
            (env._m0*env._ipos0 + .005*np.array([.02,0,0]))/total)
        np.testing.assert_allclose(env.model.body_inertia[env.drone_bid],
            env._J0 + env._m0*.005/total*np.array([0,.02**2,.02**2]))
        np.testing.assert_allclose(env._last_f, np.full(4,total*env.gravity/4))
        assert env.mass == config.vehicle.mass  # Policy's hover bias stays nominal.
        controls=Controls(); events=[]
        controls.process(['1']*10,env,0,events.append)
        env.step(np.zeros(4))
        np.testing.assert_allclose(env._last_f,env.nominal_thrust*np.array([.8,1,1,1]))
        np.testing.assert_allclose(env._last_q_actual,env.nominal_reaction_torque*np.array([.8,1,1,1]))
        controls.process(['0'],env,1,events.append)
        assert env.model.body_mass[env.drone_bid] == pytest.approx(total)
    finally: env.close()


def test_fixed_payload_checkpoint_and_saved_metadata(config,policy):
    import yaml
    source = load_config(ROOT/'configs/eval_e2e_interactive_payload.yaml')
    source = replace(source, paths=config.paths)
    before = sha256(CHECKPOINT)
    root,trace,end = run(source,policy,duration=.05,headless=True,realtime=False)
    runtime = yaml.safe_load(next((root/'config').glob('*runtime-resolved*.yaml')).read_text())
    assert runtime['payload']['mass_kg'] == .005
    assert runtime['payload']['offset_body_xy_m'] == [.02,0.]
    assert runtime['payload']['randomize'] is False
    assert runtime['payload']['total_mass_kg'] > runtime['payload']['policy_hover_mass_kg']
    assert trace.sample_count == 5 and end['end_reason'] == 'duration'
    assert sha256(CHECKPOINT) == before


def test_nominal_real_checkpoint_exact_trajectory(config,policy):
    before = sha256(CHECKPOINT)
    nominal = CrazyflieResidualEnv(config=config)
    interactive = InteractiveEnv(config=config)
    try:
        a,b = initialize(nominal),initialize(interactive)
        assert a.snapshot() == b.snapshot()
        policy.bind(nominal)
        for _ in range(80):
            action_a = policy.predict(a.current_observation())
            action_b = policy.predict(b.current_observation())
            np.testing.assert_array_equal(action_a,action_b)
            x,y=nominal.step(action_a),interactive.step(action_b)
            np.testing.assert_array_equal(x[0],y[0]); assert x[1:4]==y[1:4]
            assert a.snapshot()==b.snapshot()
    finally:
        nominal.close(); interactive.close()
    assert sha256(CHECKPOINT)==before


def test_target_events_never_reset_state_or_motors(config):
    env = InteractiveEnv(config=config)
    try:
        adapter=initialize(env); controls=Controls(.02); events=[]
        before=adapter.snapshot()
        keys=KeyEvents()
        keys(ord('W')); keys(ord('D')); keys(ord('R'))
        assert adapter.snapshot()==before  # callback queues only
        controls.process(keys.drain(),env,0,events.append)
        np.testing.assert_allclose(env.pos_des,[.02,.02,1.02])
        after=adapter.snapshot()
        for key in before.keys()-{'reference','observation'}:
            assert after[key]==before[key]
        np.testing.assert_allclose(adapter.current_observation()[:3],[-.02]*3)
        assert [e['key'] for e in events]==['W','D','R']
        assert all(e['simulation_time']==0 and e['control_step']==0 for e in events)
    finally: env.close()


def test_counts_independence_restore_and_pause_queue(config):
    env=InteractiveEnv(config=config)
    try:
        adapter=initialize(env); c=Controls(); events=[]
        before=adapter.snapshot()
        c.process(['1']*10,env,0,events.append)
        np.testing.assert_array_equal(env.motor_effectiveness,[.8,1,1,1])
        c.process(['1']*5+['3'],env,0,events.append)
        np.testing.assert_array_equal(env.motor_effectiveness,[.7,1,.98,1])
        c.process(['2']*80,env,0,events.append)
        assert env.motor_effectiveness[1]==0
        c.process(['0'],env,0,events.append)
        assert adapter.snapshot()==before
        c.process([' ','W','4'],env,0,events.append)
        assert c.paused and len(c.pending)==2
        assert adapter.snapshot()==before
        c.process([' '],env,0,events.append)
        assert not c.paused and not c.pending
        assert env.pos_des[0]==.02 and env.motor_effectiveness[3]==.98
        assert [e['event_type'] for e in events[-3:]]==['resume','target','degrade']
    finally: env.close()


def test_numbering_and_once_only_post_actuator_force_torque(config):
    a,b=InteractiveEnv(config=config),CrazyflieResidualEnv(config=config)
    try:
        initialize(a); initialize(b)
        mapping=rotor_mapping(a)
        assert [m['force_actuator'] for m in mapping]==[f'motor{i}_force' for i in range(4)]
        positions=np.array([m['position_body_m'] for m in mapping])
        np.testing.assert_allclose(a.B[0],positions[:,1],atol=6e-6)
        np.testing.assert_allclose(a.B[1],-positions[:,0],atol=6e-6)
        lam=np.array([.8,.7,0,1]); a.motor_effectiveness=lam
        wrench=np.array([.001,-.0005,.0001,config.vehicle.mass*config.vehicle.gravity])
        for _ in range(20):
            # Compare signals before plant integration, including every actuator update.
            a._apply_control(wrench); b._apply_control(wrench)
            np.testing.assert_array_equal(a.nominal_thrust,b._last_f)
            np.testing.assert_array_equal(a._last_f,lam*b._last_f)
            np.testing.assert_array_equal(a._last_q_actual,lam*b._last_q_actual)
            np.testing.assert_array_equal(a.data.ctrl[a.act_force],a._last_f)
            np.testing.assert_array_equal(a.data.ctrl[a.act_torque],a._last_q_actual)
            np.testing.assert_array_equal(a._last_omega,b._last_omega)
            np.testing.assert_array_equal(a._actuator.last_f,b._actuator.last_f)
            np.testing.assert_array_equal(a.B,b.B)
        a.motor_effectiveness[:]=1
        a._apply_control(wrench); b._apply_control(wrench)
        np.testing.assert_array_equal(a.data.ctrl,b.data.ctrl)
    finally: a.close(); b.close()


@pytest.mark.parametrize('exit_kind',['duration','quit','keyboard_interrupt','exception','window_closed','physical','pause'])
def test_saved_exit_paths(config,policy,exit_kind):
    tick=0
    def hook(step,keys,env,controls):
        nonlocal tick
        tick+=1
        if step==0 and tick==1:
            keys(ord('1')); keys(ord('0'))
        if exit_kind=='pause':
            if tick==2:
                keys(ord(' ')); keys(ord('W'))
            elif tick==3:
                assert controls.paused and env._step==1
                assert env.data.time==pytest.approx(.01)
                keys(ord(' '))
        if step==2:
            if exit_kind=='quit': keys(ord('Q'))
            if exit_kind=='keyboard_interrupt': raise KeyboardInterrupt
            if exit_kind=='exception': raise RuntimeError('injected original error')
            if exit_kind=='physical':
                # Test injection only: trigger the unchanged altitude guard.
                env.data.qpos[2]=.1
    class ClosedViewer:
        def __init__(self,env,keys): self.count=0
        def is_running(self):
            self.count+=1
            return self.count<3
        def update(self,*args): pass
        def close(self): pass
    args=dict(duration=.04,headless=exit_kind!='window_closed',realtime=False,boundary_hook=hook,viewer_factory=ClosedViewer)
    if exit_kind=='exception':
        with pytest.raises(RuntimeError,match='injected original error'):
            run(config,policy,**args)
        root=next((config.paths.artifact_root/'runs').iterdir())
    else:
        root,trace,end=run(config,policy,**args)
        expected={'physical':'min_altitude','pause':'duration'}.get(exit_kind,exit_kind)
        assert end['end_reason']==expected
        assert (trace.sample_count==4) if exit_kind in ('duration','pause') else (trace.sample_count<=3)
    manifest=json.loads(next((root/'manifests').glob('*manifest*.json')).read_text())
    evaluation=json.loads(next((root/'metrics').glob('*evaluation*.json')).read_text())
    assert set(evaluation['policies'])=={'e2e'}
    assert (root/'events.csv').is_file() and (root/'rollout.csv').is_file()
    assert (root/'plots/ppo.png').is_file() and (root/'trace.npz').is_file()
    assert (root/'plots/motor_effectiveness.png').is_file()
    assert manifest['result']['checkpoint_hash_unchanged']
    assert not list((root/'models').glob('*.zip'))
    with (root/'rollout.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert 'motor_effectiveness_1' in rows[0] and 'motor_thrust_before_effectiveness_4' in rows[0]
    for row in rows:
        for i in range(1,5):
            assert float(row[f'motor_thrust_applied_{i}'])==float(row[f'motor_thrust_before_effectiveness_{i}'])*float(row[f'motor_effectiveness_{i}'])


def test_unlimited_passes_eight_seconds_and_empty_exit(config,policy):
    def finish(step,keys,env,controls):
        assert env.max_steps==float('inf')
        if step==805: keys(ord('Q'))
    root,trace,end=run(config,policy,headless=True,realtime=False,boundary_hook=finish)
    assert trace.sample_count==805 and end['actual_duration_sec']==8.05
    assert not end['truncated']
    def immediate(step,keys,env,controls): keys(ord('Q'))
    root,trace,end=run(config,policy,headless=True,realtime=False,boundary_hook=immediate)
    assert trace.sample_count==0 and not (root/'plots/ppo.png').exists()
    metrics=json.loads(next((root/'metrics').glob('*evaluation*.json')).read_text())
    assert metrics['policies']['e2e']['position_rmse'] is None


def test_config_cli_validation_and_import_safety(config):
    with pytest.raises(ValueError,match='e2e'):
        evaluation_config(replace(config,environment=replace(config.environment,control_mode='residual')))
    for value in (0,-.02,float('nan')):
        with pytest.raises(ValueError): Controls(value)
    with pytest.raises(SystemExit): main(['--model',str(CHECKPOINT),'--headless'])
    import subprocess,sys
    subprocess.run([sys.executable,'-c',"import view_e2e_interactive,sys; assert 'mujoco' not in sys.modules; assert 'stable_baselines3' not in sys.modules"],check=True)


def test_one_sample_and_pending_quit(config,policy):
    root,trace,end=run(config,policy,duration=.001,headless=True,realtime=False)
    assert trace.sample_count==1 and end['actual_duration_sec']==.01
    def hook(step,keys,env,controls):
        for key in (' ','W','1','Q'): keys(ord(key))
    root,trace,end=run(config,policy,headless=True,realtime=False,boundary_hook=hook)
    assert trace.sample_count==0
    with (root/'events.csv').open() as f: events=list(csv.DictReader(f))
    pending=[r for r in events if r['event_type']=='pending_unapplied']
    assert [r['key'] for r in pending]==['W','1']
    assert all(r['applied']=='False' for r in pending)


def test_non_e2e_metadata_and_shape_rejected(config,tmp_path):
    from crazyflie_rl.dr_policy import training_manifest
    _,saved=training_manifest(CHECKPOINT)
    for key,value in (('control_mode','residual'),('observation_shape',[16]),('action_shape',[3])):
        metadata=dict(saved);metadata[key]=value
        file=tmp_path/f'{key}.json';file.write_text(json.dumps(metadata))
        with pytest.raises(SystemExit):
            main(['--model',str(CHECKPOINT),'--manifest',str(file),'--headless','--duration','.01'])


def test_render_copy_and_reserved_visual_keys_leave_plant_unchanged(config):
    import copy,mujoco
    from contextlib import nullcontext
    from crazyflie_rl.interactive_eval import LiveViewer
    env=InteractiveEnv(config=config)
    try:
        adapter=initialize(env)
        view=LiveViewer.__new__(LiveViewer)
        view.mj=mujoco;view.model=copy.copy(env.model);view.data=mujoco.MjData(view.model)
        view.spec=mujoco.mjtState.mjSTATE_INTEGRATION
        view.state=np.empty(mujoco.mj_stateSize(env.model,view.spec))
        view.geomgroup=np.array([1,1,1,0,0,0],np.uint8)
        opt=SimpleNamespace(geomgroup=np.zeros(6,np.uint8))
        view.handle=SimpleNamespace(lock=nullcontext,opt=opt,user_scn=None,
                                    set_texts=lambda *args:None,sync=lambda:None)
        view.data.qpos[:3]=[12,13,14]
        view.model.opt.gravity[:]=0  # A native GUI edit affects only the display model.
        before=adapter.snapshot()
        view.update(env,Controls())
        assert adapter.snapshot()==before
        np.testing.assert_array_equal(view.data.qpos,env.data.qpos)
        np.testing.assert_array_equal(opt.geomgroup[:5],view.geomgroup[:5])
        assert env.model.opt.gravity[2]!=0
    finally:env.close()

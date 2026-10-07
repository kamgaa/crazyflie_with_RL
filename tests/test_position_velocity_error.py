"""Position-generated velocity contract, COM physics, and legacy isolation."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import mujoco
import pytest

from crazyflie_rl.config import load_config, E2EVelocityConfig
from crazyflie_rl.environment import CrazyflieResidualEnv
from crazyflie_rl.interactive_eval import InteractiveEnv, evaluation_config, Controls
from crazyflie_rl.dr_transfer import EvaluationAdapter, Case
from crazyflie_rl.dr_policy import policy_raw_observation, load_frozen_policy
from crazyflie_rl.velocity_reference import desired_velocity, velocity_semantics, validate_velocity_metadata

ROOT=Path(__file__).resolve().parents[1]
SETTINGS=E2EVelocityConfig('position_error',4.,1.5)

@pytest.mark.parametrize('error',[[0,0,0],[.1,-.2,.05],[1,2,3],[-100,0,0]])
def test_reference_direction_and_norm(error):
    e=np.array(error,dtype=float);v=desired_velocity(e,SETTINGS)
    assert np.linalg.norm(v)<=1.5+1e-14
    if np.linalg.norm(e)==0: np.testing.assert_array_equal(v,0)
    else:
        np.testing.assert_allclose(v/np.linalg.norm(v),-e/np.linalg.norm(e))
        np.testing.assert_allclose(v,-4*e*min(1,1.5/np.linalg.norm(4*e)))

def config(new=True):
    return load_config(ROOT/('configs/e2e_train_position_velocity_error.yaml' if new else 'configs/e2e_train.yaml'))

def init(env):
    return EvaluationAdapter(env).reset_to_case_initial_state(Case('interactive',8,goal=(.05,0,1)),42)

@pytest.mark.parametrize('new',[False,True])
def test_observation_reward_post_state_and_decomposition(new):
    env=CrazyflieResidualEnv(config=config(new))
    try:
        init(env)
        before=env.desired_velocity(env._read_state()[0])
        obs,reward,_,_,info=env.step([.1,-.1,0,.1])
        p,q,v,w=env._read_state();des=env.desired_velocity(p)
        np.testing.assert_allclose(obs[:3],p-env.pos_des,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(obs[3:6],v-des,rtol=1e-6,atol=1e-8)
        assert obs.shape==(15,) and np.isfinite(obs).all()
        assert info['reward_terms']['velocity']==pytest.approx(-env.velocity_weight*np.dot(v-des,v-des))
        assert sum(x for k,x in info['reward_terms'].items() if k!='total')==pytest.approx(reward)
        np.testing.assert_array_equal(info['desired_velocity'],des)
        if new: assert not np.array_equal(before,des)
        else:
            np.testing.assert_array_equal(obs[3:6],v.astype(np.float32))
            assert info['reward_terms']['velocity']==pytest.approx(-env.velocity_weight*np.dot(v,v))
    finally:env.close()

def test_velocity_matching_zero_cost_and_direction(monkeypatch):
    import crazyflie_rl.environment as module
    env=CrazyflieResidualEnv(config=config())
    try:
        init(env);p,q,v,w=env._read_state();des=env.desired_velocity(p)
        monkeypatch.setattr(module.mujoco,'mj_step',lambda *args:None)
        costs=[]
        for actual in (des,-des):
            env.data.qvel[:3]=actual
            _,_,_,_,info=env.step(np.zeros(4));costs.append(-info['reward_terms']['velocity'])
        assert costs[0]==0 and costs[1]>costs[0]
    finally:env.close()

def test_training_adapter_interactive_observations_and_no_reset():
    a,b=CrazyflieResidualEnv(config=evaluation_config(config())),InteractiveEnv(config=evaluation_config(config()))
    try:
        init(a);init(b)
        np.testing.assert_array_equal(a._obs(*a._read_state()),EvaluationAdapter(b).current_observation())
        before=EvaluationAdapter(b).snapshot()
        Controls().process(['W'],b,0,lambda event:None)
        after=EvaluationAdapter(b).snapshot()
        for k in before.keys()-{'reference','observation'}:assert before[k]==after[k]
        np.testing.assert_allclose(after['observation'][3:6],[-.28,0,0],atol=1e-7)
        obs=EvaluationAdapter(b).current_observation()
        np.testing.assert_array_equal(policy_raw_observation(obs,[0,0,0],config=config()),obs)
        with pytest.raises(ValueError,match='double-subtract'):
            policy_raw_observation(obs,[0,0,0],'error',config=config())
    finally:a.close();b.close()

def test_contract_rejects_legacy_and_gain_mismatch():
    with pytest.raises(ValueError,match='semantics mismatch'):validate_velocity_metadata({},config())
    with pytest.raises(ValueError,match='semantics mismatch'):
        validate_velocity_metadata({'velocity_semantics':velocity_semantics(config())},config(False))
    changed=replace(config(),environment=replace(config().environment,e2e_velocity=replace(SETTINGS,position_gain=3)))
    with pytest.raises(ValueError,match='semantics mismatch'):
        validate_velocity_metadata({'velocity_semantics':velocity_semantics(config())},changed)


def test_legacy_nominal_exact_against_checkout_source():
    import subprocess
    from types import ModuleType
    source=subprocess.check_output(['git','show','HEAD:crazyflie_rl/environment.py'],cwd=ROOT,text=True)
    module=ModuleType('crazyflie_rl._pre_velocity_contract');module.__package__='crazyflie_rl'
    exec(compile(source,'<previous checkout>','exec'),module.__dict__)
    a,b=CrazyflieResidualEnv(config=config(False)),module.CrazyflieResidualEnv(config=config(False))
    try:
        np.testing.assert_array_equal(a.reset(seed=42)[0],b.reset(seed=42)[0])
        for k in range(100):
            action=(.03*np.sin(k*.1+np.arange(4))).astype(np.float32)
            x,y=a.step(action),b.step(action)
            for left,right in zip(x[:4],y[:4]):np.testing.assert_array_equal(left,right)
    finally:a.close();b.close()


def test_general_evaluation_uses_same_observation_and_keeps_real_velocity():
    from crazyflie_rl.eval_cli import EvaluationRunner
    from crazyflie_rl.missions import mission_from_experiment
    c=evaluation_config(config())
    c=replace(c,mission=replace(c.mission,hover=replace(c.mission.hover,target=(.05,0,1),duration=.03)))
    mission=mission_from_experiment(c)
    class Recorder:
        def __init__(self):self.observations=[]
        def predict(self,obs,deterministic):
            self.observations.append(obs.copy());return np.zeros(4),None
    policy=Recorder()
    runner=EvaluationRunner(c,None,headless=True,realtime=False,camera_tracking=False,mission=mission)
    trace=runner.run(policy,'residual','test')
    assert trace.error is None and trace.sample_count==3
    np.testing.assert_allclose(policy.observations[0][:6],[-.05,0,0,-.2,0,0],atol=1e-7)
    np.testing.assert_allclose(trace.linear_velocity-trace.desired_velocity,trace.velocity_error)
    np.testing.assert_allclose(trace.time_post,trace.time_sec+.01)
    np.testing.assert_allclose(trace.desired_velocity[0],desired_velocity(trace.position[0]-trace.reference_position[0],SETTINGS),atol=1e-8)

@pytest.mark.parametrize('offset',[(.03,0),(.02,-.03)])
def test_payload_full_inertia_engine_com_and_no_double_gravity(offset):
    c=config();c=replace(c,environment=replace(c.environment,payload=replace(c.environment.payload,mass=.005,offset=offset)))
    env=CrazyflieResidualEnv(config=c)
    try:
        init(env);m,d,b=env.model,env.data,env.drone_bid
        center=np.array([*offset,0])*.005/(env._m0+.005)
        np.testing.assert_allclose(d.xipos[b]-d.xpos[b],center,atol=1e-15)
        assert m.body_subtreemass[b]==pytest.approx(m.body_mass.sum())
        from crazyflie_rl.environment import rotmat_from_quat_wxyz
        r=rotmat_from_quat_wxyz(m.body_iquat[b]);actual=r@np.diag(m.body_inertia[b])@r.T
        off=np.array([*offset,0]);mu=env._m0*.005/(env._m0+.005)
        np.testing.assert_allclose(actual,np.diag(env._J0)+mu*(np.dot(off,off)*np.eye(3)-np.outer(off,off)),atol=1e-16)
        tau=np.cross(off,[0,0,-.005*env.gravity])
        # Solve static wrench using physical sites, not allocator approximation.
        sites=np.array([m.site_pos[m.actuator_trnid[i,0]] for i in env.act_force])
        B=np.array([sites[:,1],-sites[:,0],env.torque_coefficient*np.array([1,-1,1,-1]),np.ones(4)])
        force=np.linalg.solve(B,np.r_[-tau,m.body_mass.sum()*env.gravity])
        d.ctrl[env.act_force]=force;d.ctrl[env.act_torque]=force*B[2]
        d.xfrc_applied[:]=0;mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.qacc[:6],0,atol=1e-9)
        env.dist_torque_body[:]=[.0001,.0002,.0003]
        env.step(np.zeros(4))
        assert np.linalg.norm(d.xfrc_applied[b,3:])==pytest.approx(np.linalg.norm(env.dist_torque_body))
        # Ordinary reset remains reproducible with a previous nonzero payload.
        init(env);np.testing.assert_allclose(d.xipos[b]-d.xpos[b],center,atol=1e-15)
    finally:env.close()

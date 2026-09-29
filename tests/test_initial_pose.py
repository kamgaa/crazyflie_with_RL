"""Bounded pose-only resets and preserved legacy contracts."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import ModuleType

import numpy as np
import pytest
import yaml

from crazyflie_rl.config import ConfigError, load_config
from crazyflie_rl.environment import CrazyflieResidualEnv
from crazyflie_rl.initial_pose import sample_initial_pose

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT/'configs/e2e_train_pose_dr_10cm_30deg_scale006.yaml'


def config():
    return load_config(PROFILE)


def environment(c=None, cls=CrazyflieResidualEnv):
    pytest.importorskip('mujoco')
    pytest.importorskip('gymnasium')
    c = c or config()
    if not Path(c.paths.mujoco_xml).is_file():
        pytest.skip('MuJoCo XML unavailable')
    return cls(config=c, seed=7)


@pytest.fixture(scope='module')
def legacy_class():
    source = subprocess.check_output(['git','show','9a5419e5a3dc73921121bdc8f08ea8241354bebd:crazyflie_rl/environment.py'],cwd=ROOT,text=True)
    module = ModuleType('crazyflie_rl._pose_legacy')
    module.__package__ = 'crazyflie_rl'
    exec(compile(source,'<legacy reset>','exec'),module.__dict__)
    return module.CrazyflieResidualEnv


def test_1200_real_resets_obey_position_and_so3_bounds():
    env=environment()
    axes=[]
    try:
        for index in range(1200):
            obs,info=env.reset(seed=123 if index==0 else None)
            delta=env.data.qpos[:3]-env.pos_des
            q=env.data.qpos[3:7].copy()
            angle=np.rad2deg(2*np.arccos(np.clip(abs(q[0])/np.linalg.norm(q),0,1)))
            assert np.linalg.norm(delta)<=.10+1e-12
            assert 0<=angle<=30+1e-10
            assert abs(np.linalg.norm(q)-1)<1e-14
            assert np.array_equal(env.data.qvel,np.zeros_like(env.data.qvel))
            assert obs.shape==(15,) and env.action_space.shape==(4,)
            assert np.array_equal(info['initial_position_offset_xyz_m'],delta)
            assert np.array_equal(info['initial_attitude_quaternion_wxyz'],q)
            assert info['initial_attitude_angle_deg']==pytest.approx(angle,abs=1e-12)
            axes.append(info['initial_attitude_axis_xyz'])
        assert np.mean(np.abs(np.asarray(axes)[:,2])>.1)>.8
        assert np.min(np.asarray(axes)[:,2])<-.8
        assert np.max(np.asarray(axes)[:,2])>.8
        json.dumps(info,allow_nan=False)
    finally:
        env.close()


@pytest.mark.parametrize('disabled,zero',[(True,False),(False,True)])
def test_explicit_disabled_and_zero_bounds_are_nominal_and_override_legacy(disabled,zero):
    c=config()
    settings=c.environment.initial_pose_randomization
    settings=replace(settings,enabled=not disabled,
                     position=replace(settings.position,max_norm_m=0 if zero else .1),
                     attitude=replace(settings.attitude,max_angle_deg=0 if zero else 30))
    c=replace(c,environment=replace(c.environment,initial_pose_randomization=settings,
                                  position_perturbation=.5,attitude_perturbation_deg=90))
    env=environment(c)
    nominal=environment(replace(c,environment=replace(c.environment,initial_pose_randomization=None,
                                                      position_perturbation=0,attitude_perturbation_deg=0)))
    try:
        for seed in range(10):
            actual,info=env.reset(seed=seed)
            expected,_=nominal.reset(seed=seed)
            assert np.array_equal(actual,expected)
            assert np.array_equal(env.data.qpos,nominal.data.qpos)
            assert np.array_equal(env.data.qpos[:3],env.pos_des)
            assert np.array_equal(env.data.qpos[3:7],[1,0,0,0])
            assert info['initial_position_error_norm_m']==0
            assert info['initial_attitude_angle_deg']==0
            for field in ('_last_f','_last_f_cmd','_last_omega','_last_motor_cmd'):
                assert np.array_equal(getattr(env,field),getattr(nominal,field))
    finally:
        env.close(); nominal.close()


def test_determinism_component_switches_and_zero_direction_fallback():
    env=environment()
    try:
        first,info=env.reset(seed=123)
        qpos=env.data.qpos.copy()
        env.reset(seed=999)
        second,again=env.reset(seed=123)
        assert np.array_equal(first,second)
        assert np.array_equal(qpos,env.data.qpos)
        assert info==again
    finally:
        env.close()
    settings=config().environment.initial_pose_randomization
    offset,q,axis,_=sample_initial_pose(np.random.default_rng(3),replace(settings,position=replace(settings.position,enabled=False)))
    assert np.array_equal(offset,np.zeros(3)) and q[3]!=0
    offset,q,axis,_=sample_initial_pose(np.random.default_rng(3),replace(settings,attitude=replace(settings.attitude,enabled=False)))
    assert np.linalg.norm(offset)>0 and np.array_equal(q,[1,0,0,0])
    _,_,axis,_=sample_initial_pose(np.random.default_rng(3),replace(settings,attitude=replace(settings.attitude,full_3d=False)))
    assert axis[2]==0
    class ZeroDirection:
        def normal(self,size): return np.zeros(size)
        def uniform(self,*args): return 1.0
    offset,q,axis,angle=sample_initial_pose(ZeroDirection(),settings)
    assert np.all(np.isfinite(q))
    assert np.linalg.norm(offset)==pytest.approx(.1)
    assert np.linalg.norm(q)==pytest.approx(1)
    assert np.array_equal(axis,[1,0,0])


@pytest.mark.parametrize('mode',['e2e','residual'])
def test_legacy_reset_rng_payload_actuator_and_trajectory_exact(legacy_class,mode):
    c=load_config(ROOT/f'configs/{mode}_train.yaml')
    c=replace(c,environment=replace(c.environment,position_perturbation=.05,attitude_perturbation_deg=10,
                                  payload=replace(c.environment.payload,randomize=True)),
              actuator=replace(c.actuator,randomization=replace(c.actuator.randomization,enabled=True)))
    old,new=environment(c,legacy_class),environment(c)
    try:
        for episode in range(30):
            a,ai=old.reset(seed=51 if episode==0 else None)
            b,bi=new.reset(seed=51 if episode==0 else None)
            assert ai==bi=={}
            assert np.array_equal(a,b)
            assert old._rng.bit_generator.state==new._rng.bit_generator.state
            assert old._actuator_rng.bit_generator.state==new._actuator_rng.bit_generator.state
            assert old._com_mw==new._com_mw
            for k in range(5):
                action=(.02*np.sin(k+np.arange(4))).astype(np.float32)
                oa=old.step(action); ob=new.step(action)
                for x,y in zip(oa[:4],ob[:4]):
                    assert np.array_equal(x,y)
                for field in ('qpos','qvel','ctrl'):
                    assert np.array_equal(getattr(old.data,field),getattr(new.data,field))
                assert np.array_equal(old._last_f,new._last_f)
                assert np.array_equal(old._last_omega,new._last_omega)
    finally:
        old.close(); new.close()


def test_new_pose_preserves_seeded_payload_and_independent_actuator_stream():
    c=config()
    c=replace(c,environment=replace(c.environment,payload=replace(c.environment.payload,randomize=True)),
              actuator=replace(c.actuator,randomization=replace(c.actuator.randomization,enabled=True)))
    dr=environment(c)
    legacy=environment(replace(c,environment=replace(c.environment,initial_pose_randomization=None)))
    try:
        for seed in range(10):
            dr.reset(seed=seed); legacy.reset(seed=seed)
            assert dr._com_mw==legacy._com_mw
            assert np.array_equal(dr._com_off3,legacy._com_off3)
            assert dr.actuator_snapshot()==legacy.actuator_snapshot()
            assert dr._actuator_rng.bit_generator.state==legacy._actuator_rng.bit_generator.state
            assert np.array_equal(dr._last_omega,legacy._last_omega)
    finally:
        dr.close(); legacy.close()


@pytest.mark.parametrize('path,value',[
    (('enabled',),1), (('position','enabled'),'true'), (('attitude','enabled'),0),
    (('attitude','full_3d'),'false'), (('position','max_norm_m'),-.1),
    (('position','max_norm_m'),float('nan')), (('position','max_norm_m'),float('inf')),
    (('position','max_norm_m'),True), (('attitude','max_angle_deg'),-1),
    (('attitude','max_angle_deg'),180), (('attitude','max_angle_deg'),float('nan')),
    (('attitude','max_angle_deg'),float('inf')), (('attitude','max_angle_deg'),True),
])
def test_config_validation(tmp_path,path,value):
    data=yaml.safe_load((ROOT/'configs/base.yaml').read_text())
    settings={'enabled':True,'position':{'enabled':True,'max_norm_m':.1},
              'attitude':{'enabled':True,'max_angle_deg':30,'full_3d':True}}
    node=settings
    for key in path[:-1]: node=node[key]
    node[path[-1]]=value
    data['environment']['initial_pose_randomization']=settings
    target=tmp_path/'config.yaml'; target.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError): load_config(target)


def test_profile_only_changes_reset_settings_and_experiment_identity():
    baseline=load_config(ROOT/'configs/e2e_train.yaml')
    dr=config()
    assert dr.training==baseline.training
    assert dr.actuator==baseline.actuator
    assert dr.vehicle==baseline.vehicle and dr.controller==baseline.controller
    assert dr.environment.reward==baseline.environment.reward
    assert dr.environment.residual_scale==baseline.environment.residual_scale==(.006,.006,.0001,.3)
    assert replace(dr.environment,initial_pose_randomization=None)==baseline.environment
    assert baseline.environment.initial_pose_randomization is None


def test_reset_diagnostic_cli(tmp_path,capsys):
    from diag_initial_pose import main
    target=tmp_path/'samples.json'
    assert main(['--samples','8','--seed','7','--output',str(target)])==0
    result=json.loads(target.read_text())
    assert result['samples']==8
    assert result['bound_violation_count']=={'position':0,'attitude':0}
    assert json.loads(capsys.readouterr().out)==result


@pytest.mark.parametrize('mode',['hover','circle','lissajous'])
@pytest.mark.parametrize('pose_dr',[False,True])
def test_real_short_evaluation_modes(mode,pose_dr):
    pytest.importorskip('mujoco')
    from crazyflie_rl.eval_cli import EvaluationRunner
    from crazyflie_rl.missions import mission_from_experiment
    c=load_config(ROOT/f'configs/view_live_{mode}_eval.yaml')
    if not Path(c.paths.mujoco_xml).is_file():
        pytest.skip('MuJoCo XML unavailable')
    mission_settings=replace(c.mission,type=mode,takeoff_sec=.1,settle_sec=.1,goto_sec=.1,post_hold_sec=.1,
                             hover=replace(c.mission.hover,duration=.2),
                             circle=replace(c.mission.circle,period=.2,laps=1,ramp_sec=0),
                             lissajous=replace(c.mission.lissajous,base_period=.2,cycles=1,ramp_sec=0))
    c=replace(c,mission=mission_settings,
              environment=replace(c.environment,initial_pose_randomization=(config().environment.initial_pose_randomization if pose_dr else None)))
    mission=mission_from_experiment(c,legacy_circle_preset=False)
    c=replace(c,environment=replace(c.environment,episode_sec=mission.total_sec))
    runner=EvaluationRunner(c,None,headless=True,realtime=False,camera_tracking=False,mission=mission)
    trace=runner.run(None,'floor','PID',control_mode='residual')
    assert trace.error is None
    assert trace.sample_count>0
    assert {'hover':'HOVER','circle':'CIRCLE','lissajous':'LISSAJOUS'}[mode] in trace.phases
    assert np.all(np.isfinite(trace.position))
    assert trace.control_input.shape==(trace.sample_count,4)

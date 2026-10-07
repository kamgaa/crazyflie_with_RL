from dataclasses import replace
import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.dr_transfer import ROOT,Case,EvaluationAdapter,Thresholds,run_case
from crazyflie_rl.environment import CrazyflieResidualEnv
from crazyflie_rl.payload_motor_eval import (CONDITIONS,DEFAULT_RECORD,condition_config,FaultObserver,
    RecordedFaultEnv,position_statistics,recovery_time,dwell,analyze,select_models,verify_signals)

@pytest.fixture
def config():
    return load_config(ROOT/'configs/eval_velocity_ab.yaml')

@pytest.fixture
def policies(config,monkeypatch):
    from stable_baselines3 import PPO
    def forbidden(*a,**k):raise AssertionError('learning forbidden')
    monkeypatch.setattr(PPO,'learn',forbidden);monkeypatch.setattr(PPO,'train',forbidden)
    return select_models(DEFAULT_RECORD,config)

@pytest.mark.parametrize('condition',CONDITIONS,ids=lambda c:c.name)
def test_payload_and_nominal_initialization(config,condition):
    cfg=condition_config(config,condition)
    env=RecordedFaultEnv(config=cfg)
    try:
        adapter=EvaluationAdapter(env);adapter.reset_to_case_initial_state(Case('hover',20,(0,0,1)),42)
        obs=FaultObserver(condition);obs.on_reset(adapter)
        assert not cfg.environment.payload.randomize and not cfg.actuator.randomization.enabled
        assert cfg.environment.position_perturbation==cfg.environment.attitude_perturbation_deg==0
        assert cfg.environment.initial_pose_randomization is None
        assert cfg.environment.payload.mass==condition.mass
        total=env._m0+condition.mass
        np.testing.assert_allclose(env.model.body_ipos[env.drone_bid],
            (env._m0*env._ipos0+condition.mass*np.r_[condition.offset,0])/total)
        assert env.model.body_mass[env.drone_bid]==pytest.approx(total)
        np.testing.assert_allclose(env._last_f, np.ones(4)*env.mass*env.gravity/4,rtol=1e-12)
        np.testing.assert_array_equal(adapter.current_observation()[3:6],np.zeros(3))
        np.testing.assert_array_equal(env.motor_effectiveness,np.ones(4))
        assert env.max_steps==2000 if cfg.environment.episode_sec==20 else True
    finally:env.close()

@pytest.mark.parametrize('index',[0,1])
def test_real_checkpoint_nominal_regression(config,policies,index):
    policy=policies[index];case=Case('hover',.15,(0,0,1))
    old=run_case(config,case,policy,42)
    observer=FaultObserver(CONDITIONS[0])
    new=run_case(config,case,policy,42,env_factory=RecordedFaultEnv,observer=observer)
    assert old[1]==new[1] and old[2:]==new[2:]
    for a,b in zip(old[0],new[0]):
        for key in ('action','observation','position','velocity','quaternion','omega','motor_thrust'):
            np.testing.assert_array_equal(a[key],b[key])
    verify_signals(new[0],observer.physics_rows,CONDITIONS[0])


def test_exact_event_and_once_only_force_torque(config,policies):
    condition=CONDITIONS[3];obs=FaultObserver(condition)
    rows,initial,error,reasons=run_case(condition_config(config,condition),Case('hover',5.02,(0,0,1)),policies[0],42,
        env_factory=RecordedFaultEnv,observer=obs)
    assert error is None and len(obs.events)==1
    assert obs.events[0]['control_step']==500 and obs.events[0]['policy_input_time']==5
    np.testing.assert_array_equal(rows[499]['motor_effectiveness'],[1,1,1,1])
    np.testing.assert_array_equal(rows[500]['motor_effectiveness'],[.8,1,1,1])
    verify_signals(rows,obs.physics_rows,condition)
    assert len(obs.physics_rows)==5*len(rows)


def test_offset_sway_identity():
    e=np.array([[1.,2.,3.],[3.,0.,-1.]])
    r=position_statistics(e)
    assert r['mean_error_xy_m']==[2,1] and r['mean_error_z_m']==1
    assert r['offset_xy_m']==pytest.approx(np.sqrt(5))
    assert r['sway_xy_rms_m']==pytest.approx(np.sqrt(2))
    assert r['rmse_xy_m']**2==pytest.approx(7)
    assert r['rmse_z_m']**2==pytest.approx(5)
    assert r['sway_z_rms_m']==2
    constant=position_statistics(np.tile([.01,-.01,-.02],(100,1)))
    assert constant['sway_xy_rms_m']<1e-16 and constant['sway_z_rms_m']<1e-16
    assert constant['offset_xy_m']==pytest.approx(np.sqrt(2)*.01)
    assert position_statistics([]) is None


def samples(end=20):
    return [dict(time=(k-1)/100,time_post=k/100,position_error_world=np.zeros(3),position=np.array([0.,0.,1.]),
        reference_post=np.array([0.,0.,1.]),reference=np.array([0.,0.,1.]),
        position_before=np.array([0.,0.,1.]),velocity_before=np.zeros(3),velocity=np.zeros(3),
        internal_velocity_error=np.zeros(3),quaternion=np.array([1.,0.,0.,0.]),omega=np.zeros(3),
        terminated=False,truncated=k==round(end*100),policy_action_at_bound=np.zeros(4,dtype=bool),
        policy_action_clipped=np.zeros(4,dtype=bool)) for k in range(1,round(end*100)+1)]


def test_recovery_axis_suffix_and_partial_null():
    rows=samples();th=Thresholds()
    for r in rows:
        if r['time_post']<6:r['position_error_world'][0]=.01
        if r['time_post']<7:r['position_error_world'][2]=-.01
    assert recovery_time(rows,th,'xy',True)==1
    assert recovery_time(rows,th,'z',True)==2
    assert recovery_time(rows,th,'3d',True)==2
    rows[-1]['position_error_world'][0]=.01
    assert recovery_time(rows,th,'xy',True) is None
    assert recovery_time(rows,th,'z',False) is None
    for r in rows:r['position_error_world'][:]=.01 if r['time_post']<19.1 else 0
    assert recovery_time(rows,th,'3d',True) is None  # Less than minimum hold.


def test_windows_do_not_move_after_failure():
    obs=FaultObserver(CONDITIONS[3]);obs.events=[{'applied':True}]
    case=Case('hover',20,(0,0,1));th=Thresholds()
    rows=samples(6);rows[-1]['terminated']=True
    result=analyze(rows,case,CONDITIONS[3],obs,None,['max_tilt'],th)
    assert result['position_rmse_total'] is None and result['partial']
    assert result['windows']['full_0_20'] is None and result['windows']['tail_18_20'] is None
    assert result['partial_observed_windows']['full_0_20']['sample_count']==600
    assert result['windows']['pre_3_5']['sample_count']==200
    assert result['partial_observed_windows']['post_5_20']['sample_count']==100
    assert result['partial_observed_windows']['tail_18_20'] is None
    assert result['recovery_xy_s'] is None and result['last_2s_speed_rms'] is None
    full=analyze(samples(),case,CONDITIONS[0],FaultObserver(CONDITIONS[0]),None,[],th)
    assert full['windows']['tail_18_20']['sample_count']==200
    assert full['windows']['post_5_20']['sample_count']==1500
    assert not full['fault_scheduled'] and full['recovery_xy_s'] is None


def test_saturation_dwell():
    r=dwell([[1,0],[1,1],[0,1],[1,1]],.002)
    assert r['duration_s_per_channel']==[.006,.006]
    assert r['longest_continuous_s_per_channel']==[.004,.006]
    assert r['duration_any_s']==.008

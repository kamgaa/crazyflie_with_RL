from dataclasses import replace

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.dr_transfer import ROOT, EvaluationAdapter
from crazyflie_rl.integral_controller import IntegralController
from crazyflie_rl.integral_validation import make_case
from crazyflie_rl.oracle_allocation import OracleAllocationEnv, static_hover
from crazyflie_rl.oracle_eval import OracleObserver
from crazyflie_rl.oracle_recovery import (COMBINATIONS, BASE_PERCENT, GAIN, AttitudeCriteria,
    EfficiencyScenario, scenario_at, suffix_latency, joint_recovery, weighted_dwell,
    refinement_candidates, boundary_report, analyze_run, clean)
from crazyflie_rl.payload_motor_eval import condition_config


def test_fixed_contract():
    assert len(COMBINATIONS)*len(BASE_PERCENT)==21
    assert (GAIN.k_xy,GAIN.k_z,GAIN.xy_limit_m,GAIN.z_limit_m)==(.2,.2,.4,.15)
    assert AttitudeCriteria()==AttitudeCriteria(5,5,.1)
    assert scenario_at('combined_motor2_70',70).offset==(0,-.03)
    assert scenario_at('combined_motor3_70',70).offset==(-.03,0)
    for percent in (70.0,.7,0,101):
        with pytest.raises(ValueError):scenario_at('combined_motor2_70',percent)


def test_suffix_hold_and_null():
    times=np.arange(5,60.01,.01)
    good=times>=10
    assert suffix_latency(times,good,True)==pytest.approx(5,abs=.011)
    assert suffix_latency(times,good,False) is None
    assert suffix_latency(times,times>=59.02,True) is None
    good[-2]=False
    assert suffix_latency(times,good,True) is None
    assert suffix_latency(times,np.ones(len(times),bool),True)==0


def test_joint_requires_all_position_attitude_channels():
    event=dict(e_true_after=np.zeros(3),velocity=np.zeros(3),quaternion=np.array([1.,0,0,0]),omega=np.zeros(3))
    base=dict(position_error_world=np.zeros(3),velocity=np.zeros(3),quaternion=np.array([1.,0,0,0]),
              omega=np.zeros(3),yaw_error_rad=0.)
    rows=[dict(base,time_post=t) for t in np.arange(5.01,60.001,.01)]
    assert joint_recovery(rows,event,True)==0
    assert joint_recovery(rows,event,False) is None
    for field,value in [('yaw_error_rad',np.radians(5.1)),('omega',np.array([.101,0,0])),
                        ('position_error_world',np.array([.006,0,0])),('velocity',np.array([.021,0,0])),
                        ('quaternion',np.array([np.cos(np.radians(3)),np.sin(np.radians(3)),0,0]))]:
        changed=[dict(r,**{field:value}) for r in rows]
        assert joint_recovery(changed,event,True) is None


def test_actual_interval_dwell_union_and_longest():
    d=weighted_dwell([[1,1],[1,0],[0,0],[0,1]],[.1,.2,.4,.3])
    np.testing.assert_allclose(d['duration_s_per_channel'],[.3,.4])
    assert d['duration_any_s']==pytest.approx(.6)  # not .7 summed over motors
    assert d['longest_continuous_any_s']==pytest.approx(.3)
    with pytest.raises(ValueError):weighted_dwell([1],[0])


def item(percent,success,terminated=False):
    return dict(efficiency_percent=percent,position_criterion_met=success,terminated=terminated,evaluated=True)


def test_nonmonotone_candidate_enumeration_not_bisection():
    rows=[item(70,False,True),item(75,True),item(80,False,True),item(85,True),item(90,True),item(95,True),item(100,True)]
    assert refinement_candidates(rows)==[84,83,82,81,79,78,77,76,74,73,72,71]
    rows.append(item(73,False,True))
    assert 73 not in refinement_candidates(rows)
    assert refinement_candidates([item(70,True),item(75,False)])==[74,73,72,71]


@pytest.mark.parametrize('name', ['combined_motor2_70','combined_motor3_70'])
@pytest.mark.parametrize('percent',[100,95,66])
def test_existing_events_static_and_state_continuity(name,percent):
    config=load_config(ROOT/'configs/eval_velocity_ab.yaml');s=scenario_at(name,percent)
    env=OracleAllocationEnv(config=condition_config(config,s),allocator_mode='oracle')
    try:
        adapter=EvaluationAdapter(env);adapter.reset_to_case_initial_state(make_case(s),42)
        ctrl=IntegralController(None,GAIN);observer=OracleObserver(s,ctrl);observer.on_reset(adapter)
        before=adapter.snapshot();eta=np.ones(4);eta[s.motor_number-1]=percent/100
        audit=clean(static_hover(env,eta))
        assert audit['feasible_static_equilibrium']
        assert audit['total_mass_kg']==pytest.approx(.048384)
        assert adapter.snapshot()==before
        ctrl.xi[:]=[.02,-.03,.01];xi=ctrl.xi.copy()
        observer.before_step(adapter,499,4.99)
        np.testing.assert_array_equal(env.allocator_efficiency,1)
        env.data.time=5.;env._step=500;before=adapter.snapshot()
        observer.before_step(adapter,500,5.)
        assert adapter.snapshot()==before
        np.testing.assert_array_equal(ctrl.xi,xi)
        np.testing.assert_array_equal(env.allocator_efficiency,eta)
        np.testing.assert_array_equal(env.motor_effectiveness,eta)
        assert len(observer.events)==1
        desired=env.allocator_matrix@np.full(4,.10)
        env._apply_control(desired)
        r=env.physics_rows[-1]
        np.testing.assert_array_equal(r['motor_thrust_actual'],eta*r['motor_thrust_nominal'])
        np.testing.assert_array_equal(r['motor_reaction_actual'],eta*r['motor_reaction_nominal'])
        np.testing.assert_allclose(r['allocation_residual_b0'],0,atol=1e-14)
    finally:env.close()

"""Numerical and information-boundary tests; no training or optimizer calls."""
from dataclasses import replace
import json
import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.fault_estimator import (ObservedState,DeliveredCommands,EstimatorSettings,
    MotorEfficiencyEstimator)
from crazyflie_rl.fault_estimation_eval import nominal_model, estimate_from_row, blocks, suffix_time
from crazyflie_rl.actuators import Cf21bFirstOrderActuatorModel
from crazyflie_rl.motor_layout import S,P,exposed_motor_index


@pytest.fixture(scope='module')
def model():
    return nominal_model(load_config('configs/eval_velocity_ab_user_frd.yaml'))[0]


def synthetic_intervals(model,motor=1,efficiency=.763,n=30):
    """Independent constant-attitude algebraic velocity increment fixture.

    This validates scalar parameter recovery, not a claimed real PPO rollout.
    Translation and midpoint Euler angular momentum use known test efficiency.
    """
    a=Cf21bFirstOrderActuatorModel(**model.actuator_kwargs)
    o=a.reset(airborne=True,episode_mass=model.initial_hover_mass_kg,gravity_m_s2=9.81)
    esc=o.motor_command.copy();I=model.inertia_com_native;inv=np.linalg.inv(I)
    eta=np.ones(4);eta[exposed_motor_index(motor,'user_frd')]=efficiency
    torque=model.torque_origin_columns_native-np.cross(np.broadcast_to(model.com_native_m,(4,3)),model.force_columns_native.T).T
    vel=np.zeros(3);omega=np.zeros(3)
    for k in range(n):
        t=k*.01
        before=ObservedState(t,np.zeros(3),[1,0,0,0],vel,omega,t)
        fs=np.array([a.apply_motor_command(esc).f_actual for _ in range(5)])
        impulse=.002*np.sum(fs*eta,axis=0)
        dv=model.force_columns_native@impulse/model.mass_kg+model.gravity_world*.01
        w=omega.copy()
        for _ in range(20):
            mid=(omega+w)/2
            w=omega+inv@(torque@impulse)-.01*inv@np.cross(mid,I@mid)
        vel=vel+np.cross(omega,model.com_native_m)+dv-np.cross(w,model.com_native_m)
        omega=w
        after=ObservedState(t+.01,np.zeros(3),[1,0,0,0],vel,omega,t+.01)
        yield before,DeliveredCommands(t,.002,np.tile(esc,(5,1))),after


def test_nominal_geometry_and_hypothesis_signs(model):
    assert model.mass_kg==pytest.approx(.043384)
    assert model.com_native_m[2]==pytest.approx(.000004*.012/.043384)
    # User FRD, positive thrust magnitude; physical XML arm, not allocator approximation.
    expected=np.array([[1,1,-1,-1],[1,-1,-1,1],[-1,1,-1,1]],float)
    expected[:2]*=.03536;expected[2]*=.00594
    np.testing.assert_allclose(S@model.torque_origin_columns_native@P,expected,atol=1e-15)
    for m in range(1,5):
        assert exposed_motor_index(m,'user_frd')==4-m
    np.testing.assert_allclose(-.02*(S@model.torque_origin_columns_native[:,3]),[-.03536*.02,-.03536*.02,.00594*.02])


@pytest.mark.parametrize('motor',[1,2,3,4])
@pytest.mark.parametrize('efficiency',[.763,.917])
def test_continuous_efficiency_solution(model,motor,efficiency):
    estimator=MotorEfficiencyEstimator(model,EstimatorSettings())
    for inputs in synthetic_intervals(model,motor,efficiency):out=estimator.update(*inputs)
    assert out['estimator_state']=='fault'
    assert out['estimated_motor']==motor
    assert out['estimated_efficiency_user'][motor-1]==pytest.approx(efficiency,abs=1e-6)


def test_healthy_is_real_selectable_hypothesis(model):
    est=MotorEfficiencyEstimator(model,EstimatorSettings())
    for args in synthetic_intervals(model,efficiency=1.):out=est.update(*args)
    assert out['estimator_state']=='healthy' and out['estimated_motor']==0
    np.testing.assert_array_equal(out['estimated_efficiency_user'],np.ones(4))


def test_confirmation_is_not_backdated(model):
    est=MotorEfficiencyEstimator(model,EstimatorSettings(window_samples=2,confirmation_samples=3))
    out=[est.update(*args) for args in synthetic_intervals(model,n=5)]
    assert [r['estimator_state'] for r in out[:3]]==['insufficient_data','uncertain','uncertain']
    assert out[3]['estimator_state']=='fault' and out[3]['estimate_time']==pytest.approx(.04)


def test_no_unbounded_state_and_inputs_not_mutated(model):
    est=MotorEfficiencyEstimator(model,EstimatorSettings())
    sequence=list(synthetic_intervals(model,n=40))
    original=[(a.velocity_origin_world.copy(),c.esc_native.copy(),b.omega_native.copy()) for a,c,b in sequence]
    for a,c,b in sequence:est.update(a,c,b)
    assert len(est.window)==20 and len(est.motor_history)<=32
    for (a,c,b),(v,u,w) in zip(sequence,original):
        np.testing.assert_array_equal(a.velocity_origin_world,v);np.testing.assert_array_equal(c.esc_native,u)
        np.testing.assert_array_equal(b.omega_native,w)
    with pytest.raises(ValueError):sequence[0][0].omega_native[0]=1


def test_no_future_dependence(model):
    inputs=list(synthetic_intervals(model,n=30));a=MotorEfficiencyEstimator(model,EstimatorSettings())
    b=MotorEfficiencyEstimator(model,EstimatorSettings());left=[];right=[]
    for k,(s,c,t) in enumerate(inputs):
        left.append(a.update(s,c,t))
        if k>=25:
            t=replace(t,velocity_origin_world=t.velocity_origin_world+[1,2,3])
            c=replace(c,esc_native=.5*c.esc_native)
        right.append(b.update(s,c,t))
    for l,r in zip(left[:25],right[:25]):
        np.testing.assert_array_equal(l['hypothesis_scores'],r['hypothesis_scores'])
        np.testing.assert_array_equal(l['estimated_efficiency_user'],r['estimated_efficiency_user'])


def test_forbidden_info_not_accepted(model):
    est=MotorEfficiencyEstimator(model,EstimatorSettings())
    with pytest.raises(TypeError):est.update({'env':object()},None,None)
    with pytest.raises(TypeError):est.update(None,None,None,efficiency=.7)


def test_out_of_order_rejected(model):
    est=MotorEfficiencyEstimator(model,EstimatorSettings());args=next(synthetic_intervals(model))
    est.update(*args)
    with pytest.raises(ValueError,match='out-of-order'):est.update(*args)


def test_evaluation_truth_and_names_cannot_affect_estimates(model):
    a=MotorEfficiencyEstimator(model,EstimatorSettings());b=MotorEfficiencyEstimator(model,EstimatorSettings())
    for before,command,after in synthetic_intervals(model,n=25):
        row=dict(time=before.time,time_post=after.time,delivered_esc_native=command.esc_native)
        for obj,suffix in [(before,'_before'),(after,'')]:
            row.update({key+suffix:value for key,value in dict(position=obj.position_world,
                quaternion=obj.quaternion_wxyz,velocity=obj.velocity_origin_world,omega=obj.omega_native).items()})
        changed=dict(row,truth_efficiency_user=np.array([.1,.2,.3,.4]),scenario='fake_motor4_00',
                     file_name='ground_truth_should_not_be_seen.csv',qacc=np.ones(10)*1e6,
                     motor_thrust_actual=np.ones(4)*123,motor_omega=np.zeros(4),fault_time=-500)
        left=estimate_from_row(a,row,.002);right=estimate_from_row(b,changed,.002)
        for key in ('hypothesis_scores','estimated_efficiency_user','hypothesis_alpha_user'):
            np.testing.assert_array_equal(left[key],right[key])
        assert left['estimator_state']==right['estimator_state']


def test_information_insufficient_and_metric_nulls(model):
    setting=EstimatorSettings(minimum_information=1e15)
    est=MotorEfficiencyEstimator(model,setting)
    for inputs in synthetic_intervals(model):out=est.update(*inputs)
    assert out['estimator_state']=='insufficient_data'
    assert suffix_time([5.01,5.02],[True,True]) is None
    assert suffix_time([5.01,5.5,6.01],[False,True,True])==5.5
    assert suffix_time([5.01,5.5,6.01],[True,True,False]) is None
    assert blocks([False,True,True,False,True])==2


def test_early_failure_fixed_windows_and_undetected_are_null(monkeypatch):
    import crazyflie_rl.fault_estimation_eval as ev
    monkeypatch.setattr(ev,'summarize',lambda *a,**k:dict(completed=False,terminated=True))
    monkeypatch.setattr(ev,'segment_statistics',lambda rows:dict(sample_count=len(rows)))
    rows=[]
    for i in range(520):
        truth=np.ones(4)
        if i>=500:truth[0]=.7
        rows.append(dict(time=i*.01,time_post=(i+1)*.01,estimated_efficiency_user=np.ones(4),
            truth_efficiency_user=truth,estimated_motor=0,candidate_motor=0,raw_best_motor=2,estimator_state='healthy',
            estimator_runtime_ms=.1,allocator_clipping_union_seconds=0.,allocator_clipping_seconds_native=np.zeros(4),
            esc_boundary_union_seconds=0.,policy_action_at_bound=np.zeros(4,bool),integral_frozen=False,yaw_error_rad=0.))
    result=ev.summarize_estimation(rows,ev.ShadowScenario('unit_partial',motor_number=1,efficiency=.7),None,[])
    assert result['confirmed_detection_time'] is None and result['detection_delay'] is None
    assert result['efficiency_settling_time'] is None
    assert result['tail_control_metrics'] is None
    assert all(v is None for v in result['estimation_windows'].values())
    assert result['null_reasons']['tail']=='physical termination before 20 seconds'

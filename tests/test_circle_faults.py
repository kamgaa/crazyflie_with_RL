from dataclasses import replace
import numpy as np
import pytest
from crazyflie_rl.circle_fault_eval import (CircleProtocol,CircleCase,CircleObserver,CircleScenario,
    CONFIG,focused_motor_checks,execute,PIDOnly,metrics)
from crazyflie_rl.config import load_config
from crazyflie_rl.interactive_eval import evaluation_config
from crazyflie_rl.fault_estimation_eval import nominal_model
from crazyflie_rl.fault_estimator import EstimatorSettings
from crazyflie_rl.estimated_allocation import EstimatedAllocationEnv
from crazyflie_rl.dr_transfer import run_case,EvaluationAdapter
from crazyflie_rl.integral_controller import IntegralController
from crazyflie_rl.integral_validation import GAINS

@pytest.fixture(scope='module')
def setup():
    c=evaluation_config(load_config(CONFIG));m,_=nominal_model(c)
    return c,m,EstimatorSettings()

@pytest.mark.parametrize('period',[5.,10.])
def test_reference_schedule_continuity_derivative(period):
    p=CircleProtocol(period)
    assert p.fault_time=={5:16,10:26}[period] and p.horizon=={5:31,10:56}[period]
    np.testing.assert_allclose(p.phase(p.fault_time)[0],4*np.pi,atol=1e-14)
    np.testing.assert_allclose(p.phase(p.horizon)[0],10*np.pi,atol=1e-14)
    for t in [0.,5.,5.3,6.,6.7,7.,8.,p.fault_time,p.horizon]:
        h=1e-5
        derivative=(p.reference(t+h)[0]-p.reference(t-h)[0])/(2*h)
        np.testing.assert_allclose(derivative,p.reference_velocity(t),atol=1e-9)
    for boundary in [5.,7.]:
        np.testing.assert_allclose(p.reference(boundary-1e-9)[0],p.reference(boundary+1e-9)[0],atol=3e-9)
        np.testing.assert_allclose(p.reference_velocity(boundary-1e-9),p.reference_velocity(boundary+1e-9),atol=3e-9)
    np.testing.assert_allclose(p.reference(8)[0],p.reference(8+period)[0],atol=1e-14)


def test_shared_motor_model_limit_and_oracle(setup):
    c,m,_=setup;r,curve=focused_motor_checks(c,m)
    assert r['nominal_limit_n']==.289 and r['oracle_inverse_component_error_n']<1e-14
    assert curve[-1]['steady_nominal_n']==pytest.approx(.289)
    assert load_config('configs/eval_velocity_ab_user_frd.yaml').vehicle.thrust_max==.20


def test_pid_wrapper_is_exact_existing_zero_residual_path(setup):
    c,m,s=setup;p=CircleProtocol(5.)
    rows,initial,obs,case,reasons=execute(c,PIDOnly(),p,m,s,'blind',False,horizon=.3)
    cfg=replace(c,environment=replace(c.environment,control_mode='residual'))
    reference,other,error,_=run_case(cfg,case,PIDOnly(),42,env_factory=EstimatedAllocationEnv)
    assert error is None
    for a,b in zip(rows,reference):
        for k in ['position','velocity','quaternion','omega','action','motor_thrust_command']:
            np.testing.assert_array_equal(a[k],b[k])
        assert not a['integral_enabled'] and not a['integral_frozen']
        np.testing.assert_array_equal(a['xi_next'],np.zeros(3))
    assert initial['position']==[1.,0.,1.]


def test_scheduled_event_preserves_state_and_current_eta(setup):
    c,m,s=setup;p=CircleProtocol(5.)
    class FixedPolicy:
        def bind(self,env):pass
        def predict(self,obs):return np.zeros(4)
    ctl=IntegralController(FixedPolicy(),GAINS[1]);scenario=CircleScenario('test',p.fault_time,True)
    observer=CircleObserver(scenario,ctl,m,s,'oracle',p)
    env=EstimatedAllocationEnv(config=c,allocation_source='oracle')
    try:
        adapter=EvaluationAdapter(env);ctl.bind(env)
        adapter.reset_to_case_initial_state(CircleCase('test',p.horizon,protocol=p),42);observer.on_reset(adapter)
        env.data.time=p.fault_time;ctl.xi[:]=[.02,-.01,.005]
        adapter.set_reference(p.reference(p.fault_time)[0]);before=adapter.snapshot();motor=env.actuator_model.omega.copy()
        observer.before_step(adapter,round(p.fault_time/.01),p.fault_time)
        assert adapter.snapshot()==before
        np.testing.assert_array_equal(motor,env.actuator_model.omega)
        np.testing.assert_array_equal(ctl.xi,[.02,-.01,.005])
        np.testing.assert_array_equal(env.motor_effectiveness,[1,1,1,.7])
        env.sync_allocator();np.testing.assert_array_equal(env.allocator_efficiency,[1,1,1,.7])
        assert observer.events[0]['native_motor_index']==3
        assert observer.events[0]['estimator_updates_preserved']==0
    finally:env.close()


def test_partial_unobserved_circle_windows_are_null(setup):
    c,m,s=setup;p=CircleProtocol(5.)
    rows,_,o,case,why=execute(c,PIDOnly(),p,m,s,'blind',True,horizon=.2)
    r=metrics(rows,o,case,why)
    assert r['windows']['hover'] is None and r['windows']['post_all'] is None
    assert r['partial_observed_windows']['hover']['samples']==20
    assert r['estimation']['detection_delay_s'] is None and not r['fault_applied']

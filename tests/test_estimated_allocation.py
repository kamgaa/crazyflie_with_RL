from dataclasses import replace
import numpy as np
import pytest
from crazyflie_rl.estimated_allocation import ConfirmedEfficiency, EstimatedAllocationEnv
from crazyflie_rl.config import load_config
from crazyflie_rl.dr_transfer import EvaluationAdapter, Case
from crazyflie_rl.fault_estimation_eval import nominal_model, SCENARIOS
from crazyflie_rl.fault_estimator import EstimatorSettings
from crazyflie_rl.estimated_allocation_eval import execute, metrics
from crazyflie_rl.motor_layout import native_from_user, exposed_motor_index
from crazyflie_rl.motor_limit_audit import audit


def update(latch,state='fault',motor=1,alpha=.743,t=1):
    return latch.update(state=state,motor=motor,hypothesis_alpha_user=np.full(4,alpha),estimate_time=t)


def test_confirmed_hold_candidate_and_recovery_rules():
    l=ConfirmedEfficiency()
    for t,state in enumerate(['insufficient_data','uncertain']):
        eta,accepted=update(l,state,alpha=.01,t=t)
        np.testing.assert_array_equal(eta,np.ones(4));assert not accepted
    update(l,t=2)
    np.testing.assert_array_equal(l.eta_user,[.743,1,1,1])
    for t,state in enumerate(['uncertain','insufficient_data'],3):
        update(l,state,motor=4,alpha=.20,t=t)
        np.testing.assert_array_equal(l.eta_user,[.743,1,1,1])
    update(l,motor=4,alpha=.831,t=5)
    np.testing.assert_array_equal(l.eta_user,[1,1,1,.831])
    update(l,'healthy',motor=0,t=6)
    np.testing.assert_array_equal(l.eta_user,np.ones(4))


def test_invalid_alpha_holds_and_zero_has_no_artificial_floor():
    l=ConfirmedEfficiency();update(l)
    for t,a in enumerate([np.nan,-.1,1.1],2):
        _,ok=update(l,alpha=a,t=t);assert not ok
        assert l.eta_user[0]==.743
    update(l,alpha=0,t=5);assert l.eta_user[0]==0
    with pytest.raises(ValueError):update(l,t=5)


@pytest.fixture
def env():
    e=EstimatedAllocationEnv(config=load_config('configs/eval_velocity_ab_user_frd.yaml'),allocation_source='estimated')
    EvaluationAdapter(e).reset_to_case_initial_state(Case('hover',20.,(0,0,1)),42)
    yield e
    e.close()


def test_unity_exact_path_and_column_order(env):
    old=env.B_pinv.copy();env.sync_allocator();np.testing.assert_array_equal(old,env.B_pinv)
    for user in range(1,5):
        eta=np.ones(4);eta[user-1]=.753
        env.set_estimated_efficiency(eta);env.sync_allocator()
        target=env.B0.copy();target[:,exposed_motor_index(user,'user_frd')]*=.753
        np.testing.assert_array_equal(env.allocator_matrix,target)
        command=np.array([.10,.11,.12,.13]);w=target@command
        np.testing.assert_allclose(target@(env.B_pinv@w),w,atol=1e-15)
    env.set_estimated_efficiency(np.ones(4));env.sync_allocator()
    np.testing.assert_array_equal(old,env.B_pinv)


def test_truth_cannot_change_estimated_allocation_and_setter_no_reset(env):
    snap=EvaluationAdapter(env).snapshot();state=env.actuator_model.omega.copy()
    env.set_estimated_efficiency([.743,1,1,1]);env.sync_allocator();inverse=env.B_pinv.copy()
    env.motor_effectiveness=np.array([.1,.2,.3,.4]);env.sync_allocator()
    np.testing.assert_array_equal(env.B_pinv,inverse)
    assert EvaluationAdapter(env).snapshot()==snap
    np.testing.assert_array_equal(env.actuator_model.omega,state)


def test_zero_efficiency_rank_is_logged_without_floor(env):
    env.set_estimated_efficiency([0,1,1,1]);env.sync_allocator()
    assert env.allocator_rank==3 and np.isfinite(env.B_pinv).all()
    assert np.all(env.allocator_matrix[:,3]==0)


@pytest.mark.parametrize('mode',['blind','oracle','estimated'])
def test_information_source_is_explicit(env,mode):
    env.allocation_source=mode
    env.set_estimated_efficiency([.831,1,1,1])
    env.motor_effectiveness=native_from_user([.70,1,1,1]);env.sync_allocator()
    expected={'blind':1.,'oracle':.70,'estimated':.831}[mode]
    assert env.allocator_efficiency[3]==expected


def test_motor_cap_and_unchanged_low_range(tmp_path):
    config=load_config('configs/eval_velocity_ab_user_frd.yaml');model,_=nominal_model(config)
    result=audit(tmp_path,model,config)
    assert result['currently_reachable_steady_upper_n']==pytest.approx(.20)
    assert result['raised_cap_only_upper_n']==pytest.approx(.289)
    assert result['inverse_unchanged_below_old_cap_max_omega_error_rad_s']==0
    assert result['hypothetical_30gf_extrapolation']['ratio']>1


def test_short_real_path_observer_schema():
    class FixedPolicy:
        def bind(self,env):pass
        def predict(self,obs):return np.zeros(4)
    config=load_config('configs/eval_velocity_ab_user_frd.yaml');model,_=nominal_model(config)
    rows,initial,observer,case,reasons=execute(config,FixedPolicy(),SCENARIOS[0],model,EstimatorSettings(),'estimated',horizon=.3)
    assert len(rows)==30
    from crazyflie_rl.estimated_allocation_eval import TRACE_KEYS,EXTRA_KEYS
    for row in rows:
        assert not set(TRACE_KEYS+EXTRA_KEYS)-row.keys()
        assert row['estimate_time']==row['time_post']
        assert row['estimate_source_time_used'] is None or row['estimate_source_time_used']<=row['time']+1e-12
        np.testing.assert_array_equal(row['allocator_efficiency_user'],np.ones(4))


def test_physical_termination_keeps_unobserved_metrics_null():
    # Deliberately unsafe synthetic action fixture, not a PPO performance result.
    class TiltFixture:
        def bind(self,env):pass
        def predict(self,obs):return np.array([1.,0.,0.,0.])
    config=load_config('configs/eval_velocity_ab_user_frd.yaml');model,_=nominal_model(config)
    rows,_,observer,case,reasons=execute(config,TiltFixture(),SCENARIOS[3],model,EstimatorSettings(),'estimated')
    r=metrics(rows,SCENARIOS[3],observer,case,reasons)
    assert r['terminated'] and not r['completed'] and r['actual_duration_sec']<5
    assert all(r['windows'][name] is None for name in ('full_0_20','post_5_20','tail_18_20'))
    assert r['partial_observed_windows']['full_0_20'] is not None
    assert r['confirmed_detection_time'] is None and r['first_allocator_compensation_time'] is None
    assert all(r[f'recovery_{name}_s'] is None for name in ('xy','z','3d','position_attitude'))
    assert r['estimation_windows']['tail_18_20'] is None


def test_preservation_baseline_is_self_contained_and_excludes_new_run(tmp_path,monkeypatch):
    import json
    import crazyflie_rl.estimated_allocation_eval as ev
    from crazyflie_rl.dr_policy import sha256
    (tmp_path/'existing.txt').write_text('preserve')
    directory=tmp_path/'new-run';directory.mkdir();(directory/'settings.json').write_text('{}')
    monkeypatch.setattr(ev,'ROOT',tmp_path)
    monkeypatch.setattr(ev.subprocess,'check_output',lambda *a,**k:b'existing.txt\0new-run/settings.json\0')
    ev.capture_protected_files(directory)
    saved=json.loads((directory/'protected_hashes_before.json').read_text())
    assert saved=={'existing.txt':sha256(tmp_path/'existing.txt')}

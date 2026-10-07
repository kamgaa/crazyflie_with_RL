from functools import partial
import json

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.dr_transfer import ROOT, EvaluationAdapter, run_case
from crazyflie_rl.integral_controller import IntegralController
from crazyflie_rl.integral_validation import GAINS, ScenarioObserver, make_case
from crazyflie_rl.payload_motor_eval import DEFAULT_RECORD, RecordedFaultEnv, condition_config, select_models
from crazyflie_rl.oracle_allocation import OracleAllocationEnv, efficiency_matrix, static_hover
from crazyflie_rl.oracle_eval import (SCENARIOS, CONFIGURATIONS, OracleObserver, static_audit,
    analyze, write_reports, verify_oracle, save_comparison_plots)


@pytest.fixture(scope='module')
def config(): return load_config(ROOT/'configs/eval_velocity_ab.yaml')


@pytest.fixture(scope='module')
def policies(config):
    from stable_baselines3 import PPO
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args,**kwargs):raise AssertionError('learning forbidden')
        patch.setattr(PPO,'learn',forbidden);patch.setattr(PPO,'train',forbidden)
        yield select_models(DEFAULT_RECORD,config)


def initialize(config, scenario, mode):
    env=OracleAllocationEnv(config=condition_config(config,scenario),allocator_mode=mode)
    adapter=EvaluationAdapter(env);adapter.reset_to_case_initial_state(make_case(scenario),42)
    observer=OracleObserver(scenario,IntegralController(None,GAINS[1]));observer.on_reset(adapter)
    json.dumps(observer.metadata,allow_nan=False)
    return env,adapter,observer


def test_fixed_matrix_and_static_equilibrium(config):
    assert len(SCENARIOS)*len(CONFIGURATIONS)*2==40
    records=static_audit({s.name:condition_config(config,s) for s in SCENARIOS})
    json.dumps(records,allow_nan=False)
    assert len(records)==8
    for r in records:
        assert r['feasible_static_equilibrium'] and r['physical_wrench_in_policy_range']
        assert all(v['expressible'] for v in r['equilibrium_commands'].values())
        assert r['total_mass_kg']==pytest.approx(.043384+r['payload_mass_kg'])
        np.testing.assert_allclose(r['probe_all_generalized_acceleration'],0,atol=1e-8)
        np.testing.assert_allclose(r['force_balance_residual_n'],0,atol=1e-14)
        np.testing.assert_allclose(r['moment_balance_residual_nm'],0,atol=1e-14)
        np.testing.assert_array_equal(r['external_torque_applied_nm'],0)
        np.testing.assert_allclose(r['composite_com_body_m'][:2],
            np.array(r['payload_offset_body_xy_m'])*r['payload_mass_kg']/r['total_mass_kg'],atol=1e-14)
    faulty=[r for r in records if r['payload_mass_kg']==.005 and min(r['efficiency'])==.7]
    assert len(faulty)==3
    for r in faulty:
        i=int(np.argmin(r['efficiency']))
        assert r['bottleneck_motor']==i+1
        independent=(.048384*9.81/4 + .005*.03*9.81/(4*.03536))/.7
        assert r['required_nominal_thrust_n'][i]==pytest.approx(independent,abs=1e-14)


def test_static_probe_does_not_change_environment_or_motor_state(config):
    env,adapter,observer=initialize(config,SCENARIOS[1],'existing')
    try:
        before=adapter.snapshot();omega=env.actuator_model.omega.copy()
        static_hover(env,[1,.7,1,1])
        assert adapter.snapshot()==before
        np.testing.assert_array_equal(env.actuator_model.omega,omega)
        np.testing.assert_array_equal(env.motor_effectiveness,1)
    finally:env.close()


@pytest.mark.parametrize('index',range(4))
def test_column_scaling_unclipped_solution_and_single_application(config,index):
    env,adapter,observer=initialize(config,SCENARIOS[0],'oracle')
    try:
        B0=env.B.copy(); eta=np.ones(4);eta[index]=.7
        env.motor_effectiveness=eta.copy();env.sync_allocator()
        np.testing.assert_array_equal(env.allocator_matrix,efficiency_matrix(B0,eta))
        np.testing.assert_array_equal(env.B,B0)
        desired=env.allocator_matrix@np.array([.08,.09,.10,.11])
        cmd=env.B_pinv@desired
        np.testing.assert_allclose(env.allocator_matrix@cmd,desired,atol=1e-15)
        env._apply_control(desired)
        r=env.physics_rows[-1]
        np.testing.assert_allclose(r['motor_thrust_command'],[.08,.09,.10,.11],atol=1e-14)
        np.testing.assert_array_equal(r['motor_thrust_actual'],eta*r['motor_thrust_nominal'])
        np.testing.assert_array_equal(r['motor_reaction_actual'],eta*r['motor_reaction_nominal'])
        np.testing.assert_allclose(r['allocation_residual_b0'],0,atol=1e-14)
        assert np.linalg.norm(r['actuator_response_residual_xml'])>1e-5  # Delay retained.
        np.testing.assert_allclose(r['allocation_residual_xml']+r['actuator_response_residual_xml'],r['total_rotor_residual_xml'],atol=1e-15)
        env.motor_effectiveness[:]=1;env.sync_allocator()
        np.testing.assert_array_equal(env.B_pinv,env.B0_pinv)
    finally:env.close()


@pytest.mark.parametrize('mode',('existing','oracle'))
@pytest.mark.parametrize('scenario',SCENARIOS[1:],ids=lambda s:s.name)
def test_event_sync_no_future_efficiency_preserves_states(config,mode,scenario):
    env,adapter,observer=initialize(config,scenario,mode)
    try:
        observer.controller.xi[:]=[.02,-.03,.01]
        observer.before_step(adapter,0,0)
        np.testing.assert_array_equal(env.allocator_efficiency,np.ones(4))
        for event in scenario.events():
            env.data.time=event.time;env._step=int(event.time/.01)
            before=adapter.snapshot();xi=observer.controller.xi.copy()
            observer.before_step(adapter,env._step,event.time)
            assert adapter.snapshot()==before
            np.testing.assert_array_equal(observer.controller.xi,xi)
            expected=np.ones(4);expected[event.motor_number-1]=event.effectiveness
            np.testing.assert_array_equal(env.motor_effectiveness,expected)
            np.testing.assert_array_equal(env.allocator_efficiency,expected if mode=='oracle' else np.ones(4))
    finally:env.close()


@pytest.mark.parametrize('index',[0,1])
@pytest.mark.parametrize('gain',GAINS,ids=lambda g:g.name)
def test_nominal_exact_old_and_oracle_policy_rollout(config,policies,index,gain):
    s=SCENARIOS[0];case=make_case(s,.15);cfg=condition_config(config,s)
    traces=[]
    for mode in ('old','existing','oracle'):
        controller=IntegralController(policies[index],gain)
        observer=(ScenarioObserver if mode=='old' else OracleObserver)(s,controller)
        factory=RecordedFaultEnv if mode=='old' else partial(OracleAllocationEnv,allocator_mode=mode)
        output=run_case(cfg,case,controller,42,env_factory=factory,observer=observer,observation_transform=controller.prepare_observation)
        assert output[2] is None
        traces.append(output)
    assert traces[0][1:]==traces[1][1:]==traces[2][1:]
    for old in traces[0][0]:
        k=round(old['time']/.01)
        for new in (traces[1][0][k],traces[2][0][k]):
            for key in old:
                if isinstance(old[key],np.ndarray):np.testing.assert_array_equal(old[key],new[key])
                else:assert old[key]==new[key]


def test_saturation_freezes_actual_allocator_and_full_output(config,policies,tmp_path):
    s=SCENARIOS[2];case=make_case(s,5.05);cfg=condition_config(config,s)
    results=[]
    for mode in ('existing','oracle'):
        controller=IntegralController(policies[0],GAINS[1]);observer=OracleObserver(s,controller)
        rows,snapshot,error,reasons=run_case(cfg,case,controller,42,
            env_factory=partial(OracleAllocationEnv,allocator_mode=mode),observer=observer,
            observation_transform=controller.prepare_observation)
        assert error is None and len(rows)==505 and len(observer.events)==1
        verify_oracle(rows,observer.physics_rows,s,mode)
        for k,r in enumerate(rows):
            physics=observer.physics_rows[k*5:(k+1)*5]
            hit=any(p['allocator_clipped'].any() or p['esc_lower'].any() or p['esc_upper'].any() for p in physics) or r['policy_action_at_bound'].any()
            assert r['integral_frozen']==hit
            if hit:np.testing.assert_array_equal(r['xi_t'],r['xi_next'])
            assert r['motor_sample_time_post']==pytest.approx(r['time_post'])
        result=analyze(rows,make_case(s),observer,error,reasons,cfg)
        assert result['windows']['tail_58_60'] is None
        result.update(label='A_best',key=mode,allocator_mode=mode,configuration=mode+'_integral_020')
        results.append(result)
    write_reports(tmp_path,results)
    assert json.loads((tmp_path/'summary.json').read_text())[0]['partial']
    text=(tmp_path/'window_metrics.csv').read_text()
    assert 'allocator_mode' in text and 'existing_integral_020' in text and 'oracle_integral_020' in text

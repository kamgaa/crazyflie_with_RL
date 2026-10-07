from dataclasses import replace
import json

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.dr_transfer import ROOT, EvaluationAdapter, Thresholds, run_case
from crazyflie_rl.integral_controller import IntegralController
from crazyflie_rl.integral_eval import read_columns, save_plots
from crazyflie_rl.integral_validation import (SCENARIOS, GAINS, ScenarioObserver, make_case,
    event_recovery, interval_statistics, analyze, verify_signals, write_event_tables)
from crazyflie_rl.payload_motor_eval import (DEFAULT_RECORD, RecordedFaultEnv, condition_config, select_models)


@pytest.fixture(scope='module')
def config():
    return load_config(ROOT/'configs/eval_velocity_ab.yaml')


@pytest.fixture(scope='module')
def policies(config):
    from stable_baselines3 import PPO
    patch = pytest.MonkeyPatch()
    def forbidden(*a, **k): raise AssertionError('learning forbidden')
    patch.setattr(PPO, 'learn', forbidden); patch.setattr(PPO, 'train', forbidden)
    try: yield select_models(DEFAULT_RECORD, config)
    finally: patch.undo()


def test_exact_predeclared_matrix_and_reference_boundary():
    assert len(SCENARIOS) == 13 and len(GAINS) == 2
    assert len(set(s.name for s in SCENARIOS)) == 13
    for s in SCENARIOS:
        case = make_case(s)
        assert case.horizon == 60
        np.testing.assert_array_equal(case.initial_position(), [0, 0, 1])
        for t in (0, 5, 29.99, 30, 30.01, 60):
            np.testing.assert_array_equal(case.reference_velocity(t), 0)
        if s.target_after30:
            np.testing.assert_array_equal(case.post_reference(30)[0], [0, 0, 1])
            np.testing.assert_array_equal(case.reference(30)[0], s.target_after30)
            np.testing.assert_array_equal(case.post_reference(30.01)[0], s.target_after30)
            assert [e.kind for e in s.events()] == ['fault', 'target_step']
        if s.restore: assert [e.kind for e in s.events()] == ['fault', 'restore']


@pytest.mark.parametrize('scenario', SCENARIOS, ids=lambda s: s.name)
def test_com_rotor_mapping_event_state_preservation(config, scenario):
    env = RecordedFaultEnv(config=condition_config(config, scenario))
    try:
        adapter = EvaluationAdapter(env); adapter.reset_to_case_initial_state(make_case(scenario), 42)
        controller = IntegralController(None, GAINS[1]); observer = ScenarioObserver(scenario, controller)
        observer.on_reset(adapter)
        controller.xi[:] = [.03, -.04, .02]
        expected_com = (env._m0*env._ipos0 + scenario.mass*np.r_[scenario.offset, 0.])/(env._m0+scenario.mass)
        np.testing.assert_allclose(env.model.body_ipos[env.drone_bid], expected_com, atol=1e-15)
        assert env.model.body_mass[env.drone_bid] == pytest.approx(env._m0+scenario.mass)
        assert env._com_mw == scenario.mass
        np.testing.assert_allclose(env._last_f, np.ones(4)*env.mass*env.gravity/4, rtol=1e-12)
        np.testing.assert_allclose([r['position_body_m'] for r in observer.metadata['rotors']],
            [[.03536, -.03536, 0], [-.03536, -.03536, 0], [-.03536, .03536, 0], [.03536, .03536, 0]], atol=1e-12)
        assert [r['site'] for r in observer.metadata['rotors']] == ['motor0', 'motor1', 'motor2', 'motor3']
        np.testing.assert_array_equal([r['motor_direction'] for r in observer.metadata['rotors']], [1,-1,1,-1])
        for event in scenario.events():
            # Controlled unit-test clock placement, not a reported flight.
            env.data.time = event.time; env._step = round(event.time/adapter.control_dt)
            before = adapter.snapshot(); xi = controller.xi.copy()
            observer.before_step(adapter, env._step, event.time)
            for key in before:
                if key not in ('reference', 'observation'): assert adapter.snapshot()[key] == before[key]
            np.testing.assert_array_equal(controller.xi, xi)
            record = observer.events[-1]
            np.testing.assert_array_equal(record['p_cmd_after'], env.pos_des+xi)
            if event.motor_number:
                expected_eff = np.ones(4); expected_eff[event.motor_number-1] = event.effectiveness
                np.testing.assert_array_equal(env.motor_effectiveness, expected_eff)
                # Direct force AND reaction signal check after the existing
                # dynamics path, for each selected motor including 2,3,4.
                env.step(np.zeros(4))
                for r in env.physics_rows[observer.physics_start:]:
                    np.testing.assert_array_equal(r['motor_thrust_actual'], expected_eff*r['motor_thrust_nominal'])
                    np.testing.assert_array_equal(r['motor_reaction_actual'], expected_eff*r['motor_reaction_nominal'])
                    np.testing.assert_array_equal(r['applied_force_ctrl'], r['motor_thrust_actual'])
                    np.testing.assert_array_equal(r['applied_torque_ctrl'], r['motor_reaction_actual'])
            else:
                np.testing.assert_array_equal(env.motor_effectiveness, [.7, 1, 1, 1])
                np.testing.assert_array_equal(adapter.current_observation()[:3],
                    (np.array(before['position'])-env.pos_des).astype(np.float32))
    finally: env.close()


@pytest.mark.parametrize('index', [0, 1])
def test_no_integral_equals_native_frozen_inference(config, policies, index):
    scenario = SCENARIOS[0]; case = make_case(scenario, .12); policy = policies[index]
    cfg = condition_config(config, scenario)
    native_observer = ScenarioObserver(scenario, IntegralController(None, GAINS[0]))
    # The same initialization/events/logger with no integral transformation or
    # finish-step hook; policy is directly the existing FrozenPolicy.
    native_observer.after_step = lambda env, row: None
    native = run_case(cfg, case, policy, 42, env_factory=RecordedFaultEnv, observer=native_observer)
    controller = IntegralController(policy, GAINS[0]); observer = ScenarioObserver(scenario, controller)
    wrapped = run_case(cfg, case, controller, 42, env_factory=RecordedFaultEnv,
                       observer=observer, observation_transform=controller.prepare_observation)
    assert native[2] is None and wrapped[2] is None
    assert native[1:] == wrapped[1:]
    for a,b in zip(native[0], wrapped[0]):
        for key in ('action','position','velocity','quaternion','omega','motor_thrust','observation','reward'):
            np.testing.assert_array_equal(a[key], b[key])


def test_event_recovery_stops_at_next_event_and_uses_right_limit(config):
    from test_payload_motor_eval import samples
    rows = samples(60)
    for r in rows:
        if r['time_post'] < 10 or r['time_post'] > 30: r['position_error_world'][0] = .05
    scenario = SCENARIOS[-2]; fault, target = scenario.events()
    record = dict(e_true_after=np.array([.01,0,0]), velocity=np.zeros(3))
    rec = event_recovery(rows, record, fault, 30., 60., Thresholds())
    assert rec == {'xy': 5., 'z': 0., '3d': 5.}
    # A later early termination cannot erase the completed first interval.
    assert event_recovery(rows[:4000], record, fault, 30., 40., Thresholds()) == rec
    assert event_recovery(rows[:2000], record, fault, 30., 20., Thresholds())['xy'] is None
    # The old goal at post-state t=30 cannot yield false zero-latency recovery.
    rows[2999]['position_error_world'][:] = 0
    new_record = dict(e_true_after=np.array([-.05,0,0]), velocity=np.zeros(3))
    assert event_recovery(rows, new_record, target, 60., 60., Thresholds())['3d'] is None
    for r in rows[3000:]:
        r['position_error_world'][:] = .05 if r['time_post'] < 59.1 else 0
    assert event_recovery(rows, new_record, target, 60., 60., Thresholds())['3d'] is None


def test_partial_windows_keep_fixed_endpoints(config):
    from test_payload_motor_eval import samples
    rows = samples(35)
    for r in rows:
        r['yaw_error_rad']=.1
        for key in ('integral_frozen','interval_allocator_clipping','interval_esc_boundary','interval_action_boundary',
                    'integral_xy_at_limit','integral_z_at_limit','integral_xy_projected','integral_z_projected'):r[key]=False
        r['xi_next']=np.zeros(3)
    full, partial = interval_statistics(rows, [], 5, 30, 35, config)
    assert full['sample_count']==2500 and partial is None
    assert full['yaw_error_mean_rad']==pytest.approx(.1)
    full, partial = interval_statistics(rows, [], 30, 60, 35, config)
    assert full is None and partial['sample_count']==500
    assert interval_statistics(rows, [], 58, 60, 35, config)==(None,None)


@pytest.mark.parametrize('scenario', SCENARIOS[-2:], ids=lambda s:s.name)
def test_real_policy_target_boundary_and_full_output_pipeline(config, policies, scenario, tmp_path):
    # One real 30s smoke per commanded axis; no training and no alternative scenario.
    policy = policies[1]; controller = IntegralController(policy, GAINS[1]); observer = ScenarioObserver(scenario, controller)
    case = make_case(scenario, 30.02); cfg = condition_config(config, scenario)
    rows,snapshot,error,reasons = run_case(cfg, case, controller, 42, env_factory=RecordedFaultEnv,
        observer=observer, observation_transform=controller.prepare_observation)
    assert error is None and len(rows)==3002 and len(observer.events)==2
    np.testing.assert_array_equal(rows[2999]['reference_post'], [0,0,1])
    np.testing.assert_array_equal(rows[3000]['reference'], scenario.target_after30)
    np.testing.assert_array_equal(rows[3000]['position_before'], rows[2999]['position'])
    np.testing.assert_array_equal(rows[3000]['xi_t'], rows[2999]['xi_next'])
    np.testing.assert_array_equal(rows[3000]['motor_effectiveness'], [.7,1,1,1])
    for r in rows[2999:]:
        e=r['position']-r['reference_post'];vdes=-4*e
        if np.linalg.norm(vdes)>1.5:vdes*=1.5/np.linalg.norm(vdes)
        np.testing.assert_allclose(r['internal_velocity_error'],r['velocity']-vdes,atol=1e-14)
    verify_signals(rows,observer.physics_rows,scenario)
    result=analyze(rows,make_case(scenario),observer,error,reasons,cfg)
    result.update(label='B_best',key='smoke')
    assert result['event_results'][0]['recovery_s']['3d'] is not None
    assert result['event_results'][1]['recovery_s']['3d'] is None
    assert result['windows']['tail_58_60'] is None
    write_event_tables(tmp_path,[result])
    from crazyflie_rl.dr_transfer import write_rollout
    write_rollout(tmp_path/'smoke.csv',rows)
    assert len(read_columns(tmp_path/'smoke.csv').time)==3002
    save_plots(tmp_path,[result],'integral_020',conditions=(scenario,),
        event_markers={scenario.name:[(5.,'fault'),(30.,'target_step')]},motor_numbers={scenario.name:1})
    assert len(list(tmp_path.glob('*.png')))==3
    assert json.loads((tmp_path/'summary.json').read_text())[0]['event_results'][0]['end_s']==30

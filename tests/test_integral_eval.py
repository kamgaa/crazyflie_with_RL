from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.dr_policy import FrozenPolicy
from crazyflie_rl.dr_transfer import ROOT, Case, Thresholds, run_case
from crazyflie_rl.integral_controller import IntegralController, IntegralSettings
from crazyflie_rl.integral_eval import IntegralObserver, window_statistics, verify_integral_rows
from crazyflie_rl.payload_motor_eval import (CONDITIONS, DEFAULT_RECORD, FaultObserver,
    RecordedFaultEnv, condition_config, select_models, recovery_time)


def controller(xy=.1, z=.1, **kwargs):
    return IntegralController(None, IntegralSettings('test', xy, z, **kwargs))


def step(c, error, dt=.01, **saturation):
    raw = np.arange(15, dtype=np.float32)
    raw[:3] = error
    original = raw.copy()
    obs = c.prepare_observation(raw, error, np.zeros(3), dt)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(obs[3:], raw[3:])
    flags = dict(allocator_clipping=False, esc_boundary=False, action_boundary=False)
    flags.update(saturation)
    return c.finish_step(**flags)


def test_sign_prestate_integration_and_control_dt():
    c = controller(.05, .2)
    for _ in range(100):
        r = step(c, [1., -2., .1])
    np.testing.assert_allclose(c.xi, [-.05, .1, -.02], atol=1e-15)
    np.testing.assert_array_equal(r['xi_t'], r['p_cmd'])
    assert r['e_actor_before'][0] > r['e_true_before'][0]
    other = controller(.05, .2)
    for _ in range(50): step(other, [1., -2., .1], .02)
    np.testing.assert_allclose(c.xi, other.xi, atol=1e-15)
    # Post-state error is never accepted by finish_step: pending PRE-state owns it.


def test_independent_projection_bounded_state_not_hidden_accumulator():
    c = controller(1, 1)
    r = step(c, [3., 4., -10.], dt=1)
    np.testing.assert_allclose(c.xi, [-.24, -.32, .15])
    assert r['integral_xy_projected'] and r['integral_z_projected']
    step(c, [3., 4., -10.], dt=1)
    np.testing.assert_allclose(c.xi, [-.24, -.32, .15])
    step(c, [-.3, -.4, 1.], dt=.1)
    np.testing.assert_allclose(c.xi, [-.21, -.28, .05])
    z_only = controller(0, 1); step(z_only, [30, 40, -10], 1)
    np.testing.assert_array_equal(z_only.xi, [0, 0, .15])


@pytest.mark.parametrize('reason', ['allocator_clipping', 'esc_boundary', 'action_boundary'])
def test_freeze_all_axes_resume_without_decay(reason):
    c = controller(); step(c, [1., 2., 3.])
    xi = c.xi.copy()
    for _ in range(10):
        r = step(c, [-1., -2., -3.], **{reason: True})
        np.testing.assert_array_equal(c.xi, xi)
        assert r['integral_frozen'] and not r['integral_update_allowed']
    assert r['integral_frozen_total_s'] == .1 and r['integral_frozen_longest_s'] == .1
    step(c, [-1., -2., -3.])
    np.testing.assert_allclose(c.xi, 0, atol=1e-17)


def test_zero_gain_bypasses_transform_and_rollout_reset():
    c = controller(0, 0); raw = np.arange(15, dtype=np.float32)
    assert c.prepare_observation(raw, [4, 5, 6], [1, 2, 3], .01) is raw
    c.finish_step(allocator_clipping=True, esc_boundary=True, action_boundary=True)
    np.testing.assert_array_equal(c.xi, 0)
    c = controller(); step(c, [1., 2., 3.]); c.reset()
    np.testing.assert_array_equal(c.xi, 0)
    assert c.pending is None and c.frozen_steps == 0


def test_raw_transform_before_nonunit_frozen_normalization():
    from gymnasium.spaces import Box
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    seen = []
    model = SimpleNamespace(predict=lambda obs, deterministic: (seen.append(obs.copy()) or np.zeros(4), None))
    # Real SB3 normalize_obs, populated without a simulator vector wrapper.
    normalizer = VecNormalize.__new__(VecNormalize)
    normalizer.norm_obs = True; normalizer.clip_obs = 2.; normalizer.epsilon = 1e-8
    normalizer.training = False; normalizer.observation_space = Box(-np.inf, np.inf, (15,))
    normalizer.obs_rms = RunningMeanStd(shape=(15,))
    normalizer.obs_rms.mean = np.arange(15)*.03
    normalizer.obs_rms.var = np.arange(15)+.2
    normalizer.obs_rms.count = 42.
    policy = FrozenPolicy(model, {}, None, normalizer)
    c = IntegralController(policy, IntegralSettings('test', .1, .1)); c.xi[:] = [.1, -.2, .05]
    raw = np.arange(15, dtype=np.float32)*.1; before = raw.copy()
    state, target = np.array([.4, .2, 1.1]), np.array([0, 0, 1.])
    transformed = c.prepare_observation(raw, state, target, .01)
    c.predict(transformed)
    expected = raw.copy(); expected[:3] = state-target-c.xi
    expected = np.clip((expected-normalizer.obs_rms.mean)/np.sqrt(normalizer.obs_rms.var+1e-8), -2, 2).astype(np.float32)
    np.testing.assert_array_equal(seen[-1], expected)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(normalizer.obs_rms.mean, np.arange(15)*.03)
    assert normalizer.obs_rms.count == 42 and not normalizer.training


@pytest.fixture(scope='module')
def config():
    return load_config(ROOT/'configs/eval_velocity_ab.yaml')


@pytest.fixture(scope='module')
def policies(config):
    from stable_baselines3 import PPO
    patch = pytest.MonkeyPatch()
    def forbidden(*a, **k): raise AssertionError('training forbidden')
    patch.setattr(PPO, 'learn', forbidden); patch.setattr(PPO, 'train', forbidden)
    try: yield select_models(DEFAULT_RECORD, config)
    finally: patch.undo()


@pytest.mark.parametrize('index', [0, 1])
def test_real_policy_no_integral_exact_and_native_reward(config, policies, index, tmp_path):
    from crazyflie_rl.integral_eval import analyze, write_tables
    policy = policies[index]; case = Case('hover', .15, (0, 0, 1))
    native = run_case(config, case, policy, 42, env_factory=RecordedFaultEnv, observer=FaultObserver(CONDITIONS[0]))
    c = IntegralController(policy, IntegralSettings('no_integral', 0, 0))
    observer = IntegralObserver(CONDITIONS[0], c)
    new = run_case(config, case, c, 42, env_factory=RecordedFaultEnv,
                   observer=observer, observation_transform=c.prepare_observation)
    assert new[2] is None and native[1:] == new[1:]
    for old, row in zip(native[0], new[0]):
        for k in ('action', 'observation', 'policy_raw_observation', 'position', 'velocity', 'quaternion',
                  'omega', 'motor_thrust', 'reward', 'reference', 'internal_velocity_error'):
            np.testing.assert_array_equal(old[k], row[k])
    verify_integral_rows(new[0], c.settings)
    result = analyze(new[0], case, CONDITIONS[0], observer, new[2], new[3], Thresholds(), config)
    result.update(label='test', key='test')
    write_tables(tmp_path, [result])
    assert (tmp_path/'integral_summary.csv').exists() and result['completed']


def test_real_fault_preserves_xi_and_true_target(config, policies):
    condition = CONDITIONS[3]; policy = policies[0]
    c = IntegralController(policy, IntegralSettings('integral_010', .1, .1))
    observer = IntegralObserver(condition, c)
    rows, snapshot, error, reasons = run_case(condition_config(config, condition), Case('hover', 5.02, (0, 0, 1)),
        c, 42, env_factory=RecordedFaultEnv, observer=observer, observation_transform=c.prepare_observation)
    assert error is None and len(observer.events) == 1
    event = observer.events[0]; assert event['control_step'] == 500
    np.testing.assert_array_equal(event['xi_before'], event['xi_after'])
    assert np.linalg.norm(event['xi_before']) > 0
    verify_integral_rows(rows, c.settings)
    for row in rows:
        np.testing.assert_array_equal(row['reference'], [0, 0, 1])
        np.testing.assert_array_equal(row['e_true_post'], row['position']-row['reference_post'])
    # The physical state at the event is continuous and uses previous xi_next.
    np.testing.assert_array_equal(rows[500]['position_before'], rows[499]['position'])
    np.testing.assert_array_equal(rows[500]['xi_t'], rows[499]['xi_next'])


def test_observer_uses_every_substep_not_actual_force(monkeypatch):
    c = controller()
    observer = IntegralObserver(CONDITIONS[0], c)
    monkeypatch.setattr(FaultObserver, 'after_step', lambda self, env, row: None)
    def signal(clipped):
        return dict(allocator_clipped=np.array([clipped, 0, 0, 0], bool),
                    esc_lower=np.zeros(4, bool), esc_upper=np.zeros(4, bool),
                    motor_effectiveness=np.zeros(4), motor_thrust_actual=np.zeros(4))
    row = dict(policy_action_at_bound=np.zeros(4, bool), position_error_world=np.ones(3))
    c.prepare_observation(np.zeros(15), np.ones(3), np.zeros(3), .01)
    observer.physics_rows = [signal(True), signal(False)]
    observer.after_step(SimpleNamespace(substeps=2), row)
    assert row['integral_frozen'] and row['interval_allocator_clipping']
    np.testing.assert_array_equal(c.xi, 0)
    c.prepare_observation(np.zeros(15), np.ones(3), np.zeros(3), .01)
    observer.physics_rows = [signal(False), signal(False)]
    observer.after_step(SimpleNamespace(substeps=2), row)
    assert row['integral_update_allowed']  # Zero actual effectiveness is irrelevant to the predicate.
    np.testing.assert_allclose(c.xi, -.001)


def test_partial_fixed_windows_and_60s_suffix():
    from test_payload_motor_eval import samples
    rows = samples(24)
    full, partial = window_statistics(rows, 24)
    assert full['tail_18_20']['sample_count'] == 200
    assert full['full_0_20']['sample_count'] == 2000
    assert full['full_0_60'] is None and full['tail_58_60'] is None
    assert partial['full_0_60']['sample_count'] == 2400
    assert partial['tail_58_60'] is None
    rows = samples(60)
    for r in rows:
        if r['time_post'] < 40: r['position_error_world'][0] = .01
    assert recovery_time(rows, Thresholds(), 'xy', True, horizon=60.) == 35.
    assert recovery_time(rows, Thresholds(), 'xy', True, horizon=20.) is None
    assert recovery_time(rows, Thresholds(), 'xy', False, horizon=60.) is None


@pytest.mark.parametrize('value', [-1, float('inf'), float('nan')])
def test_invalid_settings(value):
    with pytest.raises(ValueError): IntegralSettings('bad', value, .1)

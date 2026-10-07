"""Independent velocity channels, compatibility, and paired A/B configuration."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from crazyflie_rl.config import load_config, ConfigError
from crazyflie_rl.dr_policy import policy_raw_observation
from crazyflie_rl.dr_transfer import Case, EvaluationAdapter
from crazyflie_rl.environment import CrazyflieResidualEnv
from crazyflie_rl.velocity_reference import (
    velocity_semantics, velocity_reward_semantics, validate_velocity_metadata)

ROOT = Path(__file__).resolve().parents[1]


def profile(name):
    return load_config(ROOT / f'configs/e2e_train_velocity_ab_{name}.yaml')


def test_ab_only_reward_mode_differs():
    a, b = (profile(name).resolved_dict() for name in ('a', 'b'))
    for d in (a, b):
        d.pop('source_path')
        for key in ('condition', 'description'): d['experiment'].pop(key)
    assert a['environment']['e2e_velocity'].pop('reward_mode') == 'absolute'
    assert b['environment']['e2e_velocity'].pop('reward_mode') == 'position_error'
    assert a == b
    c = profile('a')
    assert c.training.seed == 42 and c.training.total_timesteps == 1000000
    assert c.environment.position_perturbation == .05
    assert c.environment.payload.mass == 0 and not c.environment.payload.randomize
    assert not c.actuator.randomization.enabled


@pytest.mark.parametrize('obs_mode,reward_mode', [
    ('absolute', 'absolute'), ('absolute', 'position_error'),
    ('position_error', 'absolute'), ('position_error', 'position_error')])
def test_four_definitions_pre_input_and_post_reward(obs_mode, reward_mode):
    c = profile('a')
    settings = replace(c.environment.e2e_velocity, observation_mode=obs_mode, reward_mode=reward_mode)
    c = replace(c, environment=replace(c.environment, e2e_velocity=settings))
    env = CrazyflieResidualEnv(config=c)
    try:
        adapter = EvaluationAdapter(env)
        adapter.reset_to_case_initial_state(Case('step-005', 8, goal=(.05, 0, 1)), 42)
        p, q, v, w = env._read_state()
        des_before = env.desired_velocity(p)
        expected = v - des_before if obs_mode == 'position_error' else v
        np.testing.assert_allclose(adapter.current_observation()[3:6], expected, atol=1e-7)
        obs, reward, _, _, info = env.step([.1, -.1, 0, .1])
        p, q, v, w = env._read_state()
        des_after = env.desired_velocity(p)
        assert not np.array_equal(des_before, des_after)
        np.testing.assert_allclose(obs[3:6], v-des_after if obs_mode == 'position_error' else v, atol=1e-7)
        expected = v-des_after if reward_mode == 'position_error' else v
        assert info['reward_raw']['velocity_reward_sq'] == pytest.approx(expected @ expected)
        assert info['reward_terms']['velocity'] == pytest.approx(-.005*(expected @ expected))
        assert sum(x for k,x in info['reward_terms'].items() if k != 'total') == pytest.approx(reward)
        assert obs.shape == (15,) and np.isfinite(obs).all()
        assert velocity_semantics(c)['mode'] == obs_mode
        assert velocity_reward_semantics(c)['mode'] == reward_mode
        if obs_mode == 'position_error':
            with pytest.raises(ValueError, match='double-subtract'):
                policy_raw_observation(obs, np.zeros(3), 'error', config=c)
        else:
            np.testing.assert_array_equal(policy_raw_observation(obs, np.zeros(3), config=c), obs)
    finally:
        env.close()


def test_loading_uses_observation_not_reward():
    a, b = profile('a'), profile('b')
    validate_velocity_metadata({}, a)
    validate_velocity_metadata({}, b)
    for current, other in ((a,b), (b,a)):
        validate_velocity_metadata({'resolved_config': current.resolved_dict()}, other)
        validate_velocity_metadata({'velocity_semantics': velocity_semantics(current),
                                    'velocity_reward_semantics': velocity_reward_semantics(current)}, other)
    d = load_config(ROOT/'configs/e2e_train_position_velocity_error.yaml')
    c = replace(d, environment=replace(d.environment,
        e2e_velocity=replace(d.environment.e2e_velocity, reward_mode='absolute')))
    validate_velocity_metadata({'velocity_semantics': velocity_semantics(d)}, c)
    with pytest.raises(ValueError, match='semantics mismatch'):
        validate_velocity_metadata({}, c)


def test_d_archived_config_meaning_unchanged():
    run = ROOT/'artifacts/runs/ppo_e2e_hover_position-velocity-error-nominal_seed42_20261001-210046'
    if not run.exists(): pytest.skip('archived D run unavailable')
    saved = yaml.safe_load(next((run/'config').glob('*.yaml')).read_text())
    current = load_config(ROOT/'configs/e2e_train_position_velocity_error.yaml')
    assert current.resolved_dict() == saved
    assert velocity_semantics(current) == velocity_reward_semantics(current)


@pytest.mark.parametrize('key', ['observation_mode', 'reward_mode'])
def test_invalid_channel_rejected(tmp_path, key):
    p = tmp_path/'bad.yaml'
    p.write_text(yaml.safe_dump({'extends': str(ROOT/'configs/e2e_train_velocity_ab_a.yaml'),
                                'environment': {'e2e_velocity': {key: 'bogus'}}}))
    with pytest.raises(ConfigError): load_config(p)


@pytest.mark.parametrize('obs_mode,reward_mode', [
    ('absolute', 'absolute'), ('absolute', 'position_error'),
    ('position_error', 'absolute'), ('position_error', 'position_error')])
def test_reward_balance_uses_reward_selection(obs_mode, reward_mode):
    from types import SimpleNamespace
    from crazyflie_rl.reward_balance import reward_balance_metrics
    c = profile('a')
    c = replace(c, environment=replace(c.environment, e2e_velocity=replace(
        c.environment.e2e_velocity, observation_mode=obs_mode, reward_mode=reward_mode)))
    trace = SimpleNamespace(sample_count=1, time_sec=np.array([.01]), phases=['HOLD'],
        position_error=np.array([.1]), position=np.array([[.1, 0, 1]]),
        reference_position=np.array([[0., 0, 1]]), control_mode='e2e',
        linear_velocity=np.array([[.2, 0, 0]]), velocity_error=np.array([[.6, 0, 0]]),
        angular_velocity=np.zeros((1,3)), attitude_deg=np.zeros((1,3)))
    result = reward_balance_metrics(trace, c)['overall']
    expected = .6 if reward_mode == 'position_error' else .2
    assert result['mean_cost']['velocity'] == pytest.approx(.005*expected**2)
    assert result['velocity_error_rms_mps'] == pytest.approx(.6)
    assert result['velocity_rms_mps'] == pytest.approx(.2)


def test_restore_adapter_is_raw_and_nonmutating():
    from diagnose_velocity_ab import RestoredAbsolutePolicy
    from crazyflie_rl.velocity_reference import ABSOLUTE_CONTRACT
    class FrozenRecorder:
        provenance = {'observation_contract': ABSOLUTE_CONTRACT}
        def predict(self, raw):
            # Non-unit frozen preprocessing must see reconstructed RAW values.
            self.received = raw.copy()
            return (raw[3:6]-np.array([1., 2., 3.]))/np.array([2., 3., 4.])
    policy = FrozenRecorder(); adapter = RestoredAbsolutePolicy(policy)
    obs = np.arange(15, dtype=np.float32); original = obs.copy()
    desired = np.array([.3, -.4, .2])
    action = adapter.predict(obs, desired)
    expected = obs.copy(); expected[3:6] += desired
    np.testing.assert_array_equal(obs, original)
    np.testing.assert_array_equal(policy.received, expected)
    np.testing.assert_allclose(action, (expected[3:6]-[1,2,3])/[2,3,4])
    policy.provenance = {'observation_contract': 'different'}
    with pytest.raises(ValueError, match='absolute-velocity'):
        RestoredAbsolutePolicy(policy)


def test_archived_policies_accept_reward_only_change():
    from diagnose_velocity_ab import BASELINE, D_RUN
    from crazyflie_rl.dr_policy import load_frozen_policy
    final = list((D_RUN/'models').glob('*final*.zip'))
    if not BASELINE.exists() or not final: pytest.skip('archived policies unavailable')
    for c in [profile('a'), profile('b')]:
        policy = load_frozen_policy('historical_baseline', str(BASELINE), c)
        assert policy.provenance['velocity_semantics'] == {'mode': 'absolute'}
    d = load_config(ROOT/'configs/e2e_train_position_velocity_error.yaml')
    c = replace(d, environment=replace(d.environment, e2e_velocity=replace(d.environment.e2e_velocity, reward_mode='absolute')))
    policy = load_frozen_policy('d_final_input_compatible_c', str(final[0]), c)
    assert policy.provenance['velocity_semantics']['mode'] == 'position_error'
    with pytest.raises(ValueError, match='semantics mismatch'):
        load_frozen_policy('invalid', str(BASELINE), d)

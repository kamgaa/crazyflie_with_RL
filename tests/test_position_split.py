"""Non-training regression coverage for independent horizontal/vertical costs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import os

import numpy as np
import pytest
import yaml

from crazyflie_rl.config import ConfigError, dump_resolved_config, load_config
from crazyflie_rl.evaluation import PolicyEvaluator, position_rmse_metrics
from crazyflie_rl.eval_cli import RolloutTrace, trace_metrics
from crazyflie_rl.reward_balance import reward_balance_metrics

ROOT = Path(__file__).resolve().parents[1]


def profile(tmp_path, **weights):
    path = tmp_path / 'split.yaml'
    path.write_text(yaml.safe_dump({
        'extends': os.path.relpath(ROOT / 'configs/e2e_train.yaml', tmp_path),
        'paths': {'project_root': str(ROOT)},
        'environment': {'reward': weights},
    }))
    return load_config(path)


@pytest.mark.parametrize('weights,expected', [
    ({}, (4, 4)),
    ({'position_xy_weight': 4, 'position_z_weight': 4}, (4, 4)),
    ({'position_xy_weight': 7}, (7, 4)),
    ({'position_z_weight': 9}, (4, 9)),
    ({'position_xy_weight': 0}, (0, 4)),
    ({'position_z_weight': 0}, (4, 0)),
    ({'position_weight': 3, 'position_z_weight': 0}, (3, 0)),
])
def test_weight_resolution_and_serialization(tmp_path, weights, expected):
    config = profile(tmp_path, **weights)
    reward = config.environment.reward
    assert (reward.effective_position_xy_weight, reward.effective_position_z_weight) == expected
    saved = dump_resolved_config(config, tmp_path / 'resolved.yaml')
    raw = yaml.safe_load(saved.read_text())['environment']['reward']
    assert (raw['position_xy_weight'], raw['position_z_weight']) == expected
    from crazyflie_rl.environment import CrazyflieResidualEnv
    env = CrazyflieResidualEnv(config=config)
    try:
        assert (env.position_xy_weight, env.position_z_weight) == expected
    finally:
        env.close()


@pytest.mark.parametrize('key', ['position_xy_weight', 'position_z_weight'])
@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, '3', None])
def test_invalid_axis_weights(tmp_path, key, value):
    with pytest.raises(ConfigError, match=key):
        profile(tmp_path, **{key: value})


def test_equal_profile_only_changes_axis_weights():
    a = load_config(ROOT / 'configs/e2e_train.yaml').resolved_dict()
    b = load_config(ROOT / 'configs/e2e_train_position_split_equal.yaml').resolved_dict()
    a.pop('source_path')
    b.pop('source_path')
    assert a == b


def test_inherited_axis_weight_overrides_legacy_fallback(tmp_path):
    parent = tmp_path / 'parent.yaml'
    parent.write_text(yaml.safe_dump({
        'extends': os.path.relpath(ROOT / 'configs/e2e_train.yaml', tmp_path),
        'environment': {'reward': {'position_xy_weight': 0}},
    }))
    child = tmp_path / 'child.yaml'
    child.write_text(yaml.safe_dump({
        'extends': 'parent.yaml',
        'environment': {'reward': {'position_weight': 6}},
    }))
    reward = load_config(child).environment.reward
    assert reward.effective_position_xy_weight == 0
    assert reward.effective_position_z_weight == 6


@pytest.mark.parametrize('mode', ['e2e', 'residual'])
@pytest.mark.parametrize('weights', [(4, 4), (7, 4), (4, 9), (0, 4), (4, 0)])
def test_axis_cost_independence_and_legacy_total(mode, weights):
    from crazyflie_rl.environment import CrazyflieResidualEnv
    config = load_config(ROOT / f'configs/{mode}_train.yaml')
    reward = replace(config.environment.reward, position_weight=4,
                     position_xy_weight=weights[0], position_z_weight=weights[1])
    config = replace(config, environment=replace(config.environment, reward=reward))
    env = CrazyflieResidualEnv(config=config)
    try:
        env.reset(seed=42)
        env.data.qpos[:3] = env.pos_des + [0.3, -0.2, 0.1]
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        _, total, terminated, _, info = env.step(np.zeros(4))
        error = env.data.qpos[:3] - env.pos_des
        expected_xy = weights[0] * sum(error[:2] ** 2)
        expected_z = weights[1] * error[2] ** 2
        assert info['reward_raw']['position_sq_xy'] == pytest.approx(sum(error[:2] ** 2))
        assert info['reward_raw']['position_sq_z'] == pytest.approx(error[2] ** 2)
        assert info['reward_costs'] == pytest.approx({'position_xy': expected_xy, 'position_z': expected_z})
        terms = info['reward_terms']
        assert terms['position'] == pytest.approx(-expected_xy - expected_z)
        legacy_position = -4 * (error @ error)
        legacy_total = sum(v for k, v in terms.items() if k not in ('position', 'total')) + legacy_position
        expected_delta = -(weights[0] - 4) * sum(error[:2] ** 2) - (weights[1] - 4) * error[2] ** 2
        assert total == pytest.approx(legacy_total + expected_delta, abs=1e-14)
        assert terms['crash'] == (-env.crash_penalty if terminated else 0)
    finally:
        env.close()


def known_trace():
    errors = np.array([[3., 4., 2.], [0., 0., -4.], [6., 8., 0.]])
    trace = RolloutTrace(
        policy='residual', label='PPO', time_sec=np.arange(3) * .01,
        position=errors, reference_position=np.zeros((3, 3)),
        position_error=np.linalg.norm(errors, axis=1), attitude_deg=np.zeros((3, 3)),
        linear_velocity=np.zeros((3, 3)), angular_velocity=np.zeros((3, 3)),
        phases=('TAKEOFF', 'HOVER', 'HOVER'), training_boundary_crossed_at=None,
        guard_boundary_crossed_at=None, terminated_at=.02, truncated_at=None, diverged_at=None,
    )
    return errors, trace


def test_rmse_whole_tail_phases_and_reward_balance():
    errors, trace = known_trace()
    metrics = trace_metrics(trace, tail_fraction=2/3)
    assert metrics['position_rmse'] == pytest.approx(np.sqrt(145/3))
    assert metrics['position_rmse_xy'] == pytest.approx(np.sqrt(125/3))
    assert metrics['position_rmse_z'] == pytest.approx(np.sqrt(20/3))
    assert metrics['phases']['TAKEOFF']['position_rmse_xy'] == 5
    assert metrics['phases']['HOVER']['position_rmse_z'] == pytest.approx(np.sqrt(8))
    assert metrics['tail_position_rmse_xy'] == pytest.approx(np.sqrt(50))
    assert metrics['trajectory_phase_rmse_z'] == pytest.approx(np.sqrt(8))
    assert metrics['terminated_at'] == .02
    config = load_config(ROOT / 'configs/e2e_train.yaml')
    config = replace(config, environment=replace(config.environment, reward=replace(
        config.environment.reward, position_xy_weight=2, position_z_weight=7)))
    balance = reward_balance_metrics(trace, config)['overall']
    assert balance['position_cost_xy_mean'] == pytest.approx(2 * 125/3)
    assert balance['position_cost_z_mean'] == pytest.approx(7 * 20/3)
    assert balance['mean_cost']['position'] == pytest.approx((2*125 + 7*20)/3)
    assert balance['mean_cost']['total'] == balance['mean_cost']['position']
    assert position_rmse_metrics([]) == {'position_rmse_xy': None, 'position_rmse_z': None}
    empty = replace(trace, time_sec=np.array([]), position=np.empty((0, 3)),
                    reference_position=np.empty((0, 3)), position_error=np.array([]),
                    phases=(), attitude_deg=np.empty((0, 3)),
                    linear_velocity=np.empty((0, 3)), angular_velocity=np.empty((0, 3)))
    empty_metrics = trace_metrics(empty, tail_fraction=.3)
    assert empty_metrics['position_rmse_xy'] is None
    assert empty_metrics['trajectory_phase_rmse_z'] is None


def test_policy_evaluator_tail_and_failure_contract():
    errors, _ = known_trace()
    class Env:
        def reset(self, seed):
            self.index = 0
            return np.zeros(15), {}
        def step(self, action):
            obs = np.zeros(15)
            obs[:3] = errors[self.index]
            obs[6:10] = [0, 1, 0, 0]  # Disqualified tail tilt stays visible.
            self.index += 1
            return obs, 0., self.index == 3, False, {}
        def close(self):
            pass
    config = load_config(ROOT / 'configs/e2e_train.yaml')
    config = replace(config, evaluation=replace(config.evaluation, episode_count=1, tail_fraction=2/3))
    result = PolicyEvaluator(config, SimpleNamespace(make=lambda seed: Env())).evaluate(None)
    assert result.position_rmse_xy == pytest.approx(np.sqrt(125/3))
    assert result.position_rmse_z == pytest.approx(np.sqrt(20/3))
    assert result.tail_position_rmse_xy == pytest.approx(np.sqrt(50))
    assert result.tail_position_rmse_z == pytest.approx(np.sqrt(8))
    assert result.score == 7  # Existing mean distance, not RMSE.
    assert result.disqualifications == 1
    assert result.mean_episode_length == 3
    from crazyflie_rl.training import _result_fields
    assert _result_fields('policy', result)['policy_position_rmse_xy'] == result.position_rmse_xy


def test_existing_checkpoint_equal_weight_trajectory():
    from stable_baselines3 import PPO
    from crazyflie_rl.environment import CrazyflieResidualEnv
    checkpoint = ROOT / 'model/ppo_best.zip'
    if not checkpoint.exists():
        pytest.skip('existing checkpoint unavailable')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    configs = [load_config(ROOT / f'configs/{name}.yaml') for name in
               ('e2e_train', 'e2e_train_position_split_equal')]
    envs = [CrazyflieResidualEnv(config=c) for c in configs]
    try:
        model = PPO.load(checkpoint, env=envs[0], device='cpu')
        assert model.observation_space.shape == (15,)
        assert model.action_space.shape == (4,)
        observations = [env.reset(seed=42)[0] for env in envs]
        for step in range(200):
            actions = [model.predict(obs, deterministic=True)[0] for obs in observations]
            np.testing.assert_array_equal(actions[0], actions[1])
            results = [env.step(action) for env, action in zip(envs, actions)]
            for i in range(4):
                np.testing.assert_array_equal(results[0][i], results[1][i])
            np.testing.assert_array_equal(envs[0].data.qpos, envs[1].data.qpos)
            np.testing.assert_array_equal(envs[0].data.qvel, envs[1].data.qvel)
            observations = [r[0] for r in results]
            if results[0][2] or results[0][3]:
                observations = [env.reset(seed=43 + step)[0] for env in envs]
    finally:
        for env in envs:
            env.close()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest

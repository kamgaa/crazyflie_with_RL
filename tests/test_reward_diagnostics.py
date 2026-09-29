"""Observer-only reward diagnostics, including the pre-instrumentation oracle."""
from dataclasses import replace
import random
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.environment import CrazyflieResidualEnv
from crazyflie_rl.training import _make_reward_diagnostics_callback

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '9a5419e5a3dc73921121bdc8f08ea8241354bebd'
COMPONENTS = ('position', 'velocity', 'tilt', 'angular_velocity', 'yaw',
              'action', 'action_rate', 'crash')
RAW = ('position_sq', 'position_sq_xy', 'position_sq_z', 'velocity_sq', 'tilt_error', 'angular_velocity_sq',
       'yaw_error_sq', 'action_sq', 'action_rate_sq')
COSTS = ('position_xy', 'position_z')
TAGS = ({f'reward_terms/{key}' for key in (*COMPONENTS, 'total')}
        | {f'reward_raw/{key}' for key in RAW}
        | {f'reward_costs/{key}' for key in COSTS}
        | {f'reward_fraction/{key}' for key in COMPONENTS})


@pytest.fixture(scope='module')
def reference_class():
    # Read an immutable pre-change source, never checkout/reset the working tree.
    source = subprocess.check_output(
        ['git', 'show', f'{BASELINE}:crazyflie_rl/environment.py'], cwd=ROOT, text=True
    )
    module = ModuleType('crazyflie_rl._reward_baseline')
    module.__package__ = 'crazyflie_rl'
    exec(compile(source, '<pre-diagnostics environment>', 'exec'), module.__dict__)
    return module.CrazyflieResidualEnv


def make_env(mode, cls=CrazyflieResidualEnv):
    pytest.importorskip('mujoco')
    pytest.importorskip('gymnasium')
    config = load_config(ROOT / 'configs' / f'{mode}_train.yaml')
    if not Path(config.paths.mujoco_xml).is_file():
        pytest.skip('MuJoCo XML unavailable')
    # Test-only nonzero weights also check that E2E overrides the config.
    config = replace(config, environment=replace(
        config.environment, reward=replace(config.environment.reward,
                                           action_weight=0.001, action_rate_weight=0.25)))
    return cls(config=config, seed=314159)


@pytest.mark.parametrize('mode', ['e2e', 'residual'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_fixed_trajectory_exact_equivalence(reference_class, mode, dtype):
    current, baseline = make_env(mode), make_env(mode, reference_class)
    try:
        a, _ = current.reset(seed=314159)
        b, _ = baseline.reset(seed=314159)
        assert np.array_equal(a, b)
        resets = 0
        for k in range(1024):
            action = (0.08 * np.sin(k * 0.17 + np.arange(4))).astype(dtype)
            if k % 127 == 126:
                action = np.array([1.2, -1.2, 0.5, -0.5], dtype=dtype)
            prev = current._prev_action.copy()
            actual = current.step(action)
            expected = baseline.step(action)
            for first, second in zip(actual[:4], expected[:4]):
                assert np.array_equal(first, second)
                assert np.asarray(first).tobytes() == np.asarray(second).tobytes()
            for field in ('qpos', 'qvel', 'ctrl'):
                assert np.array_equal(getattr(current.data, field), getattr(baseline.data, field))
            for field in ('_last_f_cmd', '_last_f', '_last_motor_cmd', '_prev_action'):
                assert np.array_equal(getattr(current, field), getattr(baseline, field))
            terms, raw = actual[4]['reward_terms'], actual[4]['reward_raw']
            assert set(terms) == set((*COMPONENTS, 'total'))
            assert set(raw) == set(RAW)
            assert terms['total'] == actual[1]
            assert sum(terms[n] for n in COMPONENTS) == pytest.approx(actual[1], rel=1e-14, abs=1e-14)
            assert terms['crash'] == (-current.crash_penalty if actual[2] else 0.0)
            assert all(type(v) is float and np.isfinite(v) for group in actual[4].values() for v in group.values())
            assert all(terms[n] <= 0 for n in COMPONENTS)
            clipped = np.clip(action, -1, 1)
            delta = clipped - prev
            assert raw['action_sq'] == float(clipped @ clipped)
            assert raw['action_rate_sq'] == float(delta @ delta)
            position, quaternion, velocity, omega = baseline._read_state()
            position_error = position - baseline.pos_des
            assert raw['position_sq'] == float(position_error @ position_error)
            assert raw['velocity_sq'] == float(velocity @ velocity)
            assert raw['tilt_error'] == float(2.0 * (quaternion[1]**2 + quaternion[2]**2))
            assert raw['angular_velocity_sq'] == float(omega @ omega)
            assert raw['yaw_error_sq'] == baseline._yaw_err(quaternion)**2
            for name, quantity, weight in (
                ('position', 'position_sq', baseline.position_weight),
                ('velocity', 'velocity_sq', baseline.velocity_weight),
                ('tilt', 'tilt_error', baseline.tilt_weight),
                ('angular_velocity', 'angular_velocity_sq', baseline.angular_velocity_weight),
                ('yaw', 'yaw_error_sq', baseline.yaw_weight),
            ):
                assert terms[name] == -(weight * raw[quantity])
            if mode == 'e2e':
                assert terms['action'] == terms['action_rate'] == 0.0
            else:
                assert terms['action'] == float(-(current.action_weight * (clipped @ clipped)))
                assert terms['action_rate'] == float(-(current.w_dact * (delta @ delta)))
                assert terms['action'] < 0
                assert terms['action_rate'] < 0
            if actual[2] or actual[3]:
                resets += 1
                a, _ = current.reset(seed=314159 + resets)
                b, _ = baseline.reset(seed=314159 + resets)
                assert np.array_equal(a, b)
        assert current._rng.bit_generator.state == baseline._rng.bit_generator.state
        assert current._actuator_rng.bit_generator.state == baseline._actuator_rng.bit_generator.state
    finally:
        current.close()
        baseline.close()


@pytest.mark.parametrize('mode', ['e2e', 'residual'])
def test_forced_crash_single_penalty(reference_class, mode):
    current, baseline, no_penalty = (make_env(mode), make_env(mode, reference_class), make_env(mode))
    try:
        for env in (current, baseline, no_penalty):
            env.reset(seed=19)
            # Force an upper-altitude crash; avoid floor-contact effects.
            env.data.qpos[2] = env.max_altitude + 1.0
            import mujoco
            mujoco.mj_forward(env.model, env.data)
        no_penalty.crash_penalty = 0.0  # Independent test-only oracle.
        action = np.zeros(4, dtype=np.float32)
        result, before, unpenalized = [env.step(action) for env in (current, baseline, no_penalty)]
        terms = result[4]['reward_terms']
        assert result[2] is True
        assert terms['crash'] == -current.crash_penalty
        assert terms['total'] == result[1] == before[1]
        assert result[1] == unpenalized[1] - current.crash_penalty
    finally:
        for env in (current, baseline, no_penalty):
            env.close()


def sample_info(scale):
    terms = {key: -scale * (i + 1) for i, key in enumerate(COMPONENTS)}
    terms['total'] = sum(terms.values())
    return {'reward_terms': terms, 'reward_raw': {key: scale * (i + 1) for i, key in enumerate(RAW)},
            'reward_costs': {key: scale * (i + 1) for i, key in enumerate(COSTS)}}


def test_callback_tensorboard_rollout_means_fractions_and_observer_only(tmp_path):
    pytest.importorskip('stable_baselines3')
    pytest.importorskip('tensorboard')
    import torch
    from stable_baselines3.common.logger import configure
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from copy import deepcopy

    logger = configure(str(tmp_path), ['tensorboard'])
    callback = _make_reward_diagnostics_callback()
    callback.model = SimpleNamespace(logger=logger)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    callback.on_rollout_start()
    infos = [sample_info(1.0), sample_info(3.0)]
    original = deepcopy(infos)
    callback.locals = {'infos': infos}
    assert callback._on_step() is True
    assert not logger.name_to_value  # No per-step logging.
    callback.locals = {'infos': [sample_info(2.0), {}]}
    assert callback._on_step() is True
    assert not logger.name_to_value
    callback.on_rollout_end()
    assert set(logger.name_to_value) == TAGS
    for key in (*COMPONENTS, 'total'):
        assert logger.name_to_value[f'reward_terms/{key}'] == sample_info(2.0)['reward_terms'][key]
    for key in RAW:
        assert logger.name_to_value[f'reward_raw/{key}'] == sample_info(2.0)['reward_raw'][key]
    for key in COSTS:
        assert logger.name_to_value[f'reward_costs/{key}'] == sample_info(2.0)['reward_costs'][key]
    fractions = [logger.name_to_value[f'reward_fraction/{key}'] for key in COMPONENTS]
    assert sum(fractions) == pytest.approx(1.0)
    assert fractions == pytest.approx([(i + 1) / 36 for i in range(8)])
    logger.dump(step=3)

    # A fresh all-zero rollout must not inherit previous sums or divide by zero.
    callback.on_rollout_start()
    callback.locals = {'infos': [sample_info(0.0)]}
    callback._on_step()
    callback.on_rollout_end()
    assert set(logger.name_to_value) == TAGS
    assert all(v == 0.0 and np.isfinite(v) for v in logger.name_to_value.values())
    logger.dump(step=4)
    callback.on_rollout_start()
    callback.locals = {}
    callback._on_step()
    callback.on_rollout_end()
    assert not logger.name_to_value
    logger.close()
    assert infos == original
    assert random.getstate() == python_rng
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), numpy_rng))
    assert torch.equal(torch.get_rng_state(), torch_rng)
    events = EventAccumulator(str(tmp_path)).Reload()
    assert set(events.Tags()['scalars']) == TAGS
    for tag in TAGS:
        assert [event.step for event in events.Scalars(tag)] == [3, 4]
    (tmp_path / 'verified_tags.txt').write_text('\n'.join(sorted(TAGS)))

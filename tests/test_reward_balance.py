"""Reward balance uses recorded trajectories without changing their producers."""
from dataclasses import fields, replace
import ast
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.eval_cli import RolloutTrace
from crazyflie_rl.reward_balance import (
    build_reward_balance_report, format_reward_balance, reward_balance_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / 'configs/view_live_circle_eval.yaml')


def trace(errors=(3.0, 4.0), phases=('GOTO', 'GOTO'), **overrides):
    n = len(errors)
    reference = np.column_stack((np.arange(n), np.zeros((n, 2))))
    values = dict(
        policy='floor', label='floor (PID)', time_sec=np.arange(n) * 0.1,
        position=reference + np.column_stack((errors, np.zeros((n, 2)))),
        reference_position=reference, position_error=np.asarray(errors),
        attitude_deg=np.tile([30.0, 20.0, 170.0], (n, 1)),
        linear_velocity=np.tile([3.0, 4.0, 0.0], (n, 1)),
        angular_velocity=np.tile([1.0, 2.0, 2.0], (n, 1)),
        phases=phases, training_boundary_crossed_at=None,
        guard_boundary_crossed_at=None, terminated_at=None,
        truncated_at=None, diverged_at=None, control_mode='residual',
    )
    values.update(overrides)
    return RolloutTrace(**values)


def test_state_metrics_costs_fractions_and_config_weights():
    c = config()
    weights = replace(c.environment.reward, position_weight=2, velocity_weight=0.3,
                      tilt_weight=4, angular_velocity_weight=0.2, yaw_weight=0.7)
    c = replace(c, environment=replace(c.environment, reward=weights, yaw_target=np.deg2rad(-170)))
    m = reward_balance_metrics(trace(), c)['phases']['GOTO']
    tilt_raw = 1 - np.cos(np.deg2rad(30)) * np.cos(np.deg2rad(20))
    tilt_deg = np.rad2deg(np.arccos(1 - tilt_raw))
    assert m['position_error_rms_m'] == pytest.approx(np.sqrt(12.5))
    assert m['position_error_mean_m'] == 3.5
    assert m['position_error_peak_m'] == 4
    assert m['position_sq_mean'] == 12.5
    assert m['velocity_rms_mps'] == m['velocity_peak_mps'] == 5
    assert m['velocity_sq_mean'] == 25
    assert m['angular_velocity_rms_radps'] == m['angular_velocity_peak_radps'] == 3
    assert m['angular_velocity_sq_mean'] == 9
    for key in ('tilt_rms_deg', 'tilt_mean_deg', 'tilt_peak_deg'):
        assert m[key] == pytest.approx(tilt_deg)
    assert m['tilt_error_mean'] == pytest.approx(tilt_raw)
    expected = dict(position=25, velocity=7.5, tilt=4*tilt_raw,
                    angular_velocity=1.8, yaw=0.7*np.deg2rad(20)**2)
    for name, value in expected.items():
        assert m['mean_cost'][name] == pytest.approx(value)
    assert m['mean_cost']['total'] == pytest.approx(sum(expected.values()))
    assert sum(m['cost_fraction'].values()) == pytest.approx(1)


def test_phase_segmentation_and_moving_reference_progress():
    # Stationary vehicle; reference alone changes error. e^T v would be zero.
    errors = np.array([3., 2., 2., 4., 1., 0.5, 0.25])
    phases = ('GOTO', 'GOTO', 'GOTO', 'CIRCLE', 'CIRCLE', 'GOTO', 'HOLD')
    t = trace(errors, phases, position=np.zeros((7, 3)),
              reference_position=np.column_stack((errors, np.zeros((7, 2)))),
              linear_velocity=np.zeros((7, 3)))
    metrics = reward_balance_metrics(t, config())
    assert set(metrics['phases']) == {'GOTO', 'CIRCLE', 'HOLD'}
    goto = metrics['phases']['GOTO']
    assert goto['sample_count'] == 4
    assert goto['progress_pair_count'] == 2  # No gap bridging to later GOTO.
    assert goto['progress_fraction'] == 0.5
    assert goto['regress_fraction'] == 0
    assert goto['mean_error_delta_m_per_step'] == -0.5
    assert goto['mean_error_reduction_rate_mps'] == pytest.approx(5)
    assert metrics['phases']['HOLD']['progress_fraction'] is None
    assert metrics['overall']['regress_fraction'] == pytest.approx(1/6)


def test_all_present_phases_and_irregular_time_rate():
    phases = ('TAKEOFF', 'SETTLE1', 'GOTO', 'SETTLE2', 'CIRCLE', 'LISSAJOUS', 'HOLD')
    assert set(reward_balance_metrics(trace(np.ones(7), phases), config())['phases']) == set(phases)
    t = trace([3., 2., 0.], ('GOTO',)*3, time_sec=np.array([0., .1, .3]))
    assert reward_balance_metrics(t, config())['overall']['mean_error_reduction_rate_mps'] == pytest.approx(10)
    t = replace(t, time_sec=np.zeros(3))
    assert reward_balance_metrics(t, config())['overall']['mean_error_reduction_rate_mps'] is None


@pytest.mark.parametrize('angles', [(0, 0, 0), (30, 20, 70), (-40, 55, -170), (180, 0, 10)])
def test_tilt_quaternion_reconstruction(angles):
    roll, pitch, yaw = np.deg2rad(angles) / 2
    qx = np.sin(roll)*np.cos(pitch)*np.cos(yaw) - np.cos(roll)*np.sin(pitch)*np.sin(yaw)
    qy = np.cos(roll)*np.sin(pitch)*np.cos(yaw) + np.sin(roll)*np.cos(pitch)*np.sin(yaw)
    t = trace(attitude_deg=np.tile(angles, (2, 1)))
    result = reward_balance_metrics(t, config())['overall']
    assert result['tilt_error_mean'] == pytest.approx(2*(qx*qx+qy*qy), abs=1e-14)


def test_zero_missing_nonfinite_and_empty_are_json_safe():
    t = trace([0., 0.], attitude_deg=np.zeros((2, 3)),
              linear_velocity=np.zeros((2, 3)), angular_velocity=np.zeros((2, 3)))
    m = reward_balance_metrics(t, config())['overall']
    assert m['mean_cost']['total'] == 0
    assert all(v == 0 for v in m['cost_fraction'].values())
    report = build_reward_balance_report([t, replace(t, policy='residual')], config())
    assert all(v is None for v in report['comparisons']['overall']['ppo_over_pid'].values())
    json.dumps(report, allow_nan=False)
    for missing in (None, np.full((2, 3), np.nan)):
        m = reward_balance_metrics(replace(t, angular_velocity=missing), config())
        assert m['overall']['angular_velocity_rms_radps'] is None
        assert m['overall']['mean_cost']['total'] is None
        json.dumps(m, allow_nan=False)
    empty = trace([], (), attitude_deg=np.empty((0, 3)),
                  linear_velocity=np.empty((0, 3)), angular_velocity=np.empty((0, 3)))
    m = reward_balance_metrics(empty, config())
    assert m['phases'] == {}
    assert m['overall']['position_error_rms_m'] is None
    json.dumps(m, allow_nan=False)


def test_same_weights_ratios_and_observer_only():
    t = trace()
    ppo = replace(t, policy='residual', label='E2E PPO', control_mode='e2e',
                  position_error=t.position_error * 2)
    snapshots = {f.name: getattr(t, f.name).copy() for f in fields(t) if isinstance(getattr(t, f.name), np.ndarray)}
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    report = build_reward_balance_report([t, ppo], config(), model='/test/model.zip')
    a, b = report['floor']['overall'], report['ppo']['overall']
    assert b['mean_cost']['position'] == 4*a['mean_cost']['position']
    assert b['mean_cost']['tilt'] == a['mean_cost']['tilt']
    ratios = report['comparisons']['phases']['GOTO']['ppo_over_pid']
    assert ratios['position_error_rms'] == 2
    assert ratios['position_cost'] == 4
    assert ratios['tilt_rms'] == ratios['angular_velocity_cost'] == 1
    assert 'GOTO' in format_reward_balance(report)
    assert report['model'] == '/test/model.zip'
    json.dumps(report, allow_nan=False)
    for name, value in snapshots.items():
        assert np.array_equal(getattr(t, name), value)
    assert random.getstate() == python_rng
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), numpy_rng))


def test_environment_raw_consistency():
    pytest.importorskip('mujoco')
    from crazyflie_rl.environment import CrazyflieResidualEnv
    from crazyflie_rl.plotting import quaternion_to_euler_deg
    c = config()
    if not Path(c.paths.mujoco_xml).is_file():
        pytest.skip('MuJoCo XML unavailable')
    env = CrazyflieResidualEnv(config=c, seed=10)
    try:
        env.reset(seed=10)
        obs, _, _, _, info = env.step(np.array([.01, -.01, .02, .03], dtype=np.float32))
        t = trace([float(np.linalg.norm(obs[:3].astype(float)))], ('GOTO',),
                  attitude_deg=quaternion_to_euler_deg(obs[6:10])[None, :],
                  linear_velocity=obs[3:6][None, :], angular_velocity=obs[10:13][None, :])
        m = reward_balance_metrics(t, c)['overall']
        for metric, raw in [('position_sq_mean', 'position_sq'), ('velocity_sq_mean', 'velocity_sq'),
                            ('tilt_error_mean', 'tilt_error'), ('angular_velocity_sq_mean', 'angular_velocity_sq'),
                            ('yaw_error_sq_mean', 'yaw_error_sq')]:
            assert m[metric] == pytest.approx(info['reward_raw'][raw], rel=2e-6, abs=2e-7)
    finally:
        env.close()


@pytest.mark.parametrize('mode', ['circle', 'lissajous'])
@pytest.mark.parametrize('policy_key', ['floor', 'residual'])
def test_runner_exact_invariance_against_pre_change_source(mode, policy_key):
    pytest.importorskip('mujoco')
    from crazyflie_rl import eval_cli
    from crazyflie_rl.missions import mission_from_experiment
    eval_cli._ensure_runtime_imports()
    source = subprocess.check_output(
        ['git', 'show', '9a5419e5a3dc73921121bdc8f08ea8241354bebd:crazyflie_rl/eval_cli.py'],
        cwd=ROOT, text=True,
    )
    old = ast.parse(source)
    runner_node = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == 'EvaluationRunner')
    current_node = next(n for n in ast.parse(Path(eval_cli.__file__).read_text()).body
                        if isinstance(n, ast.ClassDef) and n.name == 'EvaluationRunner')
    # The only permitted runner change is cached episode-mass metadata.
    for call in ast.walk(current_node):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'RolloutTrace':
            call.keywords = [kw for kw in call.keywords if kw.arg != 'episode_mass_kg']
    assert ast.dump(runner_node) == ast.dump(current_node)
    namespace = dict(vars(eval_cli))
    future = ast.parse('from __future__ import annotations').body
    exec(compile(ast.Module(body=[*future, runner_node], type_ignores=[]), '<baseline runner>', 'exec'), namespace)
    c = config()
    if not Path(c.paths.mujoco_xml).is_file():
        pytest.skip('MuJoCo XML unavailable')
    mission_config = replace(c.mission, type=mode, takeoff_sec=.1, settle_sec=.1,
                             goto_sec=.1, post_hold_sec=.1,
                             circle=replace(c.mission.circle, period=.2, laps=1, ramp_sec=0),
                             lissajous=replace(c.mission.lissajous, base_period=.2, cycles=1, ramp_sec=0))
    c = replace(c, mission=mission_config)
    mission = mission_from_experiment(c, legacy_circle_preset=False)

    class FixedPolicy:
        def predict(self, observation, deterministic):
            return np.array([.01, -.02, .03, .04], dtype=np.float32), None

    traces = []
    for cls in (namespace['EvaluationRunner'], eval_cli.EvaluationRunner):
        runner = cls(c, None, headless=True, realtime=False, camera_tracking=False, mission=mission)
        t = runner.run(None if policy_key == 'floor' else FixedPolicy(), policy_key, policy_key,
                       control_mode='residual' if policy_key == 'floor' else 'e2e')
        traces.append(t)
        reward_balance_metrics(t, c)
    for field in fields(traces[0]):
        if field.name == 'episode_mass_kg':
            assert traces[0].episode_mass_kg is None
            assert traces[1].episode_mass_kg > 0
            continue
        a, b = (getattr(t, field.name) for t in traces)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
            assert a.tobytes() == b.tobytes()
        else:
            assert a == b

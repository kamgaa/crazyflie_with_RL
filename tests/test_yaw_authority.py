"""Yaw diagnostics must distinguish PID effort from E2E normalized action."""
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
from crazyflie_rl.yaw_authority import (
    analyze_yaw_authority, build_yaw_authority_report,
    format_yaw_authority, save_yaw_authority_plot,
)

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / 'configs/view_live_circle_eval.yaml')


def trace(policy='residual', mode='e2e', phases=('GOTO', 'CIRCLE', 'CIRCLE', 'HOLD')):
    n = len(phases)
    command = np.zeros((n, 4))
    command[:, 2] = np.arange(1, n+1) * 1e-4
    actual = command * 0.5
    actions = np.zeros((n, 4))
    actions[:, 2] = np.linspace(-1, 1, n)
    return RolloutTrace(
        policy=policy, label=policy, control_mode=mode, time_sec=np.arange(n)*.01,
        position=np.zeros((n, 3)), reference_position=np.zeros((n, 3)),
        position_error=np.zeros(n), attitude_deg=np.tile([0., 0., 170.], (n, 1)),
        phases=phases, training_boundary_crossed_at=None, guard_boundary_crossed_at=None,
        terminated_at=None, truncated_at=None, diverged_at=None,
        control_input=actions, wrench_command=command, wrench_actual=actual,
        allocation_error=command-actual,
    )


def test_yaw_wrap_against_nonzero_reference():
    c = config()
    c = replace(c, environment=replace(c.environment, yaw_target=np.deg2rad(-170)))
    t = trace()
    t.attitude_deg[:, 2] = [170, -170, 190, -190]
    m = analyze_yaw_authority(t, c)['overall']
    assert m['yaw_rms_deg'] == pytest.approx(np.sqrt(200))
    assert m['yaw_mean_abs_deg'] == pytest.approx(10)
    assert m['yaw_peak_abs_deg'] == pytest.approx(20)


def test_torque_extraction_percentiles_tracking_and_phases():
    t = trace()
    result = analyze_yaw_authority(t, config())
    assert set(result['phases']) == {'GOTO', 'CIRCLE', 'HOLD'}
    assert result['phases']['CIRCLE']['sample_count'] == 2
    for name, field in [('cmd', t.wrench_command), ('actual', t.wrench_actual)]:
        m = result['overall']
        expected = np.abs(field[:, 2])
        assert m[f'tau_z_{name}_rms_nm'] == pytest.approx(np.sqrt(np.mean(expected**2)))
        assert m[f'tau_z_{name}_abs_mean_nm'] == pytest.approx(np.mean(expected))
        assert m[f'tau_z_{name}_abs_p95_nm'] == pytest.approx(np.percentile(expected, 95))
        assert m[f'tau_z_{name}_abs_p99_nm'] == pytest.approx(np.percentile(expected, 99))
        assert m[f'tau_z_{name}_abs_max_nm'] == max(expected)
    # Prove stored allocation_error is reused, rather than recalculated.
    stored = np.full((4, 4), 0.123)
    m = analyze_yaw_authority(replace(t, allocation_error=stored), config())['overall']
    assert m['tau_z_tracking_error_rms_nm'] == pytest.approx(.123)
    assert m['tau_z_tracking_error_abs_p95_nm'] == pytest.approx(.123)
    assert m['tau_z_tracking_error_abs_max_nm'] == pytest.approx(.123)
    fallback = analyze_yaw_authority(replace(t, allocation_error=None), config())
    assert fallback['overall']['tau_z_tracking_error_rms_nm'] == result['overall']['tau_z_tracking_error_rms_nm']


def test_action_statistics_and_strict_saturation_thresholds():
    action = np.array([0, .8, -.9, .95, -.99, 1.])
    t = trace(phases=('CIRCLE',)*len(action))
    t.control_input[:, 2] = action
    m = analyze_yaw_authority(t, config())['overall']
    assert m['u_tau_z_rms'] == pytest.approx(np.sqrt(np.mean(action**2)))
    assert m['u_tau_z_abs_mean'] == np.mean(np.abs(action))
    assert m['u_tau_z_abs_p95'] == np.percentile(np.abs(action), 95)
    assert m['u_tau_z_abs_p99'] == np.percentile(np.abs(action), 99)
    assert m['u_tau_z_abs_max'] == 1
    for key, count in [('0p8', 4), ('0p9', 3), ('0p95', 2), ('0p99', 1)]:
        assert m[f'u_tau_z_fraction_gt_{key}'] == count / 6


@pytest.mark.parametrize('policy,mode', [('floor', 'residual'), ('floor', 'e2e'), ('residual', 'residual'), ('residual', None)])
def test_pid_and_non_e2e_actions_are_not_yaw_effort(policy, mode):
    t = trace(policy, mode)
    t.control_input[:, 2] = 1  # Even a misleading nonzero residual is ineligible.
    result = analyze_yaw_authority(t, config())
    assert not result['normalized_yaw_action_applicable']
    m = result['overall']
    assert all(value is None for key, value in m.items() if key.startswith('u_tau_z'))
    assert m['tau_z_cmd_abs_max_nm'] == .0004


def test_correlations_and_zero_variance_guard():
    t = trace()
    m = analyze_yaw_authority(t, config())['overall']
    assert m['tau_z_actual_vs_cmd_correlation'] == pytest.approx(1)
    assert m['u_tau_z_vs_tau_z_actual_correlation'] == pytest.approx(1)
    t.wrench_actual[:, 2] = 1e-5 + np.arange(4)*1e-14
    m = analyze_yaw_authority(t, config())['overall']
    assert m['tau_z_actual_vs_cmd_correlation'] is None
    assert m['u_tau_z_vs_tau_z_actual_correlation'] is None
    assert analyze_yaw_authority(trace(phases=('HOLD',)), config())['overall']['tau_z_actual_vs_cmd_correlation'] is None


def test_configured_scale_ratios_and_zero_scale_guard():
    c = config()
    c = replace(c, environment=replace(c.environment, residual_scale=(.02, .02, .002, .3)))
    pid = trace('floor', 'residual')
    report = build_yaw_authority_report([pid, trace()], c, model='policy.zip')
    assert report['configured_tau_z_action_scale_nm'] == .002
    assert report['policy_tau_z_max_abs_nm'] == .002
    ratio = report['comparisons']['phases']['CIRCLE']
    assert ratio['pid_tau_z_p95_over_e2e_action_scale'] == pytest.approx(np.percentile([.0002, .0003],95)/.002)
    assert ratio['pid_tau_z_p99_over_e2e_action_scale'] == pytest.approx(np.percentile([.0002, .0003],99)/.002)
    c = replace(c, environment=replace(c.environment, residual_scale=(.02,.02,0.,.3)))
    report = build_yaw_authority_report([pid], c)
    assert all(v is None for v in report['comparisons']['overall'].values())
    assert 'N/A' in format_yaw_authority(report)


def test_json_missing_nonfinite_empty_and_observer_only():
    t = trace()
    saved = {f.name: getattr(t,f.name).copy() for f in fields(t) if isinstance(getattr(t,f.name),np.ndarray)}
    for name in saved:
        getattr(t, name).setflags(write=False)
    py_rng, np_rng = random.getstate(), np.random.get_state()
    report = build_yaw_authority_report([t], config())
    json.dumps(report, allow_nan=False)
    for name, expected in saved.items():
        assert np.array_equal(getattr(t,name), expected)
    assert random.getstate() == py_rng
    assert all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),np_rng))
    for value in (None, np.full((4,4),np.nan)):
        m = analyze_yaw_authority(replace(t,wrench_actual=value),config())
        assert m['overall']['tau_z_actual_rms_nm'] is None
        json.dumps(m, allow_nan=False)
    empty = analyze_yaw_authority(trace(phases=()),config())
    assert empty['phases'] == {}
    assert empty['overall']['yaw_rms_deg'] is None
    json.dumps(empty, allow_nan=False)


def test_three_panel_plot_has_only_e2e_action_and_refuses_overwrite(tmp_path, monkeypatch):
    from crazyflie_rl import plotting
    plt = plotting._pyplot()
    original = plt.subplots
    captured = []
    def subplots(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(plt, 'subplots', subplots)
    pid, ppo = trace('floor','residual'), trace()
    path = save_yaw_authority_plot(tmp_path/'yaw.png', [pid,ppo], config())
    assert path.read_bytes().startswith(b'\x89PNG')
    fig, axes = captured[0]
    assert len(axes) == 3
    assert axes[2].get_ylim() == (-1,1)
    lines = {line.get_label():line for line in axes[2].lines}
    assert 'E2E PPO yaw action' in lines
    assert '+0.9' in lines and '-0.9' in lines
    assert not any('PID' in label for label in lines)
    np.testing.assert_array_equal(lines['E2E PPO yaw action'].get_ydata(), ppo.control_input[:,2])
    with pytest.raises(FileExistsError):
        save_yaw_authority_plot(path, [pid,ppo], config())
    save_yaw_authority_plot(tmp_path/'pid_only.png', [pid], config())
    assert not any(line.get_label() == 'E2E PPO yaw action' for line in captured[-1][1][2].lines)


@pytest.mark.parametrize('policy_key', ['floor', 'residual'])
def test_actual_runner_and_motor_signals_exact_invariance(policy_key):
    pytest.importorskip('mujoco')
    from crazyflie_rl import eval_cli
    from crazyflie_rl.missions import mission_from_experiment
    eval_cli._ensure_runtime_imports()
    source = subprocess.check_output(
        ['git','show','9a5419e5a3dc73921121bdc8f08ea8241354bebd:crazyflie_rl/eval_cli.py'],
        cwd=ROOT,text=True,
    )
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='EvaluationRunner')
    current = next(n for n in ast.parse(Path(eval_cli.__file__).read_text()).body if isinstance(n,ast.ClassDef) and n.name=='EvaluationRunner')
    # The only permitted runner change is cached episode-mass metadata.
    for call in ast.walk(current):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'RolloutTrace':
            call.keywords = [kw for kw in call.keywords if kw.arg != 'episode_mass_kg']
    assert ast.dump(node)==ast.dump(current)
    namespace = dict(vars(eval_cli))
    future = ast.parse('from __future__ import annotations').body
    exec(compile(ast.Module(body=[*future,node],type_ignores=[]),'<before yaw diagnostics>','exec'),namespace)
    c=config()
    if not Path(c.paths.mujoco_xml).is_file():
        pytest.skip('MuJoCo XML unavailable')
    c=replace(c,mission=replace(c.mission,takeoff_sec=.1,settle_sec=.1,goto_sec=.1,post_hold_sec=.1,
                               circle=replace(c.mission.circle,period=.2,laps=1,ramp_sec=0)))
    mission=mission_from_experiment(c,legacy_circle_preset=False)
    class Policy:
        def predict(self, observation, deterministic):
            return np.array([.01,-.02,.6,.03],dtype=np.float32),None
    traces=[]
    for cls in (namespace['EvaluationRunner'],eval_cli.EvaluationRunner):
        runner=cls(c,None,headless=True,realtime=False,camera_tracking=False,mission=mission)
        t=runner.run(None if policy_key=='floor' else Policy(),policy_key,policy_key,
                     control_mode='residual' if policy_key=='floor' else 'e2e')
        traces.append(t)
        analyze_yaw_authority(t,c)
        np.testing.assert_array_equal(t.wrench_actual[:,2],np.sum(t.reaction_torque_nm,axis=1))
        np.testing.assert_array_equal(t.allocation_error,t.wrench_command-t.wrench_actual)
    for field in fields(traces[0]):
        if field.name == 'episode_mass_kg':
            assert traces[0].episode_mass_kg is None
            assert traces[1].episode_mass_kg > 0
            continue
        a,b=[getattr(t,field.name) for t in traces]
        if isinstance(a,np.ndarray):
            assert np.array_equal(a,b)
            assert a.tobytes()==b.tobytes()
        else:
            assert a==b

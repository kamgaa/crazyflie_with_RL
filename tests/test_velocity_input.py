"""Inference-only velocity intervention: analytic reference and frozen preprocessing."""
from dataclasses import replace
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.dr_policy import FrozenPolicy, VELOCITY_SLICE, policy_raw_observation, sha256
from crazyflie_rl.dr_transfer import ROOT, EvaluationAdapter, make_cases, motion_metrics, build_parser
from crazyflie_rl.environment import CrazyflieResidualEnv
from crazyflie_rl.missions import CircleMission, ramped_phase_velocity


@pytest.fixture
def config():
    return load_config(ROOT/'configs/eval_dr_transfer.yaml')


@pytest.mark.parametrize('direction', ['ccw', 'cw'])
def test_analytic_velocity_matches_position_derivative(config, direction):
    mission = CircleMission.from_parameters(config.mission,
        replace(config.mission.circle, direction=direction, start_angle_deg=27))
    # Includes cosine TAKEOFF/GOTO, ramp, cruise and HOLD away from boundaries.
    for t in (1., 5., 7., 11., 12.1, 12.7, 13.9, 14.1, 17.3, 23.8, 24.2, 25.5):
        h = 1e-5
        derivative = (mission.reference(t+h)[0]-mission.reference(t-h)[0])/(2*h)
        np.testing.assert_allclose(mission.reference_velocity(t), derivative, atol=2e-9, rtol=2e-8)


def test_air_offset_ramp_hold_and_zero_step_velocity(config):
    original, air, step = make_cases(config, ['circle','circle-air','step-005'])
    for t in np.arange(1401)*.01:
        np.testing.assert_array_equal(air.reference_velocity(t), original.reference_velocity(t+12))
    np.testing.assert_array_equal(air.reference_velocity(0), 0)
    assert np.linalg.norm(air.reference_velocity(1)) == pytest.approx(np.pi/10)
    assert np.linalg.norm(air.reference_velocity(2)) == pytest.approx(np.pi/5)
    assert np.linalg.norm(air.reference_velocity(12-1e-8)) == pytest.approx(np.pi/5)
    for t in (12, 12.01, 14):
        np.testing.assert_array_equal(air.reference_velocity(t), 0)
    for t in (0, .01, 1, 8):
        np.testing.assert_array_equal(step.reference_velocity(t), 0)
    assert ramped_phase_velocity(0, 2, 0) == 2
    with pytest.raises(ValueError):
        ramped_phase_velocity(0, 2, -1)


def test_velocity_transform_is_raw_only_and_nonmutating(config):
    env = CrazyflieResidualEnv(config=config)
    try:
        adapter = EvaluationAdapter(env)
        adapter.reset_to_case_initial_state(make_cases(config,['circle-air'])[0],42)
        env.data.qvel[:3] = [1,2,3]
        raw = adapter.current_observation()
        before = adapter.snapshot()
        original = raw.copy()
        absolute = policy_raw_observation(raw, [.1,.2,.3])
        assert absolute is raw
        error = policy_raw_observation(raw, [.1,.2,.3], 'error')
        np.testing.assert_allclose(error[VELOCITY_SLICE], [.9,1.8,2.7])
        np.testing.assert_array_equal(error[:3], original[:3])
        np.testing.assert_array_equal(error[6:], original[6:])
        np.testing.assert_array_equal(raw, original)
        assert adapter.snapshot() == before
        assert not np.shares_memory(error, raw)
        np.testing.assert_array_equal(policy_raw_observation(raw, np.zeros(3), 'error'), absolute)
    finally:
        env.close()


def test_transform_precedes_frozen_normalization_and_clipping(config, tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    class Recorder:
        def predict(self, obs, deterministic):
            assert deterministic
            self.seen = obs.copy()
            return obs[:4].copy(), None
    env = CrazyflieResidualEnv(config=config)
    norm = VecNormalize(DummyVecEnv([lambda: env]), clip_obs=2.)
    norm.obs_rms.mean[:] = np.arange(15)/2
    norm.obs_rms.var[:] = np.arange(15)+2
    file = tmp_path/'stats.pkl'
    norm.save(file)
    digest = sha256(file)
    model = Recorder()
    policy = FrozenPolicy(model, {'normalization': {'sha256':digest}}, file)
    try:
        policy.bind(env)
        raw = np.linspace(-2,8,15).astype(np.float32)
        reference = np.array([1., -3., 10.])
        before = (policy.normalizer.obs_rms.mean.copy(), policy.normalizer.obs_rms.var.copy(), policy.normalizer.obs_rms.count)
        changed = policy_raw_observation(raw,reference,'error')
        policy.predict(changed)
        expected = np.clip((changed-before[0])/np.sqrt(before[1]+policy.normalizer.epsilon),-2,2).astype(np.float32)
        np.testing.assert_array_equal(model.seen,expected)
        wrong = policy.normalizer.normalize_obs(raw).copy()
        wrong[VELOCITY_SLICE] -= reference
        assert not np.allclose(wrong,model.seen)
        action_abs = policy.predict(policy_raw_observation(raw,np.zeros(3),'absolute'))
        normalized_abs = model.seen.copy()
        action_err = policy.predict(policy_raw_observation(raw,np.zeros(3),'error'))
        np.testing.assert_array_equal(normalized_abs, model.seen)
        np.testing.assert_array_equal(action_abs, action_err)
        np.testing.assert_array_equal(before[0],policy.normalizer.obs_rms.mean)
        np.testing.assert_array_equal(before[1],policy.normalizer.obs_rms.var)
        assert before[2] == policy.normalizer.obs_rms.count
        assert not policy.normalizer.training and not policy.normalizer.norm_reward
    finally:
        env.close()
    assert sha256(file) == digest


def test_motion_metrics_use_post_time_reference(config):
    case = make_cases(config,['circle-air'])[0]
    rows = [dict(time_post=t, velocity=case.reference_velocity(t)+[3,4,2], quaternion=np.array([1,0,0,0]))
            for t in (.5,1.,3.)]
    metrics = motion_metrics(rows,case)
    assert metrics['velocity_rmse'] == pytest.approx(np.sqrt(29))
    assert metrics['roll_rms_deg'] == metrics['pitch_rms_deg'] == 0
    assert all(value is None for value in motion_metrics([],case).values())


def test_cli_absolute_default_and_multiple_modes():
    args = ['--model','a=a.zip','--model','b=b.zip']
    assert build_parser().parse_args(args).velocity_inputs == ['absolute']
    assert build_parser().parse_args(args+['--velocity-inputs','absolute','error']).velocity_inputs == ['absolute','error']


def test_saved_actual_comparison():
    """Read-only validation of the explicitly supplied real evaluation output."""
    target = os.environ.get('DR_VELOCITY_RESULTS')
    if not target:
        pytest.skip('set DR_VELOCITY_RESULTS to the real two-model/two-mode run')
    directory = Path(target)
    manifest = json.loads((directory/'manifest.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    assert manifest['status'] == 'completed' and len(summary) == 8
    assert manifest['velocity_inputs'] == ['absolute', 'error']
    assert manifest['initial_snapshots_equal'] and manifest['reference_sequences_equal']
    assert all(sha256(m['path']) == m['sha256'] for m in manifest['models'])
    def rows(path):
        with path.open(newline='') as f:
            return list(csv.DictReader(f))
    def equal_columns(a, b, excluded=()):
        assert len(a) == len(b)
        keys = set(a[0]) & set(b[0]) - set(excluded)
        assert keys
        for x, y in zip(a,b):
            assert {k:x[k] for k in keys} == {k:y[k] for k in keys}
    for case in ('circle-air', 'step-005'):
        snapshots = list(manifest['initial_snapshots'][case].values())
        assert len(snapshots) == 4 and all(s == snapshots[0] for s in snapshots)
        reference = np.load(directory/f'{case}-reference.npz')
        old_directory = ROOT/'artifacts/runs'/('dr-transfer-x4ib_6hk' if case == 'circle-air' else 'dr-transfer-8rfoihws')
        old_summary = json.loads((old_directory/'summary.json').read_text())
        for label in ('baseline','posdr'):
            absolute = rows(directory/f'{case}-{label}--velocity-absolute.csv')
            error = rows(directory/f'{case}-{label}--velocity-error.csv')
            equal_columns(absolute, rows(old_directory/f'{case}-{label}.csv'))
            old = next(r for r in old_summary if r['label']==label and r['case']==case)
            new = next(r for r in summary if r['label']==label and r['case']==case and r['observation_velocity_mode']=='absolute')
            for key in old:
                if key == 'circle_phase':
                    for metric, value in old[key].items():
                        assert new[key][metric] == value
                else:
                    assert new[key] == old[key]
            if case == 'step-005':
                equal_columns(absolute, error, {'observation_velocity_mode'})
            for mode, data in (('absolute',absolute), ('error',error)):
                for k, row in enumerate(data):
                    vref = np.array([float(row[f'reference_velocity_{i}']) for i in range(3)])
                    np.testing.assert_array_equal(vref,reference['reference_velocity'][k])
                    np.testing.assert_array_equal([float(row[f'reference_velocity_post_{i}']) for i in range(3)],
                                                  reference['reference_velocity'][k+1])
                    raw = np.array([float(row[f'observation_{i}']) for i in range(15)],np.float32)
                    expected = policy_raw_observation(raw,vref,mode)
                    np.testing.assert_array_equal([float(row[f'policy_raw_observation_{i}']) for i in range(15)],expected)
                    np.testing.assert_array_equal([float(row[f'policy_raw_velocity_{i}']) for i in range(3)],expected[VELOCITY_SLICE])
                    assert row['policy_input_time'] == row['time']

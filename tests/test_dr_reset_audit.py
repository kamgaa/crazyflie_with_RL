"""Archived-reset replay checks, no optimization or policy rollout."""
from dataclasses import replace
import json

import numpy as np
import pytest

from crazyflie_rl.dr_reset_audit import archived_config, distribution_contract, sample_resets, inspect_rollouts
from crazyflie_rl.dr_transfer import ROOT

SOURCE = ROOT/'artifacts/runs/dr-transfer-8rfoihws'


@pytest.fixture
def models():
    if not (SOURCE/'manifest.json').is_file():
        pytest.skip('archived comparison unavailable')
    from pathlib import Path
    return {m['label']: Path(m['path']) for m in json.loads((SOURCE/'manifest.json').read_text())['models']}


def test_archived_settings_and_theoretical_bounds(models):
    configs = {label: archived_config(path)[0] for label, path in models.items()}
    baseline = distribution_contract(configs['baseline'])
    posdr = distribution_contract(configs['posdr'])
    assert baseline['sampler'] == 'legacy' and baseline['bound_parameter_m'] == .05
    assert baseline['theoretical_norm_max_m'] == pytest.approx(np.sqrt(3)*.05)
    assert baseline['theoretical_norm_rms_m'] == .05
    assert posdr['sampler'] == 'new' and posdr['bound_parameter_m'] == .15
    assert posdr['theoretical_norm_rms_m'] == pytest.approx(.15/np.sqrt(3))
    for config in configs.values():
        assert config.training.seed is None
        assert not distribution_contract(config)['attitude_randomized']
        assert not config.actuator.randomization.enabled
        assert config.environment.reward.effective_position_xy_weight == 10
        assert config.environment.reward.effective_position_z_weight == 6


@pytest.mark.parametrize('label', ['baseline', 'posdr'])
def test_actual_resets_advance_rng_and_are_reproducible(models, label):
    config, _, _ = archived_config(models[label])
    report, arrays = sample_resets(config, 128, 42)
    _, repeated = sample_resets(config, 128, 42)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], repeated[key])
    assert report['unique_position_count'] == 128
    assert report['all_level'] and report['all_zero_velocity'] and report['motor_omega_constant']
    assert report['hover_equilibrium_verified']
    assert report['norm']['max'] <= distribution_contract(config)['theoretical_norm_max_m']


def test_explicit_disabled_new_sampler_overrides_legacy(models):
    config, _, _ = archived_config(models['posdr'])
    config = replace(config, environment=replace(config.environment,
                     initial_pose_randomization=replace(config.environment.initial_pose_randomization, enabled=False)))
    assert config.environment.position_perturbation == .05
    assert distribution_contract(config)['bound_parameter_m'] == 0
    _, arrays = sample_resets(config, 8, 42)
    np.testing.assert_array_equal(arrays['position_offset'], 0)


def test_existing_tail_offset_variance_decomposition(models):
    for label in models:
        report = inspect_rollouts(SOURCE, label)
        tail = report['step_005_tail']
        assert tail['sample_count'] == 200
        assert tail['position_rmse_total_m']**2 == pytest.approx(tail['mean_offset_mse_m2']+tail['fluctuation_mse_m2'])
        assert tail['mean_offset_fraction_of_mse'] > .99
        final = report['step_050_final_transition']
        assert final['post_time_s'] - final['control_time_s'] == pytest.approx(.01)
        assert final['allocator_saturation'].startswith('unmeasured:')

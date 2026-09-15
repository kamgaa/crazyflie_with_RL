from __future__ import annotations

import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.controllers import rotmat_from_quat_wxyz
from crazyflie_rl.environment import CrazyflieResidualEnv, initial_state_curriculum_limits
from crazyflie_rl.factories import EnvironmentFactory
from crazyflie_rl.recovery import (
    _attitude_success_matrix,
    _write_csv,
    _write_heatmap,
    build_parser,
    consecutive_recovery_time,
    recovery_cases,
    recovery_ranking_key,
    recovery_success_diagnostics,
    reduced_recovery_cases,
    select_best_recovery,
    summarize_recovery,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _runtime():
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")


def test_initial_perturb_profile_resolves_to_e2e_legacy_contract() -> None:
    config = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")
    assert config.control_mode == "e2e"
    assert config.environment.reward.mode == "legacy"
    assert config.environment.initial_state_randomization.enabled is True
    assert config.actuator.model == "cf21b_first_order"
    assert config.observation_shape == (15,)
    assert config.action_shape == (4,)
    assert config.training.seed == 42
    assert config.environment.reward.e2e_torque_xy_weight == 0.0
    assert config.environment.reward.e2e_torque_yaw_weight == 0.0


def test_legacy_recovery_v2_profile_contract() -> None:
    config = load_config(CONFIGS / "e2e_train_legacy_initial_perturb_v2.yaml")
    reward = config.environment.reward
    assert config.control_mode == "e2e"
    assert reward.mode == "legacy"
    assert reward.lyapunov.potential_shaping_enabled is False
    assert reward.lyapunov.decay_penalty_enabled is False
    assert reward.e2e_torque_xy_weight == 0.0
    assert reward.e2e_torque_yaw_weight == 0.0
    assert config.environment.policy_hz == 100.0
    assert config.vehicle.physics_hz == 500.0
    assert config.environment.episode_sec == 8.0
    assert config.actuator.model == "cf21b_first_order"
    assert config.observation_shape == (15,)
    assert config.action_shape == (4,)
    assert config.training.seed == 42
    assert config.training.ppo.learning_rate == 1e-4
    assert config.training.policy_initialization_required is True
    assert config.evaluation.recovery.enabled is True
    assert config.evaluation.recovery.evaluation_interval == 100_000
    nominal_ppo = asdict(load_config(CONFIGS / "e2e_train.yaml").training.ppo)
    recovery_ppo = asdict(config.training.ppo)
    assert recovery_ppo.pop("learning_rate") == pytest.approx(1e-4)
    assert nominal_ppo.pop("learning_rate") == pytest.approx(3e-4)
    assert recovery_ppo == nominal_ppo


@pytest.mark.parametrize(
    ("step", "nominal", "tilt", "horizontal", "vertical"),
    [
        (0, 0.8, 5.0, 0.02, 0.02),
        (100_000, 0.8, 5.0, 0.02, 0.02),
        (200_000, 0.8, 5.0, 0.02, 0.02),
        (350_000, 0.65, 10.0, 0.06, 0.035),
        (500_000, 0.5, 15.0, 0.10, 0.05),
        (750_000, 0.25, 27.5, 11.0 / 60.0, 11.0 / 120.0),
        (799_999, 0.200001, 29.99995, 0.1999996666666667, 0.09999983333333334),
        (800_000, 0.2, 30.0, 0.20, 0.10),
        (800_001, 0.2, 30.0, 0.20, 0.10),
        (1_000_000, 0.2, 30.0, 0.20, 0.10),
        (2_000_000, 0.2, 30.0, 0.20, 0.10),
    ],
)
def test_legacy_recovery_v2_piecewise_curriculum(
    step: int,
    nominal: float,
    tilt: float,
    horizontal: float,
    vertical: float,
) -> None:
    settings = load_config(
        CONFIGS / "e2e_train_legacy_initial_perturb_v2.yaml"
    ).environment.initial_state_randomization
    limits = initial_state_curriculum_limits(settings, step)
    assert limits["nominal_reset_probability"] == pytest.approx(nominal)
    assert limits["maximum_tilt_deg"] == pytest.approx(tilt)
    assert limits["maximum_horizontal_offset_m"] == pytest.approx(horizontal)
    assert limits["maximum_vertical_offset_m"] == pytest.approx(vertical)


def test_legacy_recovery_v2_nominal_sampling_probability() -> None:
    settings = load_config(
        CONFIGS / "e2e_train_legacy_initial_perturb_v2.yaml"
    ).environment.initial_state_randomization
    fake_environment = SimpleNamespace(
        _initial_state_randomization=settings,
        _curriculum_global_step=0,
        _rng=np.random.default_rng(42),
        pos_des=np.array([0.0, 0.0, 1.0]),
        yaw_des=0.0,
    )
    nominal = [
        CrazyflieResidualEnv._curriculum_reset_pose(fake_environment)[2][
            "nominal_reset"
        ]
        for _ in range(10_000)
    ]
    assert np.mean(nominal) == pytest.approx(0.8, abs=0.015)


@pytest.mark.parametrize(
    ("step", "tilt", "horizontal", "vertical"),
    [
        (0, 5.0, 0.02, 0.02),
        (150_000, 17.5, 0.11, 0.06),
        (300_000, 30.0, 0.20, 0.10),
        (900_000, 30.0, 0.20, 0.10),
    ],
)
def test_absolute_timestep_curriculum_limits(
    step: int, tilt: float, horizontal: float, vertical: float
) -> None:
    settings = load_config(
        CONFIGS / "e2e_train_initial_perturb.yaml"
    ).environment.initial_state_randomization
    limits = initial_state_curriculum_limits(settings, step)
    assert limits["maximum_tilt_deg"] == pytest.approx(tilt)
    assert limits["maximum_horizontal_offset_m"] == pytest.approx(horizontal)
    assert limits["maximum_vertical_offset_m"] == pytest.approx(vertical)


def test_disabled_randomization_preserves_existing_reset_exactly() -> None:
    _runtime()
    config = load_config(CONFIGS / "e2e_train.yaml")
    first = EnvironmentFactory(config).make(seed=123)
    second = EnvironmentFactory(config).make(
        seed=123, initial_state_randomization_enabled=False
    )
    try:
        first_observation, first_info = first.reset(seed=123)
        second_observation, second_info = second.reset(seed=123)
        np.testing.assert_array_equal(first_observation, second_observation)
        np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
        np.testing.assert_array_equal(first.data.qvel, second.data.qvel)
        np.testing.assert_array_equal(first._last_f, second._last_f)
        assert first_info == second_info
        assert first_info["payload"]["mass_kg"] == 0.
        assert first_info["physics_model_version"] == "rigid_point_payload_v2"
    finally:
        first.close()
        second.close()


def test_same_seed_reproduces_reset_sequence() -> None:
    _runtime()
    config = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")
    first = EnvironmentFactory(config).make(seed=42)
    second = EnvironmentFactory(config).make(seed=42)
    try:
        first_sequence = [first.reset(seed=42 if i == 0 else None)[1] for i in range(20)]
        second_sequence = [second.reset(seed=42 if i == 0 else None)[1] for i in range(20)]
        assert first_sequence == second_sequence
    finally:
        first.close()
        second.close()


def test_sampled_pose_bounds_quaternion_and_hover_actuator_state() -> None:
    _runtime()
    config = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")
    environment = EnvironmentFactory(config).make(seed=7)
    hover = config.vehicle.mass * config.vehicle.gravity / 4.0
    try:
        environment.set_curriculum_global_step(300_000)
        for index in range(200):
            observation, info = environment.reset(seed=7 if index == 0 else None)
            offset = np.asarray(info["initial_position_offset_xyz_m"], dtype=float)
            assert np.linalg.norm(offset[:2]) <= 0.20 + 1e-12
            assert abs(offset[2]) <= 0.10 + 1e-12
            assert info["initial_tilt_deg"] <= 30.0 + 1e-12
            quaternion = observation[6:10].astype(float)
            assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-7)
            assert quaternion[0] >= 0.0
            rotation = rotmat_from_quat_wxyz(quaternion / np.linalg.norm(quaternion))
            measured_tilt = np.degrees(
                np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))
            )
            assert measured_tilt == pytest.approx(info["initial_tilt_deg"], abs=1e-5)
            np.testing.assert_allclose(
                info["initial_requested_motor_thrust"], hover, atol=1e-12
            )
            np.testing.assert_allclose(
                info["initial_actual_motor_thrust"], hover, atol=1e-12
            )
            assert observation.shape == (15,)
            assert environment.action_space.shape == (4,)
    finally:
        environment.close()


def test_legacy_recovery_v2_airborne_reset_starts_actuator_at_hover() -> None:
    _runtime()
    config = load_config(CONFIGS / "e2e_train_legacy_initial_perturb_v2.yaml")
    environment = EnvironmentFactory(config).make(seed=42)
    hover = config.vehicle.mass * config.vehicle.gravity / 4.0
    try:
        _observation_value, info = environment.reset(seed=42)
        np.testing.assert_allclose(
            info["initial_requested_motor_thrust"], hover, atol=1e-12
        )
        np.testing.assert_allclose(
            info["initial_actual_motor_thrust"], hover, atol=1e-12
        )
    finally:
        environment.close()


@pytest.mark.parametrize(("probability", "expected"), [(1.0, True), (0.0, False)])
def test_nominal_reset_probability_extremes(probability: float, expected: bool) -> None:
    _runtime()
    config = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")
    settings = replace(
        config.environment.initial_state_randomization,
        nominal_reset_probability=probability,
    )
    configured = replace(
        config,
        environment=replace(config.environment, initial_state_randomization=settings),
    )
    environment = EnvironmentFactory(configured).make(seed=11)
    try:
        for index in range(20):
            _observation, info = environment.reset(seed=11 if index == 0 else None)
            assert info["nominal_reset"] is expected
            if expected:
                np.testing.assert_array_equal(info["initial_position_offset_xyz_m"], [0, 0, 0])
                assert info["initial_tilt_deg"] == 0.0
    finally:
        environment.close()


def test_short_deterministic_legacy_rollout_is_finite() -> None:
    _runtime()
    config = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")
    environment = EnvironmentFactory(config).make(seed=42, track_curriculum_steps=True)
    try:
        observation, info = environment.reset(seed=42)
        assert info["curriculum_global_step"] == 0
        for _ in range(10):
            observation, reward, terminated, truncated, _info = environment.step(
                np.zeros(4, dtype=np.float32)
            )
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert environment._curriculum_global_step > 0
    finally:
        environment.close()


def test_legacy_mode_excludes_lyapunov_and_torque_penalties() -> None:
    _runtime()
    config = load_config(CONFIGS / "e2e_train_legacy_initial_perturb_v2.yaml")
    altered_reward = replace(
        config.environment.reward,
        e2e_torque_xy_weight=123.0,
        e2e_torque_yaw_weight=456.0,
        lyapunov=replace(
            config.environment.reward.lyapunov,
            potential_shaping_enabled=True,
            decay_penalty_enabled=True,
        ),
    )
    altered = replace(
        config,
        environment=replace(config.environment, reward=altered_reward),
    )
    baseline = EnvironmentFactory(config).make(seed=42)
    comparison = EnvironmentFactory(altered).make(seed=42)
    action = np.array([0.3, -0.4, 0.5, 0.2], dtype=np.float32)
    try:
        baseline_observation, _ = baseline.reset(seed=42)
        comparison_observation, _ = comparison.reset(seed=42)
        np.testing.assert_array_equal(baseline_observation, comparison_observation)
        baseline_step = baseline.step(action)
        comparison_step = comparison.step(action)
        np.testing.assert_array_equal(baseline_step[0], comparison_step[0])
        assert baseline_step[1] == comparison_step[1]
        assert baseline_step[2:] == comparison_step[2:]
        assert "reward_terms" not in baseline_step[4]
    finally:
        baseline.close()
        comparison.close()


def test_existing_profiles_keep_legacy_reset_and_ppo_defaults() -> None:
    e2e = load_config(CONFIGS / "e2e_train.yaml")
    residual = load_config(CONFIGS / "residual_train.yaml")
    for config in (e2e, residual):
        assert config.environment.reward.mode == "legacy"
        assert config.environment.initial_state_randomization.enabled is False
        assert config.environment.initial_state_randomization.curriculum_breakpoints == ()
        assert config.training.policy_initialization_required is False
        assert config.evaluation.recovery.enabled is False
        assert config.training.ppo.learning_rate == pytest.approx(3e-4)


def test_recovery_requires_one_continuous_second() -> None:
    assert consecutive_recovery_time([True] * 99, policy_hz=100.0) is None
    assert consecutive_recovery_time([True] * 100, policy_hz=100.0) == 1.0
    sequence = [True] * 50 + [False] + [True] * 100
    assert consecutive_recovery_time(sequence, policy_hz=100.0) == 1.51


def test_transient_only_20111505_pattern_is_terminal_window_failure() -> None:
    result = recovery_success_diagnostics(
        [True] * 100 + [False] * 100,
        policy_hz=100.0,
        terminated=False,
        truncated=True,
    )
    assert result["success"] is False
    assert result["recovery_time_s"] is None
    assert result["ever_dwell_success"] is True
    assert result["ever_dwell_recovery_time_s"] == 1.0
    assert result["success_but_final_window_failure"] is True


def test_middle_dwell_followed_by_exit_is_terminal_window_failure() -> None:
    result = recovery_success_diagnostics(
        [False] * 50 + [True] * 100 + [False] * 50,
        policy_hz=100.0,
        terminated=False,
        truncated=True,
    )
    assert result["success"] is False
    assert result["ever_dwell_success"] is True
    assert result["terminal_valid_suffix_steps"] == 0


def test_final_continuous_second_is_success_and_uses_last_entry_time() -> None:
    result = recovery_success_diagnostics(
        [True] * 100 + [False] + [True] * 100,
        policy_hz=100.0,
        terminated=False,
        truncated=True,
    )
    assert result["success"] is True
    assert result["recovery_time_s"] == pytest.approx(2.01)
    assert result["terminal_valid_suffix_steps"] == 100


def test_terminated_trajectory_fails_even_with_valid_terminal_window() -> None:
    result = recovery_success_diagnostics(
        [True] * 100,
        policy_hz=100.0,
        terminated=True,
        truncated=False,
    )
    assert result["terminal_window_satisfied"] is True
    assert result["success"] is False
    assert result["recovery_time_s"] is None


def test_time_limit_truncation_can_succeed_with_valid_terminal_window() -> None:
    result = recovery_success_diagnostics(
        [False] * 100 + [True] * 100,
        policy_hz=100.0,
        terminated=False,
        truncated=True,
    )
    assert result["success"] is True
    assert result["recovery_time_s"] == pytest.approx(2.0)


def test_nominal_valid_for_entire_episode_recovers_at_one_second() -> None:
    result = recovery_success_diagnostics(
        [True] * 800,
        policy_hz=100.0,
        terminated=False,
        truncated=True,
    )
    assert result["success"] is True
    assert result["recovery_time_s"] == pytest.approx(1.0)


def test_recovery_grid_and_best_selection_contract() -> None:
    cases = recovery_cases()
    assert len(cases) == 62
    assert sum(case.category == "attitude" for case in cases) == 56
    assert sum(case.category == "position" for case in cases) == 6
    assert sum(case.tilt_deg == 0.0 for case in cases) == 14
    summaries = [
        {
            "model": "slower.zip",
            "success_rate": 0.9,
            "mean_recovery_time_s": 2.0,
            "mean_maximum_position_error_m": 0.1,
        },
        {
            "model": "faster.zip",
            "success_rate": 0.9,
            "mean_recovery_time_s": 1.5,
            "mean_maximum_position_error_m": 0.2,
        },
        {
            "model": "lower-rate.zip",
            "success_rate": 0.8,
            "mean_recovery_time_s": 1.0,
            "mean_maximum_position_error_m": 0.01,
        },
    ]
    # The requested donor ordering compares position excursion before recovery
    # time after nominal/rate/termination/category-rate ties.
    assert select_best_recovery(summaries)["model"] == "slower.zip"


def test_reduced_recovery_grid_and_nominal_required_ranking() -> None:
    cases = reduced_recovery_cases()
    assert len(cases) == 19
    assert sum(case.tilt_deg == 0.0 and case.category == "attitude" for case in cases) == 1
    assert sum(case.tilt_deg > 0.0 for case in cases) == 12
    assert sum(case.category == "position" for case in cases) == 6
    no_nominal = {
        "model": "high-rate-without-nominal.zip",
        "nominal_hover_success": False,
        "overall_success_rate": 0.99,
        "mean_recovery_time_s": 1.0,
        "mean_maximum_position_error_m": 0.01,
    }
    valid = {
        "model": "nominal-valid.zip",
        "nominal_hover_success": True,
        "overall_success_rate": 0.5,
        "mean_recovery_time_s": 2.0,
        "mean_maximum_position_error_m": 0.2,
    }
    assert recovery_ranking_key(valid) < recovery_ranking_key(no_nominal)
    assert select_best_recovery([no_nominal, valid])["model"] == "nominal-valid.zip"


def _synthetic_recovery_results(success_names: set[str]) -> list[dict[str, object]]:
    return [
        {
            "category": case.category,
            "name": case.name,
            "tilt_deg": case.tilt_deg,
            "success": case.name in success_names,
            "recovery_time_s": 1.0 if case.name in success_names else None,
            "maximum_position_error_m": 0.1,
        }
        for case in recovery_cases()
    ]


def test_recovery_summary_separates_rates_and_counts_nominal_once() -> None:
    cases = recovery_cases()
    nominal = {case.name for case in cases if case.category == "attitude" and case.tilt_deg == 0.0}
    attitude = [case.name for case in cases if case.category == "attitude" and case.tilt_deg > 0.0]
    position = [case.name for case in cases if case.category == "position"]
    successes = nominal | set(attitude[:24]) | set(position[:3])

    summary = summarize_recovery("policy.zip", _synthetic_recovery_results(successes))

    assert summary["nominal_hover_success"] is True
    assert summary["attitude_perturbation_success_count"] == 24
    assert summary["attitude_perturbation_success_rate"] == pytest.approx(0.5)
    assert summary["position_perturbation_success_count"] == 3
    assert summary["position_perturbation_success_rate"] == pytest.approx(0.5)
    assert summary["aggregate_condition_count"] == 55
    assert summary["success_count"] == 28
    assert summary["overall_success_rate"] == pytest.approx(28 / 55)
    assert summary["success_rate"] == summary["overall_success_rate"]


def test_summary_csv_json_and_heatmap_use_terminal_window_success(
    tmp_path: Path,
) -> None:
    results = _synthetic_recovery_results(set())
    stable_name = "roll_plus_5deg"
    transient_name = "roll_plus_0deg"
    for result in results:
        if result["name"] == stable_name:
            result["success"] = True
            result["recovery_time_s"] = 7.0
        if result["name"] == transient_name:
            result["ever_dwell_success"] = True
            result["success_but_final_window_failure"] = True
            result["success"] = False

    summary = summarize_recovery("policy.zip", results)
    assert summary["nominal_hover_success"] is False
    assert summary["attitude_perturbation_success_count"] == 1
    assert summary["success_but_final_window_failure"] is True
    assert summary["success_but_final_window_failure_count"] == 1

    cases_csv = tmp_path / "recovery-cases.csv"
    _write_csv(cases_csv, results)
    with cases_csv.open(newline="", encoding="utf-8") as handle:
        csv_rows = {row["name"]: row for row in csv.DictReader(handle)}
    assert csv_rows[transient_name]["success"] == "False"
    assert csv_rows[stable_name]["success"] == "True"

    report_path = tmp_path / "recovery.json"
    report_path.write_text(json.dumps({"case_results": results}), encoding="utf-8")
    json_rows = {
        row["name"]: row
        for row in json.loads(report_path.read_text(encoding="utf-8"))["case_results"]
    }
    assert json_rows[transient_name]["success"] is False
    assert json_rows[stable_name]["success"] is True

    heatmap_values = _attitude_success_matrix(results)
    assert heatmap_values[0, 0] == 0.0
    assert heatmap_values[0, 1] == 1.0
    heatmap_path = tmp_path / "recovery-success-heatmap.png"
    _write_heatmap(heatmap_path, [(Path("policy.zip"), results)])
    assert heatmap_path.is_file()
    assert heatmap_path.stat().st_size > 0


def test_best_recovery_ranking_is_not_biased_by_nominal_duplicate_rows() -> None:
    cases = recovery_cases()
    nominal = {case.name for case in cases if case.category == "attitude" and case.tilt_deg == 0.0}
    perturbations = [case.name for case in cases if case.tilt_deg > 0.0]
    nominal_only = summarize_recovery(
        "nominal-only.zip", _synthetic_recovery_results(nominal)
    )
    two_perturbations = summarize_recovery(
        "two-perturbations.zip",
        _synthetic_recovery_results(nominal | set(perturbations[:2])),
    )

    assert nominal_only["overall_success_rate"] == pytest.approx(1 / 55)
    assert two_perturbations["overall_success_rate"] == pytest.approx(3 / 55)
    assert select_best_recovery([nominal_only, two_perturbations])["model"] == (
        "two-perturbations.zip"
    )


def test_recovery_parser_exposes_opt_in_pid_floor() -> None:
    args = build_parser().parse_args(
        ["--config", "profile.yaml", "--model", "policy.zip", "--include-pid-floor"]
    )
    assert args.include_pid_floor is True


def test_recovery_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "evaluate_recovery.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert "--model" in completed.stdout
    assert "--include-pid-floor" in completed.stdout

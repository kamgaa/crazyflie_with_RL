from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.controllers import geometric_attitude_error, rotmat_from_quat_wxyz
from crazyflie_rl.e2e_diagnostics import (
    _load_policy,
    _validate_loaded_policy_contract,
    actual_wrench_from_environment,
    deterministic_rollout,
    normalized_action_to_physical_wrench,
    probe_policy_case,
    requested_motor_thrusts,
    selected_failure_rows,
    set_exact_hover_state,
    yaw_pitch_quaternion_wxyz,
)
from crazyflie_rl.environment import (
    reward_component_consistency,
    termination_diagnostics,
)
from crazyflie_rl.factories import EnvironmentFactory


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "e2e_train_lyapunov.yaml"
LEGACY_CONFIG = ROOT / "configs" / "e2e_train_legacy_initial_perturb_v2.yaml"


class ZeroPolicy:
    def predict(self, observation, deterministic=False):
        assert np.asarray(observation).shape == (15,)
        assert deterministic is True
        return np.zeros(4, dtype=np.float32), None


@pytest.fixture
def e2e_environment():
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    environment = EnvironmentFactory(load_config(CONFIG)).make(seed=1000)
    try:
        yield environment
    finally:
        environment.close()


def test_exact_nominal_hover_state_and_observation(e2e_environment) -> None:
    observation = set_exact_hover_state(e2e_environment, seed=1000)
    position, quaternion, velocity, omega = e2e_environment._read_state()
    np.testing.assert_allclose(position, e2e_environment.pos_des, atol=1e-12)
    np.testing.assert_allclose(velocity, 0.0, atol=1e-12)
    np.testing.assert_allclose(omega, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        quaternion,
        yaw_pitch_quaternion_wxyz(e2e_environment.yaw_des, 0.0),
        atol=1e-12,
    )
    expected = np.concatenate(
        [np.zeros(6), quaternion, np.zeros(3), [0.0, 1.0]]
    ).astype(np.float32)
    np.testing.assert_array_equal(observation, expected)
    hover = e2e_environment.mass * e2e_environment.gravity / 4.0
    np.testing.assert_allclose(e2e_environment._last_f, hover, atol=1e-12)


@pytest.mark.parametrize("angle_deg", [5.0, -5.0])
def test_pitch_perturbation_so3_error_sign_and_magnitude(
    e2e_environment, angle_deg: float
) -> None:
    set_exact_hover_state(
        e2e_environment, seed=1000, pitch_rad=np.deg2rad(angle_deg)
    )
    state = e2e_environment._read_state()
    error = e2e_environment._tracking_error(*state)
    assert error.attitude[1] == pytest.approx(np.sin(np.deg2rad(angle_deg)))
    np.testing.assert_allclose(error.attitude[[0, 2]], 0.0, atol=1e-12)


@pytest.mark.parametrize("omega_y", [1.0, -1.0])
def test_body_rate_perturbation_is_measured_in_body_frame(
    e2e_environment, omega_y: float
) -> None:
    observation = set_exact_hover_state(
        e2e_environment,
        seed=1000,
        body_angular_velocity=(0.0, omega_y, 0.0),
    )
    state = e2e_environment._read_state()
    np.testing.assert_allclose(state[3], [0.0, omega_y, 0.0], atol=1e-12)
    np.testing.assert_allclose(observation[10:13], [0.0, omega_y, 0.0], atol=1e-12)


def test_normalized_action_to_wrench_mapping_and_hover_offset(e2e_environment) -> None:
    action = np.array([0.3, -0.4, 0.5, 0.0])
    wrench = normalized_action_to_physical_wrench(e2e_environment, action)
    expected = np.asarray(e2e_environment.residual_scale) * action
    expected[3] += e2e_environment.mass * e2e_environment.gravity
    np.testing.assert_allclose(wrench, expected, rtol=0.0, atol=1e-15)
    assert wrench[0] == pytest.approx(0.0066)
    assert wrench[1] == pytest.approx(-0.0088)
    assert wrench[2] == pytest.approx(0.00005)


def test_requested_and_actual_motor_thrust_and_wrench_are_distinct(
    e2e_environment,
) -> None:
    set_exact_hover_state(e2e_environment, seed=1000)
    wrench = normalized_action_to_physical_wrench(
        e2e_environment, [0.3, -0.4, 0.5, 0.0]
    )
    requested = requested_motor_thrusts(e2e_environment, wrench)
    before = e2e_environment._last_f.copy()
    e2e_environment._apply_control(wrench)
    np.testing.assert_allclose(e2e_environment._last_f_cmd, requested, atol=1e-12)
    assert not np.array_equal(requested, e2e_environment._last_f)
    assert not np.array_equal(before, e2e_environment._last_f)
    reconstructed = actual_wrench_from_environment(e2e_environment)
    np.testing.assert_allclose(
        reconstructed, e2e_environment._last_wrench_actual, atol=1e-12
    )


def test_probe_uses_real_observation_and_deterministic_predict(e2e_environment) -> None:
    result = probe_policy_case(
        e2e_environment,
        ZeroPolicy(),
        case="nominal",
        seed=1000,
    )
    assert len(result.observation) == 15
    np.testing.assert_allclose(result.normalized_action, 0.0)
    assert result.physical_commanded_wrench[3] == pytest.approx(
        e2e_environment.mass * e2e_environment.gravity
    )


@pytest.mark.parametrize(
    ("position", "quaternion", "expected"),
    [
        ([0.0, 0.0, 0.019], [1.0, 0.0, 0.0, 0.0], "below_minimum_altitude"),
        ([0.0, 0.0, 2.501], [1.0, 0.0, 0.0, 0.0], "above_maximum_altitude"),
        (
            [0.0, 0.0, 1.0],
            yaw_pitch_quaternion_wxyz(0.0, np.deg2rad(61.0)),
            "excessive_tilt",
        ),
        ([1.501, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], "excessive_position_error"),
    ],
)
def test_each_termination_reason(position, quaternion, expected) -> None:
    diagnostic = termination_diagnostics(
        position=position,
        position_reference=[0.0, 0.0, 1.0],
        quaternion_wxyz=quaternion,
        minimum_altitude=0.02,
        maximum_altitude=2.5,
        maximum_tilt_rad=np.deg2rad(60.0),
        maximum_position_error=1.5,
    )
    assert expected in diagnostic["termination_reasons"]


def test_reward_components_sum_to_total_and_expose_actuator_diagnostics(
    e2e_environment,
) -> None:
    set_exact_hover_state(e2e_environment, seed=1000)
    action = np.array([0.3, -0.4, 0.5, 0.9], dtype=np.float32)
    _observation, reward, _terminated, _truncated, info = e2e_environment.step(action)
    terms = info["reward_terms"]
    assert terms["reward_component_sum"] == pytest.approx(reward)
    assert terms["total_reward"] == pytest.approx(reward)
    assert terms["reward_components_consistent"] is True
    assert terms["e2e_torque_cost"] == pytest.approx(0.03)
    assert terms["e2e_torque_reward"] == pytest.approx(-0.03)
    control = info["control_diagnostics"]
    np.testing.assert_allclose(control["normalized_action"], action)
    np.testing.assert_allclose(
        control["actual_applied_wrench"], e2e_environment._last_wrench_actual
    )


def test_float32_training_transition_reward_consistency_regression() -> None:
    """Reproduce the seed-42 rate01 transition that tripped the old 1e-12 check."""

    reward = np.float32(-0.5982349514961243)
    components = {
        "state_reward": -0.43009354481132733,
        "potential_reward": -0.1431462067520063,
        "decay_reward": -0.022582156728236526,
        "nontracking_reward": -0.0,
        "e2e_torque_reward": -0.0024130108393728734,
        "legacy_tracking_reward": 0.0,
        "crash_or_ood_reward": 0.0,
    }
    result = reward_component_consistency(reward, components)
    assert result["component_sum"] == pytest.approx(-0.598234919130943)
    assert result["difference"] == pytest.approx(3.236518131277677e-08)
    assert result["reward_dtype"] == "float32"
    assert abs(result["difference"]) < result["absolute_tolerance"]
    assert result["consistent"] is True

    missing_torque = dict(components)
    missing_torque["e2e_torque_reward"] = 0.0
    invalid = reward_component_consistency(reward, missing_torque)
    assert invalid["consistent"] is False


def test_existing_termination_boolean_and_episode_length_are_unchanged(
    e2e_environment,
) -> None:
    set_exact_hover_state(e2e_environment, seed=1000)
    e2e_environment.max_steps = 1
    _obs, _reward, terminated, truncated, info = e2e_environment.step(np.zeros(4))
    assert terminated is False
    assert truncated is True
    assert info["termination_reasons"] == []


@pytest.mark.parametrize(
    "reason",
    [
        "below_minimum_altitude",
        "above_maximum_altitude",
        "excessive_tilt",
        "excessive_position_error",
    ],
)
def test_terminal_reward_components_cover_every_existing_guard(
    e2e_environment, reason: str
) -> None:
    mujoco = pytest.importorskip("mujoco")
    set_exact_hover_state(e2e_environment, seed=1000)
    if reason == "below_minimum_altitude":
        e2e_environment.data.qpos[2] = 0.01
    elif reason == "above_maximum_altitude":
        e2e_environment.data.qpos[2] = 2.51
    elif reason == "excessive_tilt":
        e2e_environment.data.qpos[3:7] = yaw_pitch_quaternion_wxyz(
            e2e_environment.yaw_des, np.deg2rad(61.0)
        )
    else:
        e2e_environment.data.qpos[0] = e2e_environment.pos_des[0] + 1.51
    mujoco.mj_forward(e2e_environment.model, e2e_environment.data)

    _obs, _reward, terminated, truncated, info = e2e_environment.step(
        np.zeros(4, dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert reason in info["termination_reasons"]
    terms = info["reward_terms"]
    assert terms["terminal_potential_zeroed"] is True
    assert terms["crash_or_ood_reward"] == -10.0
    assert terms["reward_components_consistent"] is True


def test_diagnostic_cli_help_is_import_safe() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "diagnose_e2e_policy.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert "--model" in completed.stdout
    assert "--seed" in completed.stdout


def test_diagnostic_model_archive_load_failure_is_hard_error(tmp_path) -> None:
    invalid_model = tmp_path / "invalid.zip"
    invalid_model.write_bytes(b"not an SB3 archive")

    with pytest.raises(ValueError, match="cannot load E2E policy model file"):
        _load_policy(invalid_model)


@pytest.mark.parametrize(
    ("observation_shape", "action_shape", "message"),
    [((13,), (4,), "loaded observation shape"), ((15,), (3,), "loaded action shape")],
)
def test_loaded_policy_space_mismatch_is_hard_error(
    observation_shape, action_shape, message
) -> None:
    policy = SimpleNamespace(
        observation_space=SimpleNamespace(shape=observation_shape),
        action_space=SimpleNamespace(shape=action_shape),
    )

    with pytest.raises(ValueError, match=message):
        _validate_loaded_policy_contract(policy, load_config(CONFIG))


def test_legacy_policy_rollout_does_not_require_lyapunov_reward_logging() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    environment = EnvironmentFactory(load_config(LEGACY_CONFIG)).make(seed=1000)
    try:
        rows = deterministic_rollout(
            environment, ZeroPolicy(), seed=1000, maximum_steps=2
        )
        selected = selected_failure_rows(rows)
    finally:
        environment.close()

    assert len(rows) == 2
    assert rows[0]["v_before"] is None
    assert rows[0]["v_after"] is None
    assert len(rows[0]["physical_commanded_wrench"]) == 4
    assert len(rows[0]["requested_motor_thrust"]) == 4
    assert len(rows[0]["actual_motor_thrust"]) == 4
    assert len(rows[0]["actual_applied_wrench"]) == 4
    assert len(selected) == 5

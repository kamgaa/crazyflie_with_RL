from __future__ import annotations

import importlib

import numpy as np
import pytest

from crazyflie_rl.actuators import (
    Cf21bFirstOrderActuatorModel,
    InstantaneousActuatorModel,
)
from crazyflie_rl.controllers import build_allocation_matrix


MOTOR_DIRECTION = np.array([1.0, -1.0, 1.0, -1.0])


def _first_order(**overrides) -> Cf21bFirstOrderActuatorModel:
    settings = {
        "dt": 0.002,
        "motor_direction": MOTOR_DIRECTION,
        "time_constant_s": 0.050,
        "steady_state_gain_rad_s": 2900.0,
    }
    settings.update(overrides)
    return Cf21bFirstOrderActuatorModel(**settings)


def test_actuator_module_imports_without_mujoco_or_gymnasium() -> None:
    module = importlib.import_module("crazyflie_rl.actuators")

    assert hasattr(module, "Cf21bFirstOrderActuatorModel")
    assert "mujoco" not in module.__dict__
    assert "gymnasium" not in module.__dict__


def test_instantaneous_model_preserves_legacy_force_and_reaction_torque() -> None:
    allocation, _ = build_allocation_matrix()
    model = InstantaneousActuatorModel(
        motor_direction=MOTOR_DIRECTION,
        allocation_matrix=allocation,
    )
    command = np.array([0.01, 0.10, 0.15, 0.20])

    output = model.apply(command)

    np.testing.assert_array_equal(output.f_cmd, command)
    np.testing.assert_array_equal(output.f_actual, command)
    np.testing.assert_allclose(
        output.q_actual,
        MOTOR_DIRECTION * 0.00594 * command,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(output.wrench_cmd, allocation @ command)
    np.testing.assert_allclose(output.wrench_actual, allocation @ command)
    np.testing.assert_allclose(output.allocation_error, np.zeros(4), atol=2e-18)
    assert np.isnan(output.motor_command).all()
    assert np.isnan(output.omega).all()


def test_first_order_exact_exponential_step_response_after_one_time_constant() -> None:
    model = _first_order()
    model.reset(airborne=False)

    for _ in range(25):  # 25 * 0.002 s == tau == 0.050 s
        output = model.apply_motor_command(np.ones(4))

    expected = 2900.0 * (1.0 - np.exp(-1.0))
    np.testing.assert_allclose(output.omega, expected, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(model.decay, np.exp(-0.002 / 0.050))


def test_first_order_motor_speed_converges_to_gain_times_normalized_command() -> None:
    model = _first_order()
    model.reset(airborne=False)
    command = np.array([0.0, 0.2, 0.5, 1.0])

    for _ in range(1000):
        output = model.apply_motor_command(command)

    np.testing.assert_allclose(output.omega, 2900.0 * command, rtol=0.0, atol=1e-8)
    assert np.all(output.motor_command >= 0.0)
    assert np.all(output.motor_command <= 1.0)
    assert np.all(np.isfinite(output.f_actual))
    assert np.all(output.f_actual >= 0.0)
    assert np.all(output.f_actual <= 0.20)


def test_inverse_is_bounded_and_forward_mapping_round_trips_positive_thrust() -> None:
    model = _first_order()
    requested = np.array([0.0, 0.01, 0.10, 0.20])

    omega = model.inverse_thrust(requested)
    achieved = model.thrust_from_omega(omega)

    assert omega[0] == 0.0
    assert np.all(omega[1:] >= model.positive_branch_min_ratio * 2900.0)
    assert np.all(omega <= model.max_ratio * 2900.0)
    np.testing.assert_allclose(achieved, requested, rtol=0.0, atol=2e-12)


def test_inverse_and_forward_mapping_saturate_without_negative_or_nan_thrust() -> None:
    model = _first_order()
    requested = np.array([-10.0, 0.0, 0.20, 10.0])

    omega = model.inverse_thrust(requested)
    achieved = model.thrust_from_omega(np.array([-1.0, 0.0, 2900.0, 1e9]))

    assert np.all(np.isfinite(omega))
    assert np.all(omega >= 0.0)
    assert np.all(omega <= model.max_ratio * model.omega_reference_rad_s)
    assert np.all(np.isfinite(achieved))
    assert np.all(achieved >= 0.0)
    assert np.all(achieved <= 0.20)
    assert omega[-1] == pytest.approx(model.inverse_thrust([0.20] * 4)[-1])


def test_legacy_reaction_torque_uses_cw_ccw_signs_and_cancels_at_equal_speed() -> None:
    model = _first_order()
    model.reset(airborne=False)

    for _ in range(500):
        output = model.apply(np.full(4, 0.10))

    np.testing.assert_allclose(
        output.q_actual,
        MOTOR_DIRECTION * 0.00594 * output.f_actual,
        rtol=0.0,
        atol=1e-15,
    )
    assert float(np.sum(output.q_actual)) == pytest.approx(0.0, abs=1e-15)


def test_apply_updates_at_each_physics_substep_and_holds_the_same_command() -> None:
    model = _first_order()
    target = np.full(4, 0.10)
    expected_motor_command = model.inverse_thrust(target) / 2900.0

    outputs = [model.apply(target) for _ in range(5)]

    assert outputs[4].omega[0] > outputs[0].omega[0]
    np.testing.assert_allclose(outputs[4].motor_command, expected_motor_command)
    np.testing.assert_allclose(
        outputs[4].omega,
        2900.0 * expected_motor_command * (1.0 - np.exp(-5.0 * 0.002 / 0.050)),
        rtol=0.0,
        atol=1e-10,
    )
    for output in outputs:
        np.testing.assert_allclose(output.f_cmd, target)


def test_ground_and_airborne_reset_have_expected_rotor_states_and_hover_thrust() -> None:
    model = _first_order()

    ground = model.reset(airborne=False)
    airborne = model.reset(airborne=True, episode_mass=0.04338, gravity_m_s2=9.81)

    np.testing.assert_array_equal(ground.omega, np.zeros(4))
    np.testing.assert_array_equal(ground.f_actual, np.zeros(4))
    assert float(np.sum(airborne.f_actual)) == pytest.approx(0.04338 * 9.81, abs=2e-12)
    assert np.all(airborne.omega > 0.0)


def test_parameter_sampling_is_reproducible_and_uses_a_caller_owned_rng() -> None:
    first = _first_order()
    second = _first_order()
    params_first = first.sample_parameters(
        np.random.default_rng(123),
        enabled=True,
        time_constant_range=(0.040, 0.060),
        steady_state_gain_range=(2320.0, 3480.0),
    )
    params_second = second.sample_parameters(
        np.random.default_rng(123),
        enabled=True,
        time_constant_range=(0.040, 0.060),
        steady_state_gain_range=(2320.0, 3480.0),
    )

    np.testing.assert_array_equal(
        params_first.time_constant_s, params_second.time_constant_s
    )
    np.testing.assert_array_equal(
        params_first.steady_state_gain_rad_s,
        params_second.steady_state_gain_rad_s,
    )
    np.testing.assert_array_equal(
        first.sampled_time_constant_s, params_first.time_constant_s
    )
    np.testing.assert_array_equal(
        first.sampled_steady_state_gain_rad_s,
        params_first.steady_state_gain_rad_s,
    )

    payload_rng = np.random.default_rng(7)
    expected_payload_draw = np.random.default_rng(7).uniform(size=3)
    first.sample_parameters(
        np.random.default_rng(99),
        enabled=True,
        time_constant_range=(0.040, 0.060),
        steady_state_gain_range=(2320.0, 3480.0),
    )
    np.testing.assert_array_equal(payload_rng.uniform(size=3), expected_payload_draw)


def test_paper_torque_model_is_opt_in_and_reports_finite_torque() -> None:
    model = _first_order(reaction_torque_model="paper_polynomial")
    model.reset(airborne=False)

    for _ in range(100):
        output = model.apply(np.full(4, 0.10))

    assert np.all(np.isfinite(output.q_actual))
    assert output.q_actual[0] > 0.0
    assert output.q_actual[1] < 0.0

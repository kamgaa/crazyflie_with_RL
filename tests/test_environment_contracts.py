from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import crazyflie_rl.environment as env_module
import crazyflie_rl.factories as factory_module
from crazyflie_rl.actuators import (
    Cf21bFirstOrderActuatorModel,
)
from crazyflie_rl.config import load_config
from crazyflie_rl.controllers import (
    ARM,
    GRAV,
    K_TAU,
    MASS,
    MOTOR_DIR,
    CascadePID,
    build_allocation_matrix,
)
from crazyflie_rl.environment import (
    DEFAULT_RESIDUAL_SCALE,
    OBSERVATION_DIM,
    CrazyflieResidualEnv,
)
from crazyflie_rl.factories import EnvironmentFactory


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _observation_only_env(mode: str) -> CrazyflieResidualEnv:
    env = CrazyflieResidualEnv.__new__(CrazyflieResidualEnv)
    env.mode = mode
    env.pos_des = np.array([0.0, 0.0, 1.0])
    env.yaw_des = -0.1
    return env


@pytest.mark.parametrize("mode", ["residual", "e2e"])
def test_both_modes_preserve_the_exact_fifteen_value_observation(mode: str) -> None:
    env = _observation_only_env(mode)
    position = np.array([0.2, -0.1, 1.4])
    velocity = np.array([1.0, 2.0, 3.0])
    quaternion = np.array([np.cos(0.2), 0.0, 0.0, np.sin(0.2)])
    omega = np.array([0.4, 0.5, 0.6])

    observation = env._obs(position, quaternion, velocity, omega)
    yaw_error = 0.4 - env.yaw_des
    expected = np.concatenate(
        [
            position - env.pos_des,
            velocity,
            quaternion,
            omega,
            [np.sin(yaw_error), np.cos(yaw_error)],
        ]
    ).astype(np.float32)

    assert OBSERVATION_DIM == 15
    assert observation.shape == (15,)
    assert observation.dtype == np.float32
    np.testing.assert_allclose(observation, expected, rtol=0.0, atol=1e-7)


def test_all_profiles_publish_the_same_observation_contract() -> None:
    for profile in CONFIGS.glob("*.yaml"):
        config = load_config(profile)
        assert config.observation_shape == (15,), profile.name
        assert config.action_shape == (4,), profile.name


@pytest.mark.parametrize("profile", ["residual_train.yaml", "e2e_train.yaml"])
def test_real_mujoco_reset_step_when_server_xml_is_available(profile: str) -> None:
    config = load_config(CONFIGS / profile)
    if env_module.gym is None or env_module.mujoco is None:
        pytest.skip("Gymnasium and MuJoCo are not installed")
    if not Path(config.paths.mujoco_xml).is_file():
        pytest.skip(f"server MuJoCo XML is unavailable: {config.paths.mujoco_xml}")

    env = EnvironmentFactory(config).make(seed=0)
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (15,)
        result = env.step(np.zeros(4, dtype=np.float32))
        assert result[0].shape == (15,)
        assert np.isfinite(result[1])
        assert isinstance(result[2], bool)
        assert isinstance(result[3], bool)
    finally:
        env.close()


def test_invalid_mode_scale_frequency_and_relative_xml_fail_before_mujoco() -> None:
    absolute_placeholder = "/not-loaded/cf21B_500.xml"
    with pytest.raises(ValueError, match="mode"):
        CrazyflieResidualEnv(absolute_placeholder, mode="absolute")
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        CrazyflieResidualEnv(absolute_placeholder, residual_scale=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="substeps"):
        CrazyflieResidualEnv(absolute_placeholder, policy_hz=1001.0)
    with pytest.raises(ValueError, match="absolute path"):
        CrazyflieResidualEnv("relative/cf21B_500.xml")


class _FakeBox:
    def __init__(self, low, high, shape=None, dtype=None):
        del high, dtype
        self.shape = tuple(shape) if shape is not None else np.asarray(low).shape


class _FakeModel:
    def __init__(self):
        self.opt = SimpleNamespace(timestep=None)
        self.body_mass = np.array([MASS])
        self.body_ipos = np.zeros((1, 3))
        self.body_iquat = np.array([[1.,0.,0.,0.]])
        self.body_inertia = np.array([[2.3951e-5, 2.3951e-5, 3.2347e-5]])
        self.sensor_adr = np.array([0])


class _FakeData:
    def __init__(self, _model):
        self.ctrl = np.zeros(16)


def test_config_values_are_wired_into_the_runtime_before_model_use(monkeypatch) -> None:
    config = load_config(CONFIGS / "residual_train.yaml")
    fake_model = _FakeModel()
    fake_mujoco = SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_path=lambda path: fake_model),
        MjData=_FakeData,
        mjtObj=SimpleNamespace(
            mjOBJ_BODY=1, mjOBJ_SENSOR=2, mjOBJ_ACTUATOR=3
        ),
        mj_name2id=lambda *_args: 0,
    )
    monkeypatch.setattr(env_module, "gym", SimpleNamespace())
    monkeypatch.setattr(env_module, "spaces", SimpleNamespace(Box=_FakeBox))
    monkeypatch.setattr(env_module, "mujoco", fake_mujoco)

    env = CrazyflieResidualEnv(config=config, seed=7)

    assert env.mode == "residual"
    assert env.xml_path == str(config.paths.mujoco_xml)
    assert env.physics_hz == config.vehicle.physics_hz
    assert env.policy_hz == config.environment.policy_hz
    assert env.substeps == 5
    assert env.max_steps == 800
    np.testing.assert_array_equal(env.residual_scale, config.environment.residual_scale)
    np.testing.assert_array_equal(env.pos_des, config.environment.position_target)
    assert env.com_bias_mass == config.environment.payload.mass == 0.010
    assert env.pid.kp_pos == config.controller.pid.kp_position
    assert env.pid.v_max == config.controller.pid.velocity_limit
    np.testing.assert_array_equal(env.pid.kp_rate, config.controller.pid.kp_rate)
    assert env.pid.mass == config.vehicle.mass
    assert env.action_space.shape == (4,)
    assert env.observation_space.shape == (15,)
    assert fake_model.opt.timestep == pytest.approx(1.0 / 500.0)


def test_legacy_constructor_overrides_take_precedence_over_config(monkeypatch) -> None:
    config = load_config(CONFIGS / "e2e_train.yaml")
    fake_model = _FakeModel()
    monkeypatch.setattr(env_module, "gym", SimpleNamespace())
    monkeypatch.setattr(env_module, "spaces", SimpleNamespace(Box=_FakeBox))
    monkeypatch.setattr(
        env_module,
        "mujoco",
        SimpleNamespace(
            MjModel=SimpleNamespace(from_xml_path=lambda path: fake_model),
            MjData=_FakeData,
            mjtObj=SimpleNamespace(
                mjOBJ_BODY=1, mjOBJ_SENSOR=2, mjOBJ_ACTUATOR=3
            ),
            mj_name2id=lambda *_args: 0,
        ),
    )

    env = CrazyflieResidualEnv(
        config=config,
        mode="residual",
        residual_scale=(0.006, 0.006, 0.0001, 0.3),
        episode_sec=36.0,
        com_bias_mass=0.005,
        pos_perturb=0.0,
        att_perturb_deg=5.0,
    )

    assert env.mode == "residual"
    np.testing.assert_array_equal(
        env.residual_scale, (0.006, 0.006, 0.0001, 0.3)
    )
    assert env.max_steps == 3600
    assert env.com_bias_mass == 0.005
    assert env.pos_perturb == 0.0
    assert env.att_perturb_deg == 5.0


def _step_only_env(
    mode: str, state: tuple[np.ndarray, ...], captured: list[np.ndarray]
) -> CrazyflieResidualEnv:
    env = _observation_only_env(mode)
    env.yaw_des = 0.0
    env.residual_scale = np.asarray(DEFAULT_RESIDUAL_SCALE)
    env.substeps = 1
    env._read_state = lambda: tuple(value.copy() for value in state)
    env.pid = lambda *_args: np.array([0.1, -0.2, 0.3, 0.4])
    env._apply_control = lambda value: captured.append(np.asarray(value).copy())
    env._com_off3 = np.array([0.1, -0.2, 0.0])
    env._com_mw = 0.01
    env.gravity = GRAV
    env.mass = MASS
    env.dist_torque_body = np.zeros(3)
    env.drone_bid = 0
    env.data = SimpleNamespace(xfrc_applied=np.zeros((1, 6)))
    env.model = SimpleNamespace()
    env._prev_action = np.zeros(4)
    env._step = 0
    env.max_steps = 100
    env.position_weight = 3.0
    env.position_xy_weight = env.position_z_weight = 3.0
    env.velocity_weight = 0.01
    env.tilt_weight = 3.0
    env.angular_velocity_weight = 0.001
    env.yaw_weight = 1.0
    env.action_weight = 0.001
    env.w_dact = 0.25
    env.crash_penalty = 10.0
    env.min_altitude = 0.2
    env.max_altitude = 2.5
    env.max_termination_tilt = np.deg2rad(60.0)
    env.max_position_error = 1.5
    return env


@pytest.mark.parametrize("mode", ["residual", "e2e"])
def test_action_clip_scale_control_composition_and_payload_torque(
    mode: str, monkeypatch
) -> None:
    monkeypatch.setattr(
        env_module, "mujoco", SimpleNamespace(mj_step=lambda *_args: None)
    )
    state = (
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
    )
    captured: list[np.ndarray] = []
    env = _step_only_env(mode, state, captured)
    raw_action = np.array([2.0, -2.0, 0.5, 0.25], dtype=np.float32)
    clipped = np.array([1.0, -1.0, 0.5, 0.25], dtype=np.float32)
    scaled = np.asarray(DEFAULT_RESIDUAL_SCALE) * clipped

    observation, reward, terminated, truncated, _ = env.step(raw_action)

    if mode == "residual":
        np.testing.assert_allclose(
            captured[0], np.array([0.1, -0.2, 0.3, 0.4]) + scaled
        )
        expected_reward = -(0.001 + 0.25) * float(clipped @ clipped)
        assert reward == pytest.approx(expected_reward)
    else:
        np.testing.assert_allclose(
            captured[0], scaled + np.array([0.0, 0.0, 0.0, MASS * GRAV])
        )
        assert reward == pytest.approx(0.0)

    # Payload gravity is handled by the engine COM, never an extra pure torque.
    np.testing.assert_allclose(
        env.data.xfrc_applied[0, 3:6],
        np.zeros(3),
    )
    assert observation.shape == (15,)
    assert terminated is False
    assert truncated is False
    np.testing.assert_array_equal(env._prev_action, clipped)
    assert env._prev_action.dtype == np.float32


@pytest.mark.parametrize(("mode", "expected_pid_calls"), [("residual", 5), ("e2e", 0)])
def test_pid_updates_each_physics_substep_while_policy_action_is_held(
    mode: str, expected_pid_calls: int, monkeypatch
) -> None:
    physics_steps: list[int] = []
    monkeypatch.setattr(
        env_module,
        "mujoco",
        SimpleNamespace(mj_step=lambda *_args: physics_steps.append(1)),
    )
    state = (
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
    )
    controls: list[np.ndarray] = []
    env = _step_only_env(mode, state, controls)
    env._com_mw = 0.0
    env._com_off3[:] = 0.0
    env.substeps = 5
    pid_calls: list[int] = []

    def pid(*_args):
        pid_calls.append(1)
        return np.array([0.1, -0.2, 0.3, 0.4])

    env.pid = pid
    env.step(np.array([0.25, -0.5, 0.75, 1.0]))

    assert len(pid_calls) == expected_pid_calls
    assert len(physics_steps) == 5
    assert len(controls) == 5
    for control in controls[1:]:
        np.testing.assert_array_equal(control, controls[0])


def test_full_reward_and_strict_termination_boundaries(monkeypatch) -> None:
    monkeypatch.setattr(
        env_module, "mujoco", SimpleNamespace(mj_step=lambda *_args: None)
    )
    angle = np.deg2rad(20.0)
    quaternion = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    state = (
        np.array([0.1, -0.2, 0.2]),  # min altitude is a non-crashing strict boundary
        quaternion,
        np.array([0.3, -0.4, 0.0]),
        np.array([0.1, 0.2, -0.3]),
    )
    env = _step_only_env("residual", state, [])
    env._com_mw = 0.0
    env._com_off3[:] = 0.0
    env.w_dact = 0.0

    _obs, reward, terminated, truncated, _ = env.step(np.zeros(4))

    position_error = state[0] - env.pos_des
    expected_cost = (
        3.0 * float(position_error @ position_error)
        + 0.01 * float(state[2] @ state[2])
        + 3.0 * 2.0 * float(quaternion[1] ** 2 + quaternion[2] ** 2)
        + 0.001 * float(state[3] @ state[3])
    )
    assert reward == pytest.approx(-expected_cost)
    assert terminated is False
    assert truncated is False

    crash_state = list(state)
    crash_state[0] = np.array([0.1, -0.2, 0.199999])
    env = _step_only_env("residual", tuple(crash_state), [])
    env._com_mw = 0.0
    env._com_off3[:] = 0.0
    _obs, crash_reward, crashed, _truncated, _ = env.step(np.zeros(4))
    crash_error = crash_state[0] - env.pos_des
    crash_cost = (
        3.0 * float(crash_error @ crash_error)
        + 0.01 * float(state[2] @ state[2])
        + 3.0 * 2.0 * float(quaternion[1] ** 2)
        + 0.001 * float(state[3] @ state[3])
    )
    assert crashed is True
    assert crash_reward == pytest.approx(-crash_cost - 10.0)


def test_wrapped_yaw_error_is_squared_in_reward(monkeypatch) -> None:
    monkeypatch.setattr(
        env_module, "mujoco", SimpleNamespace(mj_step=lambda *_args: None)
    )
    # 3.5 rad wraps to 3.5 - 2*pi, and must not affect the tilt term.
    yaw = 3.5
    state = (
        np.array([0.0, 0.0, 1.0]),
        np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]),
        np.zeros(3),
        np.zeros(3),
    )
    env = _step_only_env("residual", state, [])
    env._com_mw = 0.0
    env._com_off3[:] = 0.0
    env.w_dact = 0.0

    _obs, reward, terminated, _truncated, _ = env.step(np.zeros(4))

    wrapped = np.arctan2(np.sin(yaw), np.cos(yaw))
    assert reward == pytest.approx(-(wrapped**2))
    assert terminated is False


def test_time_limit_truncates_on_the_exact_max_step(monkeypatch) -> None:
    monkeypatch.setattr(
        env_module, "mujoco", SimpleNamespace(mj_step=lambda *_args: None)
    )
    state = (
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
    )
    env = _step_only_env("e2e", state, [])
    env._com_mw = 0.0
    env._com_off3[:] = 0.0
    env.max_steps = 1

    _obs, _reward, terminated, truncated, _ = env.step(np.zeros(4))

    assert terminated is False
    assert truncated is True


def test_motor_allocation_order_and_signs() -> None:
    allocation, inverse = build_allocation_matrix()
    expected = np.array(
        [
            [-ARM, -ARM, +ARM, +ARM],
            [-ARM, +ARM, +ARM, -ARM],
            [+K_TAU, -K_TAU, +K_TAU, -K_TAU],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(allocation, expected)
    np.testing.assert_allclose(allocation @ inverse, np.eye(4), atol=1e-12)
    np.testing.assert_array_equal(MOTOR_DIR, [1.0, -1.0, 1.0, -1.0])


def test_motor_thrust_clip_and_reaction_torque_application() -> None:
    config = load_config(CONFIGS / "residual_train.yaml")
    env = _actuator_only_env(config)
    env.reset_actuator_state(airborne=False, resample_parameters=True)

    # A 2 N collective request maps to 0.5 N per motor before the preserved
    # per-motor clip, so all four desired thrust commands saturate at 0.20 N.
    # The required first-order plant then applies delayed actual force/torque.
    # The first two milliseconds remain below the deliberately conservative
    # positive branch.  Advance several 500 Hz substeps to observe force.
    for _ in range(5):
        env._apply_control(np.array([0.0, 0.0, 0.0, 2.0]))

    np.testing.assert_allclose(env._last_f_cmd, np.full(4, 0.20))
    assert np.all(env._last_f > 0.0)
    assert np.all(env._last_f < 0.20)
    np.testing.assert_allclose(env.data.ctrl[0:4], env._last_f)
    np.testing.assert_allclose(
        env.data.ctrl[4:8], MOTOR_DIR * K_TAU * env._last_f
    )


def _actuator_only_env(config, *, seed: int = 0) -> CrazyflieResidualEnv:
    """Build just enough environment state to test allocator/actuator wiring."""

    env = CrazyflieResidualEnv.__new__(CrazyflieResidualEnv)
    env.B, env.B_pinv = build_allocation_matrix(
        config.vehicle.arm_length,
        config.vehicle.motor_direction,
        config.vehicle.torque_coefficient,
    )
    env.dt_phys = 1.0 / config.vehicle.physics_hz
    env.motor_direction = np.asarray(config.vehicle.motor_direction, dtype=float)
    env.thrust_min = config.vehicle.thrust_min
    env.thrust_max = config.vehicle.thrust_max
    env.torque_coefficient = config.vehicle.torque_coefficient
    env.gravity = config.vehicle.gravity
    env._actuator_config = config.actuator
    env._actuator_reset_rpm_mode = config.actuator.reset_rpm_mode
    env._actuator_randomization = config.actuator.randomization
    env._actuator_rng = env_module._actuator_rng(seed)
    env.model = SimpleNamespace(body_mass=np.array([config.vehicle.mass]))
    env.drone_bid = 0
    env.data = SimpleNamespace(
        ctrl=np.zeros(8),
        qpos=np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    env.act_force = [0, 1, 2, 3]
    env.act_torque = [4, 5, 6, 7]
    env._actuator = env._make_actuator_model(config.actuator)
    env.actuator_model = env._actuator
    env._record_actuator_output(env._actuator.last_output, np.zeros(4))
    return env


def test_default_environment_always_uses_the_bldc_actuator_path() -> None:
    config = load_config(CONFIGS / "residual_train.yaml")
    env = _actuator_only_env(config)
    env.reset_actuator_state(airborne=False, resample_parameters=True)

    for _ in range(5):
        env._apply_control(np.array([0.0, 0.0, 0.0, 2.0]))

    assert isinstance(env.actuator_model, Cf21bFirstOrderActuatorModel)
    np.testing.assert_allclose(env._last_f_cmd, np.full(4, 0.20))
    assert np.all(env._last_f >= 0.0)
    assert np.all(env._last_f < env._last_f_cmd)
    np.testing.assert_allclose(env.data.ctrl[0:4], env._last_f)
    np.testing.assert_allclose(
        env.data.ctrl[4:8], MOTOR_DIR * K_TAU * env._last_f
    )
    assert np.all(np.isfinite(env._last_motor_cmd))
    assert np.all(np.isfinite(env._last_omega))
    assert env.actuator_snapshot()["enabled"] is True


def test_required_bldc_applies_delayed_actual_force_and_reports_wrenches() -> None:
    config = load_config(CONFIGS / "residual_train.yaml")
    env = _actuator_only_env(config)
    env.reset_actuator_state(airborne=False, resample_parameters=True)
    requested_wrench = np.array([0.0, 0.0, 0.0, config.vehicle.mass * config.vehicle.gravity])

    for _ in range(5):
        env._apply_control(requested_wrench)

    assert isinstance(env.actuator_model, Cf21bFirstOrderActuatorModel)
    np.testing.assert_allclose(
        env._last_f_cmd,
        np.full(4, config.vehicle.mass * config.vehicle.gravity / 4.0),
    )
    assert np.all(env._last_f >= 0.0)
    assert np.all(env._last_f < env._last_f_cmd)
    np.testing.assert_allclose(env.data.ctrl[0:4], env._last_f)
    np.testing.assert_allclose(env.data.ctrl[4:8], env._last_q_actual)
    np.testing.assert_allclose(env._last_wrench_cmd, requested_wrench)
    assert env._last_wrench_actual[3] < requested_wrench[3]
    np.testing.assert_allclose(
        env._last_allocation_error,
        env._last_wrench_cmd - env._last_wrench_actual,
    )


def test_no_config_environment_path_uses_required_bldc_defaults() -> None:
    config = load_config(CONFIGS / "residual_train.yaml")
    env = _actuator_only_env(config)

    model = env._make_actuator_model(None)

    assert isinstance(model, Cf21bFirstOrderActuatorModel)
    assert model.nominal_parameters.time_constant_s.tolist() == [0.050] * 4
    assert model.nominal_parameters.steady_state_gain_rad_s.tolist() == [2900.0] * 4


def test_control_refuses_a_missing_required_actuator() -> None:
    env = CrazyflieResidualEnv.__new__(CrazyflieResidualEnv)
    env.B, env.B_pinv = build_allocation_matrix()
    env.thrust_min = 0.0
    env.thrust_max = 0.20

    with pytest.raises(RuntimeError, match="required CF2.1 first-order actuator"):
        env._apply_control(np.zeros(4))


def test_bldc_reset_uses_ground_zero_or_episode_mass_hover_equilibrium() -> None:
    config = load_config(CONFIGS / "cf21b_actuator_eval.yaml")
    env = _actuator_only_env(config)

    env.reset_actuator_state(airborne=False, resample_parameters=True)
    np.testing.assert_array_equal(env._last_omega, np.zeros(4))
    np.testing.assert_array_equal(env._last_f, np.zeros(4))

    episode_mass = config.vehicle.mass + 0.010
    env.model.body_mass[env.drone_bid] = episode_mass
    env.reset_actuator_state(airborne=True, resample_parameters=True)

    assert np.sum(env._last_f) == pytest.approx(
        episode_mass * config.vehicle.gravity,
        abs=2e-12,
    )
    assert np.all(env._last_omega > 0.0)


def test_bldc_randomization_uses_a_reproducible_dedicated_rng_stream() -> None:
    config = load_config(CONFIGS / "cf21b_actuator_eval.yaml")
    randomized = replace(
        config.actuator,
        randomization=replace(config.actuator.randomization, enabled=True),
    )
    config = replace(config, actuator=randomized)
    first = _actuator_only_env(config, seed=31)
    second = _actuator_only_env(config, seed=31)

    first.reset_actuator_state(airborne=True, resample_parameters=True)
    second.reset_actuator_state(airborne=True, resample_parameters=True)

    first_snapshot = first.actuator_snapshot()
    second_snapshot = second.actuator_snapshot()
    assert first_snapshot["sampled_time_constant_s"] == second_snapshot[
        "sampled_time_constant_s"
    ]
    assert first_snapshot["sampled_steady_state_gain_rad_s"] == second_snapshot[
        "sampled_steady_state_gain_rad_s"
    ]
    assert first_snapshot["sampled_time_constant_s"] != [0.050] * 4


def test_pid_hover_and_integrator_contract() -> None:
    pid = CascadePID(1.0 / 500.0)
    hover = pid(
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        0.0,
    )
    np.testing.assert_allclose(hover, [0.0, 0.0, 0.0, MASS * GRAV], atol=1e-15)

    for _ in range(2000):
        pid(
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([-100.0, -100.0, -100.0]),
            np.zeros(3),
            np.ones(3),
        )
    np.testing.assert_array_equal(pid._i_vel, np.full(3, 2.0))
    # Master accumulated the rate integral without anti-windup; ki_rate is zero.
    assert np.all(np.isfinite(pid._i_rate))


def test_com_mass_offset_and_full_inertia_update() -> None:
    env = CrazyflieResidualEnv(config=load_config(CONFIGS/'e2e_train.yaml'))
    env._m0 = MASS
    env._ipos0 = np.array([0.01, -0.02, 0.0])
    env._J0 = np.array([2.0e-5, 3.0e-5, 4.0e-5])
    payload_mass = 0.01
    offset = np.array([0.03, -0.04])

    env._set_com_bias(payload_mass, offset)

    total = MASS + payload_mass
    reduced = MASS * payload_mass / total
    expected_ipos = (
        MASS * env._ipos0 + payload_mass * np.array([0.03, -0.04, 0.0])
    ) / total
    delta=np.r_[offset,0]-env._ipos0
    expected_inertia = np.diag(env._J0)+reduced*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
    b=env.drone_bid
    rotation=env_module.rotmat_from_quat_wxyz(env.model.body_iquat[b])
    assert env.model.body_mass[b] == pytest.approx(total)
    np.testing.assert_allclose(env.model.body_ipos[b], expected_ipos)
    np.testing.assert_allclose(rotation@np.diag(env.model.body_inertia[b])@rotation.T, expected_inertia,atol=1e-16)
    env.close()


def test_payload_randomization_preserves_master_rng_draw_order(monkeypatch) -> None:
    fake_mujoco = SimpleNamespace(
        mj_resetData=lambda *_args: None,
        mj_forward=lambda *_args: None,
    )
    monkeypatch.setattr(env_module, "mujoco", fake_mujoco)
    env = _observation_only_env("residual")
    env.model = SimpleNamespace()
    env.data = SimpleNamespace(
        xfrc_applied=np.zeros((1, 6)),
        qpos=np.zeros(7),
        qvel=np.zeros(6),
    )
    env.com_bias_randomize = True
    env.com_bias_mass = 0.0
    env.com_bias_offset = np.zeros(2)
    env.random_radius_min = 0.02
    env.random_radius_max = 0.10
    env.random_torque_fraction = 0.5
    env.random_mass_max = 0.015
    env.tau_max_rp = ARM * (2 * (2 * 0.20) - MASS * GRAV)
    env.gravity = GRAV
    env.pos_perturb = 0.15
    env.att_perturb_deg = 5.0
    env.pid = SimpleNamespace(reset=lambda: None)
    env._step = 9
    env._prev_action = np.ones(4)
    captured: dict[str, np.ndarray | float] = {}
    env._set_com_bias = lambda mass, offset: captured.update(
        mass=float(mass), offset=np.asarray(offset).copy()
    )
    env._read_state = lambda: (
        env.data.qpos[0:3].copy(),
        env.data.qpos[3:7].copy(),
        np.zeros(3),
        np.zeros(3),
    )
    # This fixture isolates the historical payload/pose RNG draw order; it
    # deliberately does not construct a MuJoCo actuator environment.
    env.reset_actuator_state = lambda **_kwargs: {}

    observation, _ = env.reset(seed=123)

    expected_rng = np.random.default_rng(123)
    theta = expected_rng.uniform(0.0, 2.0 * np.pi)
    radius = expected_rng.uniform(0.02, 0.10)
    torque = expected_rng.uniform(0.0, 0.5 * env.tau_max_rp)
    expected_mass = min(torque / (radius * GRAV), 0.015)
    expected_offset = np.array([radius * np.cos(theta), radius * np.sin(theta)])
    expected_position = env.pos_des + expected_rng.uniform(-0.15, 0.15, 3)
    angle = np.radians(5.0) * expected_rng.uniform(0.0, 1.0)
    axis = expected_rng.normal(size=3)
    axis[2] = 0.0
    axis /= np.linalg.norm(axis) + 1e-9
    expected_quaternion = np.concatenate(
        [[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis]
    )

    assert captured["mass"] == pytest.approx(expected_mass)
    np.testing.assert_allclose(captured["offset"], expected_offset)
    np.testing.assert_allclose(env.data.qpos[0:3], expected_position)
    np.testing.assert_allclose(env.data.qpos[3:7], expected_quaternion)
    assert observation.shape == (15,)
    assert env._step == 0
    np.testing.assert_array_equal(env._prev_action, np.zeros(4))


def test_environment_factory_uses_config_seed_and_explicit_overrides(monkeypatch) -> None:
    config = load_config(CONFIGS / "e2e_train.yaml")
    seeded_training = replace(config.training, seed=42)
    config = replace(config, training=seeded_training)
    captured: dict[str, object] = {}

    class CapturingEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(factory_module, "CrazyflieResidualEnv", CapturingEnv)
    factory = EnvironmentFactory(config)
    result = factory.make(mode="residual", episode_sec=12.0)

    assert isinstance(result, CapturingEnv)
    assert captured["config"] is config
    assert captured["seed"] == 42
    assert captured["mode"] == "residual"
    assert captured["episode_sec"] == 12.0

    factory.make(seed=7)
    assert captured["seed"] == 7

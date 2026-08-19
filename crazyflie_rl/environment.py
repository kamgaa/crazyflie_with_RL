"""Gymnasium environment for PID-residual and end-to-end Crazyflie control.

The environment preserves the executable master-branch contracts, including
the 15-value observation in *both* modes, 500 Hz PID updates, motor allocation,
payload torque injection, reward, and reset random-number draw order.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .actuators import Cf21bFirstOrderActuatorModel
from .controllers import (
    ARM,
    GRAV,
    J_DIAG,
    K_TAU,
    MASS,
    MOTOR_DIR,
    PHYSICS_HZ,
    TAU_MAX_RP,
    THRUST_MAX,
    THRUST_MIN,
    CascadePID,
    _build_B,
    build_allocation_matrix,
    quat_normalize_wxyz,
    rotmat_from_quat_wxyz,
)

if TYPE_CHECKING:
    from .config import ExperimentConfig


try:  # Importing the module remains safe on config-only/test machines.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - depends on the execution image
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

try:
    import mujoco
except ImportError:  # pragma: no cover - depends on the execution image
    mujoco = None  # type: ignore[assignment]


_GymEnv = gym.Env if gym is not None else object

DEFAULT_RESIDUAL_SCALE = (0.022, 0.022, 0.0001, 0.3)
OBSERVATION_DIM = 15
ACTION_DIM = 4
CONTROL_MODES = frozenset({"residual", "e2e"})
_ACTUATOR_SEED_SALT = 0xCF21B


def _selected(explicit: Any, configured: Any, legacy: Any) -> Any:
    if explicit is not None:
        return explicit
    if configured is not None:
        return configured
    return legacy


def _finite_scalar(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector with shape ({length},)") from exc
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} values must all be finite")
    return result.copy()


def _validated_action(value: Any) -> np.ndarray:
    """Validate an action without changing its numeric dtype.

    Stable-Baselines3 supplies float32 actions.  Master kept that dtype for the
    normalized action penalty, so coercing to float64 here would introduce a
    small but avoidable reward difference.
    """

    try:
        result = np.asarray(value)
        finite = np.isfinite(result)
    except (TypeError, ValueError) as exc:
        raise ValueError("action must be a finite numeric vector with shape (4,)") from exc
    if result.shape != (ACTION_DIM,):
        raise ValueError(f"action must have shape (4,), got {result.shape}")
    if not np.issubdtype(result.dtype, np.number) or not np.all(finite):
        raise ValueError("action values must all be finite numbers")
    return result.copy()


def _actuator_rng(seed: int | None) -> np.random.Generator:
    """Create an RNG stream independent from the preserved reset RNG.

    Payload and pose sampling intentionally continue to use ``self._rng`` in
    their historical order.  Opt-in actuator randomisation receives this
    separately derived stream instead, so a fixed reset seed remains fully
    reproducible without perturbing any legacy draws.
    """

    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([int(seed), _ACTUATOR_SEED_SALT]))


def _absolute_xml_path(value: Any) -> str:
    if value is None:
        raise ValueError("xml_path is required when no ExperimentConfig is supplied")
    text = str(value)
    # A POSIX server path is intentionally accepted unchanged even when config
    # validation is run on Windows, where pathlib does not regard it as native.
    if not text or not (Path(text).is_absolute() or text.startswith("/")):
        raise ValueError(
            "xml_path must be an absolute path; relative/fallback XML lookup is forbidden"
        )
    return text


def _require_runtime_dependencies() -> None:
    missing: list[str] = []
    if gym is None or spaces is None:
        missing.append("gymnasium")
    if mujoco is None:
        missing.append("mujoco")
    if missing:
        raise RuntimeError(
            "CrazyflieResidualEnv requires runtime package(s): " + ", ".join(missing)
        )


class CrazyflieResidualEnv(_GymEnv):
    """MuJoCo Crazyflie environment with preserved residual/E2E behavior."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        policy_hz: float | None = None,
        episode_sec: float | None = None,
        residual_scale: Sequence[float] | None = None,
        mode: str | None = None,
        com_bias_mass: float | None = None,
        com_bias_offset: Sequence[float] | None = None,
        com_bias_randomize: bool | None = None,
        pos_perturb: float | None = None,
        att_perturb_deg: float | None = None,
        seed: int | None = None,
        *,
        config: ExperimentConfig | None = None,
        position_target: Sequence[float] | None = None,
        yaw_target: float | None = None,
    ):
        if gym is not None:
            super().__init__()

        configured_vehicle = config.vehicle if config is not None else None
        configured_environment = config.environment if config is not None else None
        configured_actuator = config.actuator if config is not None else None
        configured_payload = (
            configured_environment.payload if configured_environment is not None else None
        )

        self.mode = str(
            _selected(
                mode,
                getattr(configured_environment, "control_mode", None),
                "residual",
            )
        )
        if self.mode not in CONTROL_MODES:
            allowed = ", ".join(sorted(CONTROL_MODES))
            raise ValueError(f"mode must be one of {{{allowed}}}, got {self.mode!r}")

        self.residual_scale = _finite_vector(
            _selected(
                residual_scale,
                getattr(configured_environment, "residual_scale", None),
                DEFAULT_RESIDUAL_SCALE,
            ),
            ACTION_DIM,
            "residual_scale",
        )

        self.physics_hz = _finite_scalar(
            getattr(configured_vehicle, "physics_hz", PHYSICS_HZ),
            "physics_hz",
            positive=True,
        )
        resolved_policy_hz = _finite_scalar(
            _selected(
                policy_hz,
                getattr(configured_environment, "policy_hz", None),
                100.0,
            ),
            "policy_hz",
            positive=True,
        )
        resolved_episode_sec = _finite_scalar(
            _selected(
                episode_sec,
                getattr(configured_environment, "episode_sec", None),
                8.0,
            ),
            "episode_sec",
            positive=True,
        )
        self.substeps = int(round(self.physics_hz / resolved_policy_hz))
        if self.substeps < 1:
            raise ValueError(
                "policy_hz is too high for physics_hz: rounded substeps must be >= 1"
            )
        self.max_steps = int(round(resolved_episode_sec * resolved_policy_hz))
        if self.max_steps < 1:
            raise ValueError("episode_sec * policy_hz must yield at least one policy step")
        self.policy_hz = resolved_policy_hz
        self.episode_sec = resolved_episode_sec
        self.dt_phys = 1.0 / self.physics_hz

        self.mass = _finite_scalar(
            getattr(configured_vehicle, "mass", MASS), "mass", positive=True
        )
        self.gravity = _finite_scalar(
            getattr(configured_vehicle, "gravity", GRAV), "gravity", positive=True
        )
        self.arm_length = _finite_scalar(
            getattr(configured_vehicle, "arm_length", ARM),
            "arm_length",
            positive=True,
        )
        self.torque_coefficient = _finite_scalar(
            getattr(configured_vehicle, "torque_coefficient", K_TAU),
            "torque_coefficient",
            positive=True,
        )
        self.motor_direction = _finite_vector(
            getattr(configured_vehicle, "motor_direction", MOTOR_DIR),
            ACTION_DIM,
            "motor_direction",
        )
        self.thrust_min = _finite_scalar(
            getattr(configured_vehicle, "thrust_min", THRUST_MIN), "thrust_min"
        )
        self.thrust_max = _finite_scalar(
            getattr(configured_vehicle, "thrust_max", THRUST_MAX),
            "thrust_max",
            positive=True,
        )
        if self.thrust_max <= self.thrust_min:
            raise ValueError("thrust_max must exceed thrust_min")
        self.inertia_diagonal = _finite_vector(
            getattr(configured_vehicle, "inertia_diagonal", J_DIAG),
            3,
            "inertia_diagonal",
        )
        self.tau_max_rp = self.arm_length * (
            2.0 * (2.0 * self.thrust_max) - self.mass * self.gravity
        )

        self.com_bias_mass = _finite_scalar(
            _selected(
                com_bias_mass, getattr(configured_payload, "mass", None), 0.0
            ),
            "com_bias_mass",
        )
        if self.com_bias_mass < 0.0:
            raise ValueError("com_bias_mass must be non-negative")
        self.com_bias_offset = _finite_vector(
            _selected(
                com_bias_offset,
                getattr(configured_payload, "offset", None),
                (0.0, 0.0),
            ),
            2,
            "com_bias_offset",
        )
        resolved_randomize = _selected(
            com_bias_randomize,
            getattr(configured_payload, "randomize", None),
            False,
        )
        if not isinstance(resolved_randomize, (bool, np.bool_)):
            raise ValueError("com_bias_randomize must be a boolean")
        self.com_bias_randomize = bool(resolved_randomize)
        self.pos_perturb = _finite_scalar(
            _selected(
                pos_perturb,
                getattr(configured_environment, "position_perturbation", None),
                0.15,
            ),
            "pos_perturb",
        )
        self.att_perturb_deg = _finite_scalar(
            _selected(
                att_perturb_deg,
                getattr(configured_environment, "attitude_perturbation_deg", None),
                5.0,
            ),
            "att_perturb_deg",
        )
        if self.pos_perturb < 0.0 or self.att_perturb_deg < 0.0:
            raise ValueError("reset perturbation limits must be non-negative")

        self.pos_des = _finite_vector(
            _selected(
                position_target,
                getattr(configured_environment, "position_target", None),
                (0.0, 0.0, 1.0),
            ),
            3,
            "position_target",
        )
        self.yaw_des = _finite_scalar(
            _selected(
                yaw_target,
                getattr(configured_environment, "yaw_target", None),
                0.0,
            ),
            "yaw_target",
        )

        random_limits = getattr(configured_payload, "randomization_limits", None)
        self.random_radius_min = _finite_scalar(
            getattr(random_limits, "radius_min", 0.02),
            "payload radius_min",
            positive=True,
        )
        self.random_radius_max = _finite_scalar(
            getattr(random_limits, "radius_max", 0.10),
            "payload radius_max",
            positive=True,
        )
        self.random_torque_fraction = _finite_scalar(
            getattr(random_limits, "torque_fraction", 0.5),
            "payload torque_fraction",
        )
        self.random_mass_max = _finite_scalar(
            getattr(random_limits, "mass_max", 0.015), "payload mass_max"
        )
        if self.random_radius_max < self.random_radius_min:
            raise ValueError("payload radius_max must be >= radius_min")
        if self.random_torque_fraction < 0.0 or self.random_mass_max < 0.0:
            raise ValueError("payload torque_fraction and mass_max must be non-negative")

        reward = getattr(configured_environment, "reward", None)
        self.position_weight = _finite_scalar(
            getattr(reward, "position_weight", 3.0), "position_weight"
        )
        self.velocity_weight = _finite_scalar(
            getattr(reward, "velocity_weight", 0.01), "velocity_weight"
        )
        self.tilt_weight = _finite_scalar(
            getattr(reward, "tilt_weight", 3.0), "tilt_weight"
        )
        self.angular_velocity_weight = _finite_scalar(
            getattr(reward, "angular_velocity_weight", 0.001),
            "angular_velocity_weight",
        )
        self.yaw_weight = _finite_scalar(
            getattr(reward, "yaw_weight", 1.0), "yaw_weight"
        )
        self.action_weight = _finite_scalar(
            getattr(reward, "action_weight", 0.001), "action_weight"
        )
        self.w_dact = _finite_scalar(
            getattr(reward, "action_rate_weight", 0.0), "action_rate_weight"
        )
        self.crash_penalty = _finite_scalar(
            getattr(reward, "crash_penalty", 10.0), "crash_penalty"
        )
        if min(
            self.position_weight,
            self.velocity_weight,
            self.tilt_weight,
            self.angular_velocity_weight,
            self.yaw_weight,
            self.action_weight,
            self.w_dact,
            self.crash_penalty,
        ) < 0.0:
            raise ValueError("reward weights and crash_penalty must be non-negative")

        termination = getattr(configured_environment, "termination", None)
        self.min_altitude = _finite_scalar(
            getattr(termination, "min_altitude", 0.2), "min_altitude"
        )
        self.max_altitude = _finite_scalar(
            getattr(termination, "max_altitude", 2.5), "max_altitude"
        )
        self.max_termination_tilt = np.deg2rad(
            _finite_scalar(
                getattr(termination, "max_tilt_deg", 60.0),
                "termination max_tilt_deg",
                positive=True,
            )
        )
        self.max_position_error = _finite_scalar(
            getattr(termination, "max_position_error", 1.5),
            "max_position_error",
            positive=True,
        )
        if self.max_altitude <= self.min_altitude:
            raise ValueError("max_altitude must exceed min_altitude")

        configured_xml = config.paths.mujoco_xml if config is not None else None
        self.xml_path = _absolute_xml_path(_selected(xml_path, configured_xml, None))

        # All validation above intentionally runs before importing/creating a
        # simulator resource, so invalid profiles fail quickly and cleanly.
        _require_runtime_dependencies()
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.dt_phys

        self.B, self.B_pinv = build_allocation_matrix(
            self.arm_length, self.motor_direction, self.torque_coefficient
        )
        self.drone_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "drone"
        )
        self.gyro_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
        )
        self.act_force = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_force"
            )
            for i in range(4)
        ]
        self.act_torque = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_torque"
            )
            for i in range(4)
        ]

        # The allocator remains unchanged: it still produces clipped desired
        # motor thrust.  The mandatory BLDC model determines what plant
        # force/torque reaches MuJoCo from that command.
        self._actuator_config = configured_actuator
        self._actuator_reset_rpm_mode = str(
            getattr(configured_actuator, "reset_rpm_mode", "auto")
        )
        self._actuator_randomization = getattr(
            configured_actuator, "randomization", None
        )
        self._actuator_rng = _actuator_rng(seed)
        self._actuator = self._make_actuator_model(configured_actuator)
        # Public read-only-by-convention handle for simulator diagnostics.
        self.actuator_model = self._actuator
        self._record_actuator_output(
            self._actuator.last_output,
            np.zeros(ACTION_DIM, dtype=float),
        )

        self._m0 = float(self.model.body_mass[self.drone_bid])
        self._ipos0 = self.model.body_ipos[self.drone_bid].copy()
        self._J0 = self.model.body_inertia[self.drone_bid].copy()

        self.pid = CascadePID(self.dt_phys, config=config)
        self.dist_torque_body = np.zeros(3)
        self._com_off3 = np.zeros(3)
        self._com_mw = 0.0

        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        high = np.full(OBSERVATION_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._prev_action = np.zeros(ACTION_DIM)

    def _set_com_bias(self, m_w: float, off_xy: Sequence[float]) -> None:
        mass = float(m_w)
        offset = np.asarray(off_xy, dtype=float)
        r_w = np.array([offset[0], offset[1], 0.0])
        self._com_off3 = r_w.copy()
        self._com_mw = mass
        total_mass = self._m0 + mass
        if total_mass <= 0.0:
            raise ValueError("base mass plus payload mass must be positive")
        new_ipos = (self._m0 * self._ipos0 + mass * r_w) / total_mass
        self.model.body_mass[self.drone_bid] = total_mass
        self.model.body_ipos[self.drone_bid] = new_ipos

        reduced_mass = (self._m0 * mass) / total_mass
        x, y = r_w[0], r_w[1]
        self.model.body_inertia[self.drone_bid] = self._J0 + reduced_mass * np.array(
            [y * y, x * x, x * x + y * y]
        )

    def _make_actuator_model(self, configured_actuator: Any | None):
        """Build a plant actuator without changing the preserved allocator."""

        # A normal runtime environment always has a validated config.  The
        # no-config legacy constructor nevertheless receives the same BLDC
        # plant defaults so no executable environment silently falls back to
        # an ideal instantaneous motor path.
        enabled = (
            True
            if configured_actuator is None
            else bool(getattr(configured_actuator, "enabled", True))
        )
        model_name = str(
            getattr(configured_actuator, "model", "cf21b_first_order")
        )
        if enabled and model_name == "cf21b_first_order":
            reaction = getattr(configured_actuator, "reaction_torque", None)
            return Cf21bFirstOrderActuatorModel(
                dt=self.dt_phys,
                motor_direction=self.motor_direction,
                thrust_min=self.thrust_min,
                thrust_max=self.thrust_max,
                time_constant_s=getattr(
                    configured_actuator, "time_constant_s", 0.050
                ),
                steady_state_gain_rad_s=getattr(
                    configured_actuator, "steady_state_gain_rad_s", 2900.0
                ),
                thrust_polynomial_coefficients=getattr(
                    configured_actuator,
                    "thrust_polynomial_coefficients",
                    (-0.23, 0.562, -0.043),
                ),
                omega_reference_rad_s=getattr(
                    configured_actuator,
                    "thrust_polynomial_omega_reference_rad_s",
                    2900.0,
                ),
                positive_branch_min_ratio=getattr(
                    configured_actuator,
                    "thrust_polynomial_positive_branch_min_ratio",
                    0.0791,
                ),
                max_ratio=getattr(
                    configured_actuator, "thrust_polynomial_max_ratio", 1.0
                ),
                reaction_torque_model=getattr(reaction, "model", "legacy_ratio"),
                legacy_ratio_m=getattr(
                    reaction, "legacy_ratio_m", self.torque_coefficient
                ),
                torque_polynomial_coefficients=getattr(
                    reaction, "polynomial_coefficients", (-3.4, 8.7, 2.9)
                ),
                torque_polynomial_scale=getattr(reaction, "polynomial_scale", 1e-4),
                rotor_inertia_kg_m2=getattr(
                    reaction, "rotor_inertia_kg_m2", 0.5e-7
                ),
                include_rotor_acceleration_torque=getattr(
                    reaction, "include_rotor_acceleration_torque", False
                ),
                allocation_matrix=self.B,
            )

        raise ValueError(
            "the runtime actuator must be enabled with model "
            "'cf21b_first_order'"
        )

    def _record_actuator_output(
        self,
        output: Any,
        wrench_command: Sequence[float] | None = None,
    ) -> None:
        """Mirror command/plant actuator quantities under stable env names."""

        f_cmd = np.asarray(output.f_cmd, dtype=float).reshape(ACTION_DIM).copy()
        f_actual = np.asarray(output.f_actual, dtype=float).reshape(ACTION_DIM).copy()
        q_actual = np.asarray(output.q_actual, dtype=float).reshape(ACTION_DIM).copy()
        if wrench_command is None:
            requested_wrench = self.B @ f_cmd
        else:
            requested_wrench = np.asarray(wrench_command, dtype=float).reshape(
                ACTION_DIM
            )
        achieved_wrench = self.B @ f_actual
        # The allocation matrix contains the legacy constant yaw ratio.  A
        # polynomial plant torque therefore needs its measured yaw component
        # written explicitly rather than silently reporting the allocator's.
        achieved_wrench[2] = float(np.sum(q_actual))

        self._last_f_cmd = f_cmd
        self._last_f = f_actual
        self._last_motor_cmd = np.asarray(
            output.motor_command, dtype=float
        ).reshape(ACTION_DIM).copy()
        self._last_omega = np.asarray(output.omega, dtype=float).reshape(
            ACTION_DIM
        ).copy()
        self._last_q_actual = q_actual
        self._last_wrench_cmd = requested_wrench.copy()
        self._last_wrench_allocated = self.B @ f_cmd
        self._last_wrench_actual = achieved_wrench.copy()
        self._last_allocation_error = (
            self._last_wrench_cmd - self._last_wrench_actual
        )

    def _write_applied_motor_controls(self) -> None:
        """Write the last plant force/torque outputs to MuJoCo controls."""

        for index in range(ACTION_DIM):
            self.data.ctrl[self.act_force[index]] = float(self._last_f[index])
        for index in range(ACTION_DIM):
            self.data.ctrl[self.act_torque[index]] = float(
                self._last_q_actual[index]
            )

    def _auto_actuator_airborne(self) -> bool:
        """Classify the already-initialised MuJoCo reset pose for rotor reset."""

        return bool(float(self.data.qpos[2]) > 0.1)

    def reset_actuator_state(
        self,
        airborne: bool | None = None,
        *,
        resample_parameters: bool = False,
    ) -> dict[str, Any]:
        """Reset mandatory motor-model state without changing the vehicle API.

        ``resample_parameters`` is deliberately restricted to the regular
        environment reset.  ``view_live`` can change an already reset pose to
        the floor and call this method with ``airborne=False`` without drawing
        a second set of per-episode parameters.
        """

        model = getattr(self, "_actuator", None)
        if model is None:
            raise RuntimeError("the required CF2.1 first-order actuator is unavailable")
        if airborne is None:
            if self._actuator_reset_rpm_mode == "zero":
                airborne = False
            elif self._actuator_reset_rpm_mode == "hover_equilibrium":
                airborne = True
            else:
                airborne = self._auto_actuator_airborne()

        if not isinstance(model, Cf21bFirstOrderActuatorModel):
            raise RuntimeError("the required CF2.1 first-order actuator is unavailable")

        randomization = self._actuator_randomization
        randomize = bool(
            resample_parameters
            and getattr(randomization, "enabled", False)
        )
        if resample_parameters and not randomize:
            model.restore_nominal_parameters()
        output = model.reset(
            airborne=bool(airborne),
            episode_mass=float(self.model.body_mass[self.drone_bid]),
            gravity_m_s2=self.gravity,
            rng=self._actuator_rng if randomize else None,
            randomize=randomize,
            time_constant_range=(
                getattr(
                    getattr(randomization, "time_constant_s", None),
                    "min",
                    0.040,
                ),
                getattr(
                    getattr(randomization, "time_constant_s", None),
                    "max",
                    0.060,
                ),
            ),
            steady_state_gain_range=(
                getattr(
                    getattr(randomization, "steady_state_gain_rad_s", None),
                    "min",
                    2320.0,
                ),
                getattr(
                    getattr(randomization, "steady_state_gain_rad_s", None),
                    "max",
                    3480.0,
                ),
            ),
        )
        self._record_actuator_output(output)
        # Reset is normally followed by a control update before physics
        # advances, but writing here also makes the public helper coherent if
        # a caller explicitly switches a running simulation to ground/air.
        self._write_applied_motor_controls()
        return self.actuator_snapshot()

    def actuator_snapshot(self) -> dict[str, Any]:
        """Return JSON-safe nominal/sampled actuator provenance and state."""

        model = getattr(self, "_actuator", None)
        configured = getattr(self, "_actuator_config", None)
        reaction = getattr(configured, "reaction_torque", None)
        result: dict[str, Any] = {
            "enabled": isinstance(model, Cf21bFirstOrderActuatorModel),
            "model": (
                "cf21b_first_order"
                if isinstance(model, Cf21bFirstOrderActuatorModel)
                else "unavailable"
            ),
            "parameter_source": getattr(
                configured, "parameter_source", "paper_candidate"
            ),
            "verification_status": getattr(
                configured, "verification_status", "unverified"
            ),
            "thrust_saturation_n": [float(self.thrust_min), float(self.thrust_max)],
            "motor_command_saturation": [0.0, 1.0],
            "randomization_enabled": bool(
                getattr(self._actuator_randomization, "enabled", False)
            ),
            "reset_rpm_mode": getattr(
                configured, "reset_rpm_mode", "auto"
            ),
            "reaction_torque_model": getattr(reaction, "model", "legacy_ratio"),
            "legacy_ratio_m": float(
                getattr(model, "legacy_ratio_m", self.torque_coefficient)
            ),
        }
        if model is None:
            return result
        if isinstance(model, Cf21bFirstOrderActuatorModel):
            nominal = model.nominal_parameters
            sampled = model.parameters
            result.update(
                {
                    "nominal_time_constant_s": nominal.time_constant_s.tolist(),
                    "nominal_steady_state_gain_rad_s": (
                        nominal.steady_state_gain_rad_s.tolist()
                    ),
                    "sampled_time_constant_s": sampled.time_constant_s.tolist(),
                    "sampled_steady_state_gain_rad_s": (
                        sampled.steady_state_gain_rad_s.tolist()
                    ),
                    "thrust_polynomial_coefficients": (
                        model.thrust_polynomial_coefficients.tolist()
                    ),
                    "thrust_omega_reference_rad_s": float(
                        model.omega_reference_rad_s
                    ),
                    "positive_branch_min_ratio": float(
                        model.positive_branch_min_ratio
                    ),
                    "max_ratio": float(model.max_ratio),
                    "reaction_torque_coefficients": (
                        model.torque_polynomial_coefficients.tolist()
                    ),
                    "reaction_torque_scale": float(model.torque_polynomial_scale),
                    "rotor_inertia_kg_m2": float(model.rotor_inertia_kg_m2),
                    "include_rotor_acceleration_torque": bool(
                        model.include_rotor_acceleration_torque
                    ),
                }
            )
        return result

    def _apply_control(self, wrench: Sequence[float]) -> None:
        wrench_command = np.asarray(wrench, dtype=float)
        motor_thrust = self.B_pinv @ wrench_command
        motor_thrust = np.clip(motor_thrust, self.thrust_min, self.thrust_max)
        model = getattr(self, "_actuator", None)
        if not isinstance(model, Cf21bFirstOrderActuatorModel):
            raise RuntimeError("the required CF2.1 first-order actuator is unavailable")
        output = model.apply(motor_thrust)
        self._record_actuator_output(output, wrench_command)
        self._write_applied_motor_controls()

    def _read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position = self.data.qpos[0:3].copy()
        quaternion = quat_normalize_wxyz(self.data.qpos[3:7].copy())
        if quaternion[0] < 0:
            quaternion = -quaternion
        velocity = self.data.qvel[0:3].copy()
        sensor_address = self.model.sensor_adr[self.gyro_sid]
        omega_body = self.data.sensordata[
            sensor_address : sensor_address + 3
        ].copy()
        return position, quaternion, velocity, omega_body

    def _yaw_err(self, quaternion: Sequence[float]) -> float:
        quat = np.asarray(quaternion, dtype=float)
        yaw = np.arctan2(
            2 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
        )
        return float(
            np.arctan2(
                np.sin(yaw - self.yaw_des), np.cos(yaw - self.yaw_des)
            )
        )

    def _obs(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        velocity: Sequence[float],
        omega_body: Sequence[float],
    ) -> np.ndarray:
        yaw_error = self._yaw_err(quaternion)
        return np.concatenate(
            [
                np.asarray(position) - self.pos_des,
                np.asarray(velocity),
                np.asarray(quaternion),
                np.asarray(omega_body),
                [np.sin(yaw_error), np.cos(yaw_error)],
            ]
        ).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options  # Preserved API: master accepted but did not use options.
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._actuator_rng = _actuator_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0

        if self.com_bias_randomize:
            theta = self._rng.uniform(0.0, 2.0 * np.pi)
            radius = self._rng.uniform(
                self.random_radius_min, self.random_radius_max
            )
            torque_cap = self.random_torque_fraction * self.tau_max_rp
            target_torque = self._rng.uniform(0.0, torque_cap)
            payload_mass = min(
                target_torque / (radius * self.gravity), self.random_mass_max
            )
            offset = np.array(
                [radius * np.cos(theta), radius * np.sin(theta)]
            )
            self._set_com_bias(payload_mass, offset)
        else:
            self._set_com_bias(self.com_bias_mass, self.com_bias_offset)

        self.data.qpos[0:3] = self.pos_des + self._rng.uniform(
            -self.pos_perturb, self.pos_perturb, 3
        )
        angle = np.radians(self.att_perturb_deg) * self._rng.uniform(0.0, 1.0)
        axis = self._rng.normal(size=3)
        axis[2] = 0.0
        axis /= np.linalg.norm(axis) + 1e-9
        half_angle = 0.5 * angle
        self.data.qpos[3:7] = np.array(
            [
                np.cos(half_angle),
                np.sin(half_angle) * axis[0],
                np.sin(half_angle) * axis[1],
                np.sin(half_angle) * axis[2],
            ]
        )
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Normal environment resets begin around ``pos_des`` (normally an
        # airborne hover pose).  The helper applies the configured reset mode
        # and optionally samples per-episode motor *parameters* on its
        # dedicated RNG stream.  The motor model itself is always active.
        self.reset_actuator_state(resample_parameters=True)
        self.pid.reset()
        self._step = 0
        self._prev_action = np.zeros(ACTION_DIM)
        position, quaternion, velocity, omega_body = self._read_state()
        return self._obs(position, quaternion, velocity, omega_body), {}

    def step(
        self, action: Sequence[float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = _validated_action(action)
        action_array = np.clip(action_array, -1.0, 1.0)
        residual = self.residual_scale * action_array

        for _ in range(self.substeps):
            position, quaternion, velocity, omega_body = self._read_state()
            if self.mode == "residual":
                pid_wrench = self.pid(
                    position,
                    quaternion,
                    velocity,
                    omega_body,
                    self.pos_des,
                    self.yaw_des,
                )
                wrench = pid_wrench + residual
            else:
                wrench = residual + np.array(
                    [0.0, 0.0, 0.0, self.mass * self.gravity]
                )
            self._apply_control(wrench)

            rotation = rotmat_from_quat_wxyz(quaternion)
            torque_com_world = np.cross(
                rotation @ self._com_off3,
                np.array([0.0, 0.0, -self._com_mw * self.gravity]),
            )
            self.data.xfrc_applied[self.drone_bid, 3:6] = (
                rotation @ self.dist_torque_body + torque_com_world
            )
            mujoco.mj_step(self.model, self.data)

        position, quaternion, velocity, omega_body = self._read_state()
        observation = self._obs(position, quaternion, velocity, omega_body)

        position_error = position - self.pos_des
        tilt_error = 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
        yaw_error = self._yaw_err(quaternion)
        action_delta = action_array - self._prev_action
        action_weight = 0.0 if self.mode == "e2e" else self.action_weight
        action_rate_weight = 0.0 if self.mode == "e2e" else self.w_dact

        cost = (
            self.position_weight * (position_error @ position_error)
            + self.velocity_weight * (velocity @ velocity)
            + self.tilt_weight * tilt_error
            + self.angular_velocity_weight * (omega_body @ omega_body)
            + self.yaw_weight * yaw_error**2
            + action_weight * (action_array @ action_array)
            + action_rate_weight * (action_delta @ action_delta)
        )
        reward = -cost
        self._prev_action = action_array.copy()

        self._step += 1
        tilt = np.arccos(
            np.clip(
                1 - 2 * (quaternion[1] ** 2 + quaternion[2] ** 2), -1, 1
            )
        )
        crashed = (
            position[2] < self.min_altitude
            or position[2] > self.max_altitude
            or tilt > self.max_termination_tilt
            or np.linalg.norm(position_error) > self.max_position_error
        )
        truncated = self._step >= self.max_steps
        if crashed:
            reward -= self.crash_penalty
        return observation, float(reward), bool(crashed), bool(truncated), {}

    def close(self) -> None:
        # MuJoCo's Python model/data objects are released by reference counting.
        close_parent = getattr(super(), "close", None)
        if callable(close_parent):
            close_parent()


__all__ = [
    "ACTION_DIM",
    "ARM",
    "CONTROL_MODES",
    "CrazyflieResidualEnv",
    "DEFAULT_RESIDUAL_SCALE",
    "GRAV",
    "J_DIAG",
    "K_TAU",
    "MASS",
    "MOTOR_DIR",
    "OBSERVATION_DIM",
    "PHYSICS_HZ",
    "TAU_MAX_RP",
    "THRUST_MAX",
    "THRUST_MIN",
    "CascadePID",
    "_build_B",
    "quat_normalize_wxyz",
    "rotmat_from_quat_wxyz",
]

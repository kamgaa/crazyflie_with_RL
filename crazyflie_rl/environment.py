"""Gymnasium environment for PID-residual and end-to-end Crazyflie control.

The environment preserves the executable master-branch contracts, including
the 15-value observation in *both* modes, 500 Hz PID updates, motor allocation,
reward and reset random-number draw order. Payloads are rigid point masses.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

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
from .observation import (
    ACTION_DIM,
    BASE_OBSERVATION_DIM,
    CURRENT_STATE_READER,
    LEGACY_STATE_READER,
    AuxiliaryObservationState,
    observation_dimension,
    observation_schema,
)
from .payload_physics import (
    compose_point_payload,
    inertia_body,
    principal_axes,
    static_hover,
    subtree_properties,
    wrench_snapshot,
)
from .physics_version import PHYSICS_MODEL_VERSION
from .rewards import (
    REWARD_MODES,
    compute_lyapunov_reward,
    normalized_tracking_vector,
    tracking_error_from_state,
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
OBSERVATION_DIM = BASE_OBSERVATION_DIM
CONTROL_MODES = frozenset({"residual", "e2e"})
_ACTUATOR_SEED_SALT = 0xCF21B
_PAYLOAD_CURRICULUM_SEED_SALT = 0x5041594C


def reward_component_consistency(
    reward: float | np.floating[Any],
    components: Mapping[str, float],
) -> dict[str, float | str | bool]:
    """Compare logged semantic components against preserved reward arithmetic.

    Stable-Baselines3 supplies float32 actions.  The legacy-compatible reward
    expression consequently becomes float32 when it subtracts the action-based
    ``nontracking_cost`` -- even when that cost is zero.  Components such as
    the Lyapunov candidate are Python float64 values.  Comparing their exact
    sum to the preserved float32 result at a hard-coded float64 tolerance would
    therefore reject valid transitions.

    The tolerance below is the standard ``gamma_n`` forward-error bound for
    ``n`` floating-point additions, scaled by the sum of component magnitudes.
    It follows the dtype of the reward value and remains tight enough to reject
    a missing or duplicated reward term.
    """

    values = tuple(float(value) for value in components.values())
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("reward components must be a non-empty finite mapping")
    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise ValueError("environment reward must be finite")
    component_sum = float(math.fsum(values))
    difference = component_sum - reward_value
    reward_dtype = np.asarray(reward).dtype
    if not np.issubdtype(reward_dtype, np.floating):
        reward_dtype = np.dtype(np.float64)
    epsilon = float(np.finfo(reward_dtype).eps)
    operation_count = len(values) + 2
    gamma_n = operation_count * epsilon / (1.0 - operation_count * epsilon)
    magnitude_sum = float(math.fsum(abs(value) for value in values))
    absolute_tolerance = gamma_n * magnitude_sum
    consistent = math.isclose(
        component_sum,
        reward_value,
        rel_tol=0.0,
        abs_tol=absolute_tolerance,
    )
    return {
        "component_sum": component_sum,
        "difference": difference,
        "absolute_tolerance": absolute_tolerance,
        "relative_error": (
            abs(difference) / magnitude_sum if magnitude_sum > 0.0 else 0.0
        ),
        "reward_dtype": reward_dtype.name,
        "consistent": consistent,
    }


def legacy_reward_breakdown(
    *,
    position_error: Sequence[float],
    velocity: Sequence[float],
    tilt_error: float,
    omega_body: Sequence[float],
    yaw_error: float,
    action: Sequence[float],
    action_delta: Sequence[float],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """Report raw legacy costs and weighted contributions without mutation."""

    position_array = np.asarray(position_error, dtype=float).reshape(3)
    velocity_array = np.asarray(velocity, dtype=float).reshape(3)
    omega_array = np.asarray(omega_body, dtype=float).reshape(3)
    action_array = np.asarray(action, dtype=float).reshape(4)
    delta_array = np.asarray(action_delta, dtype=float).reshape(4)
    raw = {
        "position_error_squared_m2": float(position_array @ position_array),
        "linear_velocity_squared_m2_s2": float(velocity_array @ velocity_array),
        "tilt_error_one_minus_cos_tilt": float(tilt_error),
        "angular_velocity_squared_rad2_s2": float(omega_array @ omega_array),
        "yaw_error_squared_rad2": float(yaw_error**2),
        "normalized_action_squared": float(action_array @ action_array),
        "normalized_action_delta_squared": float(delta_array @ delta_array),
    }
    weighted = {
        "position": float(weights["position"]) * raw["position_error_squared_m2"],
        "linear_velocity": float(weights["linear_velocity"])
        * raw["linear_velocity_squared_m2_s2"],
        "tilt": float(weights["tilt"]) * raw["tilt_error_one_minus_cos_tilt"],
        "angular_velocity": float(weights["angular_velocity"])
        * raw["angular_velocity_squared_rad2_s2"],
        "yaw": float(weights["yaw"]) * raw["yaw_error_squared_rad2"],
        "action": float(weights["action"]) * raw["normalized_action_squared"],
        "action_rate": float(weights["action_rate"])
        * raw["normalized_action_delta_squared"],
    }
    return {
        "raw_costs": raw,
        "weights": {key: float(value) for key, value in weights.items()},
        "weighted_costs": weighted,
        "weighted_cost_sum": float(math.fsum(weighted.values())),
        "time_coefficient": None,
        "cost_clipping": None,
    }


def initial_state_curriculum_limits(
    settings: Any, global_step: int
) -> dict[str, float]:
    """Interpolate opt-in reset limits from one absolute global timestep."""

    if isinstance(global_step, (bool, np.bool_)) or not isinstance(
        global_step, (int, np.integer)
    ):
        raise ValueError("curriculum global step must be an integer")
    step = max(0, int(global_step))
    breakpoints = tuple(getattr(settings, "curriculum_breakpoints", ()))
    if breakpoints:
        first_step = int(breakpoints[0].global_step)
        last_step = int(breakpoints[-1].global_step)
        if first_step != 0 or last_step <= 0:
            raise ValueError(
                "curriculum breakpoints must start at zero and end after zero"
            )
        lower = upper = breakpoints[-1]
        segment_fraction = 1.0
        if step <= first_step:
            lower = upper = breakpoints[0]
            segment_fraction = 0.0
        elif step < last_step:
            for candidate_lower, candidate_upper in zip(breakpoints, breakpoints[1:]):
                if step <= int(candidate_upper.global_step):
                    lower = candidate_lower
                    upper = candidate_upper
                    width = int(upper.global_step) - int(lower.global_step)
                    if width <= 0:
                        raise ValueError(
                            "curriculum breakpoint steps must be strictly increasing"
                        )
                    segment_fraction = (step - int(lower.global_step)) / width
                    break

        def interpolate_breakpoint(name: str) -> float:
            lower_value = float(getattr(lower, name))
            upper_value = float(getattr(upper, name))
            return lower_value + segment_fraction * (upper_value - lower_value)

        return {
            "fraction": float(min(step / last_step, 1.0)),
            "segment_fraction": float(segment_fraction),
            "nominal_reset_probability": interpolate_breakpoint(
                "nominal_reset_probability"
            ),
            "maximum_tilt_deg": interpolate_breakpoint("max_tilt_deg"),
            "maximum_horizontal_offset_m": interpolate_breakpoint(
                "max_horizontal_offset_m"
            ),
            "maximum_vertical_offset_m": interpolate_breakpoint(
                "max_vertical_offset_m"
            ),
        }

    end_step = int(settings.curriculum_end_step)
    if end_step <= 0:
        raise ValueError("curriculum_end_step must be positive")
    fraction = min(step / end_step, 1.0)

    def interpolate(initial_name: str, final_name: str) -> float:
        initial = float(getattr(settings, initial_name))
        final = float(getattr(settings, final_name))
        return initial + fraction * (final - initial)

    return {
        "fraction": float(fraction),
        "maximum_tilt_deg": interpolate("initial_max_tilt_deg", "final_max_tilt_deg"),
        "maximum_horizontal_offset_m": interpolate(
            "initial_max_horizontal_offset_m", "final_max_horizontal_offset_m"
        ),
        "maximum_vertical_offset_m": interpolate(
            "initial_max_vertical_offset_m", "final_max_vertical_offset_m"
        ),
    }


def payload_curriculum_stage(settings: Any, global_step: int) -> Any:
    """Select one half-open payload stage from the new run's absolute step."""

    if isinstance(global_step, (bool, np.bool_)) or not isinstance(
        global_step, (int, np.integer)
    ):
        raise ValueError("payload curriculum global step must be an integer")
    step = max(0, int(global_step))
    stages = tuple(getattr(settings, "stages", ()))
    if not stages:
        raise ValueError("payload curriculum has no stages")
    for stage in stages:
        start = int(stage.start_step)
        end = None if stage.end_step is None else int(stage.end_step)
        if step >= start and (end is None or step < end):
            return stage
    raise ValueError(f"no payload curriculum stage covers global step {step}")


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
        raise ValueError(
            f"{name} must be a numeric vector with shape ({length},)"
        ) from exc
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
        raise ValueError(
            "action must be a finite numeric vector with shape (4,)"
        ) from exc
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
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), _ACTUATOR_SEED_SALT])
    )


def _payload_curriculum_rng(seed: int | None) -> np.random.Generator:
    """Create the opt-in payload stream without consuming legacy pose draws."""

    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), _PAYLOAD_CURRICULUM_SEED_SALT])
    )


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


def termination_diagnostics(
    *,
    position: Sequence[float],
    position_reference: Sequence[float],
    quaternion_wxyz: Sequence[float],
    minimum_altitude: float,
    maximum_altitude: float,
    maximum_tilt_rad: float,
    maximum_position_error: float,
) -> dict[str, Any]:
    """Describe the existing termination predicates without changing them."""

    actual_position = np.asarray(position, dtype=float).reshape(3)
    reference_position = np.asarray(position_reference, dtype=float).reshape(3)
    quaternion = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
    position_error_norm = float(np.linalg.norm(actual_position - reference_position))
    tilt_rad = float(
        np.arccos(
            np.clip(
                1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2),
                -1.0,
                1.0,
            )
        )
    )
    reasons: list[str] = []
    if actual_position[2] < minimum_altitude:
        reasons.append("below_minimum_altitude")
    if actual_position[2] > maximum_altitude:
        reasons.append("above_maximum_altitude")
    if tilt_rad > maximum_tilt_rad:
        reasons.append("excessive_tilt")
    if position_error_norm > maximum_position_error:
        reasons.append("excessive_position_error")
    return {
        "termination_reasons": reasons,
        "terminal_altitude": float(actual_position[2]),
        "terminal_tilt_deg": float(np.degrees(tilt_rad)),
        "terminal_position_error_norm": position_error_norm,
    }


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
        initial_state_randomization_enabled: bool | None = None,
        payload_curriculum_enabled: bool | None = None,
        track_curriculum_steps: bool = False,
        exact_attitude_perturbation: bool = False,
    ):
        if gym is not None:
            super().__init__()

        configured_vehicle = config.vehicle if config is not None else None
        configured_environment = config.environment if config is not None else None
        configured_actuator = config.actuator if config is not None else None
        configured_payload = (
            configured_environment.payload
            if configured_environment is not None
            else None
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
            raise ValueError(
                "episode_sec * policy_hz must yield at least one policy step"
            )
        self.policy_hz = resolved_policy_hz
        self.episode_sec = resolved_episode_sec
        self.dt_phys = 1.0 / self.physics_hz
        self.policy_dt = self.dt_phys * self.substeps
        self._observation_config = getattr(configured_environment, "observation", None)
        self.observation_schema = observation_schema(self._observation_config)
        self.state_reader = str(self.observation_schema["state_reader"])

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
            _selected(com_bias_mass, getattr(configured_payload, "mass", None), 0.0),
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
        self._payload_curriculum = getattr(configured_payload, "curriculum", None)
        configured_payload_curriculum_enabled = bool(
            getattr(self._payload_curriculum, "enabled", False)
        )
        resolved_payload_curriculum_enabled = (
            configured_payload_curriculum_enabled
            if payload_curriculum_enabled is None
            else payload_curriculum_enabled
        )
        if not isinstance(resolved_payload_curriculum_enabled, (bool, np.bool_)):
            raise ValueError("payload_curriculum_enabled must be a boolean")
        self.payload_curriculum_enabled = bool(resolved_payload_curriculum_enabled)
        if self.payload_curriculum_enabled and self.com_bias_randomize:
            raise ValueError(
                "payload curriculum and legacy payload randomization are mutually exclusive"
            )
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

        self._initial_state_randomization = getattr(
            configured_environment, "initial_state_randomization", None
        )
        configured_initial_enabled = bool(
            getattr(self._initial_state_randomization, "enabled", False)
        )
        self.initial_state_randomization_enabled = bool(
            configured_initial_enabled
            if initial_state_randomization_enabled is None
            else initial_state_randomization_enabled
        )
        if self.initial_state_randomization_enabled and self.mode != "e2e":
            raise ValueError("initial-state curriculum is supported only in e2e mode")
        if not isinstance(track_curriculum_steps, (bool, np.bool_)):
            raise ValueError("track_curriculum_steps must be a boolean")
        self._track_curriculum_steps = bool(track_curriculum_steps)
        if not isinstance(exact_attitude_perturbation, (bool, np.bool_)):
            raise ValueError("exact_attitude_perturbation must be a boolean")
        self._exact_attitude_perturbation = bool(exact_attitude_perturbation)
        self._curriculum_global_step = 0
        self._last_reset_info: dict[str, Any] = {}

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
            raise ValueError(
                "payload torque_fraction and mass_max must be non-negative"
            )

        reward = getattr(configured_environment, "reward", None)
        self.reward_mode = str(getattr(reward, "mode", "legacy"))
        if self.reward_mode not in REWARD_MODES:
            allowed = ", ".join(sorted(REWARD_MODES))
            raise ValueError(
                f"reward mode must be one of {{{allowed}}}, got {self.reward_mode!r}"
            )
        self._reward_config = reward
        self.reward_gamma = _finite_scalar(
            getattr(
                getattr(getattr(config, "training", None), "ppo", None),
                "gamma",
                0.99,
            ),
            "training.ppo.gamma",
        )
        if not 0.0 <= self.reward_gamma <= 1.0:
            raise ValueError("training.ppo.gamma must be within [0, 1]")
        if self.reward_mode != "legacy" and self._reward_config is None:
            raise ValueError("Lyapunov reward modes require a typed reward config")
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
        self.e2e_torque_xy_weight = _finite_scalar(
            getattr(reward, "e2e_torque_xy_weight", 0.0),
            "e2e_torque_xy_weight",
        )
        self.e2e_torque_yaw_weight = _finite_scalar(
            getattr(reward, "e2e_torque_yaw_weight", 0.0),
            "e2e_torque_yaw_weight",
        )
        self.crash_penalty = _finite_scalar(
            getattr(reward, "crash_penalty", 10.0), "crash_penalty"
        )
        if (
            min(
                self.position_weight,
                self.velocity_weight,
                self.tilt_weight,
                self.angular_velocity_weight,
                self.yaw_weight,
                self.action_weight,
                self.w_dact,
                self.e2e_torque_xy_weight,
                self.e2e_torque_yaw_weight,
                self.crash_penalty,
            )
            < 0.0
        ):
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
        self.wrench_command_reference = (
            "body frame; torque about the nominal allocator origin defined by "
            "the fixed rotor-arm geometry (not the payload-shifted combined CoM)"
        )
        self.drone_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "drone"
        )
        self.gyro_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
        )
        self._configure_state_reader()
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
        self._iquat0 = self.model.body_iquat[self.drone_bid].copy()
        self._inertia_body0 = inertia_body(self.model, self.drone_bid)
        self.physics_model_version = PHYSICS_MODEL_VERSION

        self.pid = CascadePID(self.dt_phys, config=config)
        self.dist_torque_body = np.zeros(3)
        self._com_off3 = np.zeros(3)
        self._com_mw = 0.0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        observation_dim = observation_dimension(self._observation_config)
        high = np.full(observation_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._auxiliary_observation = (
            None
            if self._observation_config is None
            else AuxiliaryObservationState(
                self._observation_config, policy_dt=self.policy_dt
            )
        )
        self._rng = np.random.default_rng(seed)
        self._payload_rng = _payload_curriculum_rng(seed)
        self._payload_curriculum_statistics: dict[str, Any] = {
            "rng_mode": "separate_seed_sequence_v1",
            "reset_count": 0,
            "accepted_count": 0,
            "payload_free_count": 0,
            "payload_center_count": 0,
            "payload_offset_count": 0,
            "candidate_attempt_count": 0,
            "rejected_candidate_count": 0,
            "rejection_reason_counts": {},
            "exhausted_reset_count": 0,
            "stage_reset_counts": {},
        }
        self._step = 0
        self._prev_action = np.zeros(ACTION_DIM)
        self._last_lyapunov_terms: dict[str, Any] | None = None

    def _set_com_bias(self, m_w: float, off_xy: Sequence[float]) -> None:
        mass = float(m_w)
        offset = np.asarray(off_xy, dtype=float)
        r_w = np.array([offset[0], offset[1], 0.0])
        self._com_off3 = r_w.copy()
        self._com_mw = mass
        total_mass = self._m0 + mass
        if total_mass <= 0.0:
            raise ValueError("base mass plus payload mass must be positive")
        total_mass, center, tensor = compose_point_payload(
            self._m0, self._ipos0, self._inertia_body0, mass, r_w
        )
        self.model.body_mass[self.drone_bid] = total_mass
        self.model.body_ipos[self.drone_bid] = center
        if mass == 0:
            # Preserve even the original principal-axis ordering/signs exactly.
            diagonal, quaternion = self._J0, self._iquat0
        else:
            diagonal, quaternion = principal_axes(tensor)
        self.model.body_inertia[self.drone_bid] = diagonal
        self.model.body_iquat[self.drone_bid] = quaternion
        # Rebuild frame shortcuts, subtree masses and reference inertias before
        # setting the episode pose. Children remain their original XML bodies.
        mujoco.mj_setConst(self.model, self.data)

    def payload_snapshot(self) -> dict[str, Any]:
        total, center, tensor, ids = subtree_properties(self)
        return {
            "physics_model_version": PHYSICS_MODEL_VERSION,
            "mass_kg": self._com_mw,
            "attachment_body_m": self._com_off3.tolist(),
            "drone_mass_kg": float(self.model.body_mass[self.drone_bid]),
            "drone_com_body_m": self.model.body_ipos[self.drone_bid].tolist(),
            "drone_inertia_body_kg_m2": inertia_body(
                self.model, self.drone_bid
            ).tolist(),
            "principal_inertia_kg_m2": self.model.body_inertia[self.drone_bid].tolist(),
            "inertial_quaternion_wxyz": self.model.body_iquat[self.drone_bid].tolist(),
            "vehicle_mass_kg": total,
            "vehicle_com_body_m": center.tolist(),
            "vehicle_locked_inertia_body_kg_m2": tensor.tolist(),
            "child_bodies_counted_once": ids[1:],
            "observation_reference": "body origin position and linear velocity in world; angular velocity in body",
            "airborne_initialization": "unchanged: equal rotor thrust from drone body mass including payload, excludes child masses; no CoM trim",
            "controller_parameters": "nominal; payload mass/CoM/inertia are not supplied",
            "rotor_dof_note": "vehicle inertia locks child joints; relative rotor energy is not included",
        }

    def physics_wrench_snapshot(self) -> dict[str, Any]:
        return wrench_snapshot(self)

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
        model_name = str(getattr(configured_actuator, "model", "cf21b_first_order"))
        if enabled and model_name == "cf21b_first_order":
            reaction = getattr(configured_actuator, "reaction_torque", None)
            return Cf21bFirstOrderActuatorModel(
                dt=self.dt_phys,
                motor_direction=self.motor_direction,
                thrust_min=self.thrust_min,
                thrust_max=self.thrust_max,
                time_constant_s=getattr(configured_actuator, "time_constant_s", 0.050),
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
                rotor_inertia_kg_m2=getattr(reaction, "rotor_inertia_kg_m2", 0.5e-7),
                include_rotor_acceleration_torque=getattr(
                    reaction, "include_rotor_acceleration_torque", False
                ),
                allocation_matrix=self.B,
            )

        raise ValueError(
            "the runtime actuator must be enabled with model 'cf21b_first_order'"
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
        self._last_motor_cmd = (
            np.asarray(output.motor_command, dtype=float).reshape(ACTION_DIM).copy()
        )
        self._last_omega = (
            np.asarray(output.omega, dtype=float).reshape(ACTION_DIM).copy()
        )
        self._last_q_actual = q_actual
        self._last_wrench_cmd = requested_wrench.copy()
        self._last_wrench_allocated = self.B @ f_cmd
        self._last_wrench_actual = achieved_wrench.copy()
        self._last_allocation_error = self._last_wrench_cmd - self._last_wrench_actual

    def _write_applied_motor_controls(self) -> None:
        """Write the last plant force/torque outputs to MuJoCo controls."""

        for index in range(ACTION_DIM):
            self.data.ctrl[self.act_force[index]] = float(self._last_f[index])
        for index in range(ACTION_DIM):
            self.data.ctrl[self.act_torque[index]] = float(self._last_q_actual[index])

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
            resample_parameters and getattr(randomization, "enabled", False)
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
            "reset_rpm_mode": getattr(configured, "reset_rpm_mode", "auto"),
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
                    "thrust_omega_reference_rad_s": float(model.omega_reference_rad_s),
                    "positive_branch_min_ratio": float(model.positive_branch_min_ratio),
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

    def _configure_state_reader(self) -> None:
        """Resolve and validate the gyro/freejoint frame contract once.

        MuJoCo free-joint rotational qvel entries are expressed in the child
        body frame.  ``body_iquat`` instead describes the principal-inertia
        frame and must not be applied to these kinematic rates.
        """

        if self.drone_bid < 0:
            raise ValueError("MuJoCo XML is missing required body 'drone'")
        if self.gyro_sid < 0:
            raise ValueError("MuJoCo XML is missing required sensor 'imu_gyro'")
        # Preserve the historical reader's minimal construction contract. Its
        # qpos/qvel layout was already fixed at zero in the legacy code, while
        # the opt-in current reader below requires and validates model metadata.
        self._freejoint_id = 0
        self._freejoint_qpos_address = 0
        self._freejoint_dof_address = 0
        if self.state_reader == LEGACY_STATE_READER:
            return
        if self.state_reader != CURRENT_STATE_READER:
            raise ValueError(f"unsupported state reader {self.state_reader!r}")
        first_joint = int(self.model.body_jntadr[self.drone_bid])
        joint_count = int(self.model.body_jntnum[self.drone_bid])
        if joint_count < 1 or first_joint < 0:
            raise ValueError("body 'drone' has no freejoint")
        if int(self.model.jnt_type[first_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("body 'drone' first joint must be a freejoint")
        self._freejoint_id = first_joint
        self._freejoint_qpos_address = int(self.model.jnt_qposadr[first_joint])
        self._freejoint_dof_address = int(self.model.jnt_dofadr[first_joint])

        if int(self.model.sensor_type[self.gyro_sid]) != int(
            mujoco.mjtSensor.mjSENS_GYRO
        ):
            raise ValueError("sensor 'imu_gyro' is not a MuJoCo gyro sensor")
        if int(self.model.sensor_objtype[self.gyro_sid]) != int(
            mujoco.mjtObj.mjOBJ_SITE
        ):
            raise ValueError("current state reader requires a site-mounted gyro")
        site_id = int(self.model.sensor_objid[self.gyro_sid])
        if site_id < 0 or int(self.model.site_bodyid[site_id]) != self.drone_bid:
            raise ValueError(
                "current state reader requires imu_gyro on a site fixed to body 'drone'"
            )
        if int(self.model.sensor_dim[self.gyro_sid]) != 3:
            raise ValueError("current state reader requires a three-axis gyro")
        site_quaternion = np.asarray(self.model.site_quat[site_id], dtype=float)
        if not np.allclose(
            site_quaternion, np.array([1.0, 0.0, 0.0, 0.0]), rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "current state reader supports only an identity-oriented IMU site; "
                f"got site quaternion {site_quaternion.tolist()}"
            )
        self._imu_site_id = site_id

    def _read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """World position/velocity of body origin, orientation, body angular rate.

        The observation reference does not follow the payload-dependent CoM.
        """
        qpos_address = int(getattr(self, "_freejoint_qpos_address", 0))
        dof_address = int(getattr(self, "_freejoint_dof_address", 0))
        position = self.data.qpos[qpos_address : qpos_address + 3].copy()
        quaternion = quat_normalize_wxyz(
            self.data.qpos[qpos_address + 3 : qpos_address + 7].copy()
        )
        if quaternion[0] < 0:
            quaternion = -quaternion
        velocity = self.data.qvel[dof_address : dof_address + 3].copy()
        if self.state_reader == CURRENT_STATE_READER:
            omega_body = self.data.qvel[dof_address + 3 : dof_address + 6].copy()
        else:
            sensor_address = int(self.model.sensor_adr[self.gyro_sid])
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
        return float(np.arctan2(np.sin(yaw - self.yaw_des), np.cos(yaw - self.yaw_des)))

    def _base_obs(
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

    def _obs(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        velocity: Sequence[float],
        omega_body: Sequence[float],
    ) -> np.ndarray:
        base = self._base_obs(position, quaternion, velocity, omega_body)
        if getattr(self, "_auxiliary_observation", None) is None:
            return base
        return self._auxiliary_observation.compose(base)

    def current_observation(self) -> np.ndarray:
        """Read the current policy input without advancing auxiliary state."""

        return self._obs(*self._read_state())

    def reset_auxiliary_observation_state(self) -> np.ndarray:
        """Reset episode-local memory after the final initial state is installed."""

        state = self._read_state()
        base = self._base_obs(*state)
        if getattr(self, "_auxiliary_observation", None) is not None:
            self._auxiliary_observation.reset(base)
        return self._obs(*state)

    def _tracking_error(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        velocity: Sequence[float],
        omega_body: Sequence[float],
    ):
        """Return the candidate error using the existing reference contract.

        The mission/environment API currently exposes position and yaw only.
        Consequently the preserved references here are zero world velocity,
        level attitude at ``yaw_des``, and zero desired body rate.  This helper
        intentionally does not instantiate or replay the stateful cascade PID.
        """

        return tracking_error_from_state(
            position=position,
            quaternion_wxyz=quaternion,
            velocity_world=velocity,
            angular_rate_body=omega_body,
            position_reference=self.pos_des,
            yaw_reference=self.yaw_des,
        )

    def set_curriculum_global_step(self, global_step: int) -> None:
        """Set the absolute training step used by the next curriculum reset."""

        if isinstance(global_step, (bool, np.bool_)) or not isinstance(
            global_step, (int, np.integer)
        ):
            raise ValueError("curriculum global step must be an integer")
        if int(global_step) < 0:
            raise ValueError("curriculum global step must be non-negative")
        self._curriculum_global_step = int(global_step)

    def payload_curriculum_statistics(self) -> dict[str, Any]:
        """Return cumulative reset-sampler counts and observed rejection ratios."""

        stats = self._payload_curriculum_statistics
        attempts = int(stats["candidate_attempt_count"])
        rejected = int(stats["rejected_candidate_count"])
        accepted = int(stats["accepted_count"])
        payload_episodes = int(stats["payload_center_count"]) + int(
            stats["payload_offset_count"]
        )
        result = {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in stats.items()
        }
        result.update(
            {
                "candidate_rejection_ratio": rejected / attempts if attempts else 0.0,
                "accepted_payload_free_ratio": (
                    int(stats["payload_free_count"]) / accepted if accepted else 0.0
                ),
                "accepted_center_ratio_within_payload": (
                    int(stats["payload_center_count"]) / payload_episodes
                    if payload_episodes
                    else 0.0
                ),
                "curriculum_global_step": int(self._curriculum_global_step),
                "physics_model_version": PHYSICS_MODEL_VERSION,
            }
        )
        return result

    def _payload_filter_rejection_reason(self, hover: Mapping[str, Any]) -> str | None:
        settings = self._payload_curriculum.feasibility_filter
        status = str(hover["physical_status"])
        if status == "infeasible":
            return "physical_infeasible"
        if status != "feasible":
            return "physical_indeterminate"
        reachable = hover["policy_allocator_reachable"]
        if reachable is False:
            return "e2e_action_unreachable"
        if reachable is not True:
            return "e2e_action_reachability_indeterminate"
        thrust = hover["equilibrium_motor_thrust_n"]
        if thrust is None:
            return "equilibrium_thrust_unavailable"
        values = np.asarray(thrust, dtype=float)
        limits = np.asarray(hover["actual_thrust_limits_n"], dtype=float)
        if np.any(
            values - limits[:, 0]
            < float(settings.minimum_lower_thrust_margin_n) - 1e-12
        ):
            return "lower_thrust_margin_below_design_minimum"
        if np.any(
            limits[:, 1] - values
            < float(settings.minimum_upper_thrust_margin_n) - 1e-12
        ):
            return "upper_thrust_margin_below_design_minimum"
        return None

    def _sample_payload_curriculum(self) -> tuple[dict[str, Any], Mapping[str, Any]]:
        """Sample, certify and install one fixed payload for the next episode."""

        settings = self._payload_curriculum
        if settings is None or not self.payload_curriculum_enabled:
            raise RuntimeError("payload curriculum is not enabled")
        stage = payload_curriculum_stage(settings, self._curriculum_global_step)
        stats = self._payload_curriculum_statistics
        stats["reset_count"] += 1
        stage_counts = stats["stage_reset_counts"]
        stage_counts[stage.name] = int(stage_counts.get(stage.name, 0)) + 1

        payload_free = bool(
            self._payload_rng.random() < float(settings.payload_free_probability)
        )
        centered = payload_free or bool(
            self._payload_rng.random() < float(stage.centered_payload_probability)
        )
        episode_class = (
            "payload_free"
            if payload_free
            else "payload_center"
            if centered
            else "payload_offset"
        )
        local_reasons: dict[str, int] = {}
        maximum_attempts = int(settings.feasibility_filter.max_resample_attempts)
        last_hover: Mapping[str, Any] | None = None
        last_candidate: dict[str, Any] | None = None
        for attempt in range(1, maximum_attempts + 1):
            stats["candidate_attempt_count"] += 1
            if payload_free:
                mass = 0.0
                radius = 0.0
                azimuth = 0.0
            else:
                mass = float(
                    self._payload_rng.uniform(stage.mass_min_kg, stage.mass_max_kg)
                )
                if centered:
                    radius = 0.0
                    azimuth = 0.0
                else:
                    # These draws are independent by construction and occur on
                    # the payload-only stream, never on the initial-pose stream.
                    radius = float(
                        self._payload_rng.uniform(
                            stage.radius_min_m, stage.radius_max_m
                        )
                    )
                    azimuth = float(self._payload_rng.uniform(0.0, 2.0 * np.pi))
            offset = np.array(
                [radius * np.cos(azimuth), radius * np.sin(azimuth)], dtype=float
            )
            self._set_com_bias(mass, offset)
            # Static certification reads current XML site geometry. A forward
            # is needed here because this happens before the reset pose forward.
            mujoco.mj_forward(self.model, self.data)
            hover = static_hover(self)
            last_hover = hover
            last_candidate = {
                "mass_kg": mass,
                "attachment_body_m": [float(offset[0]), float(offset[1]), 0.0],
                "radius_m": radius,
                "azimuth_rad": azimuth,
            }
            reason = self._payload_filter_rejection_reason(hover)
            if reason is None:
                stats["accepted_count"] += 1
                stats[f"{episode_class}_count"] += 1
                thrust = np.asarray(hover["equilibrium_motor_thrust_n"], dtype=float)
                limits = np.asarray(hover["actual_thrust_limits_n"], dtype=float)
                return (
                    {
                        "enabled": True,
                        "global_step_at_reset": int(self._curriculum_global_step),
                        "stage": {
                            "name": stage.name,
                            "start_step": int(stage.start_step),
                            "end_step": (
                                None if stage.end_step is None else int(stage.end_step)
                            ),
                        },
                        "episode_class": episode_class,
                        "payload_free_probability": float(
                            settings.payload_free_probability
                        ),
                        "centered_payload_probability": float(
                            stage.centered_payload_probability
                        ),
                        "attempt_count": attempt,
                        "rejected_candidate_count_before_accept": attempt - 1,
                        "rejection_reason_counts_this_reset": dict(local_reasons),
                        "accepted_candidate": last_candidate,
                        "acceptance": {
                            "physical_status": hover["physical_status"],
                            "e2e_action_reachable": hover["policy_allocator_reachable"],
                            "minimum_lower_thrust_margin_n": float(
                                np.min(thrust - limits[:, 0])
                            ),
                            "minimum_upper_thrust_margin_n": float(
                                np.min(limits[:, 1] - thrust)
                            ),
                            "design_minimum_lower_thrust_margin_n": float(
                                settings.feasibility_filter.minimum_lower_thrust_margin_n
                            ),
                            "design_minimum_upper_thrust_margin_n": float(
                                settings.feasibility_filter.minimum_upper_thrust_margin_n
                            ),
                            "margin_design_note": (
                                settings.feasibility_filter.margin_design_note
                            ),
                        },
                        "rng": {
                            "mode": settings.rng_mode,
                            "stream": "payload_only",
                            "initial_state_stream_unchanged": True,
                        },
                        "fixed_within_episode": bool(settings.fixed_within_episode),
                        "attachment_z_m": float(settings.attachment_z_m),
                        "cumulative_statistics": self.payload_curriculum_statistics(),
                    },
                    hover,
                )
            stats["rejected_candidate_count"] += 1
            local_reasons[reason] = local_reasons.get(reason, 0) + 1
            all_reasons = stats["rejection_reason_counts"]
            all_reasons[reason] = int(all_reasons.get(reason, 0)) + 1

        stats["exhausted_reset_count"] += 1
        raise RuntimeError(
            "payload curriculum exhausted max_resample_attempts without an "
            "acceptable sample; no nominal fallback was applied: "
            f"step={self._curriculum_global_step}, stage={stage.name!r}, "
            f"attempts={maximum_attempts}, reasons={local_reasons}, "
            f"last_candidate={last_candidate}, last_hover={last_hover}"
        )

    def set_physics_substep_observer(self, observer: Any | None) -> Any | None:
        """Install an opt-in observer called after each MuJoCo physics step.

        The observer is intentionally absent from normal training/evaluation.
        When supplied, it receives keyword arguments ``environment``,
        ``policy_step_index``, ``substep_index``, ``state_before``, and
        ``state_after``.  Returning the previous observer makes temporary
        diagnostic instrumentation straightforward without changing control,
        reward, actuator, or reset behavior.
        """

        if observer is not None and not callable(observer):
            raise TypeError("physics substep observer must be callable or None")
        previous = getattr(self, "_physics_substep_observer", None)
        self._physics_substep_observer = observer
        return previous

    def _curriculum_reset_pose(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        settings = self._initial_state_randomization
        if settings is None:
            raise RuntimeError("initial-state randomization config is unavailable")
        limits = initial_state_curriculum_limits(settings, self._curriculum_global_step)
        nominal_probability = float(
            limits.get("nominal_reset_probability", settings.nominal_reset_probability)
        )
        nominal = bool(self._rng.random() < nominal_probability)
        if nominal:
            offset = np.zeros(3)
            tilt_axis = np.zeros(3)
            tilt_rad = 0.0
        else:
            position_angle = self._rng.uniform(0.0, 2.0 * np.pi)
            position_radius = limits["maximum_horizontal_offset_m"] * np.sqrt(
                self._rng.random()
            )
            offset = np.array(
                [
                    position_radius * np.cos(position_angle),
                    position_radius * np.sin(position_angle),
                    self._rng.uniform(
                        -limits["maximum_vertical_offset_m"],
                        limits["maximum_vertical_offset_m"],
                    ),
                ]
            )
            axis_angle = self._rng.uniform(0.0, 2.0 * np.pi)
            tilt_axis = np.array(
                [np.cos(axis_angle), np.sin(axis_angle), 0.0], dtype=float
            )
            tilt_rad = np.deg2rad(limits["maximum_tilt_deg"]) * self._rng.random()

        yaw_half = 0.5 * self.yaw_des
        tilt_half = 0.5 * tilt_rad
        yaw_quaternion = np.array(
            [np.cos(yaw_half), 0.0, 0.0, np.sin(yaw_half)], dtype=float
        )
        tilt_quaternion = np.array(
            [
                np.cos(tilt_half),
                np.sin(tilt_half) * tilt_axis[0],
                np.sin(tilt_half) * tilt_axis[1],
                0.0,
            ],
            dtype=float,
        )
        w1, x1, y1, z1 = yaw_quaternion
        w2, x2, y2, z2 = tilt_quaternion
        quaternion = quat_normalize_wxyz(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
        if quaternion[0] < 0.0:
            quaternion = -quaternion
        info = {
            "nominal_reset": nominal,
            "initial_position_offset_xyz_m": offset.tolist(),
            "initial_tilt_deg": float(np.degrees(tilt_rad)),
            "initial_tilt_axis_xyz": tilt_axis.tolist(),
            "curriculum_global_step": int(self._curriculum_global_step),
            "curriculum_fraction": limits["fraction"],
            "active_position_randomization_limit": {
                "horizontal_radius_m": limits["maximum_horizontal_offset_m"],
                "vertical_abs_m": limits["maximum_vertical_offset_m"],
            },
            "active_tilt_limit_deg": limits["maximum_tilt_deg"],
        }
        if tuple(getattr(settings, "curriculum_breakpoints", ())):
            info.update(
                {
                    "curriculum_segment_fraction": limits["segment_fraction"],
                    "active_nominal_reset_probability": nominal_probability,
                }
            )
        return self.pos_des + offset, quaternion, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options  # Preserved API: master accepted but did not use options.
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._actuator_rng = _actuator_rng(seed)
            self._payload_rng = _payload_curriculum_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0

        payload_curriculum_info: dict[str, Any] = {
            "enabled": False,
            "rng": {
                "mode": "legacy_shared_reset_rng",
                "stream": "legacy",
                "initial_state_stream_unchanged": not self.com_bias_randomize,
            },
        }
        if getattr(self, "payload_curriculum_enabled", False):
            payload_curriculum_info, _accepted_hover = self._sample_payload_curriculum()
        elif self.com_bias_randomize:
            theta = self._rng.uniform(0.0, 2.0 * np.pi)
            radius = self._rng.uniform(self.random_radius_min, self.random_radius_max)
            torque_cap = self.random_torque_fraction * self.tau_max_rp
            target_torque = self._rng.uniform(0.0, torque_cap)
            payload_mass = min(
                target_torque / (radius * self.gravity), self.random_mass_max
            )
            offset = np.array([radius * np.cos(theta), radius * np.sin(theta)])
            self._set_com_bias(payload_mass, offset)
        else:
            self._set_com_bias(self.com_bias_mass, self.com_bias_offset)

        reset_info: dict[str, Any] = {}
        initial_randomization_enabled = bool(
            getattr(self, "initial_state_randomization_enabled", False)
        )
        if initial_randomization_enabled:
            reset_position, reset_quaternion, reset_info = self._curriculum_reset_pose()
            self.data.qpos[0:3] = reset_position
            self.data.qpos[3:7] = reset_quaternion
        else:
            # Keep the historical reset expression and RNG draw order exactly
            # when the new feature is disabled.
            self.data.qpos[0:3] = self.pos_des + self._rng.uniform(
                -self.pos_perturb, self.pos_perturb, 3
            )
            angle_sample = self._rng.uniform(0.0, 1.0)
            angle = np.radians(self.att_perturb_deg) * (
                1.0
                if getattr(self, "_exact_attitude_perturbation", False)
                else angle_sample
            )
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
        self.reset_actuator_state(
            airborne=True if initial_randomization_enabled else None,
            resample_parameters=True,
        )
        mujoco.mj_forward(self.model, self.data)
        active_observation_schema = dict(
            getattr(self, "observation_schema", observation_schema(None))
        )
        reset_info.update(
            {
                "physics_model_version": PHYSICS_MODEL_VERSION,
                "observation_schema": active_observation_schema,
                "state_reader": getattr(self, "state_reader", LEGACY_STATE_READER),
                "trace_schema_version": active_observation_schema[
                    "trace_schema_version"
                ],
                "payload": self.payload_snapshot(),
                "payload_curriculum": payload_curriculum_info,
                "static_hover": static_hover(self),
                "actuator_initial_state": {
                    "parameters": self.actuator_snapshot(),
                    "actual_thrust_n": self._last_f.tolist(),
                    "requested_thrust_n": self._last_f_cmd.tolist(),
                    "omega_rad_s": self._last_omega.tolist(),
                    "reaction_torque_nm": self._last_q_actual.tolist(),
                },
            }
        )
        if initial_randomization_enabled:
            reset_info.update(
                {
                    "initial_requested_motor_thrust": self._last_f_cmd.tolist(),
                    "initial_actual_motor_thrust": self._last_f.tolist(),
                }
            )
        self.pid.reset()
        self._step = 0
        self._prev_action = np.zeros(ACTION_DIM)
        # Candidate terms are transition-local.  Clearing diagnostics here
        # prevents episode state leakage, and each vectorized environment owns
        # its own independent instance of this field.
        self._last_lyapunov_terms = None
        observation = self.reset_auxiliary_observation_state()
        if getattr(self, "_auxiliary_observation", None) is not None:
            reset_info["auxiliary_observation"] = (
                self._auxiliary_observation.diagnostics()
            )
        self._last_reset_info = dict(reset_info)
        return observation, reset_info

    def step(
        self, action: Sequence[float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = _validated_action(action)
        action_array = np.clip(action_array, -1.0, 1.0)
        residual = self.residual_scale * action_array
        reward_mode = str(getattr(self, "reward_mode", "legacy"))
        state_before: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = (
            None
        )
        transition_base_before: np.ndarray | None = None
        transition_position_before: np.ndarray | None = None
        # Fully constructed environments always define policy_dt. The fallback
        # retains lightweight legacy unit-test doubles that predate timing logs.
        policy_dt = float(getattr(self, "policy_dt", 0.01))
        physics_step_start_time = float(
            getattr(self.data, "time", self._step * policy_dt)
        )
        physics_substep_observer = getattr(self, "_physics_substep_observer", None)

        for substep_index in range(self.substeps):
            position, quaternion, velocity, omega_body = self._read_state()
            if substep_index == 0:
                transition_base_before = self._base_obs(
                    position, quaternion, velocity, omega_body
                )
                transition_position_before = position.copy()
                if reward_mode != "legacy":
                    state_before = (position, quaternion, velocity, omega_body)
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
                wrench = residual + np.array([0.0, 0.0, 0.0, self.mass * self.gravity])
            self._apply_control(wrench)

            rotation = rotmat_from_quat_wxyz(quaternion)
            self.data.xfrc_applied[self.drone_bid, 3:6] = (
                rotation @ self.dist_torque_body
            )
            mujoco.mj_step(self.model, self.data)
            if physics_substep_observer is not None:
                physics_substep_observer(
                    environment=self,
                    policy_step_index=int(self._step),
                    substep_index=int(substep_index),
                    state_before=(position, quaternion, velocity, omega_body),
                    state_after=self._read_state(),
                )

        position, quaternion, velocity, omega_body = self._read_state()
        if getattr(self, "_auxiliary_observation", None) is not None:
            if transition_base_before is None or transition_position_before is None:
                raise RuntimeError("policy transition is missing its start state")
            self._auxiliary_observation.advance(
                base_observation_before=transition_base_before,
                applied_action=action_array,
                reference_minus_position_before=(
                    self.pos_des - transition_position_before
                ),
            )
        observation = self._obs(position, quaternion, velocity, omega_body)
        physics_step_end_time = float(
            getattr(self.data, "time", physics_step_start_time + policy_dt)
        )

        position_error = position - self.pos_des
        tilt_error = 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
        yaw_error = self._yaw_err(quaternion)
        action_delta = action_array - self._prev_action
        action_weight = 0.0 if self.mode == "e2e" else self.action_weight
        action_rate_weight = 0.0 if self.mode == "e2e" else self.w_dact
        normalized_torque = action_array[:3].copy()
        if self.mode == "e2e":
            e2e_torque_xy_cost = getattr(self, "e2e_torque_xy_weight", 0.0) * float(
                normalized_torque[0] ** 2 + normalized_torque[1] ** 2
            )
            e2e_torque_yaw_cost = getattr(self, "e2e_torque_yaw_weight", 0.0) * float(
                normalized_torque[2] ** 2
            )
            e2e_torque_cost = e2e_torque_xy_cost + e2e_torque_yaw_cost
        else:
            e2e_torque_xy_cost = 0.0
            e2e_torque_yaw_cost = 0.0
            e2e_torque_cost = 0.0

        legacy_cost: float | np.floating[Any] | None = None
        if reward_mode in {"legacy", "legacy_plus_lyapunov"}:
            # Keep the exact expression and operation order used before reward
            # modes were introduced.  In legacy mode this is the complete old
            # reward path, including float32 action arithmetic when supplied by
            # Stable-Baselines3.
            legacy_cost = (
                self.position_weight * (position_error @ position_error)
                + self.velocity_weight * (velocity @ velocity)
                + self.tilt_weight * tilt_error
                + self.angular_velocity_weight * (omega_body @ omega_body)
                + self.yaw_weight * yaw_error**2
                + action_weight * (action_array @ action_array)
                + action_rate_weight * (action_delta @ action_delta)
            )
        nontracking_cost = action_weight * (
            action_array @ action_array
        ) + action_rate_weight * (action_delta @ action_delta)
        self._prev_action = action_array.copy()

        self._step += 1
        termination = termination_diagnostics(
            position=position,
            position_reference=self.pos_des,
            quaternion_wxyz=quaternion,
            minimum_altitude=self.min_altitude,
            maximum_altitude=self.max_altitude,
            maximum_tilt_rad=self.max_termination_tilt,
            maximum_position_error=self.max_position_error,
        )
        crashed = bool(termination["termination_reasons"])
        truncated = self._step >= self.max_steps
        info: dict[str, Any] = {}

        if getattr(self, "_auxiliary_observation", None) is not None:
            info["observation_diagnostics"] = self._auxiliary_observation.diagnostics()
            info["transition_timing"] = {
                "trace_schema_version": getattr(
                    self,
                    "observation_schema",
                    observation_schema(None),
                )["trace_schema_version"],
                "physics_step_start_time_s": physics_step_start_time,
                "physics_step_end_time_s": physics_step_end_time,
                "policy_command_applied_interval_s": [
                    physics_step_start_time,
                    physics_step_end_time,
                ],
                "physics_substeps": int(self.substeps),
                "physics_dt_s": float(self.dt_phys),
                "policy_dt_s": policy_dt,
            }

        if reward_mode == "legacy":
            if legacy_cost is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("legacy reward cost was not computed")
            reward = -legacy_cost
        else:
            if state_before is None:  # pragma: no cover - substeps is validated >= 1
                raise RuntimeError("candidate reward is missing the pre-action state")
            if self._reward_config is None:
                raise RuntimeError("candidate reward is missing its typed config")
            error_before = self._tracking_error(*state_before)
            error_after = self._tracking_error(
                position, quaternion, velocity, omega_body
            )
            terms = compute_lyapunov_reward(
                error_before=error_before,
                error_after=error_after,
                dt=self.dt_phys * self.substeps,
                gamma=self.reward_gamma,
                config=self._reward_config,
                terminated=bool(crashed),
                truncated=bool(truncated),
            )
            if reward_mode == "lyapunov":
                reward = terms.reward_total - nontracking_cost - e2e_torque_cost
                legacy_tracking_reward = 0.0
            else:
                if legacy_cost is None:  # pragma: no cover - defensive invariant
                    raise RuntimeError("legacy-plus reward cost was not computed")
                reward = -legacy_cost + terms.reward_total - e2e_torque_cost
                legacy_tracking_cost = legacy_cost - nontracking_cost
                legacy_tracking_reward = -float(legacy_tracking_cost)

            normalized_after = normalized_tracking_vector(
                error_after, self._reward_config.lyapunov
            )
            motor_thrust_command = np.asarray(
                getattr(self, "_last_f_cmd", np.full(ACTION_DIM, np.nan)),
                dtype=float,
            ).reshape(ACTION_DIM)
            saturated = np.isclose(
                motor_thrust_command, self.thrust_min, rtol=0.0, atol=1e-12
            ) | np.isclose(motor_thrust_command, self.thrust_max, rtol=0.0, atol=1e-12)
            saturation_fraction = (
                float(np.mean(saturated))
                if np.all(np.isfinite(motor_thrust_command))
                else float("nan")
            )
            reward_terms = terms.as_dict()
            reward_terms.update(
                {
                    "mode": reward_mode,
                    "gamma": float(self.reward_gamma),
                    "dt": float(self.dt_phys * self.substeps),
                    "normalized_tracking_error_norm": float(
                        np.linalg.norm(normalized_after)
                    ),
                    "position_error": error_after.position.tolist(),
                    "velocity_error": error_after.velocity.tolist(),
                    "attitude_error": error_after.attitude.tolist(),
                    "angular_rate_error": error_after.angular_rate.tolist(),
                    "nontracking_reward": -float(nontracking_cost),
                    "e2e_torque_xy_cost": float(e2e_torque_xy_cost),
                    "e2e_torque_yaw_cost": float(e2e_torque_yaw_cost),
                    "e2e_torque_cost": float(e2e_torque_cost),
                    "e2e_torque_reward": -float(e2e_torque_cost),
                    "normalized_torque": normalized_torque.tolist(),
                    "legacy_tracking_reward": legacy_tracking_reward,
                    "crash_penalty": -float(self.crash_penalty) if crashed else 0.0,
                    "crash_or_ood_reward": (
                        -float(self.crash_penalty) if crashed else 0.0
                    ),
                    "actuator_saturation_fraction": saturation_fraction,
                    "termination_reasons": list(termination["termination_reasons"]),
                    "terminal_altitude": termination["terminal_altitude"],
                    "terminal_tilt_deg": termination["terminal_tilt_deg"],
                    "terminal_position_error_norm": termination[
                        "terminal_position_error_norm"
                    ],
                    "terminated": bool(crashed),
                    "truncated": bool(truncated),
                }
            )
            info["control_diagnostics"] = {
                "wrench_reference": self.wrench_command_reference,
                "normalized_action": action_array.tolist(),
                "physical_commanded_wrench": np.asarray(
                    getattr(self, "_last_wrench_cmd", np.full(ACTION_DIM, np.nan)),
                    dtype=float,
                )
                .reshape(ACTION_DIM)
                .tolist(),
                "requested_motor_thrust": np.asarray(
                    getattr(self, "_last_f_cmd", np.full(ACTION_DIM, np.nan)),
                    dtype=float,
                )
                .reshape(ACTION_DIM)
                .tolist(),
                "actual_motor_thrust": np.asarray(
                    getattr(self, "_last_f", np.full(ACTION_DIM, np.nan)),
                    dtype=float,
                )
                .reshape(ACTION_DIM)
                .tolist(),
                "actual_applied_wrench": np.asarray(
                    getattr(self, "_last_wrench_actual", np.full(ACTION_DIM, np.nan)),
                    dtype=float,
                )
                .reshape(ACTION_DIM)
                .tolist(),
            }
            info["reward_terms"] = reward_terms

        if crashed:
            reward -= self.crash_penalty
        if (
            getattr(self, "_auxiliary_observation", None) is not None
            and legacy_cost is not None
        ):
            breakdown = legacy_reward_breakdown(
                position_error=position_error,
                velocity=velocity,
                tilt_error=tilt_error,
                omega_body=omega_body,
                yaw_error=yaw_error,
                action=action_array,
                action_delta=action_delta,
                weights={
                    "position": self.position_weight,
                    "linear_velocity": self.velocity_weight,
                    "tilt": self.tilt_weight,
                    "angular_velocity": self.angular_velocity_weight,
                    "yaw": self.yaw_weight,
                    "action": action_weight,
                    "action_rate": action_rate_weight,
                },
            )
            info["legacy_reward_terms"] = {
                "mode": reward_mode,
                **breakdown,
                "legacy_cost": float(legacy_cost),
                "weighted_cost_sum_minus_legacy_cost": float(
                    breakdown["weighted_cost_sum"] - float(legacy_cost)
                ),
                "crash_penalty": float(self.crash_penalty if crashed else 0.0),
                "total_reward": float(reward),
            }
            info.setdefault(
                "control_diagnostics",
                {
                    "wrench_reference": self.wrench_command_reference,
                    "normalized_action": action_array.tolist(),
                    "physical_commanded_wrench": np.asarray(
                        getattr(self, "_last_wrench_cmd", np.full(ACTION_DIM, np.nan)),
                        dtype=float,
                    )
                    .reshape(ACTION_DIM)
                    .tolist(),
                    "requested_motor_thrust": np.asarray(
                        getattr(self, "_last_f_cmd", np.full(ACTION_DIM, np.nan)),
                        dtype=float,
                    )
                    .reshape(ACTION_DIM)
                    .tolist(),
                    "actual_motor_thrust": np.asarray(
                        getattr(self, "_last_f", np.full(ACTION_DIM, np.nan)),
                        dtype=float,
                    )
                    .reshape(ACTION_DIM)
                    .tolist(),
                    "actual_applied_wrench": np.asarray(
                        getattr(
                            self,
                            "_last_wrench_actual",
                            np.full(ACTION_DIM, np.nan),
                        ),
                        dtype=float,
                    )
                    .reshape(ACTION_DIM)
                    .tolist(),
                },
            )
        if reward_mode != "legacy":
            reward_terms = info["reward_terms"]
            reward_terms.update(
                {
                    "state_reward": float(reward_terms["state_cost"]),
                    "potential_reward": float(reward_terms["potential_shaping"]),
                    "decay_reward": float(reward_terms["decay_penalty"]),
                    "total_reward": float(reward),
                    "environment_reward": float(reward),
                }
            )
            components = {
                "state_reward": float(reward_terms["state_reward"]),
                "potential_reward": float(reward_terms["potential_reward"]),
                "decay_reward": float(reward_terms["decay_reward"]),
                "nontracking_reward": float(reward_terms["nontracking_reward"]),
                "e2e_torque_reward": float(reward_terms["e2e_torque_reward"]),
                "legacy_tracking_reward": float(reward_terms["legacy_tracking_reward"]),
                "crash_or_ood_reward": float(reward_terms["crash_or_ood_reward"]),
            }
            consistency = reward_component_consistency(reward, components)
            reward_terms["reward_components"] = components
            reward_terms["reward_component_sum"] = consistency["component_sum"]
            reward_terms["reward_component_difference"] = consistency["difference"]
            reward_terms["reward_component_tolerance"] = consistency[
                "absolute_tolerance"
            ]
            reward_terms["reward_component_relative_error"] = consistency[
                "relative_error"
            ]
            reward_terms["reward_arithmetic_dtype"] = consistency["reward_dtype"]
            reward_terms["reward_components_consistent"] = consistency["consistent"]
            if not reward_terms["reward_components_consistent"]:
                raise FloatingPointError(
                    "logged reward components do not sum to the environment reward: "
                    f"environment_reward={float(reward)!r}, "
                    f"component_sum={consistency['component_sum']!r}, "
                    f"difference={consistency['difference']!r}, "
                    f"tolerance={consistency['absolute_tolerance']!r}, "
                    f"reward_dtype={consistency['reward_dtype']!r}, "
                    f"mode={reward_mode!r}, terminated={bool(crashed)!r}, "
                    f"truncated={bool(truncated)!r}, "
                    f"termination_reasons={termination['termination_reasons']!r}, "
                    f"reward_terms={reward_terms!r}"
                )
            self._last_lyapunov_terms = info["reward_terms"].copy()
            info.update(termination)
        elif crashed:
            # Preserve the legacy non-terminal ``info == {}`` contract while
            # exposing the reason whenever its unchanged terminal predicate fires.
            info.update(termination)
        if getattr(self, "_track_curriculum_steps", False):
            self._curriculum_global_step += 1
        return observation, float(reward), bool(crashed), bool(truncated), info

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
    "legacy_reward_breakdown",
    "reward_component_consistency",
    "initial_state_curriculum_limits",
    "payload_curriculum_stage",
    "termination_diagnostics",
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

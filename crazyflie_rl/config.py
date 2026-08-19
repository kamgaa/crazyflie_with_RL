"""Typed, immutable experiment configuration loading and validation.

Loading this module never imports MuJoCo, Gymnasium, or Stable-Baselines3 and
never creates runtime resources.  Profiles may inherit another YAML file with
``extends``; mappings are merged recursively while sequences are replaced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePath, PurePosixPath
import math
from typing import Any, Mapping, Sequence


class ConfigError(ValueError):
    """Raised when an experiment profile is incomplete or invalid."""


class MissingResourceError(FileNotFoundError):
    """Raised when an explicitly configured runtime resource is unavailable."""


@dataclass(frozen=True)
class PathsConfig:
    mujoco_xml: Path | PurePosixPath
    project_root: Path
    artifact_root: Path
    legacy_model_root: Path
    legacy_tensorboard_root: Path


@dataclass(frozen=True)
class VehicleConfig:
    mass: float
    gravity: float
    arm_length: float
    motor_direction: tuple[float, float, float, float]
    thrust_min: float
    thrust_max: float
    torque_coefficient: float
    inertia_diagonal: tuple[float, float, float]
    physics_hz: float


@dataclass(frozen=True)
class ActuatorParameterRangeConfig:
    """Closed numeric range used for one independent actuator parameter."""

    min: float
    max: float


@dataclass(frozen=True)
class ActuatorRandomizationConfig:
    """Optional per-episode actuator variation on a dedicated RNG stream."""

    enabled: bool
    time_constant_s: ActuatorParameterRangeConfig
    steady_state_gain_rad_s: ActuatorParameterRangeConfig


@dataclass(frozen=True)
class ReactionTorqueConfig:
    """Plant-side propeller reaction-torque model configuration.

    ``legacy_ratio`` deliberately remains separate from the allocator's yaw
    ratio.  The latter lives in :class:`VehicleConfig` as
    ``torque_coefficient`` and is retained unchanged for checkpoint
    compatibility.
    """

    model: str
    legacy_ratio_m: float
    polynomial_coefficients: tuple[float, float, float]
    polynomial_scale: float
    rotor_inertia_kg_m2: float
    include_rotor_acceleration_torque: bool


@dataclass(frozen=True)
class ActuatorConfig:
    """Required first-order BLDC actuator dynamics configuration.

    The thrust coefficients are ordered ``(cubic, quadratic, linear)`` for
    ``F(r) = c3*r**3 + c2*r**2 + c1*r``, where
    ``r = omega / thrust_polynomial_omega_reference_rad_s``.  They are a
    paper-candidate mapping, not a verified hardware calibration.
    """

    enabled: bool
    model: str
    time_constant_s: float
    steady_state_gain_rad_s: float
    thrust_polynomial_coefficients: tuple[float, float, float]
    thrust_polynomial_omega_reference_rad_s: float
    thrust_polynomial_positive_branch_min_ratio: float
    thrust_polynomial_max_ratio: float
    parameter_source: str
    verification_status: str
    reaction_torque: ReactionTorqueConfig
    reset_rpm_mode: str
    randomization: ActuatorRandomizationConfig


@dataclass(frozen=True)
class PIDConfig:
    kp_position: float
    velocity_limit: float
    kp_velocity: float
    ki_velocity: float
    kd_velocity: float
    kp_attitude: float
    kp_rate: tuple[float, float, float]
    ki_rate: tuple[float, float, float]
    kd_rate: tuple[float, float, float]
    max_tilt_deg: float
    max_torque: float
    max_force: float
    integrator_limit: float


@dataclass(frozen=True)
class ControllerConfig:
    pid: PIDConfig


@dataclass(frozen=True)
class PayloadRandomizationConfig:
    radius_min: float
    radius_max: float
    torque_fraction: float
    mass_max: float


@dataclass(frozen=True)
class PayloadConfig:
    randomize: bool
    mass: float
    offset: tuple[float, float]
    randomization_limits: PayloadRandomizationConfig


@dataclass(frozen=True)
class RewardConfig:
    position_weight: float
    velocity_weight: float
    tilt_weight: float
    angular_velocity_weight: float
    yaw_weight: float
    action_weight: float
    action_rate_weight: float
    crash_penalty: float


@dataclass(frozen=True)
class TerminationConfig:
    min_altitude: float
    max_altitude: float
    max_tilt_deg: float
    max_position_error: float


@dataclass(frozen=True)
class EnvironmentConfig:
    control_mode: str
    policy_hz: float
    episode_sec: float
    residual_scale: tuple[float, float, float, float]
    position_target: tuple[float, float, float]
    yaw_target: float
    position_perturbation: float
    attitude_perturbation_deg: float
    payload: PayloadConfig
    reward: RewardConfig
    termination: TerminationConfig


@dataclass(frozen=True)
class PPOConfig:
    policy: str
    n_steps: int
    batch_size: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    n_epochs: int
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    clip_range: float
    target_kl: float | None
    log_std_init: float
    net_arch: tuple[int, ...]
    device: str
    verbose: int


@dataclass(frozen=True)
class TrainingConfig:
    seed: int | None
    total_timesteps: int
    ppo: PPOConfig


@dataclass(frozen=True)
class EvaluationConfig:
    episode_count: int
    seed_start: int
    deterministic: bool
    tail_fraction: float
    tilt_limit_deg: float
    evaluation_interval: int


@dataclass(frozen=True)
class CirclePresetConfig:
    key: str
    description: str
    radius: float
    period: float


@dataclass(frozen=True)
class HoverParameters:
    """Immutable parameters for one fixed-position reference mission."""

    target: tuple[float, float, float]
    yaw_deg: float
    duration: float


@dataclass(frozen=True)
class CircleParameters:
    """Immutable parameters for an explicitly configured circle mission."""

    center_xy: tuple[float, float]
    radius: float
    period: float
    laps: float
    start_angle_deg: float
    direction: str
    ramp_sec: float

    @property
    def clockwise(self) -> bool:
        """Return the boolean form expected by trajectory calculations."""

        return self.direction == "cw"


@dataclass(frozen=True)
class LissajousParameters:
    """Immutable parameters for a planar Lissajous reference mission."""

    center_xy: tuple[float, float]
    amplitude_xy: tuple[float, float]
    frequency_ratio: tuple[int, int]
    phase_deg: float
    base_period: float
    cycles: float
    ramp_sec: float


@dataclass(frozen=True)
class MissionConfig:
    type: str
    hover_altitude: float
    goto_xy: tuple[float, float]
    takeoff_sec: float
    settle_sec: float
    goto_sec: float
    circle_ramp_sec: float
    number_of_laps: float
    post_hold_sec: float
    force_floor_start: bool
    circle_presets: tuple[CirclePresetConfig, ...]
    hover: HoverParameters
    circle: CircleParameters
    lissajous: LissajousParameters

    def circle_preset(self, key: str) -> CirclePresetConfig:
        for preset in self.circle_presets:
            if preset.key == str(key):
                return preset
        allowed = ", ".join(preset.key for preset in self.circle_presets)
        raise ConfigError(f"unknown circle preset {key!r}; choose one of: {allowed}")


@dataclass(frozen=True)
class ExperimentMetadata:
    name: str
    condition: str
    description: str
    timezone: str


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully resolved experiment data with no mutable list/array references."""

    version: int
    paths: PathsConfig
    vehicle: VehicleConfig
    actuator: ActuatorConfig
    controller: ControllerConfig
    environment: EnvironmentConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    mission: MissionConfig
    experiment: ExperimentMetadata
    source_path: Path

    @property
    def profile_name(self) -> str:
        return self.source_path.stem

    @property
    def control_mode(self) -> str:
        return self.environment.control_mode

    @property
    def observation_shape(self) -> tuple[int]:
        # master constructs the same 15-value observation in both modes.
        return (15,)

    @property
    def action_shape(self) -> tuple[int]:
        return (4,)

    def resolved_dict(self) -> dict[str, Any]:
        payload = _serializable(asdict(self))
        payload["source_path"] = str(self.source_path)
        return payload

    def require_runtime_resources(self) -> None:
        if not Path(self.paths.mujoco_xml).is_file():
            raise MissingResourceError(
                "Configured MuJoCo XML is unavailable: "
                f"{self.paths.mujoco_xml}. The project intentionally does not "
                "search for, copy, or fabricate an alternative XML/mesh tree."
            )


_TOP_LEVEL_KEYS = {
    "version", "paths", "vehicle", "actuator", "controller", "environment",
    "training", "evaluation", "mission", "experiment",
}


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise RuntimeError("PyYAML is required to load experiment profiles") from exc
    return yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    yaml = _yaml_module()
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return loaded


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {key: _copy_value(value) for key, value in base.items()}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = _copy_value(value)
    return result


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _load_inherited(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"cyclic configuration inheritance: {chain}")
    current = _read_yaml(path)
    parent_name = current.pop("extends", None)
    if parent_name is None:
        return current
    if not isinstance(parent_name, str) or not parent_name.strip():
        raise ConfigError(f"extends must be a non-empty relative path in {path}")
    parent = Path(parent_name)
    if parent.is_absolute():
        raise ConfigError(f"extends must be relative in {path}")
    return _deep_merge(_load_inherited(path.parent / parent, (*stack, path)), current)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _keys(
    value: Mapping[str, Any], allowed: set[str], field: str, required: set[str] | None = None
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in {field}: {', '.join(unknown)}")
    missing = sorted((required if required is not None else allowed) - set(value))
    if missing:
        raise ConfigError(f"missing required key(s) in {field}: {', '.join(missing)}")


def _number(value: Any, field: str, *, minimum: float | None = None, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field} must be finite, got {value!r}")
    if positive and result <= 0.0:
        raise ConfigError(f"{field} must be positive, got {value!r}")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{field} must be at least {minimum}, got {value!r}")
    return result


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}, got {value!r}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean, got {value!r}")
    return value


def _text(value: Any, field: str, *, choices: set[str] | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    result = value.strip()
    if choices is not None and result not in choices:
        raise ConfigError(f"{field} must be one of {sorted(choices)}, got {result!r}")
    return result


def _numbers(value: Any, length: int, field: str, *, positive: bool = False) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ConfigError(f"{field} must contain exactly {length} values")
    return tuple(_number(item, f"{field}[{index}]", positive=positive) for index, item in enumerate(value))


def _integers(value: Any, length: int, field: str, *, minimum: int = 0) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ConfigError(f"{field} must contain exactly {length} values")
    return tuple(
        _integer(item, f"{field}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _foreign_posix_absolute(
    text: str, *, native_is_absolute: bool
) -> PurePosixPath | None:
    """Preserve a POSIX absolute path inspected on a non-POSIX host."""

    if PurePosixPath(text).is_absolute() and not native_is_absolute:
        return PurePosixPath(text)
    return None


def _path(value: Any, field: str, root: Path) -> Path | PurePosixPath:
    text = _text(value, field)
    candidate = Path(text).expanduser()
    # On the Linux server the protected XML path is naturally absolute.  The
    # startswith check also avoids rebasing POSIX absolute paths when profiles
    # are inspected from Windows.
    foreign_posix = _foreign_posix_absolute(
        text, native_is_absolute=candidate.is_absolute()
    )
    if foreign_posix is not None:
        # Preserve the protected Linux-server path byte-for-byte when a config
        # is inspected on Windows.  It becomes a concrete Path on Linux.
        return foreign_posix
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


def _serializable(value: Any) -> Any:
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, tuple):
        return [_serializable(item) for item in value]
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    return value


def _build_config(data: Mapping[str, Any], source_path: Path) -> ExperimentConfig:
    _keys(data, _TOP_LEVEL_KEYS, "configuration")
    version = _integer(data["version"], "version", minimum=1)

    raw_paths = _mapping(data["paths"], "paths")
    path_keys = {"mujoco_xml", "project_root", "artifact_root", "legacy_model_root", "legacy_tensorboard_root"}
    _keys(raw_paths, path_keys, "paths")
    inferred_root = source_path.parent.parent.resolve()
    project_root = _path(raw_paths["project_root"], "paths.project_root", inferred_root)
    paths = PathsConfig(
        mujoco_xml=_path(raw_paths["mujoco_xml"], "paths.mujoco_xml", project_root),
        project_root=project_root,
        artifact_root=_path(raw_paths["artifact_root"], "paths.artifact_root", project_root),
        legacy_model_root=_path(raw_paths["legacy_model_root"], "paths.legacy_model_root", project_root),
        legacy_tensorboard_root=_path(raw_paths["legacy_tensorboard_root"], "paths.legacy_tensorboard_root", project_root),
    )

    raw_vehicle = _mapping(data["vehicle"], "vehicle")
    vehicle_keys = {"mass", "gravity", "arm_length", "motor_direction", "thrust_min", "thrust_max", "torque_coefficient", "inertia_diagonal", "physics_hz"}
    _keys(raw_vehicle, vehicle_keys, "vehicle")
    vehicle = VehicleConfig(
        mass=_number(raw_vehicle["mass"], "vehicle.mass", positive=True),
        gravity=_number(raw_vehicle["gravity"], "vehicle.gravity", positive=True),
        arm_length=_number(raw_vehicle["arm_length"], "vehicle.arm_length", positive=True),
        motor_direction=_numbers(raw_vehicle["motor_direction"], 4, "vehicle.motor_direction"),  # type: ignore[arg-type]
        thrust_min=_number(raw_vehicle["thrust_min"], "vehicle.thrust_min", minimum=0.0),
        thrust_max=_number(raw_vehicle["thrust_max"], "vehicle.thrust_max", positive=True),
        torque_coefficient=_number(raw_vehicle["torque_coefficient"], "vehicle.torque_coefficient", positive=True),
        inertia_diagonal=_numbers(raw_vehicle["inertia_diagonal"], 3, "vehicle.inertia_diagonal", positive=True),  # type: ignore[arg-type]
        physics_hz=_number(raw_vehicle["physics_hz"], "vehicle.physics_hz", positive=True),
    )
    if vehicle.thrust_max <= vehicle.thrust_min:
        raise ConfigError("vehicle.thrust_max must exceed vehicle.thrust_min")

    raw_actuator = _mapping(data["actuator"], "actuator")
    actuator_keys = {
        "enabled",
        "model",
        "time_constant_s",
        "steady_state_gain_rad_s",
        "thrust_polynomial",
        "parameter_source",
        "verification_status",
        "reaction_torque",
        "reset_rpm_mode",
        "randomization",
    }
    _keys(raw_actuator, actuator_keys, "actuator")
    actuator_enabled = _boolean(raw_actuator["enabled"], "actuator.enabled")
    if not actuator_enabled:
        raise ConfigError(
            "actuator.enabled must be true because the motor model is required"
        )
    actuator_model = _text(
        raw_actuator["model"],
        "actuator.model",
        choices={"cf21b_first_order"},
    )

    raw_thrust_polynomial = _mapping(
        raw_actuator["thrust_polynomial"], "actuator.thrust_polynomial"
    )
    thrust_polynomial_keys = {
        "coefficients",
        "omega_reference_rad_s",
        "positive_branch_min_ratio",
        "max_ratio",
    }
    _keys(
        raw_thrust_polynomial,
        thrust_polynomial_keys,
        "actuator.thrust_polynomial",
    )
    thrust_polynomial_coefficients = _numbers(
        raw_thrust_polynomial["coefficients"],
        3,
        "actuator.thrust_polynomial.coefficients",
    )
    thrust_polynomial_omega_reference_rad_s = _number(
        raw_thrust_polynomial["omega_reference_rad_s"],
        "actuator.thrust_polynomial.omega_reference_rad_s",
        positive=True,
    )
    thrust_polynomial_positive_branch_min_ratio = _number(
        raw_thrust_polynomial["positive_branch_min_ratio"],
        "actuator.thrust_polynomial.positive_branch_min_ratio",
        minimum=0.0,
    )
    thrust_polynomial_max_ratio = _number(
        raw_thrust_polynomial["max_ratio"],
        "actuator.thrust_polynomial.max_ratio",
        positive=True,
    )
    if (
        thrust_polynomial_positive_branch_min_ratio
        > thrust_polynomial_max_ratio
    ):
        raise ConfigError(
            "actuator.thrust_polynomial.positive_branch_min_ratio must not exceed "
            "actuator.thrust_polynomial.max_ratio"
        )

    raw_reaction_torque = _mapping(
        raw_actuator["reaction_torque"], "actuator.reaction_torque"
    )
    reaction_torque_keys = {
        "model",
        "legacy_ratio_m",
        "polynomial_coefficients",
        "polynomial_scale",
        "rotor_inertia_kg_m2",
        "include_rotor_acceleration_torque",
    }
    _keys(raw_reaction_torque, reaction_torque_keys, "actuator.reaction_torque")
    reaction_torque = ReactionTorqueConfig(
        model=_text(
            raw_reaction_torque["model"],
            "actuator.reaction_torque.model",
            choices={"legacy_ratio", "paper_polynomial"},
        ),
        legacy_ratio_m=_number(
            raw_reaction_torque["legacy_ratio_m"],
            "actuator.reaction_torque.legacy_ratio_m",
            positive=True,
        ),
        polynomial_coefficients=_numbers(
            raw_reaction_torque["polynomial_coefficients"],
            3,
            "actuator.reaction_torque.polynomial_coefficients",
        ),  # type: ignore[arg-type]
        polynomial_scale=_number(
            raw_reaction_torque["polynomial_scale"],
            "actuator.reaction_torque.polynomial_scale",
            positive=True,
        ),
        rotor_inertia_kg_m2=_number(
            raw_reaction_torque["rotor_inertia_kg_m2"],
            "actuator.reaction_torque.rotor_inertia_kg_m2",
            positive=True,
        ),
        include_rotor_acceleration_torque=_boolean(
            raw_reaction_torque["include_rotor_acceleration_torque"],
            "actuator.reaction_torque.include_rotor_acceleration_torque",
        ),
    )

    raw_randomization = _mapping(
        raw_actuator["randomization"], "actuator.randomization"
    )
    randomization_keys = {"enabled", "time_constant_s", "steady_state_gain_rad_s"}
    _keys(raw_randomization, randomization_keys, "actuator.randomization")

    def _actuator_range(value: Any, field: str) -> ActuatorParameterRangeConfig:
        mapping = _mapping(value, field)
        _keys(mapping, {"min", "max"}, field)
        minimum = _number(mapping["min"], f"{field}.min", positive=True)
        maximum = _number(mapping["max"], f"{field}.max", positive=True)
        if maximum < minimum:
            raise ConfigError(f"{field}.max must be >= {field}.min")
        return ActuatorParameterRangeConfig(min=minimum, max=maximum)

    actuator_randomization = ActuatorRandomizationConfig(
        enabled=_boolean(
            raw_randomization["enabled"], "actuator.randomization.enabled"
        ),
        time_constant_s=_actuator_range(
            raw_randomization["time_constant_s"],
            "actuator.randomization.time_constant_s",
        ),
        steady_state_gain_rad_s=_actuator_range(
            raw_randomization["steady_state_gain_rad_s"],
            "actuator.randomization.steady_state_gain_rad_s",
        ),
    )
    actuator = ActuatorConfig(
        enabled=actuator_enabled,
        model=actuator_model,
        time_constant_s=_number(
            raw_actuator["time_constant_s"], "actuator.time_constant_s", positive=True
        ),
        steady_state_gain_rad_s=_number(
            raw_actuator["steady_state_gain_rad_s"],
            "actuator.steady_state_gain_rad_s",
            positive=True,
        ),
        thrust_polynomial_coefficients=thrust_polynomial_coefficients,  # type: ignore[arg-type]
        thrust_polynomial_omega_reference_rad_s=thrust_polynomial_omega_reference_rad_s,
        thrust_polynomial_positive_branch_min_ratio=(
            thrust_polynomial_positive_branch_min_ratio
        ),
        thrust_polynomial_max_ratio=thrust_polynomial_max_ratio,
        parameter_source=_text(
            raw_actuator["parameter_source"], "actuator.parameter_source"
        ),
        verification_status=_text(
            raw_actuator["verification_status"],
            "actuator.verification_status",
            choices={"unverified", "verified"},
        ),
        reaction_torque=reaction_torque,
        reset_rpm_mode=_text(
            raw_actuator["reset_rpm_mode"],
            "actuator.reset_rpm_mode",
            choices={"auto", "zero", "hover_equilibrium"},
        ),
        randomization=actuator_randomization,
    )

    raw_controller = _mapping(data["controller"], "controller")
    _keys(raw_controller, {"pid"}, "controller")
    raw_pid = _mapping(raw_controller["pid"], "controller.pid")
    pid_keys = {"kp_position", "velocity_limit", "kp_velocity", "ki_velocity", "kd_velocity", "kp_attitude", "kp_rate", "ki_rate", "kd_rate", "max_tilt_deg", "max_torque", "max_force", "integrator_limit"}
    _keys(raw_pid, pid_keys, "controller.pid")
    pid = PIDConfig(
        kp_position=_number(raw_pid["kp_position"], "controller.pid.kp_position", minimum=0.0),
        velocity_limit=_number(raw_pid["velocity_limit"], "controller.pid.velocity_limit", positive=True),
        kp_velocity=_number(raw_pid["kp_velocity"], "controller.pid.kp_velocity", minimum=0.0),
        ki_velocity=_number(raw_pid["ki_velocity"], "controller.pid.ki_velocity", minimum=0.0),
        kd_velocity=_number(raw_pid["kd_velocity"], "controller.pid.kd_velocity", minimum=0.0),
        kp_attitude=_number(raw_pid["kp_attitude"], "controller.pid.kp_attitude", minimum=0.0),
        kp_rate=_numbers(raw_pid["kp_rate"], 3, "controller.pid.kp_rate"),  # type: ignore[arg-type]
        ki_rate=_numbers(raw_pid["ki_rate"], 3, "controller.pid.ki_rate"),  # type: ignore[arg-type]
        kd_rate=_numbers(raw_pid["kd_rate"], 3, "controller.pid.kd_rate"),  # type: ignore[arg-type]
        max_tilt_deg=_number(raw_pid["max_tilt_deg"], "controller.pid.max_tilt_deg", positive=True),
        max_torque=_number(raw_pid["max_torque"], "controller.pid.max_torque", positive=True),
        max_force=_number(raw_pid["max_force"], "controller.pid.max_force", positive=True),
        integrator_limit=_number(raw_pid["integrator_limit"], "controller.pid.integrator_limit", positive=True),
    )

    raw_environment = _mapping(data["environment"], "environment")
    environment_keys = {"control_mode", "policy_hz", "episode_sec", "residual_scale", "position_target", "yaw_target", "position_perturbation", "attitude_perturbation_deg", "payload", "reward", "termination"}
    _keys(raw_environment, environment_keys, "environment")
    raw_payload = _mapping(raw_environment["payload"], "environment.payload")
    _keys(raw_payload, {"randomize", "mass", "offset", "randomization_limits"}, "environment.payload")
    raw_limits = _mapping(raw_payload["randomization_limits"], "environment.payload.randomization_limits")
    _keys(raw_limits, {"radius_min", "radius_max", "torque_fraction", "mass_max"}, "environment.payload.randomization_limits")
    limits = PayloadRandomizationConfig(
        radius_min=_number(raw_limits["radius_min"], "environment.payload.randomization_limits.radius_min", positive=True),
        radius_max=_number(raw_limits["radius_max"], "environment.payload.randomization_limits.radius_max", positive=True),
        torque_fraction=_number(raw_limits["torque_fraction"], "environment.payload.randomization_limits.torque_fraction", minimum=0.0),
        mass_max=_number(raw_limits["mass_max"], "environment.payload.randomization_limits.mass_max", minimum=0.0),
    )
    if limits.radius_max < limits.radius_min:
        raise ConfigError("payload randomization radius_max must be >= radius_min")
    payload = PayloadConfig(
        randomize=_boolean(raw_payload["randomize"], "environment.payload.randomize"),
        mass=_number(raw_payload["mass"], "environment.payload.mass", minimum=0.0),
        offset=_numbers(raw_payload["offset"], 2, "environment.payload.offset"),  # type: ignore[arg-type]
        randomization_limits=limits,
    )
    raw_reward = _mapping(raw_environment["reward"], "environment.reward")
    reward_keys = {"position_weight", "velocity_weight", "tilt_weight", "angular_velocity_weight", "yaw_weight", "action_weight", "action_rate_weight", "crash_penalty"}
    _keys(raw_reward, reward_keys, "environment.reward")
    reward = RewardConfig(**{
        key: _number(raw_reward[key], f"environment.reward.{key}", minimum=0.0)
        for key in reward_keys
    })
    raw_termination = _mapping(raw_environment["termination"], "environment.termination")
    termination_keys = {"min_altitude", "max_altitude", "max_tilt_deg", "max_position_error"}
    _keys(raw_termination, termination_keys, "environment.termination")
    termination = TerminationConfig(
        min_altitude=_number(raw_termination["min_altitude"], "environment.termination.min_altitude"),
        max_altitude=_number(raw_termination["max_altitude"], "environment.termination.max_altitude"),
        max_tilt_deg=_number(raw_termination["max_tilt_deg"], "environment.termination.max_tilt_deg", positive=True),
        max_position_error=_number(raw_termination["max_position_error"], "environment.termination.max_position_error", positive=True),
    )
    if termination.max_altitude <= termination.min_altitude:
        raise ConfigError("termination.max_altitude must exceed min_altitude")
    environment = EnvironmentConfig(
        control_mode=_text(raw_environment["control_mode"], "environment.control_mode", choices={"residual", "e2e"}),
        policy_hz=_number(raw_environment["policy_hz"], "environment.policy_hz", positive=True),
        episode_sec=_number(raw_environment["episode_sec"], "environment.episode_sec", positive=True),
        residual_scale=_numbers(raw_environment["residual_scale"], 4, "environment.residual_scale", positive=True),  # type: ignore[arg-type]
        position_target=_numbers(raw_environment["position_target"], 3, "environment.position_target"),  # type: ignore[arg-type]
        yaw_target=_number(raw_environment["yaw_target"], "environment.yaw_target"),
        position_perturbation=_number(raw_environment["position_perturbation"], "environment.position_perturbation", minimum=0.0),
        attitude_perturbation_deg=_number(raw_environment["attitude_perturbation_deg"], "environment.attitude_perturbation_deg", minimum=0.0),
        payload=payload,
        reward=reward,
        termination=termination,
    )
    substeps = round(vehicle.physics_hz / environment.policy_hz)
    if substeps < 1:
        raise ConfigError("environment.policy_hz is too high for vehicle.physics_hz")

    raw_training = _mapping(data["training"], "training")
    _keys(raw_training, {"seed", "total_timesteps", "ppo"}, "training")
    seed_value = raw_training["seed"]
    seed = None if seed_value is None else _integer(seed_value, "training.seed")
    raw_ppo = _mapping(raw_training["ppo"], "training.ppo")
    ppo_keys = {"policy", "n_steps", "batch_size", "learning_rate", "gamma", "gae_lambda", "n_epochs", "ent_coef", "vf_coef", "max_grad_norm", "clip_range", "target_kl", "log_std_init", "net_arch", "device", "verbose"}
    _keys(raw_ppo, ppo_keys, "training.ppo")
    raw_arch = raw_ppo["net_arch"]
    if isinstance(raw_arch, (str, bytes)) or not isinstance(raw_arch, Sequence) or not raw_arch:
        raise ConfigError("training.ppo.net_arch must be a non-empty integer list")
    net_arch = tuple(_integer(width, f"training.ppo.net_arch[{index}]", minimum=1) for index, width in enumerate(raw_arch))
    target_kl_value = raw_ppo["target_kl"]
    target_kl = None if target_kl_value is None else _number(target_kl_value, "training.ppo.target_kl", positive=True)
    ppo = PPOConfig(
        policy=_text(raw_ppo["policy"], "training.ppo.policy"),
        n_steps=_integer(raw_ppo["n_steps"], "training.ppo.n_steps", minimum=1),
        batch_size=_integer(raw_ppo["batch_size"], "training.ppo.batch_size", minimum=1),
        learning_rate=_number(raw_ppo["learning_rate"], "training.ppo.learning_rate", positive=True),
        gamma=_number(raw_ppo["gamma"], "training.ppo.gamma", minimum=0.0),
        gae_lambda=_number(raw_ppo["gae_lambda"], "training.ppo.gae_lambda", minimum=0.0),
        n_epochs=_integer(raw_ppo["n_epochs"], "training.ppo.n_epochs", minimum=1),
        ent_coef=_number(raw_ppo["ent_coef"], "training.ppo.ent_coef", minimum=0.0),
        vf_coef=_number(raw_ppo["vf_coef"], "training.ppo.vf_coef", minimum=0.0),
        max_grad_norm=_number(raw_ppo["max_grad_norm"], "training.ppo.max_grad_norm", minimum=0.0),
        clip_range=_number(raw_ppo["clip_range"], "training.ppo.clip_range", positive=True),
        target_kl=target_kl,
        log_std_init=_number(raw_ppo["log_std_init"], "training.ppo.log_std_init"),
        net_arch=net_arch,
        device=_text(raw_ppo["device"], "training.ppo.device"),
        verbose=_integer(raw_ppo["verbose"], "training.ppo.verbose"),
    )
    if not 0.0 <= ppo.gamma <= 1.0 or not 0.0 <= ppo.gae_lambda <= 1.0:
        raise ConfigError("training.ppo gamma and gae_lambda must be within [0, 1]")
    training = TrainingConfig(
        seed=seed,
        total_timesteps=_integer(raw_training["total_timesteps"], "training.total_timesteps", minimum=1),
        ppo=ppo,
    )

    raw_evaluation = _mapping(data["evaluation"], "evaluation")
    evaluation_keys = {"episode_count", "seed_start", "deterministic", "tail_fraction", "tilt_limit_deg", "evaluation_interval"}
    _keys(raw_evaluation, evaluation_keys, "evaluation")
    evaluation = EvaluationConfig(
        episode_count=_integer(raw_evaluation["episode_count"], "evaluation.episode_count", minimum=1),
        seed_start=_integer(raw_evaluation["seed_start"], "evaluation.seed_start"),
        deterministic=_boolean(raw_evaluation["deterministic"], "evaluation.deterministic"),
        tail_fraction=_number(raw_evaluation["tail_fraction"], "evaluation.tail_fraction", positive=True),
        tilt_limit_deg=_number(raw_evaluation["tilt_limit_deg"], "evaluation.tilt_limit_deg", positive=True),
        evaluation_interval=_integer(raw_evaluation["evaluation_interval"], "evaluation.evaluation_interval"),
    )
    if evaluation.tail_fraction > 1.0:
        raise ConfigError("evaluation.tail_fraction must not exceed 1.0")

    raw_mission = _mapping(data["mission"], "mission")
    legacy_mission_keys = {
        "type", "hover_altitude", "goto_xy", "takeoff_sec", "settle_sec",
        "goto_sec", "circle_ramp_sec", "number_of_laps", "post_hold_sec",
        "force_floor_start", "circle_presets",
    }
    mission_keys = legacy_mission_keys | {"hover", "circle", "lissajous"}
    _keys(raw_mission, mission_keys, "mission")
    raw_presets = raw_mission["circle_presets"]
    if isinstance(raw_presets, (str, bytes)) or not isinstance(raw_presets, Sequence):
        raise ConfigError("mission.circle_presets must be a list")
    presets: list[CirclePresetConfig] = []
    for index, item in enumerate(raw_presets):
        mapping = _mapping(item, f"mission.circle_presets[{index}]")
        _keys(mapping, {"key", "description", "radius", "period"}, f"mission.circle_presets[{index}]")
        presets.append(CirclePresetConfig(
            key=_text(mapping["key"], f"mission.circle_presets[{index}].key"),
            description=_text(mapping["description"], f"mission.circle_presets[{index}].description"),
            radius=_number(mapping["radius"], f"mission.circle_presets[{index}].radius", positive=True),
            period=_number(mapping["period"], f"mission.circle_presets[{index}].period", positive=True),
        ))
    if len({preset.key for preset in presets}) != len(presets):
        raise ConfigError("mission.circle_presets keys must be unique")

    raw_hover = _mapping(raw_mission["hover"], "mission.hover")
    _keys(raw_hover, {"target", "yaw_deg", "duration"}, "mission.hover")
    hover = HoverParameters(
        target=_numbers(raw_hover["target"], 3, "mission.hover.target"),  # type: ignore[arg-type]
        yaw_deg=_number(raw_hover["yaw_deg"], "mission.hover.yaw_deg"),
        duration=_number(raw_hover["duration"], "mission.hover.duration", positive=True),
    )
    if hover.target[2] < 0.0:
        raise ConfigError("mission.hover.target altitude must be non-negative")

    raw_circle = _mapping(raw_mission["circle"], "mission.circle")
    circle_keys = {
        "center_xy", "radius", "period", "laps", "start_angle_deg",
        "direction", "ramp_sec",
    }
    _keys(raw_circle, circle_keys, "mission.circle")
    circle = CircleParameters(
        center_xy=_numbers(raw_circle["center_xy"], 2, "mission.circle.center_xy"),  # type: ignore[arg-type]
        radius=_number(raw_circle["radius"], "mission.circle.radius", positive=True),
        period=_number(raw_circle["period"], "mission.circle.period", positive=True),
        laps=_number(raw_circle["laps"], "mission.circle.laps", positive=True),
        start_angle_deg=_number(
            raw_circle["start_angle_deg"], "mission.circle.start_angle_deg"
        ),
        direction=_text(
            raw_circle["direction"], "mission.circle.direction", choices={"cw", "ccw"}
        ),
        ramp_sec=_number(raw_circle["ramp_sec"], "mission.circle.ramp_sec", minimum=0.0),
    )

    raw_lissajous = _mapping(raw_mission["lissajous"], "mission.lissajous")
    lissajous_keys = {
        "center_xy", "amplitude_xy", "frequency_ratio", "phase_deg",
        "base_period", "cycles", "ramp_sec",
    }
    _keys(raw_lissajous, lissajous_keys, "mission.lissajous")
    amplitude_xy = _numbers(
        raw_lissajous["amplitude_xy"],
        2,
        "mission.lissajous.amplitude_xy",
    )
    if any(amplitude < 0.0 for amplitude in amplitude_xy):
        raise ConfigError("mission.lissajous.amplitude_xy values must be non-negative")
    if amplitude_xy == (0.0, 0.0):
        raise ConfigError("mission.lissajous.amplitude_xy values must not both be zero")
    lissajous = LissajousParameters(
        center_xy=_numbers(
            raw_lissajous["center_xy"], 2, "mission.lissajous.center_xy"
        ),  # type: ignore[arg-type]
        amplitude_xy=amplitude_xy,  # type: ignore[arg-type]
        frequency_ratio=_integers(
            raw_lissajous["frequency_ratio"],
            2,
            "mission.lissajous.frequency_ratio",
            minimum=1,
        ),  # type: ignore[arg-type]
        phase_deg=_number(raw_lissajous["phase_deg"], "mission.lissajous.phase_deg"),
        base_period=_number(
            raw_lissajous["base_period"], "mission.lissajous.base_period", positive=True
        ),
        cycles=_number(raw_lissajous["cycles"], "mission.lissajous.cycles", positive=True),
        ramp_sec=_number(
            raw_lissajous["ramp_sec"], "mission.lissajous.ramp_sec", minimum=0.0
        ),
    )
    mission = MissionConfig(
        type=_text(
            raw_mission["type"],
            "mission.type",
            choices={"hover", "circle", "lissajous"},
        ),
        hover_altitude=_number(raw_mission["hover_altitude"], "mission.hover_altitude", minimum=0.0),
        goto_xy=_numbers(raw_mission["goto_xy"], 2, "mission.goto_xy"),  # type: ignore[arg-type]
        takeoff_sec=_number(raw_mission["takeoff_sec"], "mission.takeoff_sec", positive=True),
        settle_sec=_number(raw_mission["settle_sec"], "mission.settle_sec", minimum=0.0),
        goto_sec=_number(raw_mission["goto_sec"], "mission.goto_sec", positive=True),
        circle_ramp_sec=_number(
            raw_mission["circle_ramp_sec"],
            "mission.circle_ramp_sec",
            minimum=0.0,
        ),
        number_of_laps=_number(raw_mission["number_of_laps"], "mission.number_of_laps", positive=True),
        post_hold_sec=_number(raw_mission["post_hold_sec"], "mission.post_hold_sec", minimum=0.0),
        force_floor_start=_boolean(raw_mission["force_floor_start"], "mission.force_floor_start"),
        circle_presets=tuple(presets),
        hover=hover,
        circle=circle,
        lissajous=lissajous,
    )

    raw_experiment = _mapping(data["experiment"], "experiment")
    experiment_keys = {"name", "condition", "description", "timezone"}
    _keys(raw_experiment, experiment_keys, "experiment")
    experiment = ExperimentMetadata(
        name=_text(raw_experiment["name"], "experiment.name"),
        condition=_text(raw_experiment["condition"], "experiment.condition"),
        description=_text(raw_experiment["description"], "experiment.description"),
        timezone=_text(raw_experiment["timezone"], "experiment.timezone"),
    )
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(experiment.timezone)
    except Exception as exc:
        raise ConfigError(f"unknown experiment.timezone: {experiment.timezone!r}") from exc

    return ExperimentConfig(
        version=version,
        paths=paths,
        vehicle=vehicle,
        actuator=actuator,
        controller=ControllerConfig(pid=pid),
        environment=environment,
        training=training,
        evaluation=evaluation,
        mission=mission,
        experiment=experiment,
        source_path=source_path,
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load, recursively merge, validate, and freeze an experiment profile."""

    source_path = Path(path).expanduser().resolve()
    if source_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigError(f"configuration file must use .yaml or .yml: {source_path}")
    return _build_config(_load_inherited(source_path), source_path)


def dump_resolved_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Serialize one immutable resolved configuration without runtime imports."""

    yaml = _yaml_module()
    target = Path(path)
    target.write_text(
        yaml.safe_dump(config.resolved_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target

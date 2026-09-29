"""Import-safe CLI and rollout lifecycle for Crazyflie flight evaluations."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, TextIO

if TYPE_CHECKING:
    import numpy as np

    from .artifacts import ArtifactManager
    from .config import ExperimentConfig
    from .missions import ReferenceMission


MODE_ALIASES = {
    "1": "hover",
    "hover": "hover",
    "2": "circle",
    "circle": "circle",
    "3": "lissajous",
    "lissajous": "lissajous",
}

DIRECTION_ALIASES = {
    "cw": "cw",
    "clockwise": "cw",
    "ccw": "ccw",
    "counterclockwise": "ccw",
}

LATEST_BEST_MODEL = "latest-best"


def _ensure_runtime_imports() -> None:
    """Load numerical/config/runtime helpers only after CLI parsing succeeds."""

    if "np" in globals():
        return
    import numpy as numpy_module

    from .artifacts import ArtifactManager as artifact_manager, stable_float as float_tag
    from .config import ConfigError as config_error, load_config as config_loader
    from .missions import (
        CircleMission as circle_mission,
        mission_from_experiment as mission_factory,
    )
    from .plotting import (
        quaternion_to_euler_deg as quaternion_converter,
        save_hover_trace as hover_plotter,
        save_policy_trace as policy_plotter,
        save_tracking_trace as tracking_plotter,
    )

    globals().update(
        np=numpy_module,
        ArtifactManager=artifact_manager,
        stable_float=float_tag,
        ConfigError=config_error,
        load_config=config_loader,
        CircleMission=circle_mission,
        mission_from_experiment=mission_factory,
        quaternion_to_euler_deg=quaternion_converter,
        save_hover_trace=hover_plotter,
        save_policy_trace=policy_plotter,
        save_tracking_trace=tracking_plotter,
    )


def normalize_mode(value: str) -> str:
    """Return the core mission name for a numeric or named CLI value."""

    key = str(value).strip().lower()
    try:
        return MODE_ALIASES[key]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown mode {value!r}; choose 1/hover, 2/circle, or 3/lissajous"
        ) from exc


def normalize_direction(value: str) -> str:
    """Normalize long and short direction spellings to ``cw`` or ``ccw``."""

    key = str(value).strip().lower()
    try:
        return DIRECTION_ALIASES[key]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown direction {value!r}; choose cw/clockwise or ccw/counterclockwise"
        ) from exc


@dataclass(frozen=True)
class RolloutTrace:
    policy: str
    label: str
    time_sec: np.ndarray
    position: np.ndarray
    attitude_deg: np.ndarray
    reference_position: np.ndarray
    position_error: np.ndarray
    phases: tuple[str, ...]
    training_boundary_crossed_at: float | None
    guard_boundary_crossed_at: float | None
    terminated_at: float | None
    truncated_at: float | None
    diverged_at: float | None
    force_floor_start_requested: bool = False
    force_floor_start_applied: bool = False
    force_floor_start_error: str | None = None
    error: str | None = None
    control_input: np.ndarray | None = None
    motor_thrust: np.ndarray | None = None
    linear_velocity: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None
    control_mode: str | None = None
    motor_thrust_command: np.ndarray | None = None
    motor_command: np.ndarray | None = None
    motor_omega_rad_s: np.ndarray | None = None
    reaction_torque_nm: np.ndarray | None = None
    wrench_command: np.ndarray | None = None
    wrench_actual: np.ndarray | None = None
    allocation_error: np.ndarray | None = None
    actuator: Mapping[str, Any] | None = None
    episode_mass_kg: float | None = None

    @property
    def sample_count(self) -> int:
        return int(self.time_sec.size)


def _cached_episode_mass(env: Any) -> float | None:
    """Reuse the operands of _set_com_bias; never read MuJoCo state here."""
    base = getattr(env, "_m0", None)
    payload = getattr(env, "_com_mw", None)
    if base is None or payload is None:
        return None
    mass = float(base) + float(payload)
    return mass if math.isfinite(mass) and mass > 0 else None


def _is_latest_best_model(value: str | Path | None) -> bool:
    """Return whether ``value`` requests the run-scoped latest best archive."""

    return value is not None and str(value).strip().casefold() == LATEST_BEST_MODEL


def _latest_best_model_path(config: ExperimentConfig) -> Path:
    """Find the newest saved ``best`` archive for this control mode.

    Selection is deliberately manifest-driven rather than filename-driven so
    residual and E2E archives cannot be mixed merely because both retain the
    same observation/action shapes.  A missing or malformed old manifest is
    ignored; an explicit ``--model`` path remains unaffected.
    """

    runs_root = Path(config.paths.artifact_root) / "runs"
    candidates: list[tuple[int, Path]] = []
    if runs_root.is_dir():
        for manifest_path in runs_root.glob("*/manifests/*manifest*.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, Mapping):
                continue
            if manifest.get("control_mode") != config.control_mode:
                continue

            models = manifest.get("models")
            best = models.get("best") if isinstance(models, Mapping) else None
            best_path = best.get("path") if isinstance(best, Mapping) else None
            if not isinstance(best_path, str) or not best_path:
                continue

            run_dir = manifest_path.parent.parent.resolve()
            relative_model = Path(best_path)
            if relative_model.is_absolute():
                continue
            model_path = (run_dir / relative_model).resolve()
            try:
                model_path.relative_to(run_dir)
            except ValueError:
                continue
            if model_path.suffix.casefold() != ".zip" or not model_path.is_file():
                continue
            candidates.append((model_path.stat().st_mtime_ns, model_path))

    if candidates:
        return max(candidates, key=lambda item: (item[0], str(item[1])))[1]
    raise FileNotFoundError(
        "No saved best PPO model matches control mode "
        f"{config.control_mode!r} under {runs_root}. Train one first or "
        "pass an explicit --model path."
    )


def _model_candidate(value: str | Path | None, config: ExperimentConfig) -> Path:
    candidate = (
        config.paths.legacy_model_root / "ppo_best.zip"
        if value is None
        else Path(value).expanduser()
    )
    return (
        (Path.cwd() / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _model_path(value: str | Path | None, config: ExperimentConfig) -> Path:
    if _is_latest_best_model(value):
        return _latest_best_model_path(config)

    candidate = _model_candidate(value, config)
    if candidate.is_file():
        return candidate
    if candidate.suffix.lower() != ".zip":
        with_zip = Path(f"{candidate}.zip")
        if with_zip.is_file():
            return with_zip
    raise FileNotFoundError(f"PPO model is unavailable: {candidate}")


def _load_policy(path: Path, config: ExperimentConfig):
    from stable_baselines3 import PPO

    policy = PPO.load(str(path), device=config.training.ppo.device)
    observed = tuple(getattr(policy.observation_space, "shape", ()))
    action = tuple(getattr(policy.action_space, "shape", ()))
    if observed != config.observation_shape:
        raise ValueError(
            f"model observation shape {observed} does not match preserved "
            f"environment contract {config.observation_shape}: {path}"
        )
    if action != config.action_shape:
        raise ValueError(
            f"model action shape {action} does not match {config.action_shape}: {path}"
        )
    return policy


def _policy_specs(
    config: ExperimentConfig,
    selected: str,
    *,
    unified: bool = False,
) -> list[tuple[str, str]]:
    if config.control_mode == "e2e":
        floor_label = (
            "floor (PID)"
            if unified
            else "E2E zero-action (gravity compensation only)"
        )
        policy_label = "E2E PPO"
    else:
        floor_label = "floor (PID)"
        policy_label = "residual (PID+RL)"
    result: list[tuple[str, str]] = []
    if selected in {"floor", "both"}:
        result.append(("floor", floor_label))
    if selected in {"residual", "both"}:
        result.append(("residual", policy_label))
    return result


def _expected_policy_control_modes(
    config: ExperimentConfig,
    selected: str,
    *,
    unified: bool,
) -> dict[str, str]:
    """Return the controller mode intended for every selected rollout."""

    return {
        policy_key: (
            "residual" if unified and policy_key == "floor" else config.control_mode
        )
        for policy_key, _label in _policy_specs(
            config,
            selected,
            unified=unified,
        )
    }


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ConfigError(f"{name} must be positive, got {value!r}")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ConfigError(f"{name} must be non-negative, got {value!r}")
    return result


def apply_runtime_overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
    mode: str,
) -> ExperimentConfig:
    """Apply CLI/prompt values with immutable dataclass replacement."""

    _ensure_runtime_imports()

    mission = config.mission
    environment = config.environment
    evaluation = config.evaluation

    if args.seed is not None:
        if int(args.seed) < 0:
            raise ConfigError(f"seed must be non-negative, got {args.seed!r}")
        evaluation = replace(evaluation, seed_start=int(args.seed))
    if args.position_perturbation is not None:
        environment = replace(
            environment,
            position_perturbation=_nonnegative(
                args.position_perturbation, "position perturbation"
            ),
        )
    if args.attitude_perturbation_deg is not None:
        environment = replace(
            environment,
            attitude_perturbation_deg=_nonnegative(
                args.attitude_perturbation_deg, "attitude perturbation"
            ),
        )

    common: dict[str, Any] = {"type": mode}
    if args.altitude is not None:
        common["hover_altitude"] = _nonnegative(args.altitude, "altitude")
    if args.takeoff_sec is not None:
        common["takeoff_sec"] = _positive(args.takeoff_sec, "takeoff duration")
    if args.settle_sec is not None:
        common["settle_sec"] = _nonnegative(args.settle_sec, "settle duration")
    if args.goto_sec is not None:
        common["goto_sec"] = _positive(args.goto_sec, "GOTO duration")
    if args.post_hold_sec is not None:
        common["post_hold_sec"] = _nonnegative(
            args.post_hold_sec, "post-HOLD duration"
        )
    if args.force_floor_start is not None:
        common["force_floor_start"] = bool(args.force_floor_start)

    if mode == "hover":
        hover = mission.hover
        if args.hover_target is None:
            target = hover.target
        else:
            target = (
                _finite(args.hover_target[0], "hover target x"),
                _finite(args.hover_target[1], "hover target y"),
                _nonnegative(args.hover_target[2], "hover target altitude"),
            )
        yaw_deg = (
            hover.yaw_deg
            if args.yaw_deg is None
            else _finite(args.yaw_deg, "yaw setpoint")
        )
        duration = (
            hover.duration
            if args.duration is None
            else _positive(args.duration, "hover duration")
        )
        hover = replace(hover, target=target, yaw_deg=yaw_deg, duration=duration)
        common.update(hover=hover, hover_altitude=target[2])
        mission = replace(mission, **common)
        environment = replace(
            environment,
            position_target=target,
            yaw_target=math.radians(yaw_deg),
            episode_sec=duration,
        )
    elif mode == "circle":
        circle = mission.circle
        center = circle.center_xy if args.center is None else tuple(
            _finite(value, "circle center") for value in args.center
        )
        radius = (
            circle.radius
            if args.radius is None
            else _positive(args.radius, "circle radius")
        )
        common_period = getattr(args, "period", None)
        period_override = (
            common_period if common_period is not None else args.circle_period
        )
        period = (
            circle.period
            if period_override is None
            else _positive(period_override, "circle period")
        )
        laps = (
            circle.laps
            if args.laps is None
            else _positive(args.laps, "circle laps")
        )
        start_angle = (
            circle.start_angle_deg
            if args.start_angle_deg is None
            else _finite(args.start_angle_deg, "circle start angle")
        )
        direction = (
            circle.direction
            if args.direction is None
            else normalize_direction(args.direction)
        )
        ramp = (
            circle.ramp_sec
            if args.ramp_sec is None
            else _nonnegative(args.ramp_sec, "circle ramp duration")
        )
        circle = replace(
            circle,
            center_xy=center,
            radius=radius,
            period=period,
            laps=laps,
            start_angle_deg=start_angle,
            direction=direction,
            ramp_sec=ramp,
        )
        start_angle_rad = math.radians(start_angle)
        goto_xy = (
            center[0] + radius * math.cos(start_angle_rad),
            center[1] + radius * math.sin(start_angle_rad),
        )
        mission = replace(mission, **common, circle=circle, goto_xy=goto_xy)
    elif mode == "lissajous":
        lissajous = mission.lissajous
        center = lissajous.center_xy if args.center is None else tuple(
            _finite(value, "Lissajous center") for value in args.center
        )
        amplitude_x = (
            lissajous.amplitude_xy[0]
            if args.amplitude_x is None
            else _nonnegative(args.amplitude_x, "amplitude x")
        )
        amplitude_y = (
            lissajous.amplitude_xy[1]
            if args.amplitude_y is None
            else _nonnegative(args.amplitude_y, "amplitude y")
        )
        if amplitude_x == 0.0 and amplitude_y == 0.0:
            raise ConfigError("Lissajous amplitudes must not both be zero")
        frequency_x = (
            lissajous.frequency_ratio[0]
            if args.frequency_x is None
            else int(args.frequency_x)
        )
        frequency_y = (
            lissajous.frequency_ratio[1]
            if args.frequency_y is None
            else int(args.frequency_y)
        )
        if frequency_x <= 0 or frequency_y <= 0:
            raise ConfigError("Lissajous frequencies must be positive integers")
        phase_deg = (
            lissajous.phase_deg
            if args.phase_deg is None
            else _finite(args.phase_deg, "Lissajous phase")
        )
        common_period = getattr(args, "period", None)
        period_override = (
            common_period if common_period is not None else args.base_period
        )
        base_period = (
            lissajous.base_period
            if period_override is None
            else _positive(period_override, "Lissajous base period")
        )
        cycles = (
            lissajous.cycles
            if args.cycles is None
            else _positive(args.cycles, "Lissajous cycles")
        )
        ramp = (
            lissajous.ramp_sec
            if args.ramp_sec is None
            else _nonnegative(args.ramp_sec, "Lissajous ramp duration")
        )
        lissajous = replace(
            lissajous,
            center_xy=center,
            amplitude_xy=(amplitude_x, amplitude_y),
            frequency_ratio=(frequency_x, frequency_y),
            phase_deg=phase_deg,
            base_period=base_period,
            cycles=cycles,
            ramp_sec=ramp,
        )
        goto_xy = (
            center[0] + amplitude_x * math.sin(math.radians(phase_deg)),
            center[1],
        )
        mission = replace(
            mission, **common, lissajous=lissajous, goto_xy=goto_xy
        )
    else:  # defensive: parser and config validation normally catch this
        raise ConfigError(f"unknown mission mode: {mode!r}")

    return replace(
        config,
        mission=mission,
        environment=environment,
        evaluation=evaluation,
    )


def _legacy_circle_condition(config: ExperimentConfig, mission: CircleMission) -> str:
    payload = config.environment.payload
    offset_x, offset_y = payload.offset
    offset_radius = math.hypot(offset_x, offset_y)
    offset_angle = (
        0.0
        if offset_radius == 0.0
        else math.degrees(math.atan2(offset_y, offset_x))
    )
    randomization = "rand" if payload.randomize else "fixed"
    return "-".join(
        (
            f"rho{stable_float(mission.preset.radius)}",
            f"T{stable_float(mission.preset.period)}",
            f"m{stable_float(payload.mass * 1000.0)}g",
            f"r{stable_float(offset_radius * 1000.0)}mm",
            f"th{stable_float(offset_angle)}deg",
            randomization,
        )
    )


def mission_condition(
    config: ExperimentConfig,
    mission: ReferenceMission,
    *,
    legacy_circle_preset: bool = False,
    path_preset: str | None = None,
) -> str:
    """Build an artifact-safe description of the effective reference path."""

    _ensure_runtime_imports()

    if legacy_circle_preset and isinstance(mission, CircleMission):
        return _legacy_circle_condition(config, mission)
    if path_preset is not None:
        key = "".join(
            character
            for character in path_preset.strip().lower().replace("_", "-")
            if character.isalnum() or character == "-"
        ).strip("-")
        key = key or "custom"
        if mission.name == "hover":
            return f"h-T{stable_float(config.mission.hover.duration)}"
        if mission.name == "circle":
            circle = config.mission.circle
            return "-".join(
                (
                    f"c-{key}",
                    f"T{stable_float(circle.period)}",
                    f"L{stable_float(circle.laps)}",
                )
            )
        lissajous = config.mission.lissajous
        return "-".join(
            (
                f"l-{key}",
                f"T{stable_float(lissajous.base_period)}",
                f"C{stable_float(lissajous.cycles)}",
            )
        )
    if mission.name == "hover":
        hover = config.mission.hover
        x, y, z = hover.target
        return "-".join(
            (
                f"x{stable_float(x)}",
                f"y{stable_float(y)}",
                f"z{stable_float(z)}",
                f"yaw{stable_float(hover.yaw_deg)}",
                f"dur{stable_float(hover.duration)}",
                f"p{stable_float(config.environment.position_perturbation)}",
                f"a{stable_float(config.environment.attitude_perturbation_deg)}",
                f"floorReq{int(config.mission.force_floor_start)}",
            )
        )
    if mission.name == "circle":
        circle = config.mission.circle
        cx, cy = circle.center_xy
        return "-".join(
            (
                f"cx{stable_float(cx)}",
                f"cy{stable_float(cy)}",
                f"r{stable_float(circle.radius)}",
                f"T{stable_float(circle.period)}",
                f"laps{stable_float(circle.laps)}",
                f"z{stable_float(config.mission.hover_altitude)}",
                f"sa{stable_float(circle.start_angle_deg)}",
                circle.direction,
                f"ramp{stable_float(circle.ramp_sec)}",
                f"to{stable_float(config.mission.takeoff_sec)}",
                f"set{stable_float(config.mission.settle_sec)}",
                f"goto{stable_float(config.mission.goto_sec)}",
                f"hold{stable_float(config.mission.post_hold_sec)}",
                f"floorReq{int(config.mission.force_floor_start)}",
                f"p{stable_float(config.environment.position_perturbation)}",
                f"a{stable_float(config.environment.attitude_perturbation_deg)}",
            )
        )
    lissajous = config.mission.lissajous
    cx, cy = lissajous.center_xy
    ax, ay = lissajous.amplitude_xy
    frequency_x, frequency_y = lissajous.frequency_ratio
    return "-".join(
        (
            f"cx{stable_float(cx)}",
            f"cy{stable_float(cy)}",
            f"Ax{stable_float(ax)}",
            f"Ay{stable_float(ay)}",
            f"a{frequency_x}",
            f"b{frequency_y}",
            f"ph{stable_float(lissajous.phase_deg)}",
            f"T{stable_float(lissajous.base_period)}",
            f"cycles{stable_float(lissajous.cycles)}",
            f"z{stable_float(config.mission.hover_altitude)}",
            f"ramp{stable_float(lissajous.ramp_sec)}",
            f"to{stable_float(config.mission.takeoff_sec)}",
            f"set{stable_float(config.mission.settle_sec)}",
            f"goto{stable_float(config.mission.goto_sec)}",
            f"hold{stable_float(config.mission.post_hold_sec)}",
            f"floorReq{int(config.mission.force_floor_start)}",
            f"p{stable_float(config.environment.position_perturbation)}",
            f"a{stable_float(config.environment.attitude_perturbation_deg)}",
        )
    )


class EvaluationRunner:
    """Own environment, viewer, mission, policy, and rollout lifecycle."""

    def __init__(
        self,
        config: ExperimentConfig,
        artifacts: ArtifactManager,
        *,
        headless: bool,
        realtime: bool,
        camera_tracking: bool,
        preset: str = "1",
        mission: ReferenceMission | None = None,
        legacy_circle_preset: bool = True,
        environment_factory: Any | None = None,
    ) -> None:
        _ensure_runtime_imports()
        if environment_factory is None:
            from .factories import EnvironmentFactory

            environment_factory = EnvironmentFactory(config)
        self.config = config
        self.artifacts = artifacts
        self.factory = environment_factory
        self.headless = bool(headless)
        self.realtime = bool(realtime)
        self.camera_tracking = bool(camera_tracking)
        self.seed = config.evaluation.seed_start
        self.mission = mission or mission_from_experiment(
            config,
            mission_type=config.mission.type,
            preset=preset,
            legacy_circle_preset=legacy_circle_preset,
        )
        # Preserve the old public attribute used by downstream scripts/tests.
        self.circle = self.mission if isinstance(self.mission, CircleMission) else None

    def _active_mission(self) -> ReferenceMission | None:
        # The fallback keeps tests/downstream code that constructed an instance
        # with ``__new__`` and assigned only ``circle`` working.
        return getattr(self, "mission", None) or getattr(self, "circle", None)

    def run(
        self,
        policy: Any | None,
        policy_key: str,
        label: str,
        *,
        control_mode: str | None = None,
    ) -> RolloutTrace:
        mission = self._active_mission()
        episode_override = None
        if mission is not None and mission.name != "hover" and not self.headless:
            # Legacy trajectory viewers reserved two seconds beyond the loop.
            episode_override = mission.total_sec + 2.0
        overrides = {}
        if episode_override is not None:
            overrides["episode_sec"] = episode_override
        if control_mode is not None:
            overrides["mode"] = control_mode
        env = self.factory.make(**overrides)
        try:
            actual_control_mode = str(
                getattr(env, "mode", control_mode or self.config.control_mode)
            )
            return self._rollout(
                env,
                policy,
                policy_key,
                label,
                control_mode=actual_control_mode,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def _rollout(
        self,
        env: Any,
        policy: Any | None,
        policy_key: str,
        label: str,
        *,
        control_mode: str | None = None,
    ) -> RolloutTrace:
        _ensure_runtime_imports()
        mission = self._active_mission()
        env.dist_torque_body = np.zeros(3)
        observation, _ = env.reset(seed=self.seed)
        force_floor_start_requested = bool(
            mission is not None and self.config.mission.force_floor_start
        )
        force_floor_start_applied = False
        force_floor_start_error: str | None = None
        if force_floor_start_requested:
            (
                force_floor_start_applied,
                force_floor_start_error,
            ) = self._force_floor_start(env)

        actuator_snapshot: dict[str, Any] | None = None
        snapshotter = getattr(env, "actuator_snapshot", None)
        if callable(snapshotter):
            try:
                snapshot = snapshotter()
                if isinstance(snapshot, Mapping):
                    actuator_snapshot = dict(snapshot)
            except Exception as exc:
                actuator_snapshot = {
                    "snapshot_error": f"{type(exc).__name__}: {exc}"
                }

        dt = float(env.dt_phys * env.substeps)
        mission_steps = (
            int(round(mission.total_sec / dt)) if mission is not None else None
        )
        is_trajectory = mission is not None and mission.name != "hover"
        times: list[float] = []
        positions: list[np.ndarray] = []
        attitudes: list[np.ndarray] = []
        references: list[np.ndarray] = []
        errors: list[float] = []
        phases: list[str] = []
        control_inputs: list[np.ndarray] = []
        motor_thrusts: list[np.ndarray] = []
        motor_thrust_commands: list[np.ndarray] = []
        motor_commands: list[np.ndarray] = []
        motor_omegas: list[np.ndarray] = []
        reaction_torques: list[np.ndarray] = []
        wrench_commands: list[np.ndarray] = []
        wrench_actuals: list[np.ndarray] = []
        allocation_errors: list[np.ndarray] = []
        linear_velocities: list[np.ndarray] = []
        angular_velocities: list[np.ndarray] = []
        terminated_at: float | None = None
        truncated_at: float | None = None
        diverged_at: float | None = None
        training_boundary_crossed_at: float | None = None
        guard_boundary_crossed_at: float | None = None
        last_phase: str | None = None
        rollout_error: str | None = None
        step_index = 0

        def actuator_vector(
            attribute: str, fallback: Any = None
        ) -> np.ndarray:
            """Read optional BLDC diagnostics without breaking legacy fakes."""

            value = getattr(env, attribute, fallback)
            try:
                return np.asarray(value, dtype=float).reshape(4).copy()
            except (TypeError, ValueError):
                return np.full(4, np.nan, dtype=float)

        @contextmanager
        def capture_rollout_error():
            nonlocal rollout_error
            try:
                yield
            except Exception as exc:
                rollout_error = f"{type(exc).__name__}: {exc}"
                time_now = step_index * dt
                phase = last_phase or "UNKNOWN"
                print(
                    f"    rollout error at t={time_now:.2f}s "
                    f"phase={phase}: {rollout_error}"
                )

        with capture_rollout_error(), self._viewer_context(env) as viewer:
            self._configure_viewer(viewer)
            while True:
                if viewer is not None and not viewer.is_running():
                    break
                if mission_steps is not None:
                    if self.headless and step_index >= mission_steps:
                        break
                    if not self.headless and step_index * dt > mission.total_sec:
                        break
                time_now = step_index * dt
                if mission is None:
                    reference = np.asarray(env.pos_des, dtype=float).copy()
                    phase = "HOVER"
                else:
                    reference, phase = mission.reference(time_now)
                    env.pos_des = np.asarray(reference, dtype=float).copy()
                if phase != last_phase:
                    print(f"    t={time_now:6.2f}s phase -> {phase}")
                    last_phase = phase

                wall_start = time.time()
                try:
                    action = (
                        policy.predict(
                            observation,
                            deterministic=self.config.evaluation.deterministic,
                        )[0]
                        if policy is not None
                        else np.zeros(4)
                    )
                    applied_action = np.clip(
                        np.asarray(action, dtype=float).reshape(4), -1.0, 1.0
                    )
                    # Keep the environment's preserved dtype/clipping behavior;
                    # ``applied_action`` is the equivalent normalized signal
                    # recorded for diagnostics.
                    observation, _reward, terminated, truncated, _ = env.step(
                        action
                    )
                    thrust = np.asarray(
                        getattr(env, "_last_f", np.full(4, np.nan)), dtype=float
                    ).reshape(4)
                    commanded_thrust = actuator_vector("_last_f_cmd", thrust)
                    motor_command = actuator_vector("_last_motor_cmd")
                    motor_omega = actuator_vector("_last_omega")
                    reaction_torque = actuator_vector("_last_q_actual")
                    wrench_command = actuator_vector("_last_wrench_cmd")
                    wrench_actual = actuator_vector("_last_wrench_actual")
                    allocation_error = actuator_vector("_last_allocation_error")
                except Exception as exc:
                    rollout_error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"    rollout error at t={time_now:.2f}s "
                        f"phase={phase}: {rollout_error}"
                    )
                    break
                if not np.all(np.isfinite(observation)):
                    diverged_at = time_now
                    print(f"    non-finite observation at t={time_now:.2f}s")
                    break

                position_error = np.asarray(observation[0:3], dtype=float)
                actual_position = position_error + np.asarray(env.pos_des, dtype=float)
                linear_velocity = np.asarray(observation[3:6], dtype=float)
                attitude = quaternion_to_euler_deg(observation[6:10])
                angular_velocity = np.asarray(observation[10:13], dtype=float)
                error_norm = float(np.linalg.norm(position_error))
                if error_norm > 0.15 and training_boundary_crossed_at is None:
                    training_boundary_crossed_at = time_now
                if error_norm > 1.5 and guard_boundary_crossed_at is None:
                    guard_boundary_crossed_at = time_now

                if (
                    is_trajectory
                    and not self.headless
                    and (
                        actual_position[2] > 5.0
                        or error_norm > 5.0
                        or abs(attitude[0]) > 80.0
                        or abs(attitude[1]) > 80.0
                    )
                ):
                    diverged_at = time_now
                    print(f"    state divergence at t={time_now:.2f}s phase={phase}")
                    break

                times.append(time_now)
                positions.append(actual_position.copy())
                attitudes.append(attitude)
                references.append(np.asarray(env.pos_des, dtype=float).copy())
                errors.append(error_norm)
                phases.append(phase)
                control_inputs.append(applied_action.copy())
                motor_thrusts.append(thrust.copy())
                motor_thrust_commands.append(commanded_thrust)
                motor_commands.append(motor_command)
                motor_omegas.append(motor_omega)
                reaction_torques.append(reaction_torque)
                wrench_commands.append(wrench_command)
                wrench_actuals.append(wrench_actual)
                allocation_errors.append(allocation_error)
                linear_velocities.append(linear_velocity.copy())
                angular_velocities.append(angular_velocity.copy())
                step_index += 1

                if terminated and terminated_at is None:
                    terminated_at = time_now
                    if is_trajectory:
                        print(
                            f"    env guard at t={time_now:.2f}s phase={phase}; "
                            "continuing trajectory mission"
                        )
                if truncated and truncated_at is None:
                    truncated_at = time_now

                self._sync_viewer(viewer, actual_position, dt, wall_start)

                if not is_trajectory and (terminated or truncated):
                    break
                if is_trajectory and not self.headless and truncated:
                    break

        return RolloutTrace(
            policy=policy_key,
            label=label,
            time_sec=np.asarray(times, dtype=float),
            position=np.asarray(positions, dtype=float).reshape((-1, 3)),
            attitude_deg=np.asarray(attitudes, dtype=float).reshape((-1, 3)),
            reference_position=np.asarray(references, dtype=float).reshape((-1, 3)),
            position_error=np.asarray(errors, dtype=float),
            phases=tuple(phases),
            training_boundary_crossed_at=training_boundary_crossed_at,
            guard_boundary_crossed_at=guard_boundary_crossed_at,
            terminated_at=terminated_at,
            truncated_at=truncated_at,
            diverged_at=diverged_at,
            force_floor_start_requested=force_floor_start_requested,
            force_floor_start_applied=force_floor_start_applied,
            force_floor_start_error=force_floor_start_error,
            error=rollout_error,
            control_input=np.asarray(control_inputs, dtype=float).reshape((-1, 4)),
            motor_thrust=np.asarray(motor_thrusts, dtype=float).reshape((-1, 4)),
            linear_velocity=np.asarray(linear_velocities, dtype=float).reshape((-1, 3)),
            angular_velocity=np.asarray(angular_velocities, dtype=float).reshape((-1, 3)),
            control_mode=control_mode,
            motor_thrust_command=np.asarray(
                motor_thrust_commands, dtype=float
            ).reshape((-1, 4)),
            motor_command=np.asarray(motor_commands, dtype=float).reshape((-1, 4)),
            motor_omega_rad_s=np.asarray(motor_omegas, dtype=float).reshape((-1, 4)),
            reaction_torque_nm=np.asarray(reaction_torques, dtype=float).reshape(
                (-1, 4)
            ),
            wrench_command=np.asarray(wrench_commands, dtype=float).reshape((-1, 4)),
            wrench_actual=np.asarray(wrench_actuals, dtype=float).reshape((-1, 4)),
            allocation_error=np.asarray(allocation_errors, dtype=float).reshape(
                (-1, 4)
            ),
            actuator=actuator_snapshot,
            episode_mass_kg=_cached_episode_mass(env),
        )

    def _viewer_context(self, env: Any):
        if self.headless:
            return nullcontext(None)
        try:
            import mujoco.viewer
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "MuJoCo viewer is unavailable; rerun with --headless"
            ) from exc
        return mujoco.viewer.launch_passive(env.model, env.data)

    def _configure_viewer(self, viewer: Any | None) -> None:
        if viewer is None:
            return
        mission = self._active_mission()
        if mission is None or mission.name == "hover":
            viewer.cam.distance = 3.7
            return
        parameters = mission.effective_parameters()
        center = np.asarray(parameters.get("center_xy", (0.0, 0.0)), dtype=float)
        with viewer.lock():
            viewer.cam.distance = 4.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.lookat[:] = np.array(
                [center[0], center[1], self.config.mission.hover_altitude]
            )

    def _sync_viewer(
        self,
        viewer: Any | None,
        actual_position: np.ndarray,
        dt: float,
        wall_start: float,
    ) -> None:
        if viewer is None:
            return
        mission = self._active_mission()
        if (
            mission is not None
            and mission.name != "hover"
            and self.camera_tracking
        ):
            with viewer.lock():
                viewer.cam.lookat[:] = (
                    0.9 * np.asarray(viewer.cam.lookat) + 0.1 * actual_position
                )
        viewer.sync()
        if self.realtime:
            slack = dt - (time.time() - wall_start)
            if slack > 0.0:
                time.sleep(slack)

    @staticmethod
    def _force_floor_start(env: Any) -> tuple[bool, str | None]:
        _ensure_runtime_imports()
        original_qpos = None
        original_qvel = None
        mujoco_module = None
        try:
            import mujoco as mujoco_module

            original_qpos = np.asarray(env.data.qpos).copy()
            original_qvel = np.asarray(env.data.qvel).copy()
            env.data.qpos[0:3] = np.array([0.0, 0.0, 0.02])
            env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            env.data.qvel[:6] = 0.0
            mujoco_module.mj_forward(env.model, env.data)
            # ``view_live`` changes the reset pose after ``env.reset``.  The
            # BLDC rotor state must therefore be explicitly told this is a
            # ground start. Keep the capability optional for lightweight test
            # environments and legacy wrappers.
            reset_actuator = getattr(env, "reset_actuator_state", None)
            if callable(reset_actuator):
                reset_actuator(airborne=False)
            return True, None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            restore_status = "reset state was not modified"
            if original_qpos is not None and original_qvel is not None:
                try:
                    env.data.qpos[:] = original_qpos
                    env.data.qvel[:] = original_qvel
                    if mujoco_module is not None:
                        mujoco_module.mj_forward(env.model, env.data)
                    reset_actuator = getattr(env, "reset_actuator_state", None)
                    if callable(reset_actuator):
                        # Restore a consistent actuator state as well as the
                        # MuJoCo pose if the floor-start transaction failed.
                        reset_actuator()
                    restore_status = "reset state restored"
                except Exception as restore_exc:
                    restore_error = (
                        f"{type(restore_exc).__name__}: {restore_exc}"
                    )
                    error += f"; restore failed: {restore_error}"
                    restore_status = "reset state restoration FAILED"
            print(
                "    warning: force_floor_start failed "
                f"({error}); {restore_status}"
            )
            return False, error


def trace_metrics(trace: RolloutTrace, tail_fraction: float) -> dict[str, Any]:
    """Return the complete, JSON-safe tracking metric contract."""

    _ensure_runtime_imports()

    result: dict[str, Any] = {
        "label": trace.label,
        "control_mode": trace.control_mode,
        "sample_count": trace.sample_count,
        "completed_sample_count": trace.sample_count,
        "duration_sec": float(trace.time_sec[-1]) if trace.sample_count else 0.0,
        "training_boundary_crossed_at": trace.training_boundary_crossed_at,
        "guard_boundary_crossed_at": trace.guard_boundary_crossed_at,
        "terminated_at": trace.terminated_at,
        "truncated_at": trace.truncated_at,
        "diverged_at": trace.diverged_at,
        "force_floor_start_requested": trace.force_floor_start_requested,
        "force_floor_start_applied": trace.force_floor_start_applied,
        "force_floor_start_error": trace.force_floor_start_error,
        "error": trace.error,
    }
    control = (
        np.asarray(trace.control_input, dtype=float).reshape((-1, 4))
        if trace.control_input is not None
        else np.empty((0, 4), dtype=float)
    )
    thrust = (
        np.asarray(trace.motor_thrust, dtype=float).reshape((-1, 4))
        if trace.motor_thrust is not None
        else np.empty((0, 4), dtype=float)
    )
    commanded_thrust = (
        np.asarray(trace.motor_thrust_command, dtype=float).reshape((-1, 4))
        if trace.motor_thrust_command is not None
        else np.empty((0, 4), dtype=float)
    )
    motor_command = (
        np.asarray(trace.motor_command, dtype=float).reshape((-1, 4))
        if trace.motor_command is not None
        else np.empty((0, 4), dtype=float)
    )
    motor_omega = (
        np.asarray(trace.motor_omega_rad_s, dtype=float).reshape((-1, 4))
        if trace.motor_omega_rad_s is not None
        else np.empty((0, 4), dtype=float)
    )
    reaction_torque = (
        np.asarray(trace.reaction_torque_nm, dtype=float).reshape((-1, 4))
        if trace.reaction_torque_nm is not None
        else np.empty((0, 4), dtype=float)
    )
    wrench_command = (
        np.asarray(trace.wrench_command, dtype=float).reshape((-1, 4))
        if trace.wrench_command is not None
        else np.empty((0, 4), dtype=float)
    )
    wrench_actual = (
        np.asarray(trace.wrench_actual, dtype=float).reshape((-1, 4))
        if trace.wrench_actual is not None
        else np.empty((0, 4), dtype=float)
    )
    allocation_error = (
        np.asarray(trace.allocation_error, dtype=float).reshape((-1, 4))
        if trace.allocation_error is not None
        else np.empty((0, 4), dtype=float)
    )

    def finite_column_summary(
        values: np.ndarray, reducer: Callable[[np.ndarray], float]
    ) -> list[float | None]:
        summary: list[float | None] = []
        for column in range(4):
            finite = values[:, column][np.isfinite(values[:, column])]
            summary.append(float(reducer(finite)) if finite.size else None)
        return summary

    result["control_input_abs_max"] = finite_column_summary(
        np.abs(control), np.max
    )
    result["motor_thrust_n_min"] = finite_column_summary(thrust, np.min)
    result["motor_thrust_n_max"] = finite_column_summary(thrust, np.max)
    result["motor_thrust_n_mean"] = finite_column_summary(thrust, np.mean)
    result["motor_thrust_command_n_min"] = finite_column_summary(
        commanded_thrust, np.min
    )
    result["motor_thrust_command_n_max"] = finite_column_summary(
        commanded_thrust, np.max
    )
    result["motor_thrust_command_n_mean"] = finite_column_summary(
        commanded_thrust, np.mean
    )
    result["motor_command_min"] = finite_column_summary(motor_command, np.min)
    result["motor_command_max"] = finite_column_summary(motor_command, np.max)
    result["motor_omega_rad_s_min"] = finite_column_summary(motor_omega, np.min)
    result["motor_omega_rad_s_max"] = finite_column_summary(motor_omega, np.max)
    result["reaction_torque_nm_min"] = finite_column_summary(
        reaction_torque, np.min
    )
    result["reaction_torque_nm_max"] = finite_column_summary(
        reaction_torque, np.max
    )

    def final_vector(values: np.ndarray) -> list[float | None]:
        if not values.size:
            return [None, None, None, None]
        return [
            float(value) if np.isfinite(value) else None
            for value in values[-1]
        ]

    result["wrench_command_final"] = final_vector(wrench_command)
    result["wrench_actual_final"] = final_vector(wrench_actual)
    result["allocation_error_final"] = final_vector(allocation_error)
    if trace.actuator is not None:
        result["actuator"] = dict(trace.actuator)
    if not trace.sample_count:
        result.update(
            position_rmse=None,
            position_rmse_xy=None,
            position_rmse_z=None,
            tail_position_rmse_xy=None,
            tail_position_rmse_z=None,
            mean_position_error=None,
            max_position_error=None,
            tail_mean_position_error=None,
            trajectory_phase_rmse=None,
            trajectory_phase_rmse_xy=None,
            trajectory_phase_rmse_z=None,
            phases={},
        )
        return result

    errors = np.asarray(trace.position_error, dtype=float)
    from .evaluation import position_rmse_metrics

    error_vectors = np.asarray(trace.position) - np.asarray(trace.reference_position)
    tail_start = int(trace.sample_count * (1.0 - tail_fraction))
    result.update(position_rmse_metrics(error_vectors))
    result.update({f"tail_{key}": value for key, value in
                   position_rmse_metrics(error_vectors[tail_start:]).items()})
    result["position_rmse"] = float(np.sqrt(np.mean(np.square(errors))))
    result["mean_position_error"] = float(np.mean(errors))
    result["max_position_error"] = float(np.max(errors))
    result["tail_mean_position_error"] = float(np.mean(errors[tail_start:]))

    phase_values = np.asarray(trace.phases)
    phase_metrics: dict[str, Any] = {}
    for phase in dict.fromkeys(trace.phases):
        selected = errors[phase_values == phase]
        phase_metrics[phase] = {
            **position_rmse_metrics(error_vectors[phase_values == phase]),
            "count": int(selected.size),
            "rmse": float(np.sqrt(np.mean(np.square(selected)))),
            "mean_position_error": float(np.mean(selected)),
            "max_position_error": float(np.max(selected)),
            "end_position_error": float(selected[-1]),
        }
    result["phases"] = phase_metrics
    trajectory_names = ("LISSAJOUS", "CIRCLE", "HOVER")
    trajectory_errors = next(
        (
            errors[phase_values == phase]
            for phase in trajectory_names
            if np.any(phase_values == phase)
        ),
        np.asarray([], dtype=float),
    )
    result["trajectory_phase_rmse"] = (
        float(np.sqrt(np.mean(np.square(trajectory_errors))))
        if trajectory_errors.size
        else None
    )
    trajectory_phase = next((phase for phase in trajectory_names if phase in phase_metrics), None)
    for axis in ("xy", "z"):
        result[f"trajectory_phase_rmse_{axis}"] = (
            phase_metrics[trajectory_phase][f"position_rmse_{axis}"]
            if trajectory_phase is not None else None
        )
    return result


# Compatibility name retained for downstream imports.
_trace_metrics = trace_metrics


def _save_trace(
    artifacts: ArtifactManager,
    config: ExperimentConfig,
    trace: RolloutTrace,
    mission: ReferenceMission,
    title_condition: str | None = None,
) -> Path:
    _ensure_runtime_imports()
    if trace.sample_count < 1:
        raise RuntimeError(f"{trace.label} rollout produced no samples")
    output = artifacts.path("plots", f"trajectory-{trace.policy}", ".png")
    if trace.force_floor_start_requested:
        floor_status = (
            "force-floor=applied"
            if trace.force_floor_start_applied
            else "force-floor=FAILED"
        )
    else:
        floor_status = "force-floor=not-requested"
    plot_tag = trace.label
    if title_condition:
        plot_tag += f"\n{title_condition}"
    plot_tag += f"\n{floor_status}"
    if mission.name == "hover":
        return save_hover_trace(
            output,
            tag=plot_tag,
            time_sec=trace.time_sec,
            position=trace.position,
            attitude_deg=trace.attitude_deg,
            hover_altitude=config.mission.hover.target[2],
            reference_position=trace.reference_position,
            position_error=trace.position_error,
        )
    return save_tracking_trace(
        output,
        tag=plot_tag,
        time_sec=trace.time_sec,
        position=trace.position,
        attitude_deg=trace.attitude_deg,
        reference_position=trace.reference_position,
        position_error=trace.position_error,
        mission_name=mission.name,
        mission_parameters=mission.effective_parameters(),
        phases=trace.phases,
    )


def _save_policy_report(
    artifacts: ArtifactManager,
    trace: RolloutTrace,
    mission: ReferenceMission,
    *,
    title_condition: str,
) -> Path:
    """Save one unified 16:9 report with a short policy-specific filename."""

    _ensure_runtime_imports()
    if trace.sample_count < 1:
        raise RuntimeError(f"{trace.label} rollout produced no samples")
    filename = "floor.png" if trace.policy == "floor" else "ppo.png"
    output = artifacts.run_dir / "plots" / filename
    return save_policy_trace(
        output,
        tag=title_condition,
        rollout=trace,
        mission_name=mission.name,
        mission_parameters=mission.effective_parameters(),
        motor_unit="N",
    )


def build_parser(default_config: str | Path, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=Path, default=Path(default_config), help="YAML profile"
    )
    parser.add_argument(
        "--mode",
        type=normalize_mode,
        default=None,
        metavar="{1,2,3,hover,circle,lissajous}",
        help="flight-test mission (unified view_live.py)",
    )
    parser.add_argument(
        "--path-preset",
        default=None,
        metavar="KEY",
        help=(
            "predefined trajectory geometry (for example: small, wide, "
            "figure8, or clover)"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "PPO archive, or latest-best for the newest matching best archive "
            "(default: <legacy_model_root>/ppo_best.zip)"
        ),
    )
    parser.add_argument(
        "--policy",
        choices=("floor", "residual", "both"),
        default="both",
        help="policy comparison (unified view_live.py always runs both)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--position-perturbation", type=float, default=None)
    parser.add_argument("--attitude-perturbation-deg", type=float, default=None)

    viewer = parser.add_mutually_exclusive_group()
    viewer.add_argument("--headless", dest="headless", action="store_true")
    viewer.add_argument("--viewer", dest="headless", action="store_false")
    parser.set_defaults(headless=False)

    pacing = parser.add_mutually_exclusive_group()
    pacing.add_argument(
        "--no-realtime", dest="no_realtime", action="store_true"
    )
    pacing.add_argument("--realtime", dest="no_realtime", action="store_false")
    parser.set_defaults(no_realtime=False)

    camera = parser.add_mutually_exclusive_group()
    camera.add_argument("--no-camera", dest="no_camera", action="store_true")
    camera.add_argument("--camera", dest="no_camera", action="store_false")
    parser.set_defaults(no_camera=False)

    floor_start = parser.add_mutually_exclusive_group()
    floor_start.add_argument(
        "--force-floor-start", dest="force_floor_start", action="store_true"
    )
    floor_start.add_argument(
        "--no-force-floor-start", dest="force_floor_start", action="store_false"
    )
    parser.set_defaults(force_floor_start=None)

    parser.add_argument("--preset", default="1", help="legacy circle preset key")
    parser.add_argument("--hover-target", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--yaw-deg", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--circle-period", type=float, default=None)
    parser.add_argument(
        "--period",
        type=float,
        default=None,
        help="seconds per circle lap or Lissajous base cycle",
    )
    parser.add_argument("--laps", type=float, default=None)
    parser.add_argument("--altitude", type=float, default=None)
    parser.add_argument("--ramp-sec", type=float, default=None)
    parser.add_argument("--start-angle-deg", type=float, default=None)
    parser.add_argument(
        "--direction",
        type=normalize_direction,
        default=None,
        metavar="{cw,clockwise,ccw,counterclockwise}",
    )
    parser.add_argument("--amplitude-x", type=float, default=None)
    parser.add_argument("--amplitude-y", type=float, default=None)
    parser.add_argument("--frequency-x", type=int, default=None)
    parser.add_argument("--frequency-y", type=int, default=None)
    parser.add_argument("--phase-deg", type=float, default=None)
    parser.add_argument("--base-period", type=float, default=None)
    parser.add_argument("--cycles", type=float, default=None)
    parser.add_argument("--takeoff-sec", type=float, default=None)
    parser.add_argument("--settle-sec", type=float, default=None)
    parser.add_argument("--goto-sec", type=float, default=None)
    parser.add_argument("--post-hold-sec", type=float, default=None)
    return parser


def _prompt(
    label: str,
    default: Any,
    convert: Callable[[str], Any],
    input_fn: Callable[[str], str],
    unit: str | None = None,
) -> Any:
    displayed_default = f"{default} {unit}" if unit else default
    while True:
        raw = input_fn(f"{label} [{displayed_default}]: ").strip()
        if not raw:
            return default
        try:
            return convert(raw)
        except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
            print(f"Invalid value: {exc}")


def _prompt_bool(
    label: str, default: bool, input_fn: Callable[[str], str]
) -> bool:
    display = "Y/n" if default else "y/N"

    def convert(value: str) -> bool:
        key = value.strip().lower()
        if key in {"y", "yes", "1", "true"}:
            return True
        if key in {"n", "no", "0", "false"}:
            return False
        raise ValueError("enter yes or no")

    while True:
        raw = input_fn(f"{label} [{display}]: ").strip()
        if not raw:
            return default
        try:
            return convert(raw)
        except ValueError as exc:
            print(f"Invalid value: {exc}")


def _cli_or_config_default(
    args: argparse.Namespace, destination: str, config_default: Any
) -> Any:
    """Prefer a parsed CLI override when choosing an interactive default."""

    cli_value = getattr(args, destination, None)
    return config_default if cli_value is None else cli_value


def prompt_runtime_options(
    args: argparse.Namespace,
    config: ExperimentConfig,
    mode: str,
    *,
    input_fn: Callable[[str], str] = input,
) -> argparse.Namespace:
    """Prompt only for the few choices an operator normally changes."""

    if mode == "hover":
        args.duration = _prompt(
            "Hover duration",
            _cli_or_config_default(
                args, "duration", config.mission.hover.duration
            ),
            float,
            input_fn,
            "s",
        )
    elif mode == "circle":
        args.period = _prompt(
            "Seconds per lap",
            _cli_or_config_default(
                args,
                "period",
                _cli_or_config_default(
                    args, "circle_period", config.mission.circle.period
                ),
            ),
            float,
            input_fn,
            "s",
        )
        args.laps = _prompt(
            "Number of laps",
            _cli_or_config_default(args, "laps", config.mission.circle.laps),
            float,
            input_fn,
        )
    else:
        args.period = _prompt(
            "Seconds per cycle",
            _cli_or_config_default(
                args,
                "period",
                _cli_or_config_default(
                    args, "base_period", config.mission.lissajous.base_period
                ),
            ),
            float,
            input_fn,
            "s",
        )
        args.cycles = _prompt(
            "Number of cycles",
            _cli_or_config_default(
                args, "cycles", config.mission.lissajous.cycles
            ),
            float,
            input_fn,
        )
    return args


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def resolve_path_preset(
    mode: str,
    value: str | None,
    path_profiles: Mapping[str, Mapping[str, str | Path]],
) -> tuple[str, Path] | None:
    """Resolve a named or numeric geometry preset for one trajectory mode."""

    available = path_profiles.get(mode)
    if not available:
        if value is not None:
            raise argparse.ArgumentTypeError(
                f"--path-preset is not available for {mode}"
            )
        return None
    keys = tuple(available)
    selected = keys[0] if value is None else str(value).strip().lower()
    for index, key in enumerate(keys, start=1):
        aliases = {str(index), key.lower(), f"{mode}-{key}".lower()}
        if selected in aliases:
            return key, Path(available[key])
    choices = ", ".join(f"{index}/{key}" for index, key in enumerate(keys, 1))
    raise argparse.ArgumentTypeError(
        f"unknown {mode} path preset {value!r}; choose {choices}"
    )


def _interactive_path_preset(
    mode: str,
    path_profiles: Mapping[str, Mapping[str, str | Path]],
    input_fn: Callable[[str], str],
    default: str | None = None,
) -> str | None:
    available = path_profiles.get(mode)
    if not available:
        return None
    print(f"\n{mode.title()} paths:")
    for index, key in enumerate(available, start=1):
        print(f"[{index}] {key}")
    resolved_default = resolve_path_preset(mode, default, path_profiles)
    if resolved_default is None:  # defensive: ``available`` is non-empty above
        return None
    default_key = resolved_default[0]
    return _prompt(
        "Select path preset",
        default_key,
        lambda value: resolve_path_preset(mode, value, path_profiles)[0],
        input_fn,
    )


def _interactive_mode(input_fn: Callable[[str], str]) -> str:
    print("=== Crazyflie trajectory test ===")
    print("[1] Hover")
    print("[2] Circle trajectory")
    print("[3] Lissajous trajectory")
    return normalize_mode(input_fn("Select mode [1-3]: "))


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_run_summary(
    config: ExperimentConfig,
    mission: ReferenceMission,
    *,
    policy: str,
    model: Path | None,
    headless: bool | None = None,
    realtime: bool | None = None,
    camera_tracking: bool | None = None,
    path_preset: str | None = None,
    unified: bool = False,
) -> str:
    """Format the effective pre-flight summary displayed to the operator."""

    rows: list[tuple[str, str]] = [("Mode", mission.name.title())]
    if unified:
        rows.extend(
            (
                ("Floor controller", "RESIDUAL (PID)"),
                ("PPO controller", config.control_mode.upper()),
            )
        )
    else:
        rows.append(("Control mode", config.control_mode.upper()))
    actuator = config.actuator
    actuator_status = f"CF2.1 first-order ({actuator.verification_status})"
    rows.append(("Motor actuator", actuator_status))
    if path_preset is not None:
        rows.append(("Path preset", path_preset))
    if mission.name == "hover":
        hover = config.mission.hover
        rows.extend(
            (
                ("Target", f"{tuple(hover.target)} m"),
                ("Yaw", f"{_format_number(hover.yaw_deg)} deg"),
                ("Duration", f"{_format_number(hover.duration)} s"),
            )
        )
    elif mission.name == "circle":
        circle = config.mission.circle
        rows.extend(
            (
                ("Center", f"{tuple(circle.center_xy)} m"),
                ("Radius", f"{_format_number(circle.radius)} m"),
                ("Period", f"{_format_number(circle.period)} s"),
                ("Laps", _format_number(circle.laps)),
                ("Start angle", f"{_format_number(circle.start_angle_deg)} deg"),
                ("Direction", circle.direction),
                ("Altitude", f"{_format_number(config.mission.hover_altitude)} m"),
                ("Ramp", f"{_format_number(circle.ramp_sec)} s"),
            )
        )
    else:
        lissajous = config.mission.lissajous
        rows.extend(
            (
                ("Center", f"{tuple(lissajous.center_xy)} m"),
                ("Amplitude", f"{tuple(lissajous.amplitude_xy)} m"),
                (
                    "Frequency ratio",
                    f"{lissajous.frequency_ratio[0]}:{lissajous.frequency_ratio[1]}",
                ),
                ("Phase", f"{_format_number(lissajous.phase_deg)} deg"),
                ("Base period", f"{_format_number(lissajous.base_period)} s"),
                ("Cycles", _format_number(lissajous.cycles)),
                ("Altitude", f"{_format_number(config.mission.hover_altitude)} m"),
                ("Ramp", f"{_format_number(lissajous.ramp_sec)} s"),
            )
        )
    if mission.name != "hover":
        rows.extend(
            (
                ("Takeoff", f"{_format_number(config.mission.takeoff_sec)} s"),
                ("Settle", f"{_format_number(config.mission.settle_sec)} s"),
                ("GOTO", f"{_format_number(config.mission.goto_sec)} s"),
                (
                    "Post-HOLD",
                    f"{_format_number(config.mission.post_hold_sec)} s",
                ),
            )
        )
    rows.append(("Force floor request", str(config.mission.force_floor_start)))
    rows.extend(
        (
            ("Policy", policy),
            ("Model", str(model) if model is not None else "n/a"),
            ("Seed", str(config.evaluation.seed_start)),
            (
                "Position perturb.",
                f"{_format_number(config.environment.position_perturbation)} m",
            ),
            (
                "Attitude perturb.",
                f"{_format_number(config.environment.attitude_perturbation_deg)} deg",
            ),
        )
    )
    if headless is not None:
        rows.append(("Display", "headless" if headless else "viewer"))
    if realtime is not None:
        rows.append(("Realtime", "enabled" if realtime else "disabled"))
    if camera_tracking is not None:
        rows.append(
            (
                "Camera tracking",
                "enabled" if camera_tracking else "disabled",
            )
        )
    return "\n".join(f"{label:<18}: {value}" for label, value in rows)


def _runtime_payload(
    config: ExperimentConfig,
    mission: ReferenceMission,
    args: argparse.Namespace,
    model: Path | None,
    *,
    unified: bool = False,
) -> dict[str, Any]:
    return {
        "mode": mission.name,
        "position_xy_weight": config.environment.reward.effective_position_xy_weight,
        "position_z_weight": config.environment.reward.effective_position_z_weight,
        "action_scale": list(config.environment.residual_scale),
        "path_preset": getattr(args, "path_preset", None),
        "mission": mission.effective_parameters(),
        "control_mode": config.control_mode,
        "policy_control_modes": _expected_policy_control_modes(
            config,
            args.policy,
            unified=unified,
        ),
        "policy": args.policy,
        "model": str(model) if model is not None else None,
        "seed": config.evaluation.seed_start,
        "headless": bool(args.headless),
        "realtime": not args.no_realtime,
        "camera_tracking": not args.no_camera,
        "position_perturbation": config.environment.position_perturbation,
        "attitude_perturbation_deg": config.environment.attitude_perturbation_deg,
        "force_floor_start_requested": config.mission.force_floor_start,
        # Retain every static actuator parameter in the runtime-resolved
        # artifact.  Per-episode sampled values are appended to each rollout
        # outcome after the environment reset has actually occurred.
        "actuator": asdict(config.actuator),
    }


def run_evaluation_cli(
    *,
    default_config: str | Path,
    description: str,
    argv: Sequence[str] | None = None,
    unified: bool = False,
    mode_profiles: Mapping[str, str | Path] | None = None,
    path_profiles: Mapping[str, Mapping[str, str | Path]] | None = None,
    always_compare: bool = False,
    stdin: TextIO | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(default_config, description)
    args = parser.parse_args(raw_argv)
    if always_compare:
        if args.policy != "both":
            print(
                "note: unified view_live.py always compares floor and PPO; "
                f"ignoring --policy {args.policy}"
            )
        args.policy = "both"
    explicit_config = _has_option(raw_argv, "--config")
    stream = sys.stdin if stdin is None else stdin
    interactive_answers = False

    preliminary_config: ExperimentConfig | None = None
    mode = args.mode
    if unified and mode is None:
        if explicit_config:
            _ensure_runtime_imports()
            preliminary_config = load_config(args.config)
            mode = normalize_mode(preliminary_config.mission.type)
        elif bool(getattr(stream, "isatty", lambda: False)()):
            try:
                mode = _interactive_mode(input_fn)
            except (EOFError, argparse.ArgumentTypeError) as exc:
                parser.error(str(exc))
            interactive_answers = True
        else:
            parser.error(
                "--mode is required when stdin is not a TTY; for example: "
                "python view_live.py --mode hover --headless"
            )

    if explicit_config and args.path_preset is not None:
        parser.error("--config and --path-preset cannot be used together")

    selected_path_profile: Path | None = None
    if path_profiles is not None and mode is not None:
        try:
            if interactive_answers:
                args.path_preset = _interactive_path_preset(
                    mode,
                    path_profiles,
                    input_fn,
                    default=args.path_preset,
                )
            if not explicit_config:
                resolved = resolve_path_preset(
                    mode, args.path_preset, path_profiles
                )
                if resolved is not None:
                    args.path_preset, selected_path_profile = resolved
                elif mode == "hover":
                    args.path_preset = "hover"
            elif mode == "hover":
                args.path_preset = "hover"
            else:
                args.path_preset = "custom"
        except (EOFError, argparse.ArgumentTypeError) as exc:
            parser.error(str(exc))

    _ensure_runtime_imports()
    if preliminary_config is None:
        config_path = args.config
        if selected_path_profile is not None:
            config_path = selected_path_profile
        elif unified and not explicit_config and mode_profiles is not None and mode:
            config_path = Path(mode_profiles[mode])
        config = load_config(config_path)
    else:
        config = preliminary_config
    if mode is None:
        mode = normalize_mode(config.mission.type)

    if interactive_answers:
        args = prompt_runtime_options(args, config, mode, input_fn=input_fn)
    config = apply_runtime_overrides(config, args, mode)
    legacy_circle_preset = not unified and mode == "circle"
    mission = mission_from_experiment(
        config,
        mission_type=mode,
        preset=args.preset,
        legacy_circle_preset=legacy_circle_preset,
    )
    if unified:
        config = replace(
            config,
            environment=replace(config.environment, episode_sec=mission.total_sec),
        )

    specs = _policy_specs(config, args.policy, unified=unified)
    reward = config.environment.reward
    print(
        f"[config] position_xy_weight={reward.effective_position_xy_weight:g} "
        f"position_z_weight={reward.effective_position_z_weight:g} "
        f"action_scale={config.environment.residual_scale}"
    )
    requires_model = any(key == "residual" for key, _label in specs)
    candidate_model = _model_candidate(args.model, config) if requires_model else None
    condition = mission_condition(
        config,
        mission,
        legacy_circle_preset=legacy_circle_preset,
        path_preset=(
            args.path_preset if path_profiles is not None else None
        ),
    )
    artifacts = ArtifactManager.create(
        config,
        command=list(sys.argv if argv is None else argv),
        condition=condition,
        mission=mission.name,
        seed=config.evaluation.seed_start,
    )
    runtime_values = _runtime_payload(
        config,
        mission,
        args,
        candidate_model,
        unified=unified,
    )
    metrics: dict[str, Any] = {
        "status": "running",
        "effective_condition": condition,
        "effective_parameters": mission.effective_parameters(),
        "path_preset": getattr(args, "path_preset", None),
        "runtime_parameters": runtime_values,
        "control_mode": config.control_mode,
        "actuator": asdict(config.actuator),
        "selected_policy": args.policy,
        "seed": config.evaluation.seed_start,
        "deterministic": config.evaluation.deterministic,
        "headless": bool(args.headless),
        "realtime": not args.no_realtime,
        "camera_tracking": not args.no_camera,
        "position_perturbation": config.environment.position_perturbation,
        "attitude_perturbation_deg": config.environment.attitude_perturbation_deg,
        "force_floor_start_requested": config.mission.force_floor_start,
        "preset": args.preset if legacy_circle_preset else None,
        "model": str(candidate_model) if candidate_model is not None else None,
        "policy_control_modes": _expected_policy_control_modes(
            config,
            args.policy,
            unified=unified,
        ),
        "policies": {},
    }
    selected_model: Path | None = None
    runtime_config_written = False
    try:
        selected_model = _model_path(args.model, config) if requires_model else None
        runtime_values = _runtime_payload(
            config,
            mission,
            args,
            selected_model,
            unified=unified,
        )
        metrics["runtime_parameters"] = runtime_values
        metrics["model"] = (
            str(selected_model) if selected_model is not None else None
        )
        print(
            "\n"
            + format_run_summary(
                config,
                mission,
                policy=args.policy,
                model=selected_model,
                headless=bool(args.headless),
                realtime=not args.no_realtime,
                camera_tracking=not args.no_camera,
                path_preset=getattr(args, "path_preset", None),
                unified=unified,
            )
        )
        artifacts.write_runtime_config(runtime_values)
        runtime_config_written = True
        config.require_runtime_resources()
        policy = (
            _load_policy(selected_model, config)
            if selected_model is not None
            else None
        )
        runner = EvaluationRunner(
            config,
            artifacts,
            headless=args.headless,
            realtime=not args.no_realtime,
            camera_tracking=not args.no_camera,
            preset=args.preset,
            mission=mission,
            legacy_circle_preset=legacy_circle_preset,
        )
        traces: list[RolloutTrace] = []
        comparison_mode = unified and len(specs) == 2
        rollout_failure: str | None = None
        for policy_key, label in specs:
            print(f"\n>>> {label}")
            trace = runner.run(
                policy if policy_key == "residual" else None,
                policy_key,
                label,
                control_mode=(
                    "residual" if unified and policy_key == "floor" else None
                ),
            )
            policy_metrics = trace_metrics(trace, config.evaluation.tail_fraction)
            metrics["policies"][policy_key] = policy_metrics
            metrics["policy_control_modes"][policy_key] = trace.control_mode
            traces.append(trace)
            if trace.sample_count:
                plot_path = (
                    _save_policy_report(
                        artifacts, trace, mission, title_condition=condition
                    )
                    if comparison_mode
                    else _save_trace(
                        artifacts,
                        config,
                        trace,
                        mission,
                        title_condition=condition,
                    )
                )
                policy_metrics["plot"] = plot_path.relative_to(
                    artifacts.run_dir
                ).as_posix()
                print(f"    saved: {plot_path}")
            if trace.error is not None and rollout_failure is None:
                rollout_failure = (
                    f"{label} rollout failed after {trace.sample_count} samples: "
                    f"{trace.error}"
                )
            if not trace.sample_count and rollout_failure is None:
                rollout_failure = f"{label} rollout produced no samples"
            if rollout_failure is not None and not comparison_mode:
                raise RuntimeError(rollout_failure)
        if rollout_failure is not None:
            raise RuntimeError(rollout_failure)
        metrics["status"] = "completed"
        metrics_path = artifacts.write_metrics("evaluation", metrics)
        if unified:
            # Post-processing only: both existing rollouts and plots are done.
            from .reward_balance import build_reward_balance_report, format_reward_balance

            balance = build_reward_balance_report(
                traces, config,
                model=str(selected_model) if selected_model is not None else None,
            )
            balance_path = artifacts.write_metrics("reward_balance", balance)
            print(format_reward_balance(balance))
            print(f"reward balance: {balance_path}")
            from .yaw_authority import (
                build_yaw_authority_report,
                format_yaw_authority,
                save_yaw_authority_plot,
            )

            yaw = build_yaw_authority_report(
                traces, config,
                model=str(selected_model) if selected_model is not None else None,
            )
            yaw_plot = save_yaw_authority_plot(
                artifacts.path("plots", "yaw_authority", ".png"), traces, config,
            )
            yaw["plot"] = yaw_plot.relative_to(artifacts.run_dir).as_posix()
            yaw_path = artifacts.write_metrics("yaw_authority", yaw)
            print(format_yaw_authority(yaw))
            print(f"yaw authority: {yaw_path}")
            from .wrench_authority import (
                build_wrench_authority_report,
                format_wrench_authority,
                save_wrench_authority_plot,
            )

            wrench = build_wrench_authority_report(
                traces, config,
                model=str(selected_model) if selected_model is not None else None,
            )
            wrench_plot = save_wrench_authority_plot(
                artifacts.path("plots", "wrench_authority", ".png"), traces, config,
            )
            wrench["plot"] = wrench_plot.relative_to(artifacts.run_dir).as_posix()
            wrench_path = artifacts.write_metrics("wrench_authority", wrench)
            print(format_wrench_authority(wrench))
            print(f"wrench authority: {wrench_path}")
        artifacts.finalize(
            "completed",
            effective_condition=condition,
            effective_parameters=mission.effective_parameters(),
            input_model=(
                str(selected_model) if selected_model is not None else None
            ),
            policy_outcomes=metrics["policies"],
            actuator_outcomes={
                policy_key: outcome.get("actuator")
                for policy_key, outcome in metrics["policies"].items()
            },
        )
        print(f"metrics: {metrics_path}")
        print(f"run: {artifacts.run_dir}")
        return 0
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        if not runtime_config_written:
            failed_runtime = dict(runtime_values)
            failed_runtime["preflight_error"] = metrics["error"]
            metrics["runtime_parameters"] = failed_runtime
            try:
                artifacts.write_runtime_config(failed_runtime)
                runtime_config_written = True
            except Exception as runtime_exc:
                metrics["runtime_config_error"] = (
                    f"{type(runtime_exc).__name__}: {runtime_exc}"
                )
        try:
            artifacts.write_metrics("evaluation", metrics)
        except FileExistsError:
            pass
        artifacts.finalize(
            "failed",
            error=metrics["error"],
            policy_outcomes=metrics["policies"],
            actuator_outcomes={
                policy_key: outcome.get("actuator")
                for policy_key, outcome in metrics["policies"].items()
            },
        )
        raise


__all__ = [
    "EvaluationRunner",
    "RolloutTrace",
    "apply_runtime_overrides",
    "build_parser",
    "format_run_summary",
    "mission_condition",
    "normalize_direction",
    "normalize_mode",
    "prompt_runtime_options",
    "resolve_path_preset",
    "run_evaluation_cli",
    "trace_metrics",
]

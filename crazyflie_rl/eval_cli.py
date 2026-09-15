"""Import-safe CLI and rollout lifecycle for Crazyflie flight evaluations."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, TextIO

from .attitude import (
    ATTITUDE_AXIS_CHOICES,
    attitude_axis_vector,
    named_attitude_quaternion_wxyz,
)
from .physics_version import (
    PHYSICS_MODEL_VERSION,
    manifest_physics_version,
    physics_comparison,
)

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
LATEST_FINAL_MODEL = "latest-final"

EVALUATION_PRESETS: dict[str, dict[str, Any]] = {
    "recovery-ppo-nominal": {
        "description": "E2E PPO nominal airborne hover for 8 seconds",
        "config": "configs/e2e_train_initial_perturb.yaml",
        "mode": "hover",
        "policy": "ppo",
        "duration": 8.0,
        "position_perturbation": 0.0,
        "attitude_perturbation_deg": 0.0,
        "seed": 1000,
        "force_floor_start": False,
        "model": LATEST_FINAL_MODEL,
    },
    "recovery-ppo-tilt30": {
        "description": "E2E PPO recovery from an exact 30 degree tilt",
        "config": "configs/e2e_train_initial_perturb.yaml",
        "mode": "hover",
        "policy": "ppo",
        "duration": 8.0,
        "position_perturbation": 0.0,
        "attitude_perturbation_deg": 30.0,
        "seed": 1000,
        "force_floor_start": False,
        "model": LATEST_FINAL_MODEL,
    },
    "recovery-pid-tilt30": {
        "description": "PID-only recovery from an exact 30 degree tilt",
        "config": "configs/e2e_train_initial_perturb.yaml",
        "mode": "hover",
        "policy": "floor",
        "duration": 8.0,
        "position_perturbation": 0.0,
        "attitude_perturbation_deg": 30.0,
        "seed": 1000,
        "force_floor_start": False,
        "model": None,
    },
    "recovery-both-tilt30": {
        "description": "PID then E2E PPO from the same exact 30 degree tilt",
        "config": "configs/e2e_train_initial_perturb.yaml",
        "mode": "hover",
        "policy": "both",
        "duration": 8.0,
        "position_perturbation": 0.0,
        "attitude_perturbation_deg": 30.0,
        "seed": 1000,
        "force_floor_start": False,
        "model": LATEST_FINAL_MODEL,
    },
    "recovery-ppo-position20cm": {
        "description": "E2E PPO hover with a 0.20 m position perturbation",
        "config": "configs/e2e_train_initial_perturb.yaml",
        "mode": "hover",
        "policy": "ppo",
        "duration": 8.0,
        "position_perturbation": 0.20,
        "attitude_perturbation_deg": 0.0,
        "seed": 1000,
        "force_floor_start": False,
        "model": LATEST_FINAL_MODEL,
    },
}


def _ensure_runtime_imports() -> None:
    """Load numerical/config/runtime helpers only after CLI parsing succeeds."""

    if "np" in globals():
        return
    import numpy as numpy_module

    from .artifacts import (
        ArtifactManager as artifact_manager,
        stable_float as float_tag,
    )
    from .config import ConfigError as config_error, load_config as config_loader
    from .missions import (
        CircleMission as circle_mission,
        mission_from_experiment as mission_factory,
    )
    from .plotting import (
        quaternion_to_euler_deg as quaternion_converter,
        save_hover_trace as hover_plotter,
        save_lyapunov_trace as lyapunov_plotter,
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
        save_lyapunov_trace=lyapunov_plotter,
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
    wrench_command_reference: str | None = None
    wrench_actual: np.ndarray | None = None
    allocation_error: np.ndarray | None = None
    actuator: Mapping[str, Any] | None = None
    lyapunov_v_before: np.ndarray | None = None
    lyapunov_v: np.ndarray | None = None
    lyapunov_delta_v: np.ndarray | None = None
    lyapunov_decay_target: np.ndarray | None = None
    normalized_tracking_error: np.ndarray | None = None
    geometric_attitude_error: np.ndarray | None = None
    angular_rate_error: np.ndarray | None = None
    actuator_saturation_fraction: np.ndarray | None = None
    terminated_transition: np.ndarray | None = None
    truncated_transition: np.ndarray | None = None
    policy_dt: float | None = None
    attitude_axis: str | None = None
    initial_axis_xyz: np.ndarray | None = None
    initial_position_xyz_m: np.ndarray | None = None
    initial_quaternion_wxyz: np.ndarray | None = None
    initial_actuator_state: Mapping[str, Any] | None = None
    reset_info: Mapping[str, Any] | None = None
    physics_wrenches: tuple[Mapping[str, Any], ...] = ()
    observation_diagnostics: tuple[Mapping[str, Any], ...] = ()
    legacy_reward_terms: tuple[Mapping[str, Any], ...] = ()
    transition_timing: tuple[Mapping[str, Any], ...] = ()

    @property
    def sample_count(self) -> int:
        return int(self.time_sec.size)


def _is_latest_best_model(value: str | Path | None) -> bool:
    """Return whether ``value`` requests the run-scoped latest best archive."""

    return value is not None and str(value).strip().casefold() == LATEST_BEST_MODEL


def _is_latest_final_model(value: str | Path | None) -> bool:
    return value is not None and str(value).strip().casefold() == LATEST_FINAL_MODEL


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


def _latest_final_model_selection(
    config: ExperimentConfig,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a provenance-complete final archive for one training profile."""

    runs_root = Path(config.paths.artifact_root) / "runs"
    accepted: list[tuple[datetime, Path, dict[str, Any]]] = []
    examined: list[str] = []
    if runs_root.is_dir():
        for manifest_path in runs_root.glob("*/manifests/*manifest*.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, Mapping):
                continue
            models = manifest.get("models")
            final = models.get("final") if isinstance(models, Mapping) else None
            final_path = final.get("path") if isinstance(final, Mapping) else None
            if not isinstance(final_path, str) or not final_path:
                continue

            run_dir = manifest_path.parent.parent.resolve()
            relative_model = Path(final_path)
            if relative_model.is_absolute():
                examined.append(f"{manifest_path}: absolute final path rejected")
                continue
            model_path = (run_dir / relative_model).resolve()
            try:
                model_path.relative_to(run_dir)
            except ValueError:
                examined.append(f"{manifest_path}: escaping final path rejected")
                continue
            examined.append(str(model_path))

            command = manifest.get("command")
            executable = (
                Path(str(command[0])).name
                if isinstance(command, list) and command
                else ""
            )
            if executable not in {"train_ppo.py", "train_ppo_02.py"}:
                continue
            if manifest.get("status") != "completed":
                continue
            if manifest.get("config_profile") != config.profile_name:
                continue
            if manifest.get("control_mode") != "e2e":
                continue
            if tuple(manifest.get("observation_shape", ())) != config.observation_shape:
                continue
            if tuple(manifest.get("action_shape", ())) != config.action_shape:
                continue
            if model_path.suffix.casefold() != ".zip" or not model_path.is_file():
                continue
            created_text = manifest.get("created_at")
            if not isinstance(created_text, str):
                continue
            try:
                created_at = datetime.fromisoformat(created_text)
            except ValueError:
                continue
            if created_at.tzinfo is None:
                continue
            provenance = {
                "selection": LATEST_FINAL_MODEL,
                "physics_model_version": manifest_physics_version(manifest),
                "manifest": str(manifest_path.resolve()),
                "run_id": manifest.get("run_id"),
                "status": manifest.get("status"),
                "created_at": created_text,
                "command": list(command),
                "config_profile": manifest.get("config_profile"),
                "control_mode": manifest.get("control_mode"),
                "observation_shape": list(manifest.get("observation_shape", ())),
                "action_shape": list(manifest.get("action_shape", ())),
                "final_timestep": final.get("timestep"),
            }
            accepted.append((created_at, model_path, provenance))

    if not accepted:
        details = "\n  ".join(examined) if examined else "(no final model records)"
        raise FileNotFoundError(
            "No unambiguous completed E2E training final matches profile "
            f"{config.profile_name!r} under {runs_root}. Candidates examined:\n  "
            f"{details}\nPass an explicit --model path."
        )
    newest_time = max(item[0] for item in accepted)
    newest = [item for item in accepted if item[0] == newest_time]
    if len(newest) != 1:
        paths = "\n  ".join(str(item[1]) for item in newest)
        raise RuntimeError(
            "Multiple matching final models have the same newest provenance "
            f"timestamp {newest_time.isoformat()}:\n  {paths}\n"
            "Pass an explicit --model path."
        )
    _created_at, model_path, provenance = newest[0]
    return model_path, provenance


def _model_provenance(model_path: Path) -> dict[str, Any]:
    """Describe an explicit archive without pretending unrecorded provenance."""

    resolved = model_path.resolve()
    run_dir = resolved.parent.parent
    manifests_dir = run_dir / "manifests"
    if manifests_dir.is_dir():
        for manifest_path in manifests_dir.glob("*manifest*.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            models = manifest.get("models") if isinstance(manifest, Mapping) else None
            if not isinstance(models, Mapping):
                continue
            for kind in ("final", "best", "best-recovery"):
                record = models.get(kind)
                relative = record.get("path") if isinstance(record, Mapping) else None
                if not isinstance(relative, str) or Path(relative).is_absolute():
                    continue
                if (run_dir / relative).resolve() != resolved:
                    continue
                return {
                    "physics_model_version": manifest_physics_version(manifest),
                    "selection": "explicit-cli",
                    "manifest": str(manifest_path.resolve()),
                    "run_id": manifest.get("run_id"),
                    "status": manifest.get("status"),
                    "created_at": manifest.get("created_at"),
                    "command": manifest.get("command"),
                    "config_profile": manifest.get("config_profile"),
                    "control_mode": manifest.get("control_mode"),
                    "observation_shape": manifest.get("observation_shape"),
                    "action_shape": manifest.get("action_shape"),
                    "model_kind": kind,
                    "model_timestep": record.get("timestep"),
                }
    return {
        "selection": "explicit-cli",
        "manifest": None,
        "verification": "unmanaged-or-unrecognized",
    }


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
    if _is_latest_final_model(value):
        return _latest_final_model_selection(config)[0]

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
    from .warm_start import validate_loaded_observation_schema

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
    validate_loaded_observation_schema(policy, config, model_path=path)
    policy.physics_model_version = _model_provenance(path).get("physics_model_version")
    return policy


def _policy_specs(
    config: ExperimentConfig,
    selected: str,
    *,
    unified: bool = False,
) -> list[tuple[str, str]]:
    if config.control_mode == "e2e":
        floor_label = (
            "floor (PID)" if unified else "E2E zero-action (gravity compensation only)"
        )
        policy_label = "E2E PPO"
    else:
        floor_label = "floor (PID)"
        policy_label = "residual (PID+RL)"
    result: list[tuple[str, str]] = []
    if selected in {"floor", "both"}:
        result.append(("floor", floor_label))
    if selected in {"ppo", "residual", "both"}:
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


def _controller_selection_text(
    config: ExperimentConfig,
    selected: str,
    *,
    unified: bool,
) -> str:
    values: list[str] = []
    for policy_key, _label in _policy_specs(config, selected, unified=unified):
        mode = "residual" if unified and policy_key == "floor" else config.control_mode
        controller = "PID floor" if policy_key == "floor" else "learned PPO"
        values.append(f"{controller}={mode.upper()}")
    return ", ".join(values)


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
        common["post_hold_sec"] = _nonnegative(args.post_hold_sec, "post-HOLD duration")
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
        center = (
            circle.center_xy
            if args.center is None
            else tuple(_finite(value, "circle center") for value in args.center)
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
        laps = circle.laps if args.laps is None else _positive(args.laps, "circle laps")
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
        center = (
            lissajous.center_xy
            if args.center is None
            else tuple(_finite(value, "Lissajous center") for value in args.center)
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
        mission = replace(mission, **common, lissajous=lissajous, goto_xy=goto_xy)
    else:  # defensive: parser and config validation normally catch this
        raise ConfigError(f"unknown mission mode: {mode!r}")

    return replace(
        config,
        mission=mission,
        environment=environment,
        evaluation=evaluation,
    )


def _disable_training_initial_state_randomization(
    config: ExperimentConfig,
) -> ExperimentConfig:
    """Return a viewer config that keeps CLI perturbations but not curriculum."""

    settings = config.environment.initial_state_randomization
    if not settings.enabled:
        return config
    return replace(
        config,
        environment=replace(
            config.environment,
            initial_state_randomization=replace(settings, enabled=False),
        ),
    )


def _legacy_circle_condition(config: ExperimentConfig, mission: CircleMission) -> str:
    payload = config.environment.payload
    offset_x, offset_y = payload.offset
    offset_radius = math.hypot(offset_x, offset_y)
    offset_angle = (
        0.0 if offset_radius == 0.0 else math.degrees(math.atan2(offset_y, offset_x))
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
        disable_initial_state_randomization: bool = False,
        exact_attitude_perturbation: bool = False,
        attitude_axis: str | None = None,
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
        self.disable_initial_state_randomization = bool(
            disable_initial_state_randomization
        )
        self.exact_attitude_perturbation = bool(exact_attitude_perturbation)
        if attitude_axis is not None:
            # Validate programmatic callers as strictly as argparse callers.
            attitude_axis_vector(attitude_axis)
        self.attitude_axis = attitude_axis
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
        if self.disable_initial_state_randomization:
            overrides["initial_state_randomization_enabled"] = False
        if self.exact_attitude_perturbation:
            overrides["exact_attitude_perturbation"] = True
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
        observation, reset_info = env.reset(seed=self.seed)
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
            if (
                force_floor_start_applied
                and getattr(env, "_auxiliary_observation", None) is not None
            ):
                observation = np.asarray(env.reset_auxiliary_observation_state())

        attitude_axis = getattr(self, "attitude_axis", None)
        initial_axis_xyz: np.ndarray | None = None
        initial_position_xyz_m: np.ndarray | None = None
        initial_quaternion_wxyz: np.ndarray | None = None
        initial_actuator_state: dict[str, Any] | None = None
        if attitude_axis is not None:
            observation = self._apply_initial_attitude_axis(
                env,
                observation,
                attitude_axis,
                self.config.environment.attitude_perturbation_deg,
            )
            initial_axis_xyz = np.asarray(
                attitude_axis_vector(attitude_axis), dtype=float
            )
            (
                initial_position_xyz_m,
                initial_quaternion_wxyz,
                initial_actuator_state,
            ) = self._capture_initial_state(env, observation)

        actuator_snapshot: dict[str, Any] | None = None
        snapshotter = getattr(env, "actuator_snapshot", None)
        if callable(snapshotter):
            try:
                snapshot = snapshotter()
                if isinstance(snapshot, Mapping):
                    actuator_snapshot = dict(snapshot)
            except Exception as exc:
                actuator_snapshot = {"snapshot_error": f"{type(exc).__name__}: {exc}"}

        dt = float(env.dt_phys * env.substeps)
        # Floor/named-attitude overrides may reinitialize the actuators after reset.
        if callable(getattr(env, "actuator_snapshot", None)):
            reset_info = dict(reset_info)
            reset_info["rollout_initial_actuator_state"] = (
                {
                    "parameters": env.actuator_snapshot(),
                    "actual_thrust_n": np.asarray(env._last_f).tolist(),
                    "omega_rad_s": np.asarray(env._last_omega).tolist(),
                }
                if hasattr(env, "_last_omega")
                else env.actuator_snapshot()
            )
        physics_wrenches = []
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
        lyapunov_v_befores: list[float] = []
        lyapunov_values: list[float] = []
        lyapunov_deltas: list[float] = []
        lyapunov_decay_targets: list[float] = []
        normalized_tracking_errors: list[float] = []
        geometric_attitude_errors: list[np.ndarray] = []
        angular_rate_errors: list[np.ndarray] = []
        actuator_saturation_fractions: list[float] = []
        terminated_transitions: list[bool] = []
        truncated_transitions: list[bool] = []
        observation_diagnostics: list[Mapping[str, Any]] = []
        legacy_reward_diagnostics: list[Mapping[str, Any]] = []
        transition_timings: list[Mapping[str, Any]] = []
        terminated_at: float | None = None
        truncated_at: float | None = None
        diverged_at: float | None = None
        training_boundary_crossed_at: float | None = None
        guard_boundary_crossed_at: float | None = None
        last_phase: str | None = None
        rollout_error: str | None = None
        step_index = 0

        def actuator_vector(attribute: str, fallback: Any = None) -> np.ndarray:
            """Read optional BLDC diagnostics without breaking legacy fakes."""

            value = getattr(env, attribute, fallback)
            try:
                return np.asarray(value, dtype=float).reshape(4).copy()
            except (TypeError, ValueError):
                return np.full(4, np.nan, dtype=float)

        def reward_scalar(terms: Mapping[str, Any], key: str) -> float:
            try:
                value = float(terms[key])
            except (KeyError, TypeError, ValueError):
                return float("nan")
            return value if np.isfinite(value) else float("nan")

        def reward_vector(terms: Mapping[str, Any], key: str) -> np.ndarray:
            try:
                value = np.asarray(terms[key], dtype=float).reshape(3)
            except (KeyError, TypeError, ValueError):
                return np.full(3, np.nan, dtype=float)
            return (
                value.copy()
                if np.all(np.isfinite(value))
                else np.full(3, np.nan, dtype=float)
            )

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
                    # The enhanced reader rebuilds only the current base
                    # feature for a changed reference. This is a pure query;
                    # history and integral remain transition-owned.
                    if getattr(env, "_auxiliary_observation", None) is not None:
                        observation = np.asarray(env.current_observation())
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
                    (
                        observation,
                        _reward,
                        terminated,
                        truncated,
                        step_info,
                    ) = env.step(action)
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
                    reward_terms_value = (
                        step_info.get("reward_terms", {})
                        if isinstance(step_info, Mapping)
                        else {}
                    )
                    reward_terms = (
                        reward_terms_value
                        if isinstance(reward_terms_value, Mapping)
                        else {}
                    )
                    if isinstance(step_info, Mapping):
                        observation_diagnostics.append(
                            dict(step_info.get("observation_diagnostics", {}))
                        )
                        legacy_reward_diagnostics.append(
                            dict(step_info.get("legacy_reward_terms", {}))
                        )
                        transition_timings.append(
                            dict(step_info.get("transition_timing", {}))
                        )
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
                state_sample_time = time_now
                if getattr(env, "_auxiliary_observation", None) is not None:
                    timing = (
                        step_info.get("transition_timing", {})
                        if isinstance(step_info, Mapping)
                        else {}
                    )
                    if isinstance(timing, Mapping):
                        state_sample_time = float(
                            timing.get("physics_step_end_time_s", time_now)
                        )
                error_norm = float(np.linalg.norm(position_error))
                if error_norm > 0.15 and training_boundary_crossed_at is None:
                    training_boundary_crossed_at = state_sample_time
                if error_norm > 1.5 and guard_boundary_crossed_at is None:
                    guard_boundary_crossed_at = state_sample_time

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
                    diverged_at = state_sample_time
                    print(f"    state divergence at t={time_now:.2f}s phase={phase}")
                    break

                times.append(state_sample_time)
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
                if callable(getattr(env, "physics_wrench_snapshot", None)):
                    physics_wrenches.append(env.physics_wrench_snapshot())
                linear_velocities.append(linear_velocity.copy())
                angular_velocities.append(angular_velocity.copy())
                lyapunov_v_befores.append(reward_scalar(reward_terms, "v_before"))
                lyapunov_values.append(reward_scalar(reward_terms, "v_after"))
                lyapunov_deltas.append(reward_scalar(reward_terms, "delta_v"))
                lyapunov_decay_targets.append(
                    reward_scalar(reward_terms, "decay_target")
                )
                normalized_tracking_errors.append(
                    reward_scalar(reward_terms, "normalized_tracking_error_norm")
                )
                geometric_attitude_errors.append(
                    reward_vector(reward_terms, "attitude_error")
                )
                angular_rate_errors.append(
                    reward_vector(reward_terms, "angular_rate_error")
                )
                actuator_saturation_fractions.append(
                    reward_scalar(reward_terms, "actuator_saturation_fraction")
                )
                terminated_transitions.append(bool(terminated))
                truncated_transitions.append(bool(truncated))
                step_index += 1

                if terminated and terminated_at is None:
                    terminated_at = state_sample_time
                    if is_trajectory:
                        print(
                            f"    env guard at t={time_now:.2f}s phase={phase}; "
                            "continuing trajectory mission"
                        )
                if truncated and truncated_at is None:
                    truncated_at = state_sample_time

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
            angular_velocity=np.asarray(angular_velocities, dtype=float).reshape(
                (-1, 3)
            ),
            control_mode=control_mode,
            motor_thrust_command=np.asarray(motor_thrust_commands, dtype=float).reshape(
                (-1, 4)
            ),
            motor_command=np.asarray(motor_commands, dtype=float).reshape((-1, 4)),
            motor_omega_rad_s=np.asarray(motor_omegas, dtype=float).reshape((-1, 4)),
            reaction_torque_nm=np.asarray(reaction_torques, dtype=float).reshape(
                (-1, 4)
            ),
            wrench_command=np.asarray(wrench_commands, dtype=float).reshape((-1, 4)),
            wrench_command_reference=str(
                getattr(
                    env,
                    "wrench_command_reference",
                    "body frame; torque about the nominal allocator origin "
                    "(not the payload-shifted combined CoM)",
                )
            ),
            wrench_actual=np.asarray(wrench_actuals, dtype=float).reshape((-1, 4)),
            allocation_error=np.asarray(allocation_errors, dtype=float).reshape(
                (-1, 4)
            ),
            actuator=actuator_snapshot,
            reset_info=reset_info,
            physics_wrenches=tuple(physics_wrenches),
            observation_diagnostics=tuple(observation_diagnostics),
            legacy_reward_terms=tuple(legacy_reward_diagnostics),
            transition_timing=tuple(transition_timings),
            lyapunov_v_before=np.asarray(lyapunov_v_befores, dtype=float),
            lyapunov_v=np.asarray(lyapunov_values, dtype=float),
            lyapunov_delta_v=np.asarray(lyapunov_deltas, dtype=float),
            lyapunov_decay_target=np.asarray(lyapunov_decay_targets, dtype=float),
            normalized_tracking_error=np.asarray(
                normalized_tracking_errors, dtype=float
            ),
            geometric_attitude_error=np.asarray(
                geometric_attitude_errors, dtype=float
            ).reshape((-1, 3)),
            angular_rate_error=np.asarray(angular_rate_errors, dtype=float).reshape(
                (-1, 3)
            ),
            actuator_saturation_fraction=np.asarray(
                actuator_saturation_fractions, dtype=float
            ),
            terminated_transition=np.asarray(terminated_transitions, dtype=bool),
            truncated_transition=np.asarray(truncated_transitions, dtype=bool),
            policy_dt=dt,
            attitude_axis=attitude_axis,
            initial_axis_xyz=initial_axis_xyz,
            initial_position_xyz_m=initial_position_xyz_m,
            initial_quaternion_wxyz=initial_quaternion_wxyz,
            initial_actuator_state=initial_actuator_state,
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
        if mission is not None and mission.name != "hover" and self.camera_tracking:
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
    def _apply_initial_attitude_axis(
        env: Any,
        observation: np.ndarray,
        attitude_axis: str,
        perturbation_deg: float,
    ) -> np.ndarray:
        """Replace only the reset quaternion with one named axis-angle pose."""

        _ensure_runtime_imports()
        import mujoco

        quaternion = np.asarray(
            named_attitude_quaternion_wxyz(attitude_axis, perturbation_deg),
            dtype=float,
        )
        original_quaternion = np.asarray(env.data.qpos[3:7], dtype=float).copy()
        try:
            env.data.qpos[3:7] = quaternion
            mujoco.mj_forward(env.model, env.data)
        except Exception:
            env.data.qpos[3:7] = original_quaternion
            try:
                mujoco.mj_forward(env.model, env.data)
            except Exception:
                pass
            raise

        reset_info = getattr(env, "_last_reset_info", None)
        if isinstance(reset_info, dict):
            reset_info.update(
                {
                    "attitude_axis": attitude_axis,
                    "initial_tilt_deg": float(perturbation_deg),
                    "initial_tilt_axis_xyz": list(attitude_axis_vector(attitude_axis)),
                    "initial_quaternion_wxyz": quaternion.tolist(),
                }
            )

        reset_auxiliary = getattr(env, "reset_auxiliary_observation_state", None)
        if callable(reset_auxiliary):
            return np.asarray(reset_auxiliary())
        state_reader = getattr(env, "_read_state", None)
        observation_builder = getattr(env, "_obs", None)
        if callable(state_reader) and callable(observation_builder):
            return np.asarray(observation_builder(*state_reader()))

        updated = np.asarray(observation).copy()
        if updated.ndim != 1 or updated.size < 10:
            raise ValueError("environment observation cannot hold a wxyz quaternion")
        updated[6:10] = quaternion
        return updated

    @staticmethod
    def _capture_initial_state(
        env: Any, observation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Capture pose and live actuator state before the first control step."""

        _ensure_runtime_imports()
        state_reader = getattr(env, "_read_state", None)
        if callable(state_reader):
            position, quaternion, _velocity, _omega = state_reader()
            initial_position = np.asarray(position, dtype=float).reshape(3).copy()
            initial_quaternion = np.asarray(quaternion, dtype=float).reshape(4).copy()
        else:
            initial_observation = np.asarray(observation, dtype=float).reshape(-1)
            if initial_observation.size < 10:
                raise ValueError("environment observation is missing its initial pose")
            initial_position = initial_observation[0:3] + np.asarray(
                env.pos_des, dtype=float
            ).reshape(3)
            initial_quaternion = initial_observation[6:10].copy()

        quaternion_norm = float(np.linalg.norm(initial_quaternion))
        if not np.isfinite(quaternion_norm) or quaternion_norm <= 0.0:
            raise ValueError("initial quaternion must have a positive finite norm")
        initial_quaternion /= quaternion_norm
        if initial_quaternion[0] < 0.0:
            initial_quaternion = -initial_quaternion

        actuator_fields = {
            "requested_motor_thrust_n": "_last_f_cmd",
            "actual_motor_thrust_n": "_last_f",
            "motor_command": "_last_motor_cmd",
            "rotor_omega_rad_s": "_last_omega",
            "reaction_torque_nm": "_last_q_actual",
        }
        actuator_state: dict[str, Any] = {}
        for field, attribute in actuator_fields.items():
            if not hasattr(env, attribute):
                continue
            value = np.asarray(getattr(env, attribute), dtype=float).reshape(4)
            actuator_state[field] = value.tolist()
        return initial_position, initial_quaternion, actuator_state

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
                    restore_error = f"{type(restore_exc).__name__}: {restore_exc}"
                    error += f"; restore failed: {restore_error}"
                    restore_status = "reset state restoration FAILED"
            print(f"    warning: force_floor_start failed ({error}); {restore_status}")
            return False, error


def trace_metrics(trace: RolloutTrace, tail_fraction: float) -> dict[str, Any]:
    """Return the complete, JSON-safe tracking metric contract."""

    _ensure_runtime_imports()

    result: dict[str, Any] = {
        "label": trace.label,
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "reset_info": dict(trace.reset_info or {}),
        "physics_wrenches": list(trace.physics_wrenches),
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
        "terminated": trace.terminated_at is not None,
        "truncated": trace.truncated_at is not None,
        "crash_or_guard_termination": trace.terminated_at is not None,
        "mean_v": None,
        "max_v": None,
        "terminal_v": None,
        "mean_delta_v": None,
        "v_decrease_transition_ratio": None,
        "decay_condition_satisfied_ratio": None,
        "integrated_tracking_error": None,
        "integrated_normalized_tracking_error": None,
        "geometric_attitude_rmse_rad": None,
        "angular_rate_rmse_rad_s": None,
        "action_rms": None,
        "residual_action_rms": None,
        "action_total_variation": None,
        "actuator_saturation_ratio": None,
    }
    observation_rows = [dict(row) for row in trace.observation_diagnostics if row]
    reward_rows = [dict(row) for row in trace.legacy_reward_terms if row]
    timing_rows = [dict(row) for row in trace.transition_timing if row]
    result["observation_schema"] = dict(
        (trace.reset_info or {}).get("observation_schema", {})
    )
    result["final_observation_diagnostics"] = (
        observation_rows[-1] if observation_rows else None
    )
    result["transition_timing_first"] = timing_rows[0] if timing_rows else None
    result["transition_timing_last"] = timing_rows[-1] if timing_rows else None
    result["legacy_reward_terms_last"] = reward_rows[-1] if reward_rows else None
    if reward_rows:
        for group_name in ("raw_costs", "weighted_costs"):
            keys = sorted(
                {
                    key
                    for row in reward_rows
                    for key in (
                        row.get(group_name, {}).keys()
                        if isinstance(row.get(group_name), Mapping)
                        else ()
                    )
                }
            )
            result[f"legacy_reward_{group_name}_mean"] = {
                key: float(
                    np.mean(
                        [
                            float(row[group_name][key])
                            for row in reward_rows
                            if isinstance(row.get(group_name), Mapping)
                            and key in row[group_name]
                        ]
                    )
                )
                for key in keys
            }
    if (
        trace.angular_velocity is not None
        and len(trace.physics_wrenches) == trace.sample_count
        and trace.sample_count > 0
    ):
        motor_torques = np.asarray(
            [
                row["motor_wrench_vehicle_com_body"][:3]
                for row in trace.physics_wrenches
            ],
            dtype=float,
        )
        external_torques = np.asarray(
            [
                row["external_applied_wrench_vehicle_com_body"][:3]
                for row in trace.physics_wrenches
            ],
            dtype=float,
        )
        angular_velocity = np.asarray(trace.angular_velocity, dtype=float)
        motor_axis_power = motor_torques * angular_velocity
        external_axis_power = external_torques * angular_velocity
        total_axis_power = motor_axis_power + external_axis_power
        result["rotational_power_vehicle_com_body"] = {
            "definition": (
                "same post-step sample: torque about actual vehicle CoM in body "
                "frame dotted with body angular velocity"
            ),
            "time_sec": trace.time_sec.tolist(),
            "angular_velocity_body_rad_s": angular_velocity.tolist(),
            "motor_torque_vehicle_com_body_nm": motor_torques.tolist(),
            "external_torque_vehicle_com_body_nm": external_torques.tolist(),
            "motor_axis_power_w": motor_axis_power.tolist(),
            "external_axis_power_w": external_axis_power.tolist(),
            "total_axis_power_w": total_axis_power.tolist(),
            "total_rotational_power_w": np.sum(total_axis_power, axis=1).tolist(),
        }
    if trace.attitude_axis is not None:
        result.update(
            {
                "attitude_axis": trace.attitude_axis,
                "initial_axis_xyz": (
                    np.asarray(trace.initial_axis_xyz, dtype=float).reshape(3).tolist()
                    if trace.initial_axis_xyz is not None
                    else None
                ),
                "initial_position_xyz_m": (
                    np.asarray(trace.initial_position_xyz_m, dtype=float)
                    .reshape(3)
                    .tolist()
                    if trace.initial_position_xyz_m is not None
                    else None
                ),
                "initial_quaternion_wxyz": (
                    np.asarray(trace.initial_quaternion_wxyz, dtype=float)
                    .reshape(4)
                    .tolist()
                    if trace.initial_quaternion_wxyz is not None
                    else None
                ),
                "initial_actuator_state": (
                    dict(trace.initial_actuator_state)
                    if trace.initial_actuator_state is not None
                    else None
                ),
            }
        )
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

    def optional_series(value: Any, *, dtype: Any = float) -> np.ndarray:
        if value is None:
            return np.empty(0, dtype=dtype)
        try:
            array = np.asarray(value, dtype=dtype).reshape(-1)
        except (TypeError, ValueError):
            return np.empty(0, dtype=dtype)
        return array if array.size == trace.sample_count else np.empty(0, dtype=dtype)

    def optional_vectors(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0, 3), dtype=float)
        try:
            array = np.asarray(value, dtype=float).reshape((-1, 3))
        except (TypeError, ValueError):
            return np.empty((0, 3), dtype=float)
        return (
            array
            if array.shape[0] == trace.sample_count
            else np.empty((0, 3), dtype=float)
        )

    v_values = optional_series(trace.lyapunov_v)
    delta_v_values = optional_series(trace.lyapunov_delta_v)
    decay_targets = optional_series(trace.lyapunov_decay_target)
    normalized_errors = optional_series(trace.normalized_tracking_error)
    attitude_errors = optional_vectors(trace.geometric_attitude_error)
    rate_errors = optional_vectors(trace.angular_rate_error)
    saturation_fractions = optional_series(trace.actuator_saturation_fraction)
    terminated_transitions = optional_series(trace.terminated_transition, dtype=bool)

    def finite_column_summary(
        values: np.ndarray, reducer: Callable[[np.ndarray], float]
    ) -> list[float | None]:
        summary: list[float | None] = []
        for column in range(4):
            finite = values[:, column][np.isfinite(values[:, column])]
            summary.append(float(reducer(finite)) if finite.size else None)
        return summary

    result["control_input_abs_max"] = finite_column_summary(np.abs(control), np.max)
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
    result["reaction_torque_nm_min"] = finite_column_summary(reaction_torque, np.min)
    result["reaction_torque_nm_max"] = finite_column_summary(reaction_torque, np.max)

    def final_vector(values: np.ndarray) -> list[float | None]:
        if not values.size:
            return [None, None, None, None]
        return [float(value) if np.isfinite(value) else None for value in values[-1]]

    result["wrench_command_final"] = final_vector(wrench_command)
    result["wrench_command_order"] = [
        "tau_x_cmd",
        "tau_y_cmd",
        "tau_z_cmd",
        "Fz_cmd",
    ]
    result["wrench_command_units"] = ["N*m", "N*m", "N*m", "N"]
    result["wrench_command_stage"] = "before_allocation"
    result["wrench_command_reference"] = trace.wrench_command_reference
    result["wrench_actual_final"] = final_vector(wrench_actual)
    result["allocation_error_final"] = final_vector(allocation_error)
    if control.size:
        result["action_rms"] = float(np.sqrt(np.mean(np.square(control))))
        result["residual_action_rms"] = (
            result["action_rms"] if trace.control_mode == "residual" else None
        )
        result["action_total_variation"] = float(
            np.sum(np.abs(np.diff(control, axis=0)))
        )
    finite_saturation = saturation_fractions[np.isfinite(saturation_fractions)]
    if finite_saturation.size:
        result["actuator_saturation_ratio"] = float(np.mean(finite_saturation))

    finite_v = np.isfinite(v_values)
    if np.any(finite_v):
        selected_v = v_values[finite_v]
        result["mean_v"] = float(np.mean(selected_v))
        result["max_v"] = float(np.max(selected_v))
        if terminated_transitions.size:
            terminal_indices = np.flatnonzero(terminated_transitions & finite_v)
            if terminal_indices.size:
                result["terminal_v"] = float(v_values[terminal_indices[0]])
    finite_delta = np.isfinite(delta_v_values)
    if np.any(finite_delta):
        selected_delta = delta_v_values[finite_delta]
        result["mean_delta_v"] = float(np.mean(selected_delta))
        result["v_decrease_transition_ratio"] = float(np.mean(selected_delta < 0.0))
    finite_decay = finite_v & np.isfinite(decay_targets)
    if np.any(finite_decay):
        result["decay_condition_satisfied_ratio"] = float(
            np.mean(v_values[finite_decay] <= decay_targets[finite_decay])
        )

    if attitude_errors.size:
        finite_attitude_rows = np.all(np.isfinite(attitude_errors), axis=1)
        if np.any(finite_attitude_rows):
            result["geometric_attitude_rmse_rad"] = float(
                np.sqrt(
                    np.mean(
                        np.sum(np.square(attitude_errors[finite_attitude_rows]), axis=1)
                    )
                )
            )
    if rate_errors.size:
        finite_rate_rows = np.all(np.isfinite(rate_errors), axis=1)
        if np.any(finite_rate_rows):
            result["angular_rate_rmse_rad_s"] = float(
                np.sqrt(
                    np.mean(np.sum(np.square(rate_errors[finite_rate_rows]), axis=1))
                )
            )
    if trace.actuator is not None:
        result["actuator"] = dict(trace.actuator)
    if not trace.sample_count:
        result.update(
            position_rmse=None,
            mean_position_error=None,
            max_position_error=None,
            tail_mean_position_error=None,
            trajectory_phase_rmse=None,
            phases={},
        )
        return result

    errors = np.asarray(trace.position_error, dtype=float)
    positive_time_deltas = np.diff(np.asarray(trace.time_sec, dtype=float))
    positive_time_deltas = positive_time_deltas[positive_time_deltas > 0.0]
    configured_dt = trace.policy_dt
    sample_dt = (
        float(configured_dt)
        if configured_dt is not None
        and np.isfinite(configured_dt)
        and configured_dt > 0.0
        else float(np.median(positive_time_deltas))
        if positive_time_deltas.size
        else 0.0
    )
    result["integrated_tracking_error"] = float(np.sum(errors) * sample_dt)
    finite_normalized = normalized_errors[np.isfinite(normalized_errors)]
    if finite_normalized.size:
        result["integrated_normalized_tracking_error"] = float(
            np.sum(finite_normalized) * sample_dt
        )
    tail_start = int(trace.sample_count * (1.0 - tail_fraction))
    result["position_rmse"] = float(np.sqrt(np.mean(np.square(errors))))
    result["mean_position_error"] = float(np.mean(errors))
    result["max_position_error"] = float(np.max(errors))
    result["tail_mean_position_error"] = float(np.mean(errors[tail_start:]))

    phase_values = np.asarray(trace.phases)
    phase_metrics: dict[str, Any] = {}
    for phase in dict.fromkeys(trace.phases):
        selected = errors[phase_values == phase]
        phase_metrics[phase] = {
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
    return result


# Compatibility name retained for downstream imports.
_trace_metrics = trace_metrics


def _attitude_axis_plot_subtitle(trace: RolloutTrace) -> str | None:
    if trace.attitude_axis is None:
        return None
    return f"Attitude axis: {trace.attitude_axis}"


def _save_trace_csv(
    artifacts: ArtifactManager,
    trace: RolloutTrace,
) -> Path:
    """Save the plotted kinematic trace with constant initial-pose metadata."""

    _ensure_runtime_imports()
    output = artifacts.ensure_available(
        artifacts.path("metrics", f"trace-{trace.policy}", ".csv")
    )
    fieldnames = (
        "time_sec",
        "phase",
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "reference_x_m",
        "reference_y_m",
        "reference_z_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "position_error_norm_m",
        "attitude_axis",
        "initial_axis_x",
        "initial_axis_y",
        "initial_axis_z",
        "initial_quaternion_w",
        "initial_quaternion_x",
        "initial_quaternion_y",
        "initial_quaternion_z",
    )
    initial_axis = (
        np.asarray(trace.initial_axis_xyz, dtype=float).reshape(3)
        if trace.initial_axis_xyz is not None
        else np.full(3, np.nan, dtype=float)
    )
    initial_quaternion = (
        np.asarray(trace.initial_quaternion_wxyz, dtype=float).reshape(4)
        if trace.initial_quaternion_wxyz is not None
        else np.full(4, np.nan, dtype=float)
    )
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(trace.sample_count):
            writer.writerow(
                {
                    "time_sec": float(trace.time_sec[index]),
                    "phase": trace.phases[index],
                    "position_x_m": float(trace.position[index, 0]),
                    "position_y_m": float(trace.position[index, 1]),
                    "position_z_m": float(trace.position[index, 2]),
                    "reference_x_m": float(trace.reference_position[index, 0]),
                    "reference_y_m": float(trace.reference_position[index, 1]),
                    "reference_z_m": float(trace.reference_position[index, 2]),
                    "roll_deg": float(trace.attitude_deg[index, 0]),
                    "pitch_deg": float(trace.attitude_deg[index, 1]),
                    "yaw_deg": float(trace.attitude_deg[index, 2]),
                    "position_error_norm_m": float(trace.position_error[index]),
                    "attitude_axis": trace.attitude_axis or "",
                    "initial_axis_x": float(initial_axis[0]),
                    "initial_axis_y": float(initial_axis[1]),
                    "initial_axis_z": float(initial_axis[2]),
                    "initial_quaternion_w": float(initial_quaternion[0]),
                    "initial_quaternion_x": float(initial_quaternion[1]),
                    "initial_quaternion_y": float(initial_quaternion[2]),
                    "initial_quaternion_z": float(initial_quaternion[3]),
                }
            )
    return output


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
    attitude_subtitle = _attitude_axis_plot_subtitle(trace)
    if attitude_subtitle is not None:
        plot_tag += f"\n{attitude_subtitle}"
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
    plot_tag = title_condition
    attitude_subtitle = _attitude_axis_plot_subtitle(trace)
    if attitude_subtitle is not None:
        plot_tag += f"\n{attitude_subtitle}"
    return save_policy_trace(
        output,
        tag=plot_tag,
        rollout=trace,
        mission_name=mission.name,
        mission_parameters=mission.effective_parameters(),
        motor_unit="N",
    )


def _save_lyapunov_report(
    artifacts: ArtifactManager,
    trace: RolloutTrace,
    *,
    title_condition: str,
) -> Path | None:
    """Save an additional candidate diagnostic when transition data exists."""

    _ensure_runtime_imports()
    required = (
        trace.lyapunov_v_before,
        trace.lyapunov_v,
        trace.lyapunov_delta_v,
        trace.lyapunov_decay_target,
    )
    if any(value is None for value in required):
        return None
    arrays = [np.asarray(value, dtype=float).reshape(-1) for value in required]
    if any(value.size != trace.sample_count for value in arrays):
        return None
    finite = np.ones(trace.sample_count, dtype=bool)
    for value in arrays:
        finite &= np.isfinite(value)
    if not np.any(finite):
        return None
    output = artifacts.path("plots", f"lyapunov-{trace.policy}", ".png")
    plot_tag = f"{title_condition} — {trace.label}"
    attitude_subtitle = _attitude_axis_plot_subtitle(trace)
    if attitude_subtitle is not None:
        plot_tag += f"\n{attitude_subtitle}"
    return save_lyapunov_trace(
        output,
        tag=plot_tag,
        time_sec=trace.time_sec,
        v_before=arrays[0],
        v_after=arrays[1],
        delta_v=arrays[2],
        decay_target=arrays[3],
    )


def build_parser(
    default_config: str | Path, description: str
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--eval-preset",
        choices=tuple(EVALUATION_PRESETS),
        default=None,
        help="reproducible named evaluation preset (distinct from legacy --preset)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="list named evaluation presets and exit",
    )
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
        choices=("floor", "ppo", "residual", "both"),
        default="both",
        help=(
            "controller selection: floor=PID only, residual=learned PPO only "
            "(legacy option name), both=PID then PPO"
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--position-perturbation", type=float, default=None)
    parser.add_argument("--attitude-perturbation-deg", type=float, default=None)
    parser.add_argument(
        "--attitude-axis",
        choices=ATTITUDE_AXIS_CHOICES,
        default=None,
        help=(
            "body-frame axis and sign for an exact initial attitude "
            "perturbation; omitted preserves the legacy seeded random axis"
        ),
    )

    viewer = parser.add_mutually_exclusive_group()
    viewer.add_argument("--headless", dest="headless", action="store_true")
    viewer.add_argument("--viewer", dest="headless", action="store_false")
    parser.set_defaults(headless=False)

    pacing = parser.add_mutually_exclusive_group()
    pacing.add_argument("--no-realtime", dest="no_realtime", action="store_true")
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


def _prompt_bool(label: str, default: bool, input_fn: Callable[[str], str]) -> bool:
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
            _cli_or_config_default(args, "duration", config.mission.hover.duration),
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
            _cli_or_config_default(args, "cycles", config.mission.lissajous.cycles),
            float,
            input_fn,
        )
    return args


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def _print_evaluation_presets(stream: TextIO) -> None:
    print("Named evaluation presets (--eval-preset):", file=stream)
    for name, values in EVALUATION_PRESETS.items():
        print(f"  {name:<28} {values['description']}", file=stream)


def _apply_evaluation_preset(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
) -> argparse.Namespace:
    """Merge a named preset below explicit CLI arguments."""

    if args.eval_preset is None:
        return args
    preset = EVALUATION_PRESETS[args.eval_preset]
    root = Path(__file__).resolve().parents[1]
    option_fields = (
        ("--config", "config"),
        ("--mode", "mode"),
        ("--policy", "policy"),
        ("--duration", "duration"),
        ("--position-perturbation", "position_perturbation"),
        ("--attitude-perturbation-deg", "attitude_perturbation_deg"),
        ("--seed", "seed"),
        ("--model", "model"),
    )
    for option, field in option_fields:
        if _has_option(raw_argv, option):
            continue
        value = preset[field]
        if field == "config":
            value = root / str(value)
        elif field == "model" and value is not None:
            value = Path(str(value))
        setattr(args, field, value)
    if not (
        _has_option(raw_argv, "--force-floor-start")
        or _has_option(raw_argv, "--no-force-floor-start")
    ):
        args.force_floor_start = bool(preset["force_floor_start"])
    if not (_has_option(raw_argv, "--headless") or _has_option(raw_argv, "--viewer")):
        args.headless = False
    if not (
        _has_option(raw_argv, "--no-realtime") or _has_option(raw_argv, "--realtime")
    ):
        args.no_realtime = False
    if not (_has_option(raw_argv, "--no-camera") or _has_option(raw_argv, "--camera")):
        args.no_camera = False
    return args


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


def _format_vector(values: Sequence[float]) -> str:
    return "[" + ", ".join(_format_number(float(value)) for value in values) + "]"


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
    attitude_axis: str | None = None,
    initial_quaternion_wxyz: Sequence[float] | None = None,
) -> str:
    """Format the effective pre-flight summary displayed to the operator."""

    rows: list[tuple[str, str]] = [("Mode", mission.name.title())]
    if unified:
        rows.extend(
            (
                ("Floor controller", "RESIDUAL (PID)"),
                ("PPO controller", config.control_mode.upper()),
                (
                    "Selected controls",
                    _controller_selection_text(config, policy, unified=True),
                ),
            )
        )
    else:
        rows.append(("Control mode", config.control_mode.upper()))
    rows.append(("Reward mode", config.environment.reward.mode))
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
    if attitude_axis is not None:
        initial_axis = attitude_axis_vector(attitude_axis)
        initial_quaternion = (
            named_attitude_quaternion_wxyz(
                attitude_axis,
                config.environment.attitude_perturbation_deg,
            )
            if initial_quaternion_wxyz is None
            else tuple(float(value) for value in initial_quaternion_wxyz)
        )
        rows.extend(
            (
                ("Attitude axis", attitude_axis),
                ("Initial axis", _format_vector(initial_axis)),
                ("Initial quaternion", _format_vector(initial_quaternion)),
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


def _initial_state_preview(
    config: ExperimentConfig,
    *,
    exact_attitude_perturbation: bool,
    attitude_axis: str | None = None,
) -> dict[str, Any]:
    """Reset a disposable environment using the same viewer reset path."""

    from .factories import EnvironmentFactory

    environment = EnvironmentFactory(config).make(
        seed=config.evaluation.seed_start,
        initial_state_randomization_enabled=False,
        exact_attitude_perturbation=exact_attitude_perturbation,
    )
    try:
        observation, _reset_info = environment.reset(seed=config.evaluation.seed_start)
        if config.mission.force_floor_start:
            applied, error = EvaluationRunner._force_floor_start(environment)
            if not applied:
                raise RuntimeError(f"initial-state preview floor start failed: {error}")
        if attitude_axis is not None:
            observation = EvaluationRunner._apply_initial_attitude_axis(
                environment,
                observation,
                attitude_axis,
                config.environment.attitude_perturbation_deg,
            )
        position, quaternion, _velocity, _omega = environment._read_state()
        yaw_half = 0.5 * float(environment.yaw_des)
        reference_conjugate = np.array(
            [np.cos(yaw_half), 0.0, 0.0, -np.sin(yaw_half)], dtype=float
        )
        w1, x1, y1, z1 = reference_conjugate
        w2, x2, y2, z2 = np.asarray(quaternion, dtype=float)
        relative = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
        if relative[0] < 0.0:
            relative = -relative
        vector_norm = float(np.linalg.norm(relative[1:]))
        total_tilt_deg = float(np.degrees(2.0 * np.arctan2(vector_norm, relative[0])))
        tilt_axis = relative[1:] / vector_norm if vector_norm > 1e-12 else np.zeros(3)
        result = {
            "position_xyz_m": position.tolist(),
            "position_offset_xyz_m": (position - environment.pos_des).tolist(),
            "quaternion_wxyz": np.asarray(quaternion, dtype=float).tolist(),
            "total_tilt_deg": total_tilt_deg,
            "tilt_axis_xyz": tilt_axis.tolist(),
            "qvel": np.asarray(environment.data.qvel, dtype=float).tolist(),
            "actuator": {
                "requested_motor_thrust_n": environment._last_f_cmd.tolist(),
                "actual_motor_thrust_n": environment._last_f.tolist(),
                "rotor_omega_rad_s": environment._last_omega.tolist(),
                "snapshot": environment.actuator_snapshot(),
            },
        }
        if attitude_axis is not None:
            result.update(
                {
                    "attitude_axis": attitude_axis,
                    "selected_axis_xyz": list(attitude_axis_vector(attitude_axis)),
                }
            )
        return result
    finally:
        environment.close()


def _format_preset_resolution(
    name: str,
    config: ExperimentConfig,
    model: Path | None,
    model_provenance: Mapping[str, Any] | None,
    initial_state: Mapping[str, Any],
    policy: str,
) -> str:
    actuator = initial_state["actuator"]
    rows = (
        ("Evaluation preset", name),
        ("Config path", str(config.source_path)),
        ("Selected model", str(model) if model is not None else "not required"),
        (
            "Model provenance",
            json.dumps(model_provenance, sort_keys=True) if model_provenance else "n/a",
        ),
        ("Controllers", _controller_selection_text(config, policy, unified=True)),
        ("Initial position", str(initial_state["position_xyz_m"])),
        (
            "Initial tilt",
            f"{float(initial_state['total_tilt_deg']):g} deg, "
            f"axis={initial_state['tilt_axis_xyz']}",
        ),
        (
            "Actuator initial",
            f"requested={actuator['requested_motor_thrust_n']} N, "
            f"actual={actuator['actual_motor_thrust_n']} N, "
            f"omega={actuator['rotor_omega_rad_s']} rad/s",
        ),
        (
            "Duration / seed",
            f"{config.mission.hover.duration:g} s / {config.evaluation.seed_start}",
        ),
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
    payload = {
        "mode": mission.name,
        "evaluation_preset": getattr(args, "eval_preset", None),
        "path_preset": getattr(args, "path_preset", None),
        "mission": mission.effective_parameters(),
        "control_mode": config.control_mode,
        "reward_mode": config.environment.reward.mode,
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "payload": asdict(config.environment.payload),
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
    attitude_axis = getattr(args, "attitude_axis", None)
    if attitude_axis is not None:
        payload.update(
            {
                "attitude_axis": attitude_axis,
                "initial_axis_xyz": list(attitude_axis_vector(attitude_axis)),
                "initial_quaternion_wxyz": list(
                    named_attitude_quaternion_wxyz(
                        attitude_axis,
                        config.environment.attitude_perturbation_deg,
                    )
                ),
            }
        )
    return payload


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
    if args.list_presets:
        _print_evaluation_presets(sys.stdout)
        return 0
    args = _apply_evaluation_preset(args, raw_argv)
    # Kept as a no-op keyword for source compatibility with older wrappers.
    # Controller selection now always follows the parsed --policy value.
    del always_compare
    explicit_config = _has_option(raw_argv, "--config")
    selected_config = explicit_config or args.eval_preset is not None
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
            if not selected_config:
                resolved = resolve_path_preset(mode, args.path_preset, path_profiles)
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
        elif unified and not selected_config and mode_profiles is not None and mode:
            config_path = Path(mode_profiles[mode])
        config = load_config(config_path)
    else:
        config = preliminary_config
    if mode is None:
        mode = normalize_mode(config.mission.type)

    if interactive_answers:
        args = prompt_runtime_options(args, config, mode, input_fn=input_fn)
    config = apply_runtime_overrides(config, args, mode)
    if unified:
        # Training curriculum is not a viewer initial condition. This immutable
        # copy preserves any explicit CLI position/attitude perturbations.
        config = _disable_training_initial_state_randomization(config)
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
    requires_model = any(key == "residual" for key, _label in specs)
    candidate_model = (
        None
        if not requires_model or _is_latest_final_model(args.model)
        else _model_candidate(args.model, config)
    )
    condition = mission_condition(
        config,
        mission,
        legacy_circle_preset=legacy_circle_preset,
        path_preset=(args.path_preset if path_profiles is not None else None),
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
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "effective_condition": condition,
        "effective_parameters": mission.effective_parameters(),
        "path_preset": getattr(args, "path_preset", None),
        "runtime_parameters": runtime_values,
        "control_mode": config.control_mode,
        "reward_mode": config.environment.reward.mode,
        "actuator": asdict(config.actuator),
        "evaluation_preset": args.eval_preset,
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
    if args.attitude_axis is not None:
        metrics.update(
            {
                "attitude_axis": args.attitude_axis,
                "initial_axis_xyz": list(attitude_axis_vector(args.attitude_axis)),
                "initial_quaternion_wxyz": list(
                    named_attitude_quaternion_wxyz(
                        args.attitude_axis,
                        config.environment.attitude_perturbation_deg,
                    )
                ),
            }
        )
    selected_model: Path | None = None
    model_provenance: dict[str, Any] | None = None
    initial_state_preview: dict[str, Any] | None = None
    runtime_config_written = False
    try:
        if requires_model and _is_latest_final_model(args.model):
            selected_model, model_provenance = _latest_final_model_selection(config)
        elif requires_model:
            selected_model = _model_path(args.model, config)
            model_provenance = _model_provenance(selected_model)
        exact_attitude_perturbation = unified and (
            args.eval_preset is not None
            or _has_option(raw_argv, "--attitude-perturbation-deg")
            or args.attitude_axis is not None
        )
        config.require_runtime_resources()
        if args.eval_preset is not None or args.attitude_axis is not None:
            initial_state_preview = _initial_state_preview(
                config,
                exact_attitude_perturbation=exact_attitude_perturbation,
                attitude_axis=args.attitude_axis,
            )
        runtime_values = _runtime_payload(
            config,
            mission,
            args,
            selected_model,
            unified=unified,
        )
        runtime_values["model_provenance"] = model_provenance
        comparison = physics_comparison(
            (model_provenance or {}).get("physics_model_version")
        )
        runtime_values["physics_provenance"] = comparison
        metrics["physics_provenance"] = comparison
        if selected_model is not None:
            print("Physics evaluation: " + comparison["physics_comparison_status"])
        runtime_values["initial_state_preview"] = initial_state_preview
        metrics["runtime_parameters"] = runtime_values
        if args.attitude_axis is not None:
            metrics["initial_state"] = initial_state_preview
        metrics["model"] = str(selected_model) if selected_model is not None else None
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
                attitude_axis=args.attitude_axis,
                initial_quaternion_wxyz=(
                    initial_state_preview["quaternion_wxyz"]
                    if initial_state_preview is not None
                    and args.attitude_axis is not None
                    else None
                ),
            )
        )
        if args.eval_preset is not None and initial_state_preview is not None:
            print(
                "\n"
                + _format_preset_resolution(
                    args.eval_preset,
                    config,
                    selected_model,
                    model_provenance,
                    initial_state_preview,
                    args.policy,
                )
            )
        artifacts.write_runtime_config(runtime_values)
        runtime_config_written = True
        policy = (
            _load_policy(selected_model, config) if selected_model is not None else None
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
            disable_initial_state_randomization=unified,
            exact_attitude_perturbation=exact_attitude_perturbation,
            attitude_axis=args.attitude_axis,
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
            if trace.attitude_axis is not None and trace.sample_count:
                trace_csv_path = _save_trace_csv(artifacts, trace)
                policy_metrics["trace_csv"] = trace_csv_path.relative_to(
                    artifacts.run_dir
                ).as_posix()
                print(f"    saved: {trace_csv_path}")
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
                lyapunov_plot_path = _save_lyapunov_report(
                    artifacts,
                    trace,
                    title_condition=condition,
                )
                if lyapunov_plot_path is not None:
                    policy_metrics["lyapunov_candidate_plot"] = (
                        lyapunov_plot_path.relative_to(artifacts.run_dir).as_posix()
                    )
                    print(f"    saved: {lyapunov_plot_path}")
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
        artifacts.finalize(
            "completed",
            effective_condition=condition,
            effective_parameters=mission.effective_parameters(),
            input_model=(str(selected_model) if selected_model is not None else None),
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
    "ATTITUDE_AXIS_CHOICES",
    "EvaluationRunner",
    "RolloutTrace",
    "apply_runtime_overrides",
    "attitude_axis_vector",
    "build_parser",
    "format_run_summary",
    "mission_condition",
    "named_attitude_quaternion_wxyz",
    "normalize_direction",
    "normalize_mode",
    "prompt_runtime_options",
    "resolve_path_preset",
    "run_evaluation_cli",
    "trace_metrics",
]

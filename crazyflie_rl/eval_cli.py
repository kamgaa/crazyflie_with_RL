"""Shared CLI and rollout lifecycle for legacy evaluation entrypoints."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

from .artifacts import ArtifactManager, stable_float
from .config import ExperimentConfig, load_config
from .missions import CircleMission, PHASES
from .plotting import (
    quaternion_to_euler_deg,
    save_circle_trace,
    save_hover_trace,
)


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

    @property
    def sample_count(self) -> int:
        return int(self.time_sec.size)


def _model_path(value: str | Path | None, config: ExperimentConfig) -> Path:
    candidate = (
        config.paths.legacy_model_root / "ppo_best.zip"
        if value is None
        else Path(value).expanduser()
    )
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
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


def _policy_specs(config: ExperimentConfig, selected: str) -> list[tuple[str, str]]:
    if config.control_mode == "e2e":
        floor_label = "E2E zero-action (gravity compensation only)"
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


def _effective_condition(
    config: ExperimentConfig, mission: CircleMission | None
) -> str:
    if mission is None:
        return config.experiment.condition
    payload = config.environment.payload
    offset_x, offset_y = payload.offset
    offset_radius = math.hypot(offset_x, offset_y)
    offset_angle = 0.0 if offset_radius == 0.0 else math.degrees(
        math.atan2(offset_y, offset_x)
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


class EvaluationRunner:
    """Own environment, viewer, policy, and artifact lifecycle for one eval run."""

    def __init__(
        self,
        config: ExperimentConfig,
        artifacts: ArtifactManager,
        *,
        headless: bool,
        realtime: bool,
        camera_tracking: bool,
        preset: str,
    ) -> None:
        from .factories import EnvironmentFactory

        self.config = config
        self.artifacts = artifacts
        self.factory = EnvironmentFactory(config)
        self.headless = bool(headless)
        self.realtime = bool(realtime)
        self.camera_tracking = bool(camera_tracking)
        self.seed = config.evaluation.seed_start
        self.circle = (
            CircleMission.from_experiment(config, preset)
            if config.mission.type == "circle"
            else None
        )

    def run(self, policy: Any | None, policy_key: str, label: str) -> RolloutTrace:
        episode_override = None
        if self.circle is not None and not self.headless:
            # Legacy viewer construction used T_TOTAL + 2 seconds.  The loop
            # itself still stops at T_TOTAL, so this only keeps truncation out
            # of the way during the mission.
            episode_override = self.circle.total_sec + 2.0
        overrides = {}
        if episode_override is not None:
            overrides["episode_sec"] = episode_override
        env = self.factory.make(**overrides)
        try:
            return self._rollout(env, policy, policy_key, label)
        finally:
            env.close()

    def _rollout(
        self, env: Any, policy: Any | None, policy_key: str, label: str
    ) -> RolloutTrace:
        env.dist_torque_body = np.zeros(3)
        observation, _ = env.reset(seed=self.seed)
        if self.circle is not None and self.config.mission.force_floor_start:
            self._force_floor_start(env)

        dt = float(env.dt_phys * env.substeps)
        circle_steps = (
            int(round(self.circle.total_sec / dt)) if self.circle is not None else None
        )
        times: list[float] = []
        positions: list[np.ndarray] = []
        attitudes: list[np.ndarray] = []
        references: list[np.ndarray] = []
        errors: list[float] = []
        phases: list[str] = []
        terminated_at: float | None = None
        truncated_at: float | None = None
        diverged_at: float | None = None
        training_boundary_crossed_at: float | None = None
        guard_boundary_crossed_at: float | None = None
        last_phase: str | None = None

        viewer_context = self._viewer_context(env)
        with viewer_context as viewer:
            self._configure_viewer(viewer)
            step_index = 0
            while True:
                if viewer is not None and not viewer.is_running():
                    break
                if self.circle is not None:
                    if self.headless and step_index >= int(circle_steps):
                        break
                    time_now = step_index * dt
                    if not self.headless and time_now > self.circle.total_sec:
                        break
                    reference, phase = self.circle.reference(time_now)
                    env.pos_des = reference.copy()
                else:
                    time_now = step_index * dt
                    reference = np.asarray(env.pos_des, dtype=float).copy()
                    phase = "HOVER"
                if phase != last_phase:
                    print(f"    t={time_now:6.2f}s phase -> {phase}")
                    last_phase = phase

                wall_start = time.time()
                action = (
                    policy.predict(
                        observation,
                        deterministic=self.config.evaluation.deterministic,
                    )[0]
                    if policy is not None
                    else np.zeros(4)
                )
                observation, _reward, terminated, truncated, _ = env.step(action)
                if not np.all(np.isfinite(observation)):
                    diverged_at = time_now
                    print(f"    non-finite observation at t={time_now:.2f}s")
                    break

                position_error = np.asarray(observation[0:3], dtype=float)
                actual_position = position_error + np.asarray(env.pos_des, dtype=float)
                attitude = quaternion_to_euler_deg(observation[6:10])
                error_norm = float(np.linalg.norm(position_error))
                if error_norm > 0.15 and training_boundary_crossed_at is None:
                    training_boundary_crossed_at = time_now
                if error_norm > 1.5 and guard_boundary_crossed_at is None:
                    guard_boundary_crossed_at = time_now

                if (
                    self.circle is not None
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
                step_index += 1

                if terminated and terminated_at is None:
                    terminated_at = time_now
                    if self.circle is not None:
                        print(
                            f"    env guard at t={time_now:.2f}s phase={phase}; "
                            "continuing legacy OOD mission"
                        )
                if truncated and truncated_at is None:
                    truncated_at = time_now

                self._sync_viewer(viewer, actual_position, dt, wall_start)

                if self.circle is None and (terminated or truncated):
                    break
                if self.circle is not None and not self.headless and truncated:
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
        if self.circle is None:
            viewer.cam.distance = 3.7
            return
        with viewer.lock():
            viewer.cam.distance = 4.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.lookat[:] = np.array([0.5, 0.0, 1.0])

    def _sync_viewer(
        self,
        viewer: Any | None,
        actual_position: np.ndarray,
        dt: float,
        wall_start: float,
    ) -> None:
        if viewer is None:
            return
        if self.circle is not None and self.camera_tracking:
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
    def _force_floor_start(env: Any) -> bool:
        try:
            import mujoco

            env.data.qpos[0:3] = np.array([0.0, 0.0, 0.02])
            env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            env.data.qvel[:6] = 0.0
            mujoco.mj_forward(env.model, env.data)
            return True
        except Exception as exc:
            print(f"    warning: force_floor_start failed ({exc}); using reset state")
            return False


def _trace_metrics(trace: RolloutTrace, tail_fraction: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": trace.label,
        "sample_count": trace.sample_count,
        "duration_sec": float(trace.time_sec[-1]) if trace.sample_count else 0.0,
        "training_boundary_crossed_at": trace.training_boundary_crossed_at,
        "guard_boundary_crossed_at": trace.guard_boundary_crossed_at,
        "terminated_at": trace.terminated_at,
        "truncated_at": trace.truncated_at,
        "diverged_at": trace.diverged_at,
    }
    if not trace.sample_count:
        result["mean_position_error"] = None
        return result
    tail_start = int(trace.sample_count * (1.0 - tail_fraction))
    result["mean_position_error"] = float(np.mean(trace.position_error))
    result["tail_mean_position_error"] = float(
        np.mean(trace.position_error[tail_start:])
    )
    if trace.phases:
        phase_values = np.asarray(trace.phases)
        result["phases"] = {}
        for phase in PHASES:
            selected = trace.position_error[phase_values == phase]
            if selected.size:
                result["phases"][phase] = {
                    "count": int(selected.size),
                    "mean_position_error": float(np.mean(selected)),
                    "max_position_error": float(np.max(selected)),
                    "end_position_error": float(selected[-1]),
                }
    return result


def _save_trace(
    artifacts: ArtifactManager,
    config: ExperimentConfig,
    trace: RolloutTrace,
    circle: CircleMission | None,
) -> Path:
    if trace.sample_count < 1:
        raise RuntimeError(f"{trace.label} rollout produced no samples")
    output = artifacts.path("plots", f"trajectory-{trace.policy}", ".png")
    if circle is None:
        return save_hover_trace(
            output,
            tag=trace.label,
            time_sec=trace.time_sec,
            position=trace.position,
            attitude_deg=trace.attitude_deg,
            hover_altitude=config.mission.hover_altitude,
        )
    return save_circle_trace(
        output,
        tag=trace.label,
        time_sec=trace.time_sec,
        position=trace.position,
        attitude_deg=trace.attitude_deg,
        reference_position=trace.reference_position,
        position_error=trace.position_error,
    )


def build_parser(default_config: str | Path, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=Path, default=Path(default_config), help="YAML profile"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="PPO archive (default: <legacy_model_root>/ppo_best.zip)",
    )
    parser.add_argument("--headless", action="store_true", help="disable viewer")
    parser.add_argument(
        "--policy",
        choices=("floor", "residual", "both"),
        default="both",
        help="policy comparison to run",
    )
    parser.add_argument("--preset", default="1", help="circle preset key")
    parser.add_argument(
        "--no-realtime", action="store_true", help="do not pace viewer to physics time"
    )
    parser.add_argument(
        "--no-camera", action="store_true", help="disable circle camera tracking"
    )
    return parser


def run_evaluation_cli(
    *,
    default_config: str | Path,
    description: str,
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser(default_config, description).parse_args(argv)
    config = load_config(args.config)
    config.require_runtime_resources()
    circle = (
        CircleMission.from_experiment(config, args.preset)
        if config.mission.type == "circle"
        else None
    )
    specs = _policy_specs(config, args.policy)
    requires_model = any(key == "residual" for key, _label in specs)
    selected_model = _model_path(args.model, config) if requires_model else None
    policy = _load_policy(selected_model, config) if selected_model is not None else None
    condition = _effective_condition(config, circle)
    artifacts = ArtifactManager.create(
        config,
        command=list(sys.argv if argv is None else argv),
        condition=condition,
        mission=config.mission.type,
        seed=config.evaluation.seed_start,
    )
    try:
        runner = EvaluationRunner(
            config,
            artifacts,
            headless=args.headless,
            realtime=not args.no_realtime,
            camera_tracking=not args.no_camera,
            preset=args.preset,
        )
        metrics: dict[str, Any] = {
            "effective_condition": condition,
            "seed": runner.seed,
            "deterministic": config.evaluation.deterministic,
            "headless": bool(args.headless),
            "realtime": not args.no_realtime,
            "camera_tracking": not args.no_camera,
            "preset": args.preset if circle is not None else None,
            "model": str(selected_model) if selected_model is not None else None,
            "policies": {},
        }
        for policy_key, label in specs:
            print(f"\n>>> {label}")
            trace = runner.run(
                policy if policy_key == "residual" else None,
                policy_key,
                label,
            )
            plot_path = _save_trace(artifacts, config, trace, runner.circle)
            policy_metrics = _trace_metrics(trace, config.evaluation.tail_fraction)
            policy_metrics["plot"] = plot_path.relative_to(artifacts.run_dir).as_posix()
            metrics["policies"][policy_key] = policy_metrics
            print(f"    saved: {plot_path}")
        metrics_path = artifacts.write_metrics("evaluation", metrics)
        artifacts.finalize(
            "completed",
            effective_condition=condition,
            input_model=str(selected_model) if selected_model is not None else None,
        )
        print(f"metrics: {metrics_path}")
        print(f"run: {artifacts.run_dir}")
        return 0
    except Exception as exc:
        artifacts.finalize("failed", error=f"{type(exc).__name__}: {exc}")
        raise


__all__ = [
    "EvaluationRunner",
    "RolloutTrace",
    "build_parser",
    "run_evaluation_cli",
]

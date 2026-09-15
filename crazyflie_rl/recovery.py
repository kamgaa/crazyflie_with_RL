"""Deterministic recovery-grid evaluation for E2E hover policies."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactManager
from .attitude import (
    ATTITUDE_AXIS_CHOICES,
    CARDINAL_ATTITUDE_AXIS_CHOICES,
    attitude_axis_vector,
)
from .config import load_config
from .controllers import quat_normalize_wxyz, rotmat_from_quat_wxyz
from .factories import EnvironmentFactory
from .physics_version import PHYSICS_MODEL_VERSION, physics_comparison


RECOVERY_HOLD_SECONDS = 1.0
POSITION_TOLERANCE_M = 0.05
VELOCITY_TOLERANCE_M_S = 0.05
TILT_TOLERANCE_DEG = 3.0
ANGULAR_RATE_TOLERANCE_RAD_S = 0.1


@dataclass(frozen=True)
class RecoveryCase:
    category: str
    name: str
    tilt_deg: float
    tilt_axis_xyz: tuple[float, float, float]
    position_offset_xyz_m: tuple[float, float, float]


def recovery_cases() -> tuple[RecoveryCase, ...]:
    directions = tuple(
        (name, attitude_axis_vector(name)) for name in ATTITUDE_AXIS_CHOICES
    )
    cases: list[RecoveryCase] = []
    # Keep the complete requested Cartesian grid, including the physically
    # equivalent zero-degree cases for every labelled direction.  This makes
    # the CSV and heatmap axes unambiguous instead of silently collapsing a
    # requested grid point into a special nominal row.
    for tilt in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        for name, axis in directions:
            cases.append(
                RecoveryCase(
                    "attitude", f"{name}_{tilt:g}deg", tilt, axis, (0.0, 0.0, 0.0)
                )
            )
    for name, offset in (
        ("x_plus_0p20m", (0.20, 0.0, 0.0)),
        ("x_minus_0p20m", (-0.20, 0.0, 0.0)),
        ("y_plus_0p20m", (0.0, 0.20, 0.0)),
        ("y_minus_0p20m", (0.0, -0.20, 0.0)),
        ("z_plus_0p10m", (0.0, 0.0, 0.10)),
        ("z_minus_0p10m", (0.0, 0.0, -0.10)),
    ):
        cases.append(RecoveryCase("position", name, 0.0, (0.0, 0.0, 0.0), offset))
    return tuple(cases)


def reduced_recovery_cases() -> tuple[RecoveryCase, ...]:
    """Return the deterministic 100k-step training-selection grid."""

    cases: list[RecoveryCase] = [
        RecoveryCase(
            "attitude",
            "nominal",
            0.0,
            attitude_axis_vector("roll_plus"),
            (0.0, 0.0, 0.0),
        )
    ]
    for tilt in (5.0, 15.0, 30.0):
        for name in CARDINAL_ATTITUDE_AXIS_CHOICES:
            cases.append(
                RecoveryCase(
                    "attitude",
                    f"{name}_{tilt:g}deg",
                    tilt,
                    attitude_axis_vector(name),
                    (0.0, 0.0, 0.0),
                )
            )
    for name, offset in (
        ("x_plus_0p20m", (0.20, 0.0, 0.0)),
        ("x_minus_0p20m", (-0.20, 0.0, 0.0)),
        ("y_plus_0p20m", (0.0, 0.20, 0.0)),
        ("y_minus_0p20m", (0.0, -0.20, 0.0)),
        ("z_plus_0p10m", (0.0, 0.0, 0.10)),
        ("z_minus_0p10m", (0.0, 0.0, -0.10)),
    ):
        cases.append(RecoveryCase("position", name, 0.0, (0.0, 0.0, 0.0), offset))
    return tuple(cases)


def consecutive_recovery_time(
    within_tolerance: Sequence[bool], *, policy_hz: float
) -> float | None:
    """Return dwell completion for the continuous valid suffix, if long enough."""

    rate = float(policy_hz)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("policy_hz must be positive and finite")
    required = int(round(RECOVERY_HOLD_SECONDS * rate))
    suffix_steps = 0
    for value in reversed(within_tolerance):
        if not bool(value):
            break
        suffix_steps += 1
    if suffix_steps < required:
        return None
    suffix_start = len(within_tolerance) - suffix_steps
    return float((suffix_start + required) / rate)


def _ever_dwell_recovery_time(
    within_tolerance: Sequence[bool], *, policy_hz: float
) -> float | None:
    """Return the former first-dwell result for diagnostic comparison only."""

    rate = float(policy_hz)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("policy_hz must be positive and finite")
    required = int(round(RECOVERY_HOLD_SECONDS * rate))
    run = 0
    for index, value in enumerate(within_tolerance):
        run = run + 1 if bool(value) else 0
        if run >= required:
            return float((index + 1) / rate)
    return None


def recovery_success_diagnostics(
    within_tolerance: Sequence[bool],
    *,
    policy_hz: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    """Classify recovery from the episode's final continuous valid window.

    A time-limit truncation is an eligible episode boundary. An absorbing
    termination always fails, even in the unlikely event that its final state
    also satisfies the recovery tolerances.
    """

    rate = float(policy_hz)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("policy_hz must be positive and finite")
    required_steps = int(round(RECOVERY_HOLD_SECONDS * rate))
    suffix_steps = 0
    for value in reversed(within_tolerance):
        if not bool(value):
            break
        suffix_steps += 1
    terminal_recovery_time = consecutive_recovery_time(within_tolerance, policy_hz=rate)
    ever_recovery_time = _ever_dwell_recovery_time(within_tolerance, policy_hz=rate)
    terminal_window_satisfied = terminal_recovery_time is not None
    success = terminal_window_satisfied and not bool(terminated)
    # This is precisely the class that the former evaluator accepted and the
    # terminal-window definition rejects. Terminated cases were already rejected.
    transient_only = (
        ever_recovery_time is not None
        and not bool(terminated)
        and not terminal_window_satisfied
    )
    return {
        "success": bool(success),
        "recovery_time_s": terminal_recovery_time if success else None,
        "terminal_window_satisfied": bool(terminal_window_satisfied),
        "terminal_valid_suffix_steps": int(suffix_steps),
        "terminal_valid_suffix_s": float(suffix_steps / rate),
        "required_dwell_steps": int(required_steps),
        "ever_dwell_success": ever_recovery_time is not None,
        "ever_dwell_recovery_time_s": ever_recovery_time,
        "success_but_final_window_failure": bool(transient_only),
        "terminated_forces_failure": bool(terminated),
        "time_limit_truncated": bool(truncated),
    }


def _axis_angle_quaternion_wxyz(
    yaw_rad: float, axis_xyz: Sequence[float], angle_rad: float
) -> np.ndarray:
    axis = np.asarray(axis_xyz, dtype=float).reshape(3)
    angle = float(angle_rad)
    if abs(angle) > 0.0:
        norm = float(np.linalg.norm(axis))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("nonzero recovery tilt requires a unit axis")
    half_yaw = 0.5 * float(yaw_rad)
    half_angle = 0.5 * angle
    q_yaw = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
    q_tilt = np.concatenate([[np.cos(half_angle)], np.sin(half_angle) * axis])
    w1, x1, y1, z1 = q_yaw
    w2, x2, y2, z2 = q_tilt
    result = quat_normalize_wxyz(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    return result if result[0] >= 0.0 else -result


def set_recovery_case(environment: Any, case: RecoveryCase, *, seed: int) -> np.ndarray:
    """Install an explicit airborne case independently of training randomization."""

    import mujoco

    environment.reset(seed=seed)
    environment.data.qpos[0:3] = environment.pos_des + np.asarray(
        case.position_offset_xyz_m, dtype=float
    )
    environment.data.qpos[3:7] = _axis_angle_quaternion_wxyz(
        environment.yaw_des,
        case.tilt_axis_xyz,
        np.deg2rad(case.tilt_deg),
    )
    environment.data.qvel[:] = 0.0
    mujoco.mj_forward(environment.model, environment.data)
    environment.reset_actuator_state(airborne=True, resample_parameters=False)
    environment.pid.reset()
    environment._step = 0
    environment._prev_action = np.zeros(4)
    reset_auxiliary = getattr(environment, "reset_auxiliary_observation_state", None)
    if callable(reset_auxiliary):
        return np.asarray(reset_auxiliary())
    return environment._obs(*environment._read_state())


def _state_metrics(environment: Any) -> dict[str, float | list[float]]:
    position, quaternion, velocity, omega = environment._read_state()
    position_error = position - environment.pos_des
    rotation = rotmat_from_quat_wxyz(quaternion)
    tilt_deg = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
    return {
        "position_error": position_error.tolist(),
        "position_error_norm_m": float(np.linalg.norm(position_error)),
        "horizontal_displacement_m": float(np.linalg.norm(position_error[:2])),
        "altitude_m": float(position[2]),
        "velocity": velocity.tolist(),
        "velocity_norm_m_s": float(np.linalg.norm(velocity)),
        "tilt_deg": tilt_deg,
        "angular_velocity": omega.tolist(),
        "angular_velocity_norm_rad_s": float(np.linalg.norm(omega)),
    }


def evaluate_recovery_case(
    environment: Any,
    policy: Any,
    case: RecoveryCase,
    *,
    seed: int,
    duration_s: float = 8.0,
) -> dict[str, Any]:
    observation = set_recovery_case(environment, case, seed=seed)
    maximum_steps = int(round(duration_s * environment.policy_hz))
    within: list[bool] = []
    saturation_samples: list[float] = []
    state_samples: list[dict[str, Any]] = []
    termination_reasons: list[str] = []
    initial_normalized_action: np.ndarray | None = None
    terminated = truncated = False
    for _step in range(maximum_steps):
        action = policy.predict(observation, deterministic=True)[0]
        if initial_normalized_action is None:
            initial_normalized_action = (
                np.asarray(action, dtype=float).reshape(4).copy()
            )
        observation, _reward, terminated, truncated, info = environment.step(action)
        state = _state_metrics(environment)
        state_samples.append(state)
        within.append(
            state["position_error_norm_m"] < POSITION_TOLERANCE_M
            and state["velocity_norm_m_s"] < VELOCITY_TOLERANCE_M_S
            and state["tilt_deg"] < TILT_TOLERANCE_DEG
            and state["angular_velocity_norm_rad_s"] < ANGULAR_RATE_TOLERANCE_RAD_S
        )
        requested = np.asarray(environment._last_f_cmd, dtype=float)
        saturated = np.isclose(
            requested, environment.thrust_min, atol=1e-12, rtol=0.0
        ) | np.isclose(requested, environment.thrust_max, atol=1e-12, rtol=0.0)
        saturation_samples.append(float(np.mean(saturated)))
        if terminated or truncated:
            termination_reasons = list(info.get("termination_reasons", []))
            break
    if not state_samples:
        raise RuntimeError("recovery case ended without a transition")
    if initial_normalized_action is None:  # pragma: no cover - same loop invariant
        raise RuntimeError("recovery case is missing its first policy action")
    recovery_diagnostics = recovery_success_diagnostics(
        within,
        policy_hz=environment.policy_hz,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )
    final = state_samples[-1]
    return {
        **asdict(case),
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "physics_provenance": physics_comparison(
            getattr(policy, "physics_model_version", None)
        ),
        "reset_info": getattr(environment, "_last_reset_info", {}),
        "seed": seed,
        "duration_s": len(state_samples) / environment.policy_hz,
        **recovery_diagnostics,
        "initial_normalized_action": initial_normalized_action.tolist(),
        "initial_normalized_action_l2": float(
            np.linalg.norm(initial_normalized_action)
        ),
        "maximum_position_error_m": max(
            sample["position_error_norm_m"] for sample in state_samples
        ),
        "maximum_horizontal_displacement_m": max(
            sample["horizontal_displacement_m"] for sample in state_samples
        ),
        "minimum_altitude_m": min(sample["altitude_m"] for sample in state_samples),
        "maximum_tilt_deg": max(sample["tilt_deg"] for sample in state_samples),
        "maximum_angular_velocity_rad_s": max(
            sample["angular_velocity_norm_rad_s"] for sample in state_samples
        ),
        "motor_saturation_fraction": float(np.mean(saturation_samples)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": termination_reasons,
        "final_position_error_xyz_m": final["position_error"],
        "final_velocity_xyz_m_s": final["velocity"],
        "final_tilt_deg": final["tilt_deg"],
        "final_angular_velocity_xyz_rad_s": final["angular_velocity"],
    }


def summarize_recovery(
    controller: str,
    results: Sequence[Mapping[str, Any]],
    *,
    controller_type: str = "e2e_ppo",
) -> dict[str, Any]:
    """Aggregate physically distinct recovery conditions.

    The case CSV retains all eight labelled zero-degree attitude rows, but they
    describe one nominal state. Only the first is used in aggregate statistics
    and ranking; nonzero attitude and position perturbations remain independent.
    """

    nominal = [
        result
        for result in results
        if result["category"] == "attitude" and float(result["tilt_deg"]) == 0.0
    ]
    attitude = [
        result
        for result in results
        if result["category"] == "attitude" and float(result["tilt_deg"]) > 0.0
    ]
    position = [result for result in results if result["category"] == "position"]
    if not nominal:
        raise ValueError("recovery results must contain a nominal zero-degree case")
    if not attitude:
        raise ValueError("recovery results must contain nonzero attitude cases")
    if not position:
        raise ValueError("recovery results must contain position cases")

    aggregate_results = [nominal[0], *attitude, *position]
    successes = [result for result in aggregate_results if result["success"]]
    recovery_times = [float(result["recovery_time_s"]) for result in successes]
    attitude_successes = sum(bool(result["success"]) for result in attitude)
    position_successes = sum(bool(result["success"]) for result in position)
    terminated_count = sum(
        bool(result.get("terminated", False)) for result in aggregate_results
    )
    truncated_count = sum(
        bool(result.get("truncated", False)) for result in aggregate_results
    )
    termination_reason_counts: Counter[str] = Counter()
    for result in aggregate_results:
        reasons = result.get("termination_reason", ())
        if isinstance(reasons, str):
            reasons = (reasons,)
        termination_reason_counts.update(str(reason) for reason in reasons)
    transient_only = [
        result
        for result in aggregate_results
        if bool(result.get("success_but_final_window_failure", False))
    ]
    overall_success_rate = len(successes) / len(aggregate_results)
    return {
        "controller": str(controller),
        "controller_type": controller_type,
        # Compatibility field retained for downstream readers of older reports.
        "model": str(controller),
        "case_count": len(results),
        "aggregate_condition_count": len(aggregate_results),
        "nominal_duplicate_row_count": len(nominal),
        "nominal_hover_success": bool(nominal[0]["success"]),
        "attitude_perturbation_case_count": len(attitude),
        "attitude_perturbation_success_count": attitude_successes,
        "attitude_perturbation_success_rate": attitude_successes / len(attitude),
        "position_perturbation_case_count": len(position),
        "position_perturbation_success_count": position_successes,
        "position_perturbation_success_rate": position_successes / len(position),
        "success_count": len(successes),
        "overall_success_rate": overall_success_rate,
        # Compatibility alias: now deliberately means the de-duplicated rate.
        "success_rate": overall_success_rate,
        "terminated_case_count": terminated_count,
        "truncated_case_count": truncated_count,
        "termination_reason_counts": dict(sorted(termination_reason_counts.items())),
        "success_but_final_window_failure": bool(transient_only),
        "success_but_final_window_failure_count": len(transient_only),
        "success_but_final_window_failure_cases": [
            str(result["name"]) for result in transient_only
        ],
        "exact_hover_initial_action_l2": (
            float(nominal[0]["initial_normalized_action_l2"])
            if nominal[0].get("initial_normalized_action_l2") is not None
            else None
        ),
        "mean_recovery_time_s": (
            float(np.mean(recovery_times)) if recovery_times else None
        ),
        "mean_maximum_position_error_m": float(
            np.mean(
                [result["maximum_position_error_m"] for result in aggregate_results]
            )
        ),
    }


def recovery_ranking_key(summary: Mapping[str, Any]) -> tuple[float, ...]:
    """Return the deterministic recovery-donor ordering key."""

    nominal_success = bool(summary.get("nominal_hover_success", True))
    success_rate = float(
        summary["overall_success_rate"]
        if "overall_success_rate" in summary
        else summary["success_rate"]
    )
    recovery_time = summary.get("mean_recovery_time_s")
    exact_hover_bias = summary.get("exact_hover_initial_action_l2")
    return (
        -float(nominal_success),
        -success_rate,
        float(summary.get("terminated_case_count", 0)),
        -float(summary.get("position_perturbation_success_rate", 0.0)),
        -float(summary.get("attitude_perturbation_success_rate", 0.0)),
        float(summary["mean_maximum_position_error_m"]),
        float(recovery_time) if recovery_time is not None else float("inf"),
        float(exact_hover_bias) if exact_hover_bias is not None else float("inf"),
    )


def select_best_recovery(summaries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not summaries:
        raise ValueError("at least one recovery summary is required")
    eligible = [
        item for item in summaries if bool(item.get("nominal_hover_success", True))
    ]
    if not eligible:
        raise ValueError(
            "no recovery candidate satisfies the required nominal hover success"
        )
    return min(eligible, key=recovery_ranking_key)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value)
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _attitude_success_matrix(
    results: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Build the exact terminal-window success matrix rendered by the heatmap."""

    direction_names = (
        "roll_plus",
        "roll_minus",
        "pitch_plus",
        "pitch_minus",
        "diagonal_pp",
        "diagonal_pm",
        "diagonal_mp",
        "diagonal_mm",
    )
    tilts = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    lookup = {str(item["name"]): bool(item["success"]) for item in results}
    return np.asarray(
        [
            [lookup[f"{direction}_{tilt:g}deg"] for tilt in tilts]
            for direction in direction_names
        ],
        dtype=float,
    )


def _write_heatmap(
    path: Path, model_results: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]]
) -> None:
    import matplotlib.pyplot as plt

    models = len(model_results)
    fig, axes = plt.subplots(models, 1, figsize=(11, 4 * models), squeeze=False)
    direction_names = (
        "roll_plus",
        "roll_minus",
        "pitch_plus",
        "pitch_minus",
        "diagonal_pp",
        "diagonal_pm",
        "diagonal_mp",
        "diagonal_mm",
    )
    tilts = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    for row, (model_path, results) in enumerate(model_results):
        values = _attitude_success_matrix(results)
        axis = axes[row, 0]
        image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
        axis.set_xticks(range(len(tilts)), [f"{tilt:g}" for tilt in tilts])
        axis.set_yticks(range(len(direction_names)), direction_names)
        axis.set_xlabel("initial tilt [deg]")
        axis.set_title(model_path.name)
    fig.colorbar(image, ax=axes[:, 0].tolist(), label="success")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic E2E hover recovery"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument(
        "--include-pid-floor",
        action="store_true",
        help="also evaluate the existing cascade PID with zero residual action",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


class _ZeroResidualPolicy:
    """Adapter that exercises the existing PID-plus-zero-residual path."""

    @staticmethod
    def predict(
        observation: Any, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        del observation, deterministic
        return np.zeros(4, dtype=np.float32), None


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not np.isclose(args.duration, 8.0):
        raise SystemExit("recovery evaluation duration must remain 8 seconds")
    config = load_config(args.config)
    if config.control_mode != "e2e" or config.environment.reward.mode != "legacy":
        raise SystemExit(
            "recovery evaluator requires the E2E legacy-reward baseline profile"
        )
    manager = ArtifactManager.create(
        config,
        command=list(sys.argv if argv is None else ["evaluate_recovery.py", *argv]),
        condition="initial-perturb-recovery-evaluation",
        mission="hover",
        seed=args.seed,
    )
    from stable_baselines3 import PPO

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    model_results: list[tuple[Path, list[dict[str, Any]]]] = []
    try:
        for model_value in args.model:
            model_path = model_value.expanduser().resolve()
            if not model_path.is_file():
                raise FileNotFoundError(f"model does not exist: {model_path}")
            policy = PPO.load(str(model_path), device="cpu")
            from .warm_start import validate_loaded_observation_schema

            validate_loaded_observation_schema(policy, config, model_path=model_path)
            from .eval_cli import _model_provenance

            policy.physics_model_version = _model_provenance(model_path).get(
                "physics_model_version"
            )
            print(
                "Physics evaluation: "
                + physics_comparison(policy.physics_model_version)[
                    "physics_comparison_status"
                ]
            )
            environment = EnvironmentFactory(config).make(
                seed=args.seed,
                initial_state_randomization_enabled=False,
            )
            try:
                results = [
                    evaluate_recovery_case(
                        environment,
                        policy,
                        case,
                        seed=args.seed,
                        duration_s=args.duration,
                    )
                    for case in recovery_cases()
                ]
            finally:
                environment.close()
            for result in results:
                all_rows.append(
                    {
                        "model": str(model_path),
                        "controller_type": "e2e_ppo",
                        **result,
                    }
                )
            summaries.append(summarize_recovery(str(model_path), results))
            model_results.append((model_path, results))

        if args.include_pid_floor:
            floor_label = "PID floor"
            environment = EnvironmentFactory(config).make(
                seed=args.seed,
                mode="residual",
                initial_state_randomization_enabled=False,
            )
            try:
                floor_results = [
                    evaluate_recovery_case(
                        environment,
                        _ZeroResidualPolicy(),
                        case,
                        seed=args.seed,
                        duration_s=args.duration,
                    )
                    for case in recovery_cases()
                ]
            finally:
                environment.close()
            for result in floor_results:
                all_rows.append(
                    {
                        "model": floor_label,
                        "controller_type": "pid_floor",
                        **result,
                    }
                )
            summaries.append(
                summarize_recovery(
                    floor_label,
                    floor_results,
                    controller_type="pid_floor",
                )
            )
            model_results.append((Path(floor_label), floor_results))

        eligible_summaries = [
            item
            for item in summaries
            if item.get("controller_type") == "e2e_ppo"
            and bool(item.get("nominal_hover_success", True))
        ]
        best = (
            dict(select_best_recovery(eligible_summaries))
            if eligible_summaries
            else None
        )
        csv_path = manager.path("metrics", "recovery-cases", ".csv")
        _write_csv(csv_path, all_rows)
        summary_csv_path = manager.path("metrics", "recovery-summary", ".csv")
        _write_csv(summary_csv_path, summaries)
        heatmap_path = manager.path("plots", "recovery-success-heatmap", ".png")
        if not args.no_plots:
            _write_heatmap(heatmap_path, model_results)
        report = {
            "physics_model_version": PHYSICS_MODEL_VERSION,
            "criteria": {
                "success_definition": "terminal_continuous_valid_suffix",
                "continuous_hold_s": RECOVERY_HOLD_SECONDS,
                "position_error_norm_lt_m": POSITION_TOLERANCE_M,
                "velocity_norm_lt_m_s": VELOCITY_TOLERANCE_M_S,
                "tilt_lt_deg": TILT_TOLERANCE_DEG,
                "angular_velocity_norm_lt_rad_s": ANGULAR_RATE_TOLERANCE_RAD_S,
                "terminated_forces_failure": True,
                "time_limit_truncation_may_succeed": True,
                "ever_dwell_success_is_diagnostic_only": True,
            },
            "duration_s": args.duration,
            "seed": args.seed,
            "summaries": summaries,
            "best_recovery": best,
            "case_results": all_rows,
            "csv": str(csv_path),
            "summary_csv": str(summary_csv_path),
            "heatmap": None if args.no_plots else str(heatmap_path),
        }
        metrics_path = manager.write_metrics("recovery-evaluation", report)
        manager.write_metrics(
            "best-recovery",
            (
                best
                if best is not None
                else {
                    "selected": None,
                    "reason": "no candidate passed nominal hover",
                }
            ),
        )
        manager.write_runtime_config(
            {
                "models": [str(path.resolve()) for path in args.model],
                "seed": args.seed,
                "duration_s": args.duration,
                "training_randomization_disabled": True,
                "include_pid_floor": bool(args.include_pid_floor),
            }
        )
        manager.finalize(
            "completed",
            recovery_metrics=str(metrics_path),
            best_recovery=best,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"recovery artifacts: {manager.run_dir}")
    except BaseException as exc:
        manager.finalize("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    return 0


__all__ = [
    "RecoveryCase",
    "consecutive_recovery_time",
    "evaluate_recovery_case",
    "recovery_success_diagnostics",
    "recovery_ranking_key",
    "recovery_cases",
    "reduced_recovery_cases",
    "select_best_recovery",
    "set_recovery_case",
    "summarize_recovery",
]

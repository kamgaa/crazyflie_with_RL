"""Fixed 14-case evaluation and best-payload ranking for payload DR runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .attitude import attitude_axis_vector
from .controllers import rotmat_from_quat_wxyz
from .physics_version import PHYSICS_MODEL_VERSION, physics_comparison
from .recovery import (
    ANGULAR_RATE_TOLERANCE_RAD_S,
    POSITION_TOLERANCE_M,
    RECOVERY_HOLD_SECONDS,
    TILT_TOLERANCE_DEG,
    VELOCITY_TOLERANCE_M_S,
    RecoveryCase,
    recovery_success_diagnostics,
    set_recovery_case,
)


BEST_PAYLOAD_RANKING = (
    "nominal_success_required",
    "payload_success_count_descending",
    "payload_completed_full_duration_count_descending",
    "payload_completed_worst_tail_position_rmse_m_ascending",
)


@dataclass(frozen=True)
class FixedPayloadEvaluationCase:
    category: str
    name: str
    payload_mass_kg: float
    attachment_xy_m: tuple[float, float]
    tilt_axis_name: str = "roll_plus"
    tilt_deg: float = 0.0
    azimuth_deg: float | None = None


def fixed_payload_evaluation_cases() -> tuple[FixedPayloadEvaluationCase, ...]:
    cases = [
        FixedPayloadEvaluationCase("nominal", "nominal_no_payload", 0.0, (0.0, 0.0)),
        FixedPayloadEvaluationCase(
            "center_payload", "center_payload_5g", 0.005, (0.0, 0.0)
        ),
        FixedPayloadEvaluationCase(
            "center_payload", "center_payload_10g", 0.010, (0.0, 0.0)
        ),
    ]
    for degrees in range(0, 360, 45):
        angle = math.radians(degrees)
        cases.append(
            FixedPayloadEvaluationCase(
                "offset_payload",
                f"payload_10g_r30mm_az{degrees:03d}",
                0.010,
                (0.030 * math.cos(angle), 0.030 * math.sin(angle)),
                azimuth_deg=float(degrees),
            )
        )
    cases.extend(
        (
            FixedPayloadEvaluationCase(
                "recovery",
                "roll_minus_15deg",
                0.0,
                (0.0, 0.0),
                "roll_minus",
                15.0,
            ),
            FixedPayloadEvaluationCase(
                "recovery",
                "roll_minus_20deg",
                0.0,
                (0.0, 0.0),
                "roll_minus",
                20.0,
            ),
            FixedPayloadEvaluationCase(
                "recovery",
                "roll_plus_20deg",
                0.0,
                (0.0, 0.0),
                "roll_plus",
                20.0,
            ),
        )
    )
    if len(cases) != 14:  # pragma: no cover - construction invariant
        raise RuntimeError("fixed payload evaluation must contain 14 cases")
    return tuple(cases)


def _equilibrium_metrics(hover: Mapping[str, Any]) -> dict[str, Any]:
    thrust_value = hover["equilibrium_motor_thrust_n"]
    if thrust_value is None:
        return {
            "equilibrium_motor_thrust_n": None,
            "equilibrium_lower_margin_per_motor_n": None,
            "equilibrium_upper_margin_per_motor_n": None,
            "equilibrium_min_lower_margin_n": None,
            "equilibrium_min_upper_margin_n": None,
        }
    thrust = np.asarray(thrust_value, dtype=float)
    limits = np.asarray(hover["actual_thrust_limits_n"], dtype=float)
    lower = thrust - limits[:, 0]
    upper = limits[:, 1] - thrust
    return {
        "equilibrium_motor_thrust_n": thrust.tolist(),
        "equilibrium_lower_margin_per_motor_n": lower.tolist(),
        "equilibrium_upper_margin_per_motor_n": upper.tolist(),
        "equilibrium_min_lower_margin_n": float(np.min(lower)),
        "equilibrium_min_upper_margin_n": float(np.min(upper)),
    }


def _initial_snapshot(environment: Any, observation: np.ndarray) -> dict[str, Any]:
    position, quaternion, velocity, omega = environment._read_state()
    return {
        "qpos": environment.data.qpos.copy().tolist(),
        "qvel": environment.data.qvel.copy().tolist(),
        "observation": np.asarray(observation, dtype=float).tolist(),
        "position_body_origin_world_m": position.tolist(),
        "quaternion_body_to_world_wxyz": quaternion.tolist(),
        "linear_velocity_body_origin_world_m_s": velocity.tolist(),
        "angular_velocity_body_rad_s": omega.tolist(),
        "payload": environment.payload_snapshot(),
        "actuator": {
            "parameters": environment.actuator_snapshot(),
            "actual_thrust_n": environment._last_f.tolist(),
            "requested_thrust_n": environment._last_f_cmd.tolist(),
            "omega_rad_s": environment._last_omega.tolist(),
            "reaction_torque_nm": environment._last_q_actual.tolist(),
        },
        "physics_wrenches": environment.physics_wrench_snapshot(),
        "external_xfrc_applied": environment.data.xfrc_applied.tolist(),
        "contact_count": int(environment.data.ncon),
    }


def evaluate_fixed_payload_case(
    environment: Any,
    policy: Any,
    case: FixedPayloadEvaluationCase,
    *,
    seed: int,
    duration_s: float,
) -> dict[str, Any]:
    recovery_case = RecoveryCase(
        category=case.category,
        name=case.name,
        tilt_deg=case.tilt_deg,
        tilt_axis_xyz=attitude_axis_vector(case.tilt_axis_name),
        position_offset_xyz_m=(0.0, 0.0, 0.0),
    )
    observation = set_recovery_case(environment, recovery_case, seed=seed)
    initial = _initial_snapshot(environment, observation)
    hover = environment._last_reset_info["static_hover"]
    # The reset info is produced immediately before the explicit pose install;
    # payload/static certification are unchanged by that pose update.
    base = {
        **asdict(case),
        "attachment_body_m": [*case.attachment_xy_m, 0.0],
        "attachment_radius_m": float(np.linalg.norm(case.attachment_xy_m)),
        "seed": int(seed),
        "deterministic": True,
        "maximum_duration_s": float(duration_s),
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "physics_provenance": physics_comparison(
            getattr(policy, "physics_model_version", None)
        ),
        "physical_status_preflight": hover["physical_status"],
        "physically_feasible_preflight": hover["physically_feasible"],
        "e2e_action_reachable_preflight": hover["policy_allocator_reachable"],
        **_equilibrium_metrics(hover),
        "initial_snapshot": initial,
    }
    if hover["physical_status"] == "infeasible":
        return {
            **base,
            "execution_status": "skipped_physical_infeasible",
            "outcome_class": "preflight_physical_infeasible",
            "success": None,
            "termination_reason": ["preflight_physical_infeasible"],
            "survival_time_s": 0.0,
            "completed_full_duration": False,
        }

    maximum_steps = int(round(float(duration_s) * environment.policy_hz))
    position_errors: list[np.ndarray] = []
    tilts: list[float] = []
    rates: list[float] = []
    actual_thrusts: list[np.ndarray] = []
    requested_thrusts: list[np.ndarray] = []
    within: list[bool] = []
    terminated = truncated = False
    last_info: Mapping[str, Any] = {}
    for _ in range(maximum_steps):
        action = policy.predict(observation, deterministic=True)[0]
        observation, _reward, terminated, truncated, last_info = environment.step(
            action
        )
        position, quaternion, velocity, omega = environment._read_state()
        error = position - environment.pos_des
        rotation = rotmat_from_quat_wxyz(quaternion)
        tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
        rate = float(np.linalg.norm(omega))
        error_norm = float(np.linalg.norm(error))
        position_errors.append(error)
        tilts.append(tilt)
        rates.append(rate)
        actual_thrusts.append(environment._last_f.copy())
        requested_thrusts.append(environment._last_f_cmd.copy())
        within.append(
            error_norm < POSITION_TOLERANCE_M
            and float(np.linalg.norm(velocity)) < VELOCITY_TOLERANCE_M_S
            and tilt < TILT_TOLERANCE_DEG
            and rate < ANGULAR_RATE_TOLERANCE_RAD_S
        )
        if terminated or truncated:
            break
    diagnostics = recovery_success_diagnostics(
        within,
        policy_hz=environment.policy_hz,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )
    positions = np.asarray(position_errors, dtype=float)
    actual = np.asarray(actual_thrusts, dtype=float)
    requested = np.asarray(requested_thrusts, dtype=float)
    completed = len(positions) == maximum_steps and not bool(terminated)
    tail = positions[-int(round(environment.policy_hz)) :] if completed else None
    lower_clip = np.isclose(requested, environment.thrust_min, atol=1e-12, rtol=0.0)
    upper_clip = np.isclose(requested, environment.thrust_max, atol=1e-12, rtol=0.0)
    reasons = list(last_info.get("termination_reasons", []))
    if truncated and not terminated:
        reasons = ["time_limit"]
    if diagnostics["success"]:
        outcome = "success"
    elif hover["policy_allocator_reachable"] is False:
        outcome = "failure_e2e_action_unreachable"
    elif terminated:
        outcome = "failure_terminated"
    else:
        outcome = "failure_terminal_dwell_not_met"
    return {
        **base,
        "execution_status": "executed",
        "outcome_class": outcome,
        **diagnostics,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": reasons,
        "steps_executed": len(positions),
        "survival_time_s": len(positions) / environment.policy_hz,
        "completed_full_duration": bool(completed),
        "position_rmse_full_m": float(
            np.sqrt(np.mean(np.sum(positions * positions, axis=1)))
        ),
        "position_rmse_last_1s_m": (
            None
            if tail is None
            else float(np.sqrt(np.mean(np.sum(tail * tail, axis=1))))
        ),
        "mean_position_error_last_1s_m": (
            None if tail is None else float(np.mean(np.linalg.norm(tail, axis=1)))
        ),
        "mean_position_error_xyz_last_1s_m": (
            None if tail is None else np.mean(tail, axis=0).tolist()
        ),
        "maximum_tilt_deg": max(tilts),
        "maximum_body_angular_rate_rad_s": max(rates),
        "requested_motor_clipping_fraction": float(np.mean(lower_clip | upper_clip)),
        "requested_motor_lower_clipping_fraction": float(np.mean(lower_clip)),
        "requested_motor_upper_clipping_fraction": float(np.mean(upper_clip)),
        "actual_thrust_min_observed_n": float(np.min(actual)),
        "actual_thrust_max_observed_n": float(np.max(actual)),
        "actual_thrust_min_lower_margin_n": float(
            np.min(actual - environment.thrust_min)
        ),
        "actual_thrust_min_upper_margin_n": float(
            np.min(environment.thrust_max - actual)
        ),
        "final_snapshot": {
            "payload": environment.payload_snapshot(),
            "actuator": environment.actuator_snapshot(),
            "actual_thrust_n": environment._last_f.tolist(),
            "requested_thrust_n": environment._last_f_cmd.tolist(),
            "omega_rad_s": environment._last_omega.tolist(),
            "physics_wrenches": environment.physics_wrench_snapshot(),
            "external_xfrc_applied": environment.data.xfrc_applied.tolist(),
            "contact_count": int(environment.data.ncon),
            "observation_diagnostics": dict(
                last_info.get("observation_diagnostics", {})
            ),
            "legacy_reward_terms": dict(last_info.get("legacy_reward_terms", {})),
            "transition_timing": dict(last_info.get("transition_timing", {})),
        },
    }


def evaluate_fixed_payload_suite(
    environment_factory: Any,
    policy: Any,
    *,
    seed: int = 1000,
    duration_s: float = 8.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in fixed_payload_evaluation_cases():
        environment = environment_factory.make(
            seed=seed,
            mode="e2e",
            com_bias_mass=case.payload_mass_kg,
            com_bias_offset=case.attachment_xy_m,
            com_bias_randomize=False,
            payload_curriculum_enabled=False,
            pos_perturb=0.0,
            att_perturb_deg=0.0,
            initial_state_randomization_enabled=False,
            track_curriculum_steps=False,
        )
        try:
            results.append(
                evaluate_fixed_payload_case(
                    environment,
                    policy,
                    case,
                    seed=seed,
                    duration_s=duration_s,
                )
            )
        finally:
            environment.close()
    return results, summarize_fixed_payload_suite(results)


def _category_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    executed = [row for row in results if row["execution_status"] == "executed"]
    return {
        "case_count": len(results),
        "executed_count": len(executed),
        "success_count": sum(row["success"] is True for row in executed),
        "completed_full_duration_count": sum(
            bool(row["completed_full_duration"]) for row in executed
        ),
        "termination_reason_counts": dict(
            Counter(reason for row in executed for reason in row["termination_reason"])
        ),
    }


def summarize_fixed_payload_suite(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    nominal = [row for row in results if row["category"] == "nominal"]
    centered = [row for row in results if row["category"] == "center_payload"]
    offset = [row for row in results if row["category"] == "offset_payload"]
    recovery = [row for row in results if row["category"] == "recovery"]
    if (
        len(nominal) != 1
        or len(centered) != 2
        or len(offset) != 8
        or len(recovery) != 3
    ):
        raise ValueError(
            "fixed payload evaluation does not contain the required 14 cases"
        )
    payload = [*centered, *offset]
    completed_payload = [
        row for row in payload if bool(row.get("completed_full_duration", False))
    ]
    tail_values = [
        float(row["position_rmse_last_1s_m"])
        for row in completed_payload
        if row.get("position_rmse_last_1s_m") is not None
    ]
    return {
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "case_count": len(results),
        "criteria": {
            "success_definition": "terminal_continuous_valid_suffix",
            "continuous_hold_s": RECOVERY_HOLD_SECONDS,
            "position_error_norm_lt_m": POSITION_TOLERANCE_M,
            "velocity_norm_lt_m_s": VELOCITY_TOLERANCE_M_S,
            "tilt_lt_deg": TILT_TOLERANCE_DEG,
            "angular_velocity_norm_lt_rad_s": ANGULAR_RATE_TOLERANCE_RAD_S,
            "terminated_forces_failure": True,
            "time_limit_truncation_may_succeed": True,
        },
        "execution_contract": {
            "payload_dr_disabled": True,
            "initial_state_randomization_disabled": True,
            "wind_and_additional_disturbance": "none",
            "deterministic_policy": True,
            "fixed_case_feasibility_filter_applied": False,
            "physical_infeasible_execution": "skipped",
            "e2e_action_unreachable_execution": "executed_and_classified_separately",
        },
        "nominal_success": nominal[0]["success"] is True,
        "payload_case_count": len(payload),
        "payload_success_count": sum(row["success"] is True for row in payload),
        "payload_completed_full_duration_count": len(completed_payload),
        "payload_completed_worst_tail_position_rmse_m": (
            max(tail_values) if tail_values else None
        ),
        "nominal": _category_summary(nominal),
        "center_payload": _category_summary(centered),
        "offset_payload": _category_summary(offset),
        "recovery": _category_summary(recovery),
        "roll_minus_20deg_success_is_required": False,
        "ranking": list(BEST_PAYLOAD_RANKING),
        "short_terminated_rmse_used_for_ranking": False,
    }


def best_payload_ranking_key(summary: Mapping[str, Any]) -> tuple[float, ...]:
    """Return the ordered key after the caller applies the nominal hard gate."""

    if not bool(summary.get("nominal_success", False)):
        raise ValueError("best-payload ranking requires nominal success")
    worst_tail = summary.get("payload_completed_worst_tail_position_rmse_m")
    return (
        -float(summary["payload_success_count"]),
        -float(summary["payload_completed_full_duration_count"]),
        float("inf") if worst_tail is None else float(worst_tail),
    )


__all__ = [
    "BEST_PAYLOAD_RANKING",
    "FixedPayloadEvaluationCase",
    "best_payload_ranking_key",
    "evaluate_fixed_payload_case",
    "evaluate_fixed_payload_suite",
    "fixed_payload_evaluation_cases",
    "summarize_fixed_payload_suite",
]

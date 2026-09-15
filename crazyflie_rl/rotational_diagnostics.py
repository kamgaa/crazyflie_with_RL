"""Physics-rate, three-axis rotational diagnostics for E2E PPO policies.

The diagnostic observes the existing environment control and actuator path.  It
does not implement a second controller, allocator, quaternion initializer, or
plant model, and its physics-substep observer is installed only for an explicit
diagnostic rollout.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from .attitude import ATTITUDE_AXIS_CHOICES, attitude_axis_vector
from .config import ExperimentConfig, load_config
from .controllers import (
    build_allocation_matrix,
    quat_normalize_wxyz,
    rotmat_from_quat_wxyz,
)
from .e2e_diagnostics import _load_policy, _validate_loaded_policy_contract
from .factories import EnvironmentFactory
from .recovery import (
    ANGULAR_RATE_TOLERANCE_RAD_S,
    POSITION_TOLERANCE_M,
    TILT_TOLERANCE_DEG,
    VELOCITY_TOLERANCE_M_S,
    RecoveryCase,
    recovery_success_diagnostics,
    set_recovery_case,
)
from .warm_start import E2EPolicyCompatibility, validate_e2e_policy_compatibility
from .physics_version import PHYSICS_MODEL_VERSION


AXIS_KEYS = ("x", "y", "z")
ANGLE_KEYS = ("roll", "pitch", "yaw")
POWER_EPSILON_W = 1e-12
TORQUE_EPSILON_NM = 1e-12
SMALL_DAMPING_FRACTION = 0.10
ALLOWED_CHECKPOINT_KINDS = frozenset({"best", "best-recovery", "final"})


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite shape ({size},) vector")
    return result.copy()


def euler_rpy_rad(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """Convert the shared body-to-world wxyz convention to roll/pitch/yaw."""

    quaternion = quat_normalize_wxyz(quaternion_wxyz)
    rotation = rotmat_from_quat_wxyz(quaternion)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def inertia_matrix_body(environment: Any) -> np.ndarray:
    """Return MuJoCo's diagonal inertial-frame tensor in the body frame."""

    diagonal = _finite_vector(
        environment.model.body_inertia[environment.drone_bid],
        3,
        "body inertia diagonal",
    )
    body_iquat = getattr(environment.model, "body_iquat", None)
    inertial_quaternion = (
        np.array([1.0, 0.0, 0.0, 0.0])
        if body_iquat is None
        else np.asarray(body_iquat[environment.drone_bid], dtype=float)
    )
    rotation_body_from_inertial = rotmat_from_quat_wxyz(
        quat_normalize_wxyz(inertial_quaternion)
    )
    result = (
        rotation_body_from_inertial
        @ np.diag(diagonal)
        @ rotation_body_from_inertial.T
    )
    if not np.allclose(result, result.T, rtol=0.0, atol=1e-15):
        raise RuntimeError("body inertia matrix is not symmetric")
    return result


def reconstruct_three_axis_torques(environment: Any) -> dict[str, np.ndarray]:
    """Read all torque stages already computed by the live environment."""

    raw = _finite_vector(environment._last_wrench_cmd, 4, "raw commanded wrench")
    allocated = _finite_vector(
        environment._last_wrench_allocated, 4, "allocated requested wrench"
    )
    actual = _finite_vector(environment._last_wrench_actual, 4, "actual motor wrench")
    snapshotter = getattr(environment, "physics_wrench_snapshot", None)
    physical = snapshotter() if callable(snapshotter) else None
    motor_com = (
        np.asarray(physical["motor_wrench_drone_com_body"][:3])
        if physical is not None
        else actual[:3]
    )
    external = (
        np.asarray(physical["external_applied_wrench_drone_com_body"][:3])
        if physical is not None
        else np.zeros(3)
    )
    return {
        "raw_commanded_torque_xyz_nm": raw[:3],
        "allocated_requested_torque_xyz_nm": allocated[:3],
        "actual_motor_torque_xyz_nm": motor_com,
        "nominal_allocator_actual_torque_xyz_nm": actual[:3],
        "external_applied_torque_xyz_nm": external,
        "command_actual_torque_error_xyz_nm": raw[:3] - actual[:3],
    }


def rotational_power(
    torque_xyz_nm: Sequence[float], omega_xyz_rad_s: Sequence[float]
) -> tuple[np.ndarray, float]:
    """Return per-axis and total rotational power from one aligned sample."""

    torque = _finite_vector(torque_xyz_nm, 3, "torque")
    omega = _finite_vector(omega_xyz_rad_s, 3, "body angular velocity")
    axis_power = torque * omega
    return axis_power, float(np.dot(torque, omega))


def integrate_axis_energy(
    power_xyz_w: Sequence[Sequence[float]], dt_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rectangle-integrate positive and negative per-axis power magnitudes."""

    power = np.asarray(power_xyz_w, dtype=float)
    dt = float(dt_s)
    if power.ndim != 2 or power.shape[1:] != (3,) or not np.all(np.isfinite(power)):
        raise ValueError("axis power must be a finite shape (N, 3) array")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("energy integration dt must be positive and finite")
    positive = np.sum(np.maximum(power, 0.0), axis=0) * dt
    dissipated = np.sum(np.maximum(-power, 0.0), axis=0) * dt
    return positive, dissipated


def integrate_total_energy(
    total_power_w: Sequence[float], dt_s: float
) -> tuple[float, float]:
    """Integrate positive and dissipative portions of total rotational power."""

    power = np.asarray(total_power_w, dtype=float)
    dt = float(dt_s)
    if power.ndim != 1 or not np.all(np.isfinite(power)):
        raise ValueError("total rotational power must be a finite vector")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("energy integration dt must be positive and finite")
    return (
        float(np.sum(np.maximum(power, 0.0)) * dt),
        float(np.sum(np.maximum(-power, 0.0)) * dt),
    )


def gyroscopic_torque_xyz_nm(
    omega_xyz_rad_s: Sequence[float], inertia_matrix: Sequence[Sequence[float]]
) -> np.ndarray:
    """Compute omega x (I omega) in body coordinates."""

    omega = _finite_vector(omega_xyz_rad_s, 3, "body angular velocity")
    inertia = np.asarray(inertia_matrix, dtype=float)
    if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
        raise ValueError("inertia matrix must be a finite shape (3, 3) array")
    return np.cross(omega, inertia @ omega)


def first_zero_crossing_s(
    time_s: Sequence[float], values: Sequence[float], *, tolerance: float = 1e-10
) -> float | None:
    """Return the first interpolated sign crossing after a nonzero excursion."""

    time = np.asarray(time_s, dtype=float)
    signal = np.asarray(values, dtype=float)
    if (
        time.ndim != 1
        or signal.ndim != 1
        or len(time) != len(signal)
        or len(time) == 0
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(signal))
    ):
        raise ValueError("zero-crossing inputs must be equal-length finite vectors")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("zero-crossing times must be strictly increasing")
    tol = float(tolerance)
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError("zero-crossing tolerance must be non-negative and finite")

    previous_index: int | None = None
    previous_sign = 0
    for index, value in enumerate(signal):
        if abs(float(value)) <= tol:
            if previous_index is not None:
                return float(time[index])
            continue
        sign = 1 if value > 0.0 else -1
        if previous_index is not None and sign != previous_sign:
            left = float(signal[previous_index])
            right = float(value)
            fraction = -left / (right - left)
            return float(
                time[previous_index]
                + fraction * (time[index] - time[previous_index])
            )
        previous_index = index
        previous_sign = sign
    return None


def estimate_actuator_delay_s(
    commanded: Sequence[float],
    actual: Sequence[float],
    dt_s: float,
    *,
    maximum_delay_s: float = 0.20,
) -> float | None:
    """Estimate causal command-to-actual lag by normalized cross-correlation."""

    command = np.asarray(commanded, dtype=float)
    response = np.asarray(actual, dtype=float)
    dt = float(dt_s)
    if (
        command.ndim != 1
        or response.ndim != 1
        or command.shape != response.shape
        or not np.all(np.isfinite(command))
        or not np.all(np.isfinite(response))
    ):
        raise ValueError("delay inputs must be equal-length finite vectors")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("delay dt must be positive and finite")
    maximum = float(maximum_delay_s)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("maximum delay must be non-negative and finite")
    if len(command) < 8 or np.ptp(command) <= 1e-15 or np.ptp(response) <= 1e-15:
        return None

    maximum_lag = min(int(round(maximum / dt)), len(command) // 2)
    best_lag: int | None = None
    best_correlation = -math.inf
    for lag in range(maximum_lag + 1):
        left = command[: len(command) - lag] if lag else command
        right = response[lag:] if lag else response
        if len(left) < 8:
            continue
        left_centered = left - np.mean(left)
        right_centered = right - np.mean(right)
        denominator = float(
            np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
        )
        if denominator <= 1e-30:
            continue
        correlation = float(np.dot(left_centered, right_centered) / denominator)
        if correlation > best_correlation + 1e-15:
            best_lag = lag
            best_correlation = correlation
    if best_lag is None or best_correlation < 0.10:
        return None
    return float(best_lag * dt)


def boolean_event_intervals(
    time_s: Sequence[float], active: Sequence[bool], dt_s: float
) -> list[dict[str, float]]:
    """Convert an aligned physics-sample mask into start/end intervals."""

    time = np.asarray(time_s, dtype=float)
    mask = np.asarray(active, dtype=bool)
    dt = float(dt_s)
    if time.ndim != 1 or mask.ndim != 1 or time.shape != mask.shape:
        raise ValueError("event times and mask must be equal-length vectors")
    if len(time) == 0:
        return []
    intervals: list[dict[str, float]] = []
    start: float | None = None
    for index, enabled in enumerate(mask):
        if enabled and start is None:
            start = max(0.0, float(time[index] - dt))
        if not enabled and start is not None:
            intervals.append({"start_s": start, "end_s": float(time[index - 1])})
            start = None
    if start is not None:
        intervals.append({"start_s": start, "end_s": float(time[-1])})
    return intervals


class PhysicsSubstepRecorder:
    """Capture the live control/actuator/state values at MuJoCo's rate."""

    def __init__(self, environment: Any):
        self.environment = environment
        self.dt_s = float(environment.dt_phys)
        self.samples: list[dict[str, Any]] = []
        self._raw_action: np.ndarray | None = None
        self._policy_step_index: int | None = None
        self._policy_sample_start = 0

    def begin_policy_step(
        self, action: Sequence[float], *, policy_step_index: int
    ) -> None:
        self._raw_action = _finite_vector(action, 4, "normalized policy action")
        self._policy_step_index = int(policy_step_index)
        self._policy_sample_start = len(self.samples)

    def __call__(
        self,
        *,
        environment: Any,
        policy_step_index: int,
        substep_index: int,
        state_before: Sequence[np.ndarray],
        state_after: Sequence[np.ndarray],
    ) -> None:
        if environment is not self.environment:
            raise RuntimeError("physics observer received an unexpected environment")
        if self._raw_action is None or self._policy_step_index is None:
            raise RuntimeError("physics observer has no active policy command")
        if int(policy_step_index) != self._policy_step_index:
            raise RuntimeError("physics observer policy-step alignment failed")

        quaternion = _finite_vector(state_after[1], 4, "quaternion")
        omega_before = _finite_vector(state_before[3], 3, "body omega before")
        omega_after = _finite_vector(state_after[3], 3, "body omega after")
        angular_acceleration = (omega_after - omega_before) / self.dt_s
        normalized_action = np.clip(self._raw_action, -1.0, 1.0)
        action_clipped_mask = self._raw_action != normalized_action

        torques = reconstruct_three_axis_torques(environment)
        commanded_power, commanded_total = rotational_power(
            torques["raw_commanded_torque_xyz_nm"], omega_after
        )
        actual_power, actual_total = rotational_power(
            torques["actual_motor_torque_xyz_nm"], omega_after
        )
        inertia = inertia_matrix_body(environment)
        gyroscopic = gyroscopic_torque_xyz_nm(omega_after, inertia)
        effective = (
            torques["actual_motor_torque_xyz_nm"]
            + torques["external_applied_torque_xyz_nm"]
            - gyroscopic
        )
        requested_motor = _finite_vector(
            environment._last_f_cmd, 4, "requested motor thrust"
        )
        actual_motor = _finite_vector(environment._last_f, 4, "actual motor thrust")
        motor_clipped_mask = np.isclose(
            requested_motor, environment.thrust_min, rtol=0.0, atol=1e-12
        ) | np.isclose(
            requested_motor, environment.thrust_max, rtol=0.0, atol=1e-12
        )

        self.samples.append(
            {
                "rotation_reference": "drone body CoM, body axes; not whole-vehicle locked inertia",
                "child_dof_note": "relative child rotor energy and joint reaction power are excluded from drone-only energy balance",
                "physics_wrenches": (
                    environment.physics_wrench_snapshot()
                    if callable(
                        getattr(environment, "physics_wrench_snapshot", None)
                    )
                    else None
                ),
                "physics_step": len(self.samples) + 1,
                "policy_step": int(policy_step_index) + 1,
                "physics_substep": int(substep_index) + 1,
                "time_s": (len(self.samples) + 1) * self.dt_s,
                "quaternion_wxyz": quaternion,
                "roll_pitch_yaw_rad": euler_rpy_rad(quaternion),
                "body_omega_xyz_rad_s": omega_after,
                "body_angular_acceleration_xyz_rad_s2": angular_acceleration,
                "normalized_action_tau_xyz": normalized_action[:3],
                "normalized_action_fz": float(normalized_action[3]),
                "raw_policy_action": self._raw_action.copy(),
                "action_clipped": bool(np.any(action_clipped_mask)),
                "action_clipped_mask": action_clipped_mask,
                **torques,
                "requested_motor_thrust_n": requested_motor,
                "actual_motor_thrust_n": actual_motor,
                "motor_thrust_clipped": bool(np.any(motor_clipped_mask)),
                "motor_thrust_clipped_mask": motor_clipped_mask,
                "commanded_axis_power_xyz_w": commanded_power,
                "actual_axis_power_xyz_w": actual_power,
                "commanded_total_rotational_power_w": commanded_total,
                "actual_total_rotational_power_w": actual_total,
                "inertia_matrix_body": inertia,
                "gyroscopic_torque_xyz_nm": gyroscopic,
                "effective_torque_xyz_nm": effective,
                "rotational_kinetic_energy_j": float(
                    0.5 * omega_after @ inertia @ omega_after
                ),
                "terminated": False,
                "truncated": False,
                "termination_reasons": [],
            }
        )

    def finish_policy_step(
        self,
        *,
        terminated: bool,
        truncated: bool,
        termination_reasons: Sequence[str],
    ) -> None:
        captured = len(self.samples) - self._policy_sample_start
        if captured != int(self.environment.substeps):
            raise RuntimeError(
                "physics observer captured "
                f"{captured} samples for {self.environment.substeps} substeps"
            )
        if self.samples:
            self.samples[-1]["terminated"] = bool(terminated)
            self.samples[-1]["truncated"] = bool(truncated)
            self.samples[-1]["termination_reasons"] = [
                str(value) for value in termination_reasons
            ]
        self._raw_action = None
        self._policy_step_index = None


def _recovery_tolerance_sample(environment: Any) -> bool:
    position, quaternion, velocity, omega = environment._read_state()
    position_error = position - environment.pos_des
    rotation = rotmat_from_quat_wxyz(quaternion)
    tilt_deg = float(
        np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
    )
    return bool(
        np.linalg.norm(position_error) < POSITION_TOLERANCE_M
        and np.linalg.norm(velocity) < VELOCITY_TOLERANCE_M_S
        and tilt_deg < TILT_TOLERANCE_DEG
        and np.linalg.norm(omega) < ANGULAR_RATE_TOLERANCE_RAD_S
    )


def validate_rotational_diagnostic_contract(
    config: ExperimentConfig,
    environment: Any,
    policy: Any,
    compatibility: E2EPolicyCompatibility,
) -> dict[str, Any]:
    """Strictly validate the policy and live rotational plant ABI."""

    _validate_loaded_policy_contract(policy, config)
    errors: list[str] = []
    if compatibility.checkpoint_kind not in ALLOWED_CHECKPOINT_KINDS:
        errors.append(
            f"checkpoint kind {compatibility.checkpoint_kind!r} is not one of "
            f"{sorted(ALLOWED_CHECKPOINT_KINDS)}"
        )
    provenance = compatibility.training_provenance
    if tuple(provenance.get("observation_shape") or ()) != tuple(
        config.observation_shape
    ):
        errors.append("manifest observation shape is absent or incompatible")
    if tuple(provenance.get("action_shape") or ()) != tuple(config.action_shape):
        errors.append("manifest action shape is absent or incompatible")
    if tuple(float(v) for v in provenance.get("action_scale") or ()) != tuple(
        float(v) for v in config.environment.residual_scale
    ):
        errors.append("manifest action scale is absent or incompatible")
    if provenance.get("actuator_model") != config.actuator.model:
        errors.append("manifest actuator model is incompatible")

    try:
        manifest = json.loads(compatibility.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read diagnostic policy manifest: {exc}") from exc
    resolved = manifest.get("resolved_config")
    if not isinstance(resolved, Mapping):
        errors.append("manifest resolved_config is missing")
        resolved = {}
    archived_actuator = manifest.get("actuator", resolved.get("actuator"))
    runtime_actuator = config.resolved_dict()["actuator"]
    if not isinstance(archived_actuator, Mapping) or dict(archived_actuator) != dict(
        runtime_actuator
    ):
        errors.append("manifest actuator configuration differs from runtime")

    archived_vehicle = resolved.get("vehicle", {})
    if not isinstance(archived_vehicle, Mapping):
        archived_vehicle = {}
    try:
        archived_allocation, archived_inverse = build_allocation_matrix(
            float(archived_vehicle["arm_length"]),
            archived_vehicle["motor_direction"],
            float(archived_vehicle["torque_coefficient"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"manifest allocation inputs are missing or invalid: {exc}")
        archived_allocation = archived_inverse = np.full((4, 4), np.nan)
    if not np.allclose(
        environment.B, archived_allocation, rtol=0.0, atol=1e-15
    ):
        errors.append("live allocation matrix differs from training provenance")
    if not np.allclose(
        environment.B_pinv, archived_inverse, rtol=0.0, atol=1e-15
    ):
        errors.append("live allocation pseudoinverse differs from training provenance")
    live_actuator = environment.actuator_snapshot()
    if not live_actuator.get("enabled") or live_actuator.get("model") != config.actuator.model:
        errors.append("live actuator model is unavailable or incompatible")
    expected_substeps = int(
        round(config.vehicle.physics_hz / config.environment.policy_hz)
    )
    if int(environment.substeps) != expected_substeps:
        errors.append(
            f"live substeps {environment.substeps} != expected {expected_substeps}"
        )
    if errors:
        raise ValueError("incompatible rotational diagnostic contract: " + "; ".join(errors))
    return {
        "checkpoint_kind": compatibility.checkpoint_kind,
        "observation_shape": list(config.observation_shape),
        "action_shape": list(config.action_shape),
        "action_scale": [float(v) for v in config.environment.residual_scale],
        "actuator_model": config.actuator.model,
        "actuator_configuration_exact_match": True,
        "allocation_matrix_exact_match": True,
        "allocation_matrix": np.asarray(environment.B, dtype=float).tolist(),
        "allocation_pseudoinverse": np.asarray(
            environment.B_pinv, dtype=float
        ).tolist(),
        "physics_hz": float(config.vehicle.physics_hz),
        "policy_hz": float(config.environment.policy_hz),
        "physics_substeps_per_policy_action": int(environment.substeps),
    }


def _event_evidence(mask: np.ndarray, time: np.ndarray) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    return {
        "detected": bool(count),
        "sample_count": count,
        "fraction_of_analysis_window": float(count / len(mask)) if len(mask) else 0.0,
        "first_time_s": float(time[np.flatnonzero(mask)[0]]) if count else None,
    }


def _classify_axis_after_crossing(
    samples: Sequence[Mapping[str, Any]],
    first_crossing_s: float | None,
    axis_index: int,
    *,
    initial_kinetic_energy_j: float,
) -> dict[str, Any]:
    key = AXIS_KEYS[axis_index]
    if first_crossing_s is None or not samples:
        return {
            "axis": key,
            "available": False,
            "first_zero_crossing_s": first_crossing_s,
            "categories": [],
        }
    time = np.asarray([row["time_s"] for row in samples], dtype=float)
    selected = time >= float(first_crossing_s)
    if not np.any(selected):
        return {
            "axis": key,
            "available": False,
            "first_zero_crossing_s": first_crossing_s,
            "categories": [],
        }
    indices = np.flatnonzero(selected)
    selected_time = time[indices]
    commanded_power = np.asarray(
        [row["commanded_axis_power_xyz_w"][axis_index] for row in samples],
        dtype=float,
    )[indices]
    actual_power = np.asarray(
        [row["actual_axis_power_xyz_w"][axis_index] for row in samples], dtype=float
    )[indices]
    kinetic = np.asarray(
        [row["rotational_kinetic_energy_j"] for row in samples], dtype=float
    )
    previous_kinetic = np.concatenate(
        [[float(initial_kinetic_energy_j)], kinetic[:-1]]
    )
    kinetic_rate = (kinetic - previous_kinetic) / float(
        samples[0]["time_s"]
    )
    kinetic_rate = kinetic_rate[indices]
    raw_torque = np.asarray(
        [row["raw_commanded_torque_xyz_nm"][axis_index] for row in samples],
        dtype=float,
    )[indices]
    allocated_torque = np.asarray(
        [row["allocated_requested_torque_xyz_nm"][axis_index] for row in samples],
        dtype=float,
    )[indices]

    damping_reference = float(
        np.max(np.abs(commanded_power[commanded_power < -POWER_EPSILON_W]))
    ) if np.any(commanded_power < -POWER_EPSILON_W) else 0.0
    small_threshold = max(POWER_EPSILON_W, SMALL_DAMPING_FRACTION * damping_reference)
    masks = [
        commanded_power > POWER_EPSILON_W,
        (commanded_power < -POWER_EPSILON_W) & (actual_power > POWER_EPSILON_W),
        (commanded_power < -POWER_EPSILON_W)
        & (actual_power < -POWER_EPSILON_W)
        & (np.abs(commanded_power) <= small_threshold)
        & (np.abs(actual_power) <= small_threshold),
        (actual_power < -POWER_EPSILON_W) & (kinetic_rate > POWER_EPSILON_W),
        np.abs(raw_torque - allocated_torque) > TORQUE_EPSILON_NM,
    ]
    definitions = [
        (
            1,
            "policy_anti_damping_command",
            "commanded axis torque multiplied by same-axis omega is positive",
        ),
        (
            2,
            "actuator_lag_or_actual_torque_sign_delay",
            "commanded axis power is negative while actual axis power is positive",
        ),
        (
            3,
            "insufficient_damping_magnitude",
            "commanded and actual axis power are negative but both are small",
        ),
        (
            4,
            "cross_axis_coupling",
            "same-axis actual power is damping while total rotational kinetic energy increases",
        ),
        (
            5,
            "control_allocation_or_motor_clipping",
            "raw commanded and allocated requested same-axis torque differ",
        ),
    ]
    categories: list[dict[str, Any]] = []
    for (identifier, name, definition), mask in zip(definitions, masks):
        evidence = _event_evidence(mask, selected_time)
        categories.append(
            {
                "id": identifier,
                "name": name,
                "definition": definition,
                **evidence,
            }
        )
    categories[2]["small_power_threshold_w"] = small_threshold
    return {
        "axis": key,
        "available": True,
        "first_zero_crossing_s": float(first_crossing_s),
        "analysis_start_s": float(selected_time[0]),
        "analysis_end_s": float(selected_time[-1]),
        "sample_count": int(len(selected_time)),
        "categories": categories,
    }


def summarize_rotational_rollout(
    samples: Sequence[Mapping[str, Any]],
    *,
    initial_condition: Mapping[str, Any],
    recovery: Mapping[str, Any],
    termination_reasons: Sequence[str],
    terminated: bool,
    truncated: bool,
    requested_duration_s: float,
    contract: Mapping[str, Any],
    compatibility: E2EPolicyCompatibility,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot summarize an empty rotational rollout")
    dt = float(samples[0]["time_s"])
    sample_time = np.asarray([row["time_s"] for row in samples], dtype=float)
    initial_rpy = _finite_vector(
        initial_condition["roll_pitch_yaw_rad"], 3, "initial roll/pitch/yaw"
    )
    initial_omega = _finite_vector(
        initial_condition["body_omega_xyz_rad_s"], 3, "initial body omega"
    )
    rpy = np.asarray([row["roll_pitch_yaw_rad"] for row in samples], dtype=float)
    omega = np.asarray([row["body_omega_xyz_rad_s"] for row in samples], dtype=float)
    time_with_initial = np.concatenate([[0.0], sample_time])
    rpy_with_initial = np.vstack([initial_rpy, rpy])
    omega_with_initial = np.vstack([initial_omega, omega])
    crossings = {
        key: first_zero_crossing_s(time_with_initial, rpy_with_initial[:, index])
        for index, key in enumerate(AXIS_KEYS)
    }
    allocated_torque = np.asarray(
        [row["allocated_requested_torque_xyz_nm"] for row in samples], dtype=float
    )
    actual_torque = np.asarray(
        [row["actual_motor_torque_xyz_nm"] for row in samples], dtype=float
    )
    torque_error = np.asarray(
        [row["command_actual_torque_error_xyz_nm"] for row in samples], dtype=float
    )
    actual_power = np.asarray(
        [row["actual_axis_power_xyz_w"] for row in samples], dtype=float
    )
    actual_total_power = np.asarray(
        [row["actual_total_rotational_power_w"] for row in samples], dtype=float
    )
    positive, dissipated = integrate_axis_energy(actual_power, dt)
    total_positive, total_dissipated = integrate_total_energy(actual_total_power, dt)
    delays = {
        key: estimate_actuator_delay_s(
            allocated_torque[:, index], actual_torque[:, index], dt
        )
        for index, key in enumerate(AXIS_KEYS)
    }
    fractions: dict[str, float | None] = {}
    for index, key in enumerate(AXIS_KEYS):
        crossing = crossings[key]
        if crossing is None:
            fractions[key] = None
            continue
        after = sample_time >= crossing
        fractions[key] = (
            float(np.mean(actual_power[after, index] > POWER_EPSILON_W))
            if np.any(after)
            else None
        )
    action_clipped = np.asarray(
        [row["action_clipped"] for row in samples], dtype=bool
    )
    motor_clipped = np.asarray(
        [row["motor_thrust_clipped"] for row in samples], dtype=bool
    )
    initial_kinetic = float(initial_condition["rotational_kinetic_energy_j"])
    per_axis_classification = {
        key: _classify_axis_after_crossing(
            samples,
            crossings[key],
            index,
            initial_kinetic_energy_j=initial_kinetic,
        )
        for index, key in enumerate(AXIS_KEYS)
    }
    reasons = [str(value) for value in termination_reasons]
    termination_reason = reasons[0] if reasons else ("time_limit" if truncated else None)
    return {
        "schema_version": 1,
        "diagnostic": "generic-3-axis-rotational-response",
        "initial_condition": dict(initial_condition),
        "success": bool(recovery["success"]),
        "termination_reason": termination_reason,
        "termination_reasons": reasons,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "requested_duration_s": float(requested_duration_s),
        "recorded_duration_s": float(sample_time[-1]),
        "physics_sample_count": int(len(samples)),
        "policy_step_count": int(samples[-1]["policy_step"]),
        "first_zero_crossing_s": crossings,
        "maximum_abs_angle_deg": {
            key: float(np.max(np.abs(np.degrees(rpy_with_initial[:, index]))))
            for index, key in enumerate(AXIS_KEYS)
        },
        "maximum_abs_omega_rad_s": {
            key: float(np.max(np.abs(omega_with_initial[:, index])))
            for index, key in enumerate(AXIS_KEYS)
        },
        "maximum_command_actual_torque_error_nm": {
            key: float(np.max(np.abs(torque_error[:, index])))
            for index, key in enumerate(AXIS_KEYS)
        },
        "estimated_actuator_delay_s": delays,
        "actuator_delay_estimator": {
            "method": "causal normalized cross-correlation",
            "input": "allocated_requested_torque_xyz_nm",
            "response": "actual_motor_torque_xyz_nm",
            "maximum_delay_s": 0.20,
        },
        "positive_energy_j": {
            key: float(positive[index]) for index, key in enumerate(AXIS_KEYS)
        },
        "dissipated_energy_j": {
            key: float(dissipated[index]) for index, key in enumerate(AXIS_KEYS)
        },
        "total_positive_rotational_energy_j": total_positive,
        "total_dissipated_rotational_energy_j": total_dissipated,
        "fraction_positive_power_after_zero_crossing": fractions,
        "post_zero_crossing_classification": {
            "power_epsilon_w": POWER_EPSILON_W,
            "torque_difference_epsilon_nm": TORQUE_EPSILON_NM,
            "small_damping_fraction_of_peak_commanded_damping": (
                SMALL_DAMPING_FRACTION
            ),
            "roll_after_first_zero_crossing": per_axis_classification["x"],
            "per_axis": per_axis_classification,
        },
        "events": {
            "action_clipping_intervals_s": boolean_event_intervals(
                sample_time, action_clipped, dt
            ),
            "motor_thrust_clipping_intervals_s": boolean_event_intervals(
                sample_time, motor_clipped, dt
            ),
            "termination_s": float(sample_time[-1]) if terminated or truncated else None,
            "termination_reason": termination_reason,
        },
        "recovery_criterion": dict(recovery),
        "contract": dict(contract),
        "policy_provenance": compatibility.as_dict(),
        "physics_model_version": PHYSICS_MODEL_VERSION,
    }


def run_rotational_rollout(
    environment: Any,
    policy: Any,
    *,
    attitude_axis: str,
    attitude_perturbation_deg: float,
    duration_s: float,
    seed: int,
    contract: Mapping[str, Any],
    compatibility: E2EPolicyCompatibility,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one exact recovery case through the unmodified E2E policy path."""

    axis = attitude_axis_vector(attitude_axis)
    case = RecoveryCase(
        category="attitude",
        name=f"{attitude_axis}_{attitude_perturbation_deg:g}deg",
        tilt_deg=float(attitude_perturbation_deg),
        tilt_axis_xyz=axis,
        position_offset_xyz_m=(0.0, 0.0, 0.0),
    )
    environment.dist_torque_body = np.zeros(3)
    observation = set_recovery_case(environment, case, seed=int(seed))
    position, quaternion, velocity, omega = environment._read_state()
    inertia = inertia_matrix_body(environment)
    initial_condition = {
        "reset_info": getattr(environment, "_last_reset_info", {}),
        "rotation_reference": "drone body CoM, full inertia in body axes; child rotor energy excluded",
        "seed": int(seed),
        "attitude_axis": attitude_axis,
        "attitude_axis_xyz": [float(value) for value in axis],
        "attitude_perturbation_deg": float(attitude_perturbation_deg),
        "position_xyz_m": np.asarray(position, dtype=float).tolist(),
        "quaternion_wxyz": np.asarray(quaternion, dtype=float).tolist(),
        "roll_pitch_yaw_rad": euler_rpy_rad(quaternion).tolist(),
        "body_omega_xyz_rad_s": np.asarray(omega, dtype=float).tolist(),
        "velocity_world_xyz_m_s": np.asarray(velocity, dtype=float).tolist(),
        "requested_motor_thrust_n": np.asarray(
            environment._last_f_cmd, dtype=float
        ).tolist(),
        "actual_motor_thrust_n": np.asarray(environment._last_f, dtype=float).tolist(),
        "inertia_matrix_body": inertia.tolist(),
        "rotational_kinetic_energy_j": float(0.5 * omega @ inertia @ omega),
    }
    recorder = PhysicsSubstepRecorder(environment)
    previous_observer = environment.set_physics_substep_observer(recorder)
    within_tolerance: list[bool] = []
    maximum_steps = max(1, int(round(float(duration_s) * environment.policy_hz)))
    terminated = truncated = False
    termination_reasons: list[str] = []
    try:
        for policy_step_index in range(maximum_steps):
            action = np.asarray(
                policy.predict(observation, deterministic=True)[0], dtype=float
            ).reshape(4)
            recorder.begin_policy_step(action, policy_step_index=policy_step_index)
            observation, _reward, terminated, truncated, info = environment.step(action)
            termination_reasons = [
                str(value) for value in info.get("termination_reasons", [])
            ]
            recorder.finish_policy_step(
                terminated=bool(terminated),
                truncated=bool(truncated),
                termination_reasons=termination_reasons,
            )
            within_tolerance.append(_recovery_tolerance_sample(environment))
            if terminated or truncated:
                break
    finally:
        environment.set_physics_substep_observer(previous_observer)
    duration_elapsed = not terminated and not truncated
    recovery = recovery_success_diagnostics(
        within_tolerance,
        policy_hz=float(environment.policy_hz),
        terminated=bool(terminated),
        truncated=bool(truncated or duration_elapsed),
    )
    summary = summarize_rotational_rollout(
        recorder.samples,
        initial_condition=initial_condition,
        recovery=recovery,
        termination_reasons=termination_reasons,
        terminated=bool(terminated),
        truncated=bool(truncated),
        requested_duration_s=float(duration_s),
        contract=contract,
        compatibility=compatibility,
    )
    summary["rollout_end"] = (
        "terminated"
        if terminated
        else "environment_time_limit"
        if truncated
        else "diagnostic_duration_elapsed"
    )
    return recorder.samples, summary


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_trace_csv(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    if not samples:
        raise ValueError("cannot write an empty physics trace")
    vector_fields = (
        "physics_wrenches",
        "external_applied_torque_xyz_nm",
        "nominal_allocator_actual_torque_xyz_nm",
        "quaternion_wxyz",
        "roll_pitch_yaw_rad",
        "body_omega_xyz_rad_s",
        "body_angular_acceleration_xyz_rad_s2",
        "normalized_action_tau_xyz",
        "raw_policy_action",
        "action_clipped_mask",
        "raw_commanded_torque_xyz_nm",
        "allocated_requested_torque_xyz_nm",
        "actual_motor_torque_xyz_nm",
        "command_actual_torque_error_xyz_nm",
        "motor_thrust_clipped_mask",
        "commanded_axis_power_xyz_w",
        "actual_axis_power_xyz_w",
        "inertia_matrix_body",
        "gyroscopic_torque_xyz_nm",
        "effective_torque_xyz_nm",
        "termination_reasons",
    )
    fieldnames = [
        "rotation_reference",
        "child_dof_note",
        "physics_wrenches",
        "external_applied_torque_xyz_nm",
        "nominal_allocator_actual_torque_xyz_nm",
        "physics_step",
        "policy_step",
        "physics_substep",
        "time_s",
        "quaternion_wxyz",
        "roll_pitch_yaw_rad",
        "body_omega_xyz_rad_s",
        "body_angular_acceleration_xyz_rad_s2",
        "normalized_action_tau_xyz",
        "normalized_action_fz",
        "raw_policy_action",
        "action_clipped",
        "action_clipped_mask",
        "raw_commanded_torque_xyz_nm",
        "allocated_requested_torque_xyz_nm",
        "actual_motor_torque_xyz_nm",
        "command_actual_torque_error_xyz_nm",
        *[f"requested_motor_thrust_{index}_n" for index in range(4)],
        *[f"actual_motor_thrust_{index}_n" for index in range(4)],
        "motor_thrust_clipped",
        "motor_thrust_clipped_mask",
        "commanded_axis_power_xyz_w",
        "actual_axis_power_xyz_w",
        "commanded_total_rotational_power_w",
        "actual_total_rotational_power_w",
        "inertia_matrix_body",
        "gyroscopic_torque_xyz_nm",
        "effective_torque_xyz_nm",
        "rotational_kinetic_energy_j",
        "terminated",
        "truncated",
        "termination_reasons",
    ]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = {key: sample[key] for key in fieldnames if key in sample}
            requested = np.asarray(sample["requested_motor_thrust_n"], dtype=float)
            actual = np.asarray(sample["actual_motor_thrust_n"], dtype=float)
            for index in range(4):
                row[f"requested_motor_thrust_{index}_n"] = float(requested[index])
                row[f"actual_motor_thrust_{index}_n"] = float(actual[index])
            for field in vector_fields:
                row[field] = json.dumps(
                    _json_value(sample.get(field)), separators=(",", ":"), allow_nan=False
                )
            writer.writerow(row)


def _mark_events(axis: Any, summary: Mapping[str, Any]) -> None:
    colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
    for key, crossing in summary["first_zero_crossing_s"].items():
        if crossing is not None:
            axis.axvline(
                crossing,
                color=colors[key],
                linestyle=":",
                linewidth=0.9,
                alpha=0.8,
                label=f"{key} zero crossing",
            )
    for index, interval in enumerate(summary["events"]["action_clipping_intervals_s"]):
        axis.axvspan(
            interval["start_s"],
            interval["end_s"],
            color="tab:purple",
            alpha=0.12,
            label="action clipping" if index == 0 else None,
        )
        axis.axvline(interval["start_s"], color="tab:purple", lw=0.7, ls="--")
        axis.axvline(interval["end_s"], color="tab:purple", lw=0.7, ls="-.")
    for index, interval in enumerate(
        summary["events"]["motor_thrust_clipping_intervals_s"]
    ):
        axis.axvspan(
            interval["start_s"],
            interval["end_s"],
            color="tab:orange",
            alpha=0.12,
            label="motor clipping" if index == 0 else None,
        )
        axis.axvline(interval["start_s"], color="tab:orange", lw=0.7, ls="--")
        axis.axvline(interval["end_s"], color="tab:orange", lw=0.7, ls="-.")
    termination_s = summary["events"]["termination_s"]
    if termination_s is not None:
        reason = summary["events"]["termination_reason"] or "termination"
        axis.axvline(
            termination_s,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"termination: {reason}",
        )


def _save_plots(
    output_directory: Path,
    samples: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.asarray([row["time_s"] for row in samples], dtype=float)
    rpy_deg = np.degrees(
        np.asarray([row["roll_pitch_yaw_rad"] for row in samples], dtype=float)
    )
    omega = np.asarray([row["body_omega_xyz_rad_s"] for row in samples], dtype=float)
    raw = np.asarray(
        [row["raw_commanded_torque_xyz_nm"] for row in samples], dtype=float
    )
    allocated = np.asarray(
        [row["allocated_requested_torque_xyz_nm"] for row in samples], dtype=float
    )
    actual = np.asarray(
        [row["actual_motor_torque_xyz_nm"] for row in samples], dtype=float
    )
    commanded_power = np.asarray(
        [row["commanded_axis_power_xyz_w"] for row in samples], dtype=float
    )
    actual_power = np.asarray(
        [row["actual_axis_power_xyz_w"] for row in samples], dtype=float
    )
    commanded_total = np.asarray(
        [row["commanded_total_rotational_power_w"] for row in samples], dtype=float
    )
    actual_total = np.asarray(
        [row["actual_total_rotational_power_w"] for row in samples], dtype=float
    )
    kinetic = np.asarray(
        [row["rotational_kinetic_energy_j"] for row in samples], dtype=float
    )
    requested_motor = np.asarray(
        [row["requested_motor_thrust_n"] for row in samples], dtype=float
    )
    actual_motor = np.asarray(
        [row["actual_motor_thrust_n"] for row in samples], dtype=float
    )
    title = (
        f"{summary['initial_condition']['attitude_axis']} "
        f"{summary['initial_condition']['attitude_perturbation_deg']:g} deg"
    )
    paths: list[Path] = []

    def finish(figure: Any, axis: Any, filename: str, ylabel: str) -> None:
        _mark_events(axis, summary)
        axis.set_xlabel("time [s]")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=7, ncol=2)
        figure.suptitle(title)
        figure.tight_layout()
        path = output_directory / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for index, label in enumerate(ANGLE_KEYS):
        ax.plot(time, rpy_deg[:, index], label=label)
    ax.axhline(0.0, color="gray", lw=0.7)
    finish(fig, ax, "01_roll_pitch_yaw.png", "angle [deg]")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for index, label in enumerate(("omega_x", "omega_y", "omega_z")):
        ax.plot(time, omega[:, index], label=label)
    ax.axhline(0.0, color="gray", lw=0.7)
    finish(fig, ax, "02_body_omega_xyz.png", "angular velocity [rad/s]")

    for index, key in enumerate(AXIS_KEYS, start=3):
        component = index - 3
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(time, raw[:, component], label="raw commanded")
        ax.plot(time, allocated[:, component], label="allocated requested")
        ax.plot(time, actual[:, component], label="actual motor")
        ax.axhline(0.0, color="gray", lw=0.7)
        finish(fig, ax, f"0{index}_torque_{key}.png", "torque [N m]")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for index, key in enumerate(AXIS_KEYS):
        ax.plot(time, actual_power[:, index], label=f"actual P_{key}")
        ax.plot(
            time,
            commanded_power[:, index],
            linestyle="--",
            alpha=0.65,
            label=f"commanded P_{key}",
        )
    ax.axhline(0.0, color="gray", lw=0.7)
    finish(fig, ax, "06_axis_rotational_power.png", "power [W]")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(time, commanded_total, label="commanded total power")
    ax.plot(time, actual_total, label="actual total power")
    ax.axhline(0.0, color="gray", lw=0.7)
    second = ax.twinx()
    second.plot(time, kinetic, color="black", alpha=0.65, label="rotational KE")
    second.set_ylabel("rotational kinetic energy [J]")
    _mark_events(ax, summary)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = second.get_legend_handles_labels()
    ax.set_xlabel("time [s]")
    ax.set_ylabel("power [W]")
    ax.grid(True, alpha=0.25)
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    path = output_directory / "07_total_power_and_rotational_ke.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for motor in range(4):
        ax.plot(time, requested_motor[:, motor], ls="--", label=f"motor {motor} requested")
        ax.plot(time, actual_motor[:, motor], label=f"motor {motor} actual")
    finish(fig, ax, "08_requested_actual_motor_thrust.png", "thrust [N]")
    return paths


def _create_output_directory(
    config: ExperimentConfig,
    *,
    requested: Path | None,
    attitude_axis: str,
    attitude_perturbation_deg: float,
    seed: int,
) -> Path:
    if requested is not None:
        path = requested.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=False)
        return path
    zone = ZoneInfo(config.experiment.timezone)
    timestamp = datetime.now(zone).strftime("%Y%m%d-%H%M%S-%f")
    degree = format(float(attitude_perturbation_deg), ".12g").replace(".", "p")
    root = config.paths.artifact_root / "diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (
        f"rotational_response_{attitude_axis}_{degree}deg_seed{int(seed)}_{timestamp}"
    )
    path.mkdir(exist_ok=False)
    return path


def write_rotational_artifacts(
    output_directory: Path,
    samples: Sequence[Mapping[str, Any]],
    summary: dict[str, Any],
    *,
    plots: bool,
) -> tuple[Path, Path, list[Path]]:
    csv_path = output_directory / "rotational_response_trace.csv"
    _write_trace_csv(csv_path, samples)
    plot_paths = _save_plots(output_directory, samples, summary) if plots else []
    summary["artifacts"] = {
        "trace_csv": str(csv_path),
        "plots": [str(path) for path in plot_paths],
    }
    json_path = output_directory / "rotational_response_summary.json"
    summary["artifacts"]["summary_json"] = str(json_path)
    json_path.write_text(
        json.dumps(_json_value(summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return json_path, csv_path, plot_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose a deterministic E2E PPO rotational response at MuJoCo's "
            "physics-substep rate"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--attitude-axis", choices=ATTITUDE_AXIS_CHOICES, required=True
    )
    parser.add_argument("--attitude-perturbation-deg", type=float, required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _validate_cli_values(args: argparse.Namespace) -> None:
    angle = float(args.attitude_perturbation_deg)
    duration = float(args.duration)
    if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
        raise ValueError("--attitude-perturbation-deg must be finite and in [0, 180]")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("--duration must be positive and finite")


def _print_classification(summary: Mapping[str, Any]) -> None:
    initial = summary["initial_condition"]
    print(
        "Rotational response: "
        f"axis={initial['attitude_axis']} "
        f"angle={initial['attitude_perturbation_deg']:g} deg "
        f"success={summary['success']} "
        f"end={summary['rollout_end']}"
    )
    print("First zero crossings [s]:", summary["first_zero_crossing_s"])
    per_axis = summary["post_zero_crossing_classification"]["per_axis"]
    for axis_key in AXIS_KEYS:
        analysis = per_axis[axis_key]
        if not analysis["available"]:
            print(
                f"Axis {axis_key} post-zero-crossing classification: "
                "unavailable (no crossing)"
            )
            continue
        print(
            f"Axis {axis_key} post-zero-crossing classification: "
            f"{analysis['analysis_start_s']:.6f}.."
            f"{analysis['analysis_end_s']:.6f} s"
        )
        for category in analysis["categories"]:
            state = "DETECTED" if category["detected"] else "not detected"
            first = category["first_time_s"]
            first_text = "n/a" if first is None else f"{first:.6f} s"
            print(
                f"  {category['id']}. {category['name']}: {state} | "
                f"samples={category['sample_count']} | first={first_text}"
            )


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_cli_values(args)
        config = load_config(args.config)
        model_path = args.model.expanduser().resolve()
        compatibility = validate_e2e_policy_compatibility(model_path, config)
        policy = _load_policy(model_path)
        environment = EnvironmentFactory(config).make(
            seed=args.seed,
            initial_state_randomization_enabled=False,
        )
        try:
            contract = validate_rotational_diagnostic_contract(
                config, environment, policy, compatibility
            )
            samples, summary = run_rotational_rollout(
                environment,
                policy,
                attitude_axis=args.attitude_axis,
                attitude_perturbation_deg=args.attitude_perturbation_deg,
                duration_s=args.duration,
                seed=args.seed,
                contract=contract,
                compatibility=compatibility,
            )
        finally:
            environment.close()
        output_directory = _create_output_directory(
            config,
            requested=args.output_dir,
            attitude_axis=args.attitude_axis,
            attitude_perturbation_deg=args.attitude_perturbation_deg,
            seed=args.seed,
        )
        json_path, csv_path, plot_paths = write_rotational_artifacts(
            output_directory, samples, summary, plots=not args.no_plots
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for warning in compatibility.compatibility_warnings:
        print(f"warning: {warning}", file=sys.stderr)
    _print_classification(summary)
    print(f"Physics samples: {summary['physics_sample_count']}")
    print(f"Trace CSV      : {csv_path}")
    print(f"Summary JSON   : {json_path}")
    print(f"Plots          : {len(plot_paths)}")
    return 0


__all__ = [
    "ALLOWED_CHECKPOINT_KINDS",
    "PhysicsSubstepRecorder",
    "boolean_event_intervals",
    "build_parser",
    "estimate_actuator_delay_s",
    "euler_rpy_rad",
    "first_zero_crossing_s",
    "gyroscopic_torque_xyz_nm",
    "inertia_matrix_body",
    "integrate_axis_energy",
    "integrate_total_energy",
    "reconstruct_three_axis_torques",
    "rotational_power",
    "run_cli",
    "run_rotational_rollout",
    "summarize_rotational_rollout",
    "validate_rotational_diagnostic_contract",
    "write_rotational_artifacts",
]

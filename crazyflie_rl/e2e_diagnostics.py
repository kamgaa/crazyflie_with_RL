"""Deterministic local probes for an E2E Crazyflie PPO policy.

This module observes the existing environment and actuator path.  It does not
change the policy input, action scaling, reward, termination thresholds, or
plant dynamics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactManager
from .config import ExperimentConfig, load_config
from .controllers import rotmat_from_quat_wxyz
from .factories import EnvironmentFactory
from .warm_start import (
    E2EPolicyCompatibility,
    validate_e2e_policy_compatibility,
)


EXPECTED_CONTROL_MODE = "e2e"


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite shape ({size},) vector")
    return result.copy()


def yaw_pitch_quaternion_wxyz(yaw_rad: float, pitch_rad: float) -> np.ndarray:
    """Return ``Rz(yaw) @ Ry(pitch)`` as a normalized wxyz quaternion."""

    yaw_half = 0.5 * float(yaw_rad)
    pitch_half = 0.5 * float(pitch_rad)
    yaw = np.array([np.cos(yaw_half), 0.0, 0.0, np.sin(yaw_half)])
    pitch = np.array([np.cos(pitch_half), 0.0, np.sin(pitch_half), 0.0])
    w1, x1, y1, z1 = yaw
    w2, x2, y2, z2 = pitch
    result = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    return result / np.linalg.norm(result)


def normalized_action_to_physical_wrench(
    environment: Any, action: Sequence[float]
) -> np.ndarray:
    """Apply the environment's existing E2E scale and hover-force offset."""

    if str(environment.mode) != EXPECTED_CONTROL_MODE:
        raise ValueError("E2E diagnostics require an environment in e2e mode")
    normalized = np.clip(_vector(action, 4, "action"), -1.0, 1.0)
    return environment.residual_scale * normalized + np.array(
        [0.0, 0.0, 0.0, environment.mass * environment.gravity], dtype=float
    )


def requested_motor_thrusts(environment: Any, wrench: Sequence[float]) -> np.ndarray:
    """Use the live allocator and its configured per-motor thrust clipping."""

    command = _vector(wrench, 4, "wrench")
    return np.clip(
        environment.B_pinv @ command,
        environment.thrust_min,
        environment.thrust_max,
    )


def actual_wrench_from_environment(environment: Any) -> np.ndarray:
    """Legacy nominal-allocator wrench, not a combined-CoM plant wrench."""

    actual = _vector(environment._last_f, 4, "actual_motor_thrust")
    wrench = environment.B @ actual
    # The actuator may use a yaw reaction-torque model that differs from the
    # allocator's constant ratio, so preserve the plant's measured yaw torque.
    wrench[2] = float(np.sum(_vector(environment._last_q_actual, 4, "q_actual")))
    return wrench


def set_exact_hover_state(
    environment: Any,
    *,
    seed: int,
    pitch_rad: float = 0.0,
    body_angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Reset and then construct one exact hover-consistent simulator state."""

    import mujoco

    environment.reset(seed=seed)
    quaternion = yaw_pitch_quaternion_wxyz(environment.yaw_des, pitch_rad)
    environment.data.qpos[0:3] = environment.pos_des
    environment.data.qpos[3:7] = quaternion
    environment.data.qvel[:] = 0.0
    omega_body = _vector(body_angular_velocity, 3, "body_angular_velocity")
    # MuJoCo freejoint rotational qvel is already expressed in the child-body
    # frame.  The principal-inertia frame in body_iquat is a separate concept.
    dof_address = int(getattr(environment, "_freejoint_dof_address", 0))
    environment.data.qvel[dof_address + 3 : dof_address + 6] = omega_body
    mujoco.mj_forward(environment.model, environment.data)
    measured = environment._read_state()[3]
    if not np.allclose(measured, omega_body, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            "failed to establish requested body angular velocity: "
            f"requested={omega_body.tolist()} measured={measured.tolist()}"
        )
    environment.reset_actuator_state(airborne=True, resample_parameters=False)
    environment.pid.reset()
    environment._step = 0
    environment._prev_action = np.zeros(4)
    environment._last_lyapunov_terms = None
    reset_auxiliary = getattr(environment, "reset_auxiliary_observation_state", None)
    if callable(reset_auxiliary):
        return np.asarray(reset_auxiliary())
    return environment._obs(*environment._read_state())


def _euler_rpy_from_quaternion(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    rotation = rotmat_from_quat_wxyz(quaternion_wxyz)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


@dataclass(frozen=True)
class ProbeResult:
    case: str
    position: list[float]
    quaternion_wxyz: list[float]
    velocity_world: list[float]
    body_angular_velocity: list[float]
    position_error: list[float]
    velocity_error: list[float]
    attitude_error: list[float]
    angular_rate_error: list[float]
    observation: list[float]
    normalized_action: list[float]
    physical_commanded_wrench: list[float]
    requested_motor_thrust: list[float]
    actual_motor_thrust: list[float]
    actual_applied_wrench: list[float]
    wrench_reference: str = "nominal allocator; physical COM wrenches are separate"
    physics_wrenches: dict[str, Any] | None = None


def probe_policy_case(
    environment: Any,
    policy: Any,
    *,
    case: str,
    seed: int,
    pitch_rad: float = 0.0,
    omega_y_rad_s: float = 0.0,
) -> ProbeResult:
    """Evaluate a deterministic policy and one actuator update at a fixed state."""

    observation = set_exact_hover_state(
        environment,
        seed=seed,
        pitch_rad=pitch_rad,
        body_angular_velocity=(0.0, omega_y_rad_s, 0.0),
    )
    state = environment._read_state()
    error = environment._tracking_error(*state)
    action = np.asarray(
        policy.predict(observation, deterministic=True)[0], dtype=float
    ).reshape(4)
    wrench = normalized_action_to_physical_wrench(environment, action)
    requested = requested_motor_thrusts(environment, wrench)
    environment._apply_control(wrench)
    np.testing.assert_allclose(environment._last_f_cmd, requested, rtol=0.0, atol=1e-12)
    actual = _vector(environment._last_f, 4, "actual_motor_thrust")
    actual_wrench = actual_wrench_from_environment(environment)
    physics = None
    if callable(getattr(environment, "physics_wrench_snapshot", None)):
        import mujoco

        # Refresh actuator_force after the diagnostic's one motor update;
        # this performs no integration and no additional motor update.
        mujoco.mj_forward(environment.model, environment.data)
        physics = environment.physics_wrench_snapshot()
    return ProbeResult(
        case=case,
        position=np.asarray(state[0], dtype=float).tolist(),
        quaternion_wxyz=np.asarray(state[1], dtype=float).tolist(),
        velocity_world=np.asarray(state[2], dtype=float).tolist(),
        body_angular_velocity=np.asarray(state[3], dtype=float).tolist(),
        position_error=error.position.tolist(),
        velocity_error=error.velocity.tolist(),
        attitude_error=error.attitude.tolist(),
        angular_rate_error=error.angular_rate.tolist(),
        observation=np.asarray(observation, dtype=float).tolist(),
        normalized_action=action.tolist(),
        physical_commanded_wrench=wrench.tolist(),
        requested_motor_thrust=requested.tolist(),
        actual_motor_thrust=actual.tolist(),
        actual_applied_wrench=actual_wrench.tolist(),
        physics_wrenches=physics,
    )


def probe_policy(environment: Any, policy: Any, *, seed: int) -> dict[str, Any]:
    """Run nominal and symmetric pitch/rate probes and calculate sensitivities."""

    cases = [
        probe_policy_case(environment, policy, case="nominal", seed=seed),
        probe_policy_case(
            environment,
            policy,
            case="pitch_plus_5_deg",
            seed=seed,
            pitch_rad=np.deg2rad(5.0),
        ),
        probe_policy_case(
            environment,
            policy,
            case="pitch_minus_5_deg",
            seed=seed,
            pitch_rad=np.deg2rad(-5.0),
        ),
        probe_policy_case(
            environment,
            policy,
            case="omega_y_plus_1_rad_s",
            seed=seed,
            omega_y_rad_s=1.0,
        ),
        probe_policy_case(
            environment,
            policy,
            case="omega_y_minus_1_rad_s",
            seed=seed,
            omega_y_rad_s=-1.0,
        ),
    ]
    by_name = {item.case: item for item in cases}
    pitch_plus = by_name["pitch_plus_5_deg"]
    pitch_minus = by_name["pitch_minus_5_deg"]
    rate_plus = by_name["omega_y_plus_1_rad_s"]
    rate_minus = by_name["omega_y_minus_1_rad_s"]
    attitude_denominator = pitch_plus.attitude_error[1] - pitch_minus.attitude_error[1]
    rate_denominator = (
        rate_plus.body_angular_velocity[1] - rate_minus.body_angular_velocity[1]
    )
    sensitivities = {
        "du_tau_y_de_R_y": (
            pitch_plus.normalized_action[1] - pitch_minus.normalized_action[1]
        )
        / attitude_denominator,
        "du_tau_y_domega_y": (
            rate_plus.normalized_action[1] - rate_minus.normalized_action[1]
        )
        / rate_denominator,
    }
    sensitivities["attitude_feedback_local_sign"] = (
        "restoring" if sensitivities["du_tau_y_de_R_y"] < 0.0 else "non-restoring"
    )
    sensitivities["rate_feedback_local_sign"] = (
        "damping" if sensitivities["du_tau_y_domega_y"] < 0.0 else "anti-damping"
    )
    return {
        "cases": [asdict(item) for item in cases],
        "sensitivities": sensitivities,
    }


def _provenance_config(
    model_path: Path, requested_config: ExperimentConfig
) -> E2EPolicyCompatibility:
    """Audit inference compatibility without treating training choices as ABI."""

    return validate_e2e_policy_compatibility(model_path, requested_config)


def _load_policy(model_path: Path) -> Any:
    """Load one PPO archive and turn archive errors into a clear hard failure."""

    from stable_baselines3 import PPO

    try:
        from .eval_cli import _model_provenance

        policy = PPO.load(str(model_path), device="cpu")
        policy.physics_model_version = _model_provenance(model_path).get(
            "physics_model_version"
        )
        return policy
    except Exception as exc:
        raise ValueError(
            f"cannot load E2E policy model file {model_path}: {exc}"
        ) from exc


def _validate_loaded_policy_contract(
    policy: Any, requested_config: ExperimentConfig
) -> None:
    """Confirm tensor shapes from the model archive, not just its manifest."""

    observation_shape = tuple(getattr(policy.observation_space, "shape", ()))
    action_shape = tuple(getattr(policy.action_space, "shape", ()))
    errors: list[str] = []
    if observation_shape != tuple(requested_config.observation_shape):
        errors.append(
            f"loaded observation shape {observation_shape} != "
            f"{tuple(requested_config.observation_shape)}"
        )
    if action_shape != tuple(requested_config.action_shape):
        errors.append(
            f"loaded action shape {action_shape} != {tuple(requested_config.action_shape)}"
        )
    if errors:
        raise ValueError("incompatible loaded E2E policy: " + "; ".join(errors))
    from .warm_start import validate_loaded_observation_schema

    validate_loaded_observation_schema(policy, requested_config)


def _rollout_row(
    environment: Any,
    *,
    step: int,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    position, quaternion, velocity, omega = environment._read_state()
    errors = environment._tracking_error(position, quaternion, velocity, omega)
    terms = dict(info.get("reward_terms", {}))
    control = dict(info.get("control_diagnostics", {}))
    commanded_wrench = control.get(
        "physical_commanded_wrench",
        np.asarray(environment._last_wrench_cmd, dtype=float).tolist(),
    )
    requested_thrust = control.get(
        "requested_motor_thrust",
        np.asarray(environment._last_f_cmd, dtype=float).tolist(),
    )
    actual_thrust = control.get(
        "actual_motor_thrust",
        np.asarray(environment._last_f, dtype=float).tolist(),
    )
    actual_wrench = control.get(
        "actual_applied_wrench",
        np.asarray(environment._last_wrench_actual, dtype=float).tolist(),
    )
    return {
        "step": step,
        "time_s": step / float(environment.policy_hz),
        "normalized_action": action.tolist(),
        "physical_commanded_wrench": commanded_wrench,
        "requested_motor_thrust": requested_thrust,
        "actual_motor_thrust": actual_thrust,
        "actual_applied_wrench": actual_wrench,
        "wrench_reference": "nominal allocator; physical COM wrenches are separate",
        "physics_wrenches": (
            environment.physics_wrench_snapshot()
            if callable(getattr(environment, "physics_wrench_snapshot", None))
            else None
        ),
        "attitude_error": errors.attitude.tolist(),
        "angular_rate_error": errors.angular_rate.tolist(),
        "position_error": errors.position.tolist(),
        "velocity_error": errors.velocity.tolist(),
        "rpy_deg": np.degrees(_euler_rpy_from_quaternion(quaternion)).tolist(),
        "body_angular_velocity": omega.tolist(),
        "v_before": terms.get("v_before"),
        "v_after": terms.get("v_after"),
        "state_reward": terms.get("state_reward"),
        "potential_reward": terms.get("potential_reward"),
        "decay_violation": terms.get("decay_violation"),
        "decay_reward": terms.get("decay_reward"),
        "e2e_torque_xy_cost": terms.get("e2e_torque_xy_cost"),
        "e2e_torque_yaw_cost": terms.get("e2e_torque_yaw_cost"),
        "e2e_torque_cost": terms.get("e2e_torque_cost"),
        "e2e_torque_reward": terms.get("e2e_torque_reward"),
        "nontracking_reward": terms.get("nontracking_reward"),
        "crash_or_ood_reward": terms.get("crash_or_ood_reward"),
        "reward_component_sum": terms.get("reward_component_sum"),
        "total_reward": reward,
        "reward_components_consistent": terms.get("reward_components_consistent"),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reasons": list(info.get("termination_reasons", [])),
        "terminal_altitude": info.get("terminal_altitude"),
        "terminal_tilt_deg": info.get("terminal_tilt_deg"),
        "terminal_position_error_norm": info.get("terminal_position_error_norm"),
    }


def deterministic_rollout(
    environment: Any, policy: Any, *, seed: int, maximum_steps: int
) -> list[dict[str, Any]]:
    observation = set_exact_hover_state(environment, seed=seed)
    rows: list[dict[str, Any]] = []
    for step in range(1, maximum_steps + 1):
        action = np.asarray(
            policy.predict(observation, deterministic=True)[0], dtype=float
        ).reshape(4)
        observation, reward, terminated, truncated, info = environment.step(action)
        rows.append(
            _rollout_row(
                environment,
                step=step,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        if terminated or truncated:
            break
    return rows


def selected_failure_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select reproducible onset/extreme/terminal rows from one short rollout."""

    if not rows:
        return []
    onset = 0
    candidate_values = [(row.get("v_before"), row.get("v_after")) for row in rows]
    if all(
        before is not None and after is not None for before, after in candidate_values
    ):
        delta = np.asarray(
            [float(after) - float(before) for before, after in candidate_values]
        )
        for index in range(max(0, len(rows) - 2)):
            if np.all(delta[index : index + 3] > 0.0):
                onset = index
                break
    pitch_action = np.asarray([abs(float(row["normalized_action"][1])) for row in rows])
    indices = [
        0,
        onset,
        int(np.argmax(pitch_action)),
        max(0, len(rows) - 2),
        len(rows) - 1,
    ]
    labels = [
        "first_step",
        "v_persistent_increase_onset",
        "maximum_abs_u_tau_y",
        "preterminal_step",
        "terminal_or_last_step",
    ]
    result: list[dict[str, Any]] = []
    for label, index in zip(labels, indices):
        selected = dict(rows[index])
        selected["selection"] = label
        result.append(selected)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic CSV: {path}")
    if not rows:
        raise ValueError("cannot write an empty diagnostic rollout")
    fieldnames = list(rows[0].keys())
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def _write_plots(
    manager: ArtifactManager, rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    import matplotlib.pyplot as plt

    time = np.asarray([row["time_s"] for row in rows], dtype=float)
    commanded = np.asarray(
        [row["physical_commanded_wrench"] for row in rows], dtype=float
    )
    actual = np.asarray([row["actual_applied_wrench"] for row in rows], dtype=float)
    requested_motor = np.asarray(
        [row["requested_motor_thrust"] for row in rows], dtype=float
    )
    actual_motor = np.asarray([row["actual_motor_thrust"] for row in rows], dtype=float)
    paths: list[str] = []

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for axis, name in enumerate(("tau_x", "tau_y", "tau_z")):
        axes[axis].plot(time, commanded[:, axis], label="commanded")
        axes[axis].plot(time, actual[:, axis], label="actual")
        axes[axis].set_ylabel(f"{name} [N m]")
        axes[axis].grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("time [s]")
    torque_path = manager.path("plots", "commanded-vs-actual-torque", ".png")
    fig.tight_layout()
    fig.savefig(torque_path, dpi=150)
    plt.close(fig)
    paths.append(str(torque_path))

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for motor in range(4):
        axes[motor].plot(time, requested_motor[:, motor], label="requested")
        axes[motor].plot(time, actual_motor[:, motor], label="actual")
        axes[motor].set_ylabel(f"motor {motor} [N]")
        axes[motor].grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("time [s]")
    motor_path = manager.path("plots", "requested-vs-actual-motor-thrust", ".png")
    fig.tight_layout()
    fig.savefig(motor_path, dpi=150)
    plt.close(fig)
    paths.append(str(motor_path))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe E2E policy feedback, actuator lag, reward terms, and termination"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"model does not exist: {model_path}")
    requested_config = load_config(args.config)
    try:
        compatibility = _provenance_config(model_path, requested_config)
        policy = _load_policy(model_path)
        _validate_loaded_policy_contract(policy, requested_config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for warning in compatibility.compatibility_warnings:
        print(f"warning: {warning}", file=sys.stderr)

    environment = EnvironmentFactory(requested_config).make(seed=args.seed)
    manager = ArtifactManager.create(
        requested_config,
        command=list(sys.argv if argv is None else ["diagnose_e2e_policy.py", *argv]),
        condition="e2e-policy-diagnostic",
        mission="hover",
        seed=args.seed,
    )
    try:
        probes = probe_policy(environment, policy, seed=args.seed)
        rows = deterministic_rollout(
            environment,
            policy,
            seed=args.seed,
            maximum_steps=min(args.steps, environment.max_steps),
        )
        selected = selected_failure_rows(rows)
        csv_path = manager.path("metrics", "rollout", ".csv")
        _write_csv(csv_path, rows)
        plots = [] if args.no_plots else _write_plots(manager, rows)
        report = {
            **compatibility.as_dict(),
            "reset_info": getattr(environment, "_last_reset_info", {}),
            "model": str(model_path),
            "training_provenance": compatibility.training_provenance,
            "requested_runtime_config": compatibility.requested_runtime_config,
            "compatibility_warnings": list(compatibility.compatibility_warnings),
            "checkpoint_kind": compatibility.checkpoint_kind,
            "saved_timestep": compatibility.saved_timestep,
            # Kept for consumers of the first diagnostic report schema. New
            # consumers should use the separated provenance/runtime fields.
            "model_provenance": compatibility.training_provenance,
            "seed": args.seed,
            "observation_contract": {
                "shape": list(requested_config.observation_shape),
                "actuator_state_included": False,
            },
            "action_scale": list(requested_config.environment.residual_scale),
            "allocation_matrix": environment.B.tolist(),
            "actuator": environment.actuator_snapshot(),
            "probes": probes,
            "rollout_steps": len(rows),
            "selected_failure_rows": selected,
            "termination_reasons": rows[-1]["termination_reasons"],
            "rollout_csv": str(csv_path),
            "plots": plots,
        }
        json_path = manager.write_metrics("e2e-policy-diagnostic", report)
        manager.write_runtime_config(
            {
                "model": str(model_path),
                "seed": args.seed,
                "steps": args.steps,
                "deterministic": True,
                "training_provenance": compatibility.training_provenance,
                "requested_runtime_config": compatibility.requested_runtime_config,
                "compatibility_warnings": list(compatibility.compatibility_warnings),
                "checkpoint_kind": compatibility.checkpoint_kind,
                "saved_timestep": compatibility.saved_timestep,
                "diagnostic_json": str(json_path),
                "rollout_csv": str(csv_path),
            }
        )
        manager.finalize(
            "completed",
            diagnostic_json=str(json_path),
            rollout_csv=str(csv_path),
            rollout_steps=len(rows),
            termination_reasons=rows[-1]["termination_reasons"],
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"diagnostic artifacts: {manager.run_dir}")
    finally:
        environment.close()
    return 0


__all__ = [
    "ProbeResult",
    "actual_wrench_from_environment",
    "deterministic_rollout",
    "normalized_action_to_physical_wrench",
    "probe_policy",
    "probe_policy_case",
    "requested_motor_thrusts",
    "run_cli",
    "selected_failure_rows",
    "set_exact_hover_state",
    "yaw_pitch_quaternion_wxyz",
]

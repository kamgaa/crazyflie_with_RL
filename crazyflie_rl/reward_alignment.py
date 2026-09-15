"""Deterministic PID-versus-E2E reward-alignment diagnostics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactManager
from .config import ExperimentConfig, load_config
from .eval_cli import _model_path, _model_provenance
from .factories import EnvironmentFactory
from .recovery import (
    ANGULAR_RATE_TOLERANCE_RAD_S,
    POSITION_TOLERANCE_M,
    TILT_TOLERANCE_DEG,
    VELOCITY_TOLERANCE_M_S,
    RecoveryCase,
    _state_metrics,
    consecutive_recovery_time,
    set_recovery_case,
)


DISCOUNTED_TERM_KEYS = {
    "discounted_state_reward": "state_reward",
    "discounted_potential_reward": "potential_reward",
    "discounted_decay_reward": "decay_reward",
    "discounted_torque_penalty": "e2e_torque_reward",
    "discounted_crash_penalty": "crash_or_ood_reward",
    "discounted_nontracking_reward": "nontracking_reward",
}


def reward_alignment_cases() -> tuple[RecoveryCase, ...]:
    cases = [RecoveryCase("attitude", "nominal", 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
    directions = (
        ("roll_plus", (1.0, 0.0, 0.0)),
        ("roll_minus", (-1.0, 0.0, 0.0)),
        ("pitch_plus", (0.0, 1.0, 0.0)),
        ("pitch_minus", (0.0, -1.0, 0.0)),
    )
    for tilt_deg in (5.0, 15.0, 30.0):
        for direction, axis in directions:
            cases.append(
                RecoveryCase(
                    "attitude",
                    f"{direction}_{tilt_deg:g}deg",
                    tilt_deg,
                    axis,
                    (0.0, 0.0, 0.0),
                )
            )
    return tuple(cases)


def discounted_reward_totals(
    transitions: Sequence[tuple[float, Mapping[str, Any]]],
    *,
    gamma: float,
) -> dict[str, float]:
    """Accumulate exactly logged per-transition reward components."""

    discount = 1.0
    totals = {output: 0.0 for output in DISCOUNTED_TERM_KEYS}
    total_return = 0.0
    for reward, terms in transitions:
        for output, source in DISCOUNTED_TERM_KEYS.items():
            totals[output] += discount * float(terms[source])
        total_return += discount * float(reward)
        discount *= float(gamma)
    totals["total_discounted_return"] = total_return
    totals["discounted_logged_component_sum"] = sum(
        totals[output] for output in DISCOUNTED_TERM_KEYS
    )
    return totals


def evaluate_reward_alignment_case(
    environment: Any,
    policy: Any | None,
    case: RecoveryCase,
    *,
    controller: str,
    seed: int,
    gamma: float,
    duration_s: float,
) -> dict[str, Any]:
    observation = set_recovery_case(environment, case, seed=seed)
    maximum_steps = int(round(duration_s * environment.policy_hz))
    transitions: list[tuple[float, Mapping[str, Any]]] = []
    within_tolerance: list[bool] = []
    termination_reasons: list[str] = []
    terminated = truncated = False

    for _step in range(maximum_steps):
        action = (
            np.zeros(4, dtype=np.float32)
            if policy is None
            else policy.predict(observation, deterministic=True)[0]
        )
        observation, reward, terminated, truncated, info = environment.step(action)
        terms = info.get("reward_terms")
        if not isinstance(terms, Mapping):
            raise RuntimeError("Lyapunov reward_terms are required for alignment")
        transitions.append((float(reward), terms))
        state = _state_metrics(environment)
        within_tolerance.append(
            state["position_error_norm_m"] < POSITION_TOLERANCE_M
            and state["velocity_norm_m_s"] < VELOCITY_TOLERANCE_M_S
            and state["tilt_deg"] < TILT_TOLERANCE_DEG
            and state["angular_velocity_norm_rad_s"] < ANGULAR_RATE_TOLERANCE_RAD_S
        )
        if terminated or truncated:
            termination_reasons = list(info.get("termination_reasons", []))
            break

    recovery_time = consecutive_recovery_time(
        within_tolerance, policy_hz=environment.policy_hz
    )
    success = recovery_time is not None and not terminated
    totals = discounted_reward_totals(transitions, gamma=gamma)
    return {
        "controller": controller,
        **asdict(case),
        "seed": seed,
        "gamma": gamma,
        "transition_count": len(transitions),
        "success": bool(success),
        "failure": not bool(success),
        "recovery_time_s": recovery_time,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": termination_reasons,
        **totals,
    }


def reward_alignment_warnings(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    by_case: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["name"]), {})[str(row["controller"])] = row
    warnings: list[str] = []
    for case_name, controllers in by_case.items():
        pid = controllers.get("pid_floor")
        ppo = controllers.get("e2e_ppo")
        if pid is None or ppo is None:
            continue
        if (
            bool(pid["success"])
            and not bool(ppo["success"])
            and float(pid["total_discounted_return"])
            < float(ppo["total_discounted_return"])
        ):
            warnings.append(
                "reward-alignment inversion: PID succeeds but PPO fails while "
                f"PID return is lower for {case_name}: "
                f"pid={float(pid['total_discounted_return']):.9g}, "
                f"ppo={float(ppo['total_discounted_return']):.9g}"
            )
    return warnings


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def _validate_config(config: ExperimentConfig) -> None:
    if config.control_mode != "e2e":
        raise ValueError("reward alignment requires an E2E profile")
    if config.environment.reward.mode != "lyapunov":
        raise ValueError("reward alignment requires reward.mode=lyapunov")


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare discounted Lyapunov reward terms for PID and E2E PPO"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "e2e_train_lyapunov_rate01_initial_perturb.yaml",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=8.0)
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not np.isfinite(args.duration) or args.duration <= 0.0:
        raise ValueError("duration must be positive and finite")
    config = load_config(args.config)
    _validate_config(config)
    config.require_runtime_resources()
    model_path = _model_path(args.model, config)

    from stable_baselines3 import PPO

    policy = PPO.load(str(model_path), device=config.training.ppo.device)
    if tuple(policy.observation_space.shape) != config.observation_shape:
        raise ValueError("model observation shape does not match config")
    if tuple(policy.action_space.shape) != config.action_shape:
        raise ValueError("model action shape does not match config")
    from .warm_start import validate_loaded_observation_schema

    validate_loaded_observation_schema(policy, config, model_path=model_path)

    manager = ArtifactManager.create(
        config,
        command=list(
            sys.argv if argv is None else ["evaluate_reward_alignment.py", *argv]
        ),
        condition="reward-alignment-diagnostic",
        mission="hover",
        seed=args.seed,
    )
    rows: list[dict[str, Any]] = []
    try:
        for controller, mode, selected_policy in (
            ("pid_floor", "residual", None),
            ("e2e_ppo", "e2e", policy),
        ):
            environment = EnvironmentFactory(config).make(
                seed=args.seed,
                mode=mode,
                initial_state_randomization_enabled=False,
            )
            try:
                rows.extend(
                    evaluate_reward_alignment_case(
                        environment,
                        selected_policy,
                        case,
                        controller=controller,
                        seed=args.seed,
                        gamma=config.training.ppo.gamma,
                        duration_s=args.duration,
                    )
                    for case in reward_alignment_cases()
                )
            finally:
                environment.close()

        warnings = reward_alignment_warnings(rows)
        for warning in warnings:
            print(f"WARNING: {warning}")
        csv_path = manager.path("metrics", "reward-alignment", ".csv")
        _write_csv(csv_path, rows)
        report = {
            "config_profile": config.profile_name,
            "model": str(model_path),
            "model_provenance": _model_provenance(model_path),
            "seed": args.seed,
            "duration_s": args.duration,
            "gamma": config.training.ppo.gamma,
            "warnings": warnings,
            "trajectories": rows,
            "csv": str(csv_path),
        }
        json_path = manager.write_metrics("reward-alignment", report)
        manager.write_runtime_config(
            {
                "model": str(model_path),
                "seed": args.seed,
                "duration_s": args.duration,
                "training_initial_state_randomization_disabled": True,
            }
        )
        manager.finalize(
            "completed",
            reward_alignment_metrics=str(json_path),
            warning_count=len(warnings),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"reward-alignment artifacts: {manager.run_dir}")
    except BaseException as exc:
        manager.finalize("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    return 0


__all__ = [
    "discounted_reward_totals",
    "evaluate_reward_alignment_case",
    "reward_alignment_cases",
    "reward_alignment_warnings",
]

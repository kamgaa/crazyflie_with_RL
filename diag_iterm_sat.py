"""PID velocity-integrator saturation diagnostic, exposed as a safe CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "residual_hover_eval.yaml"
SATURATION_EPSILON = 0.02


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PID floor and residual PPO I-term saturation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="PPO archive (default: <legacy_model_root>/ppo_best.zip)",
    )
    return parser


def _rollout(config: Any, factory: Any, policy: Any | None, label: str):
    import numpy as np
    from crazyflie_rl.plotting import quaternion_to_euler_deg

    env = factory.make(episode_sec=config.environment.episode_sec + 1.0)
    try:
        observation, _ = env.reset(seed=config.evaluation.seed_start)
        # This assignment is deliberately retained from the legacy script,
        # even though the configured target already has the same value.
        env.pos_des = np.asarray(config.environment.position_target, dtype=float).copy()
        current_mass = float(env.model.body_mass[env.drone_bid])
        print(
            f"[{label}] mass={current_mass:.5f}kg "
            f"(nominal {env._m0:.5f}, delta={current_mass - env._m0:+.5f})"
        )
        dt = float(env.dt_phys * env.substeps)
        sample_count = int(round(config.environment.episode_sec / dt))
        rows: list[list[float]] = []
        for index in range(sample_count):
            action = (
                policy.predict(
                    observation,
                    deterministic=config.evaluation.deterministic,
                )[0]
                if policy is not None
                else np.zeros(4)
            )
            observation, _reward, _terminated, _truncated, _ = env.step(action)
            position = observation[0:3] + env.pos_des
            roll, pitch, _yaw = quaternion_to_euler_deg(observation[6:10])
            i_vel_x, i_vel_y, i_vel_z = env.pid._i_vel
            rows.append(
                [
                    index * dt,
                    float(position[2]),
                    float(roll),
                    float(pitch),
                    float(i_vel_x),
                    float(i_vel_y),
                    float(i_vel_z),
                    *np.asarray(action, dtype=float).tolist(),
                ]
            )
        return np.asarray(rows, dtype=float)
    finally:
        env.close()


def _summary(config: Any, rows, label: str) -> dict[str, Any]:
    import numpy as np

    time_sec = rows[:, 0]
    window_sec = config.environment.episode_sec * config.evaluation.tail_fraction
    steady = time_sec >= (config.environment.episode_sec - window_sec)
    limit = config.controller.pid.integrator_limit
    threshold = limit - SATURATION_EPSILON
    integrator = rows[:, 4:7]
    action = rows[:, 7:11]

    def saturation(values) -> float:
        return 100.0 * float(np.mean(np.abs(values) > threshold))

    result = {
        "label": label,
        "sample_count": int(rows.shape[0]),
        "steady_state_window_sec": float(window_sec),
        "z_mean": float(np.mean(rows[steady, 1])),
        "z_sag": float(
            config.environment.position_target[2] - np.mean(rows[steady, 1])
        ),
        "roll_mean_deg": float(np.mean(rows[steady, 2])),
        "pitch_mean_deg": float(np.mean(rows[steady, 3])),
        "i_vel_mean": np.mean(integrator[steady], axis=0).tolist(),
        "i_vel_saturation_percent": [
            saturation(integrator[:, index]) for index in range(3)
        ],
        "action_mean": np.mean(action[steady], axis=0).tolist(),
    }
    print(f"\n=== [{label}] steady-state summary ===")
    print(
        f"  z={result['z_mean']:.4f}m sag={result['z_sag']:+.4f}m, "
        f"roll/pitch={result['roll_mean_deg']:+.2f}/{result['pitch_mean_deg']:+.2f}deg"
    )
    print(f"  I-term saturation x/y/z={result['i_vel_saturation_percent']}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from crazyflie_rl.artifacts import ArtifactManager
    from crazyflie_rl.config import load_config
    from crazyflie_rl.eval_cli import _load_policy, _model_path
    from crazyflie_rl.factories import EnvironmentFactory
    from crazyflie_rl.plotting import save_iterm_diagnostic

    config = load_config(args.config)
    config.require_runtime_resources()
    model_path = _model_path(args.model, config)
    model = _load_policy(model_path, config)
    factory = EnvironmentFactory(config)
    artifacts = ArtifactManager.create(
        config,
        command=list(sys.argv if argv is None else argv),
        condition=config.experiment.condition,
        mission="iterm-saturation-diagnostic",
        seed=config.evaluation.seed_start,
    )
    try:
        floor = _rollout(config, factory, None, "floor")
        residual = _rollout(config, factory, model, "residual")
        floor_summary = _summary(config, floor, "floor")
        residual_summary = _summary(config, residual, "residual")
        plot_path = save_iterm_diagnostic(
            artifacts.path("plots", "iterm-saturation", ".png"),
            floor=floor,
            policy=residual,
            integrator_limit=config.controller.pid.integrator_limit,
            force_scale=config.environment.residual_scale[3],
            payload_mass=config.environment.payload.mass,
            gravity=config.vehicle.gravity,
            hover_altitude=config.environment.position_target[2],
        )
        metrics_path = artifacts.write_metrics(
            "iterm-saturation",
            {
                "model": str(model_path),
                "seed": config.evaluation.seed_start,
                "floor": floor_summary,
                "residual": residual_summary,
                "plot": plot_path.relative_to(artifacts.run_dir).as_posix(),
            },
        )
        artifacts.finalize("completed", input_model=str(model_path))
        print(f"saved: {plot_path}")
        print(f"metrics: {metrics_path}")
        return 0
    except Exception as exc:
        artifacts.finalize("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

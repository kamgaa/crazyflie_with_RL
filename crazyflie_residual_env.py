"""Backward-compatible entrypoint for the Crazyflie Gym environment.

Runtime implementation lives in :mod:`crazyflie_rl.environment`; importing
this module only exposes symbols and never creates MuJoCo state or starts a
rollout.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from crazyflie_rl.controllers import (
    ARM,
    GRAV,
    J_DIAG,
    K_TAU,
    MASS,
    MOTOR_DIR,
    PHYSICS_HZ,
    TAU_MAX_RP,
    THRUST_MAX,
    THRUST_MIN,
    CascadePID,
    _build_B,
    build_allocation_matrix,
    quat_normalize_wxyz,
    rotmat_from_quat_wxyz,
)
from crazyflie_rl.environment import (
    ACTION_DIM,
    CONTROL_MODES,
    DEFAULT_RESIDUAL_SCALE,
    OBSERVATION_DIM,
    CrazyflieResidualEnv,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a zero-action smoke rollout using an explicit config profile."""

    import argparse

    from crazyflie_rl.config import load_config
    from crazyflie_rl.factories import EnvironmentFactory

    parser = argparse.ArgumentParser(
        description="Run a config-driven Crazyflie zero-action smoke rollout."
    )
    parser.add_argument(
        "--config", required=True, help="YAML experiment profile to resolve"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="optional reset/factory seed override"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    env = EnvironmentFactory(config).make(seed=args.seed)
    try:
        observation, _ = env.reset(seed=args.seed)
        print(
            f"mode={env.mode}, obs={observation.shape}, action={env.action_space.shape}, "
            f"substeps={env.substeps}, max_steps={env.max_steps}"
        )
        errors: list[float] = []
        for step_index in range(env.max_steps):
            observation, _reward, terminated, truncated, _ = env.step(
                np.zeros(ACTION_DIM)
            )
            errors.append(float(np.linalg.norm(observation[0:3])))
            if terminated or truncated:
                status = "terminated" if terminated else "truncated"
                print(f"{status} at policy step {step_index + 1}")
                break
        if errors:
            print(
                "position error: "
                f"start={errors[0]:.6f}, end={errors[-1]:.6f}, min={min(errors):.6f}"
            )
    finally:
        env.close()
    return 0


__all__ = [
    "ACTION_DIM",
    "ARM",
    "CONTROL_MODES",
    "CascadePID",
    "CrazyflieResidualEnv",
    "DEFAULT_RESIDUAL_SCALE",
    "GRAV",
    "J_DIAG",
    "K_TAU",
    "MASS",
    "MOTOR_DIR",
    "OBSERVATION_DIM",
    "PHYSICS_HZ",
    "TAU_MAX_RP",
    "THRUST_MAX",
    "THRUST_MIN",
    "_build_B",
    "build_allocation_matrix",
    "main",
    "quat_normalize_wxyz",
    "rotmat_from_quat_wxyz",
]


if __name__ == "__main__":
    raise SystemExit(main())

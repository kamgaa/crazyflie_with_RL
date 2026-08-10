"""Official E2E PPO training command.

Importing this module is inert.  Use ``--dry-run`` to inspect the fully
resolved effective configuration without importing or creating runtime and
artifact resources.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "e2e_train.yaml"


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the active E2E Crazyflie PPO profile."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML profile (default: configs/e2e_train.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved effective config without runtime creation",
    )
    parser.add_argument(
        "--total-timesteps",
        type=_positive_integer,
        help="override training.total_timesteps for this run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from dataclasses import replace
    import json

    from crazyflie_rl.config import load_config

    config = load_config(args.config)
    if args.total_timesteps is not None:
        config = replace(
            config,
            training=replace(
                config.training,
                total_timesteps=args.total_timesteps,
            ),
        )

    if args.dry_run:
        print(
            json.dumps(
                config.resolved_dict(),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    # Set this before the first SB3/Torch import, matching master execution.
    import os

    if config.training.ppo.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from crazyflie_rl.training import PPOTrainer

    PPOTrainer(config).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

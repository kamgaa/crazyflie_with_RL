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
    parser.add_argument(
        "--init-policy-from",
        type=Path,
        help=(
            "initialize only policy parameters from a manifest-validated "
            "E2E PPO archive (not resume training)"
        ),
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

    if config.training.policy_initialization_required and args.init_policy_from is None:
        if config.environment.payload.curriculum.enabled:
            raise SystemExit(
                "this payload-DR profile requires an explicit "
                "--init-policy-from archive; donor discovery and checkpoint "
                "reselection are intentionally disabled"
            )
        from crazyflie_rl.warm_start import discover_legacy_e2e_donors

        candidates = discover_legacy_e2e_donors(config)
        if candidates:
            candidate_lines = "\n".join(
                f"  {item.model_path} ({item.model_kind}, step "
                f"{item.model_timestep}, manifest {item.manifest_path})"
                for item in candidates
                if item.model_kind == "final"
            )
            if not candidate_lines:
                candidate_lines = "\n".join(
                    f"  {item.model_path} ({item.model_kind}, step "
                    f"{item.model_timestep})"
                    for item in candidates
                )
            raise SystemExit(
                "this recovery fine-tuning profile requires "
                "--init-policy-from. Compatible donor candidates:\n"
                f"{candidate_lines}"
            )
        raise SystemExit(
            "no compatible nominal legacy E2E donor was found. Train "
            "configs/e2e_train.yaml first, then pass its original artifact "
            "model with --init-policy-from"
        )

    # Set this before the first SB3/Torch import, matching master execution.
    import os

    if config.training.ppo.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from crazyflie_rl.training import PPOTrainer

    PPOTrainer(config, init_policy_from=args.init_policy_from).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

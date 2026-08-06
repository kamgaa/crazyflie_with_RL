"""Official PPO training command-line entrypoint.

Importing this module is intentionally inert. Runtime configuration and the
training stack are loaded only from :func:`main`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "e2e_train.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Crazyflie PPO policy from a resolved config profile."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "Training profile to load. Defaults to configs/e2e_train.yaml to "
            "preserve the active train_ppo_02.py behavior."
        ),
    )
    return parser


def _configure_runtime() -> None:
    """Preserve the original CPU-only training behavior at execution time."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _configure_runtime()

    # Keep configuration, MuJoCo, NumPy, Torch and SB3 imports out of module
    # import so importing this entrypoint cannot construct an environment,
    # create a model, start training, or write artifacts.
    from crazyflie_rl.config import load_config
    from crazyflie_rl.training import ExperimentRunner

    config = load_config(args.config)
    ExperimentRunner(config).train()


if __name__ == "__main__":
    main()

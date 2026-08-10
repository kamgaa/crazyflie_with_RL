"""Payload circle evaluation, retained as a thin import-safe CLI wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "residual_circle_eval.yaml"


def main(argv: Sequence[str] | None = None) -> int:
    from crazyflie_rl.eval_cli import run_evaluation_cli

    return run_evaluation_cli(
        default_config=DEFAULT_CONFIG,
        description="Evaluate PID and residual PPO on the payload circle mission.",
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())

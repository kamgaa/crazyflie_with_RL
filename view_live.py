"""Legacy E2E hover viewer, now a config-driven import-safe CLI wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "e2e_hover_eval.yaml"


def main(argv: Sequence[str] | None = None) -> int:
    from crazyflie_rl.eval_cli import run_evaluation_cli

    return run_evaluation_cli(
        default_config=DEFAULT_CONFIG,
        description=(
            "Evaluate the legacy E2E hover policy. Its floor is zero policy "
            "action plus gravity compensation, not cascade PID."
        ),
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())

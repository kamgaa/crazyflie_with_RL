"""Import-safe entry point for deterministic E2E recovery evaluation."""

from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from crazyflie_rl.recovery import run_cli

    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())

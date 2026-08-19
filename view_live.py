"""Unified, import-safe Hover/Circle/Lissajous flight-test wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "e2e_hover_eval.yaml"
MODE_CONFIGS = {
    "hover": DEFAULT_CONFIG.parent / "view_live_hover_eval.yaml",
    "circle": DEFAULT_CONFIG.parent / "view_live_circle_eval.yaml",
    "lissajous": DEFAULT_CONFIG.parent / "view_live_lissajous_eval.yaml",
}

# Geometry is deliberately kept in versioned YAML profiles.  Operators choose
# a recognizable path and only tune its traversal time/repetitions at runtime.
PATH_PROFILES = {
    "circle": {
        "small": DEFAULT_CONFIG.parent / "view_live_circle_eval.yaml",
        "wide": DEFAULT_CONFIG.parent / "view_live_circle_wide_eval.yaml",
    },
    "lissajous": {
        "figure8": DEFAULT_CONFIG.parent / "view_live_lissajous_eval.yaml",
        "clover": DEFAULT_CONFIG.parent / "view_live_lissajous_clover_eval.yaml",
    },
}


def main(argv: Sequence[str] | None = None) -> int:
    from crazyflie_rl.eval_cli import run_evaluation_cli

    return run_evaluation_cli(
        default_config=DEFAULT_CONFIG,
        description=(
            "Run an interactive or scripted Crazyflie Hover, Circle, or "
            "Lissajous flight test. The E2E floor remains zero policy action "
            "plus gravity compensation, not cascade PID."
        ),
        argv=argv,
        unified=True,
        mode_profiles=MODE_CONFIGS,
        path_profiles=PATH_PROFILES,
        always_compare=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Render the preserved hand-recorded learning curve as a run artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "residual_train.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the legacy 10g learning curve.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import numpy as np

    from crazyflie_rl.artifacts import ArtifactManager
    from crazyflie_rl.config import load_config
    from crazyflie_rl.plotting import save_learning_curve

    config = load_config(args.config)
    steps = np.arange(20_000, 300_001, 20_000, dtype=int)
    residual_error = np.asarray(
        [
            0.0278,
            0.0275,
            0.0271,
            0.0267,
            0.0269,
            0.0292,
            0.0310,
            0.0323,
            0.0392,
            0.0562,
            0.0612,
            0.0819,
            0.1046,
            0.1155,
            0.1037,
        ],
        dtype=float,
    )
    floor_error = 0.0277
    artifacts = ArtifactManager.create(
        config,
        command=list(sys.argv if argv is None else argv),
        condition="legacy-10g-learning-curve",
        mission="diagnostic",
    )
    try:
        plot_path = save_learning_curve(
            artifacts.path("plots", "learning-curve", ".png"),
            steps=steps,
            residual_error=residual_error,
            floor_error=floor_error,
        )
        metrics_path = artifacts.write_metrics(
            "learning-curve",
            {
                "steps": steps.tolist(),
                "residual_position_error": residual_error.tolist(),
                "floor_position_error": floor_error,
                "highlight_window": [40_000, 100_000],
                "annotated_best_step": 80_000,
                "plot": plot_path.relative_to(artifacts.run_dir).as_posix(),
            },
        )
        artifacts.finalize("completed")
        print(f"saved: {plot_path}")
        print(f"metrics: {metrics_path}")
        return 0
    except Exception as exc:
        artifacts.finalize("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

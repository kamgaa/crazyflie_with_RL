"""Plot PPO entropy/KL diagnostics without import-time file creation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "residual_train.yaml"
SCALAR_TAGS = (
    "train/entropy_loss",
    "train/approx_kl",
    "train/std",
    "train/clip_fraction",
)
SOURCE_PROVENANCE = "legacy-tensorboard-control-mode-condition-seed-unverified"


def find_event_dir(root: str | Path) -> Path:
    """Preserve the legacy direct/first-child TensorBoard run selection."""

    directory = Path(root)
    if any(directory.glob("events.out.*")):
        return directory
    for child in sorted(directory.glob("*")):
        if child.is_dir() and any(child.glob("events.out.*")):
            return child
    raise FileNotFoundError(f"no TensorBoard event file found in {directory}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot PPO entropy/KL diagnostics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="event directory/root (default: config paths.legacy_tensorboard_root)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from crazyflie_rl.artifacts import ArtifactManager
    from crazyflie_rl.config import load_config
    from crazyflie_rl.plotting import save_entropy_diagnostic

    config = load_config(args.config)
    event_dir = find_event_dir(
        config.paths.legacy_tensorboard_root if args.logdir is None else args.logdir
    )
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", ()))
    series: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for tag in SCALAR_TAGS:
        if tag not in available:
            series[tag] = None
            continue
        events = accumulator.Scalars(tag)
        series[tag] = (
            np.asarray([event.step for event in events], dtype=int),
            np.asarray([event.value for event in events], dtype=float),
        )

    artifacts = ArtifactManager.create(
        config,
        command=list(sys.argv if argv is None else argv),
        condition="entropy-kl-source-unverified",
        mission="diagnostic",
    )
    try:
        plot_path = save_entropy_diagnostic(
            artifacts.path("plots", "entropy-kl", ".png"), series
        )
        metrics_path = artifacts.write_metrics(
            "entropy-kl",
            {
                "event_dir": str(event_dir.resolve()),
                "source_provenance": SOURCE_PROVENANCE,
                "source_control_mode": None,
                "source_condition": None,
                "source_seed": None,
                "diagnostic_config_profile": config.profile_name,
                "available_scalar_tags": sorted(available),
                "requested_scalar_tags": list(SCALAR_TAGS),
                "plot": plot_path.relative_to(artifacts.run_dir).as_posix(),
            },
        )
        artifacts.finalize(
            "completed",
            input_tensorboard={
                "event_dir": str(event_dir.resolve()),
                "provenance": SOURCE_PROVENANCE,
                "control_mode": None,
                "condition": None,
                "seed": None,
            },
        )
        print(f"event dir: {event_dir}")
        print("source provenance: legacy control mode/condition/seed unverified")
        print(f"saved: {plot_path}")
        print(f"metrics: {metrics_path}")
        return 0
    except Exception as exc:
        artifacts.finalize("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

"""
diag_entropy_kl.py

330k 붕괴가 PPO 탐색-수렴 동역학 때문인지 확인:
  train/entropy_loss, train/approx_kl, train/std, train/clip_fraction 을
  residual 붕괴 시점(~330k)과 같은 x축에 겹쳐 본다.

전제: train_ppo_02.py 에서 PPO(..., tensorboard_log=LOGDIR, verbose=1) 로 학습했어야 함.
  -> TB 로그가 없으면 아래 [대안] 참고 (330k 만 재현하며 콜백으로 수집).
"""
from datetime import datetime, timezone
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from crazyflie_rl.config import load_config

# ===== train_ppo_02.py 의 run-scoped TensorBoard 로그 =====
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = load_config(PROJECT_ROOT / "configs" / "residual_train.yaml")
MODE = CONFIG.control_mode
ARTIFACT_ROOT = CONFIG.resolve_path("artifact_root")
LOGDIR = ARTIFACT_ROOT / "runs"
SEED = CONFIG.seed
SEED_TAG = "unset" if SEED is None else f"{SEED:04d}"
CONDITION = CONFIG.condition
UTC_DATE = datetime.now(timezone.utc).strftime("%Y%m%d")
OUTDIR = ARTIFACT_ROOT / "runs" / "diagnostics" / "plots"
#COLLAPSE = 330000


def _artifact_plot_path(filename):
    source = Path(filename)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source.stem).strip("-._") or "plot"
    condition = re.sub(r"[^A-Za-z0-9._-]+", "-", CONDITION).strip("-._") or "unspecified"
    suffix = source.suffix or ".png"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    return OUTDIR / (
        f"{stem}__mode-{MODE}__condition-{condition}"
        f"__seed-{SEED_TAG}__date-{UTC_DATE}{suffix}"
    )

def find_run(root):
    """Return the latest event directory below the configured artifact runs root."""
    root = Path(root)
    event_files = sorted(root.rglob("events.out.*"), reverse=True)
    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files were found below configured runs root: {root}"
        )
    return event_files[0].parent


def main():
    run = find_run(LOGDIR)
    output = _artifact_plot_path("diag_entropy_kl.png")
    print("event dir:", run)
    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    avail = ea.Tags()["scalars"]
    print("available scalar tags:")
    for t in avail:
        print("   ", t)

    def series(tag):
        if tag not in avail:
            return None, None
        ev = ea.Scalars(tag)
        return np.array([e.step for e in ev]), np.array([e.value for e in ev])

    tags = ["train/entropy_loss", "train/approx_kl", "train/std", "train/clip_fraction"]
    fig, axes = plt.subplots(len(tags), 1, figsize=(10, 12), sharex=True)
    for ax, tag in zip(axes, tags):
        s, v = series(tag)
        #ax.axvline(COLLAPSE, ls="--", c="r", lw=1.8, label="collapse ~330k")
        if s is None:
            ax.set_title(f"{tag}  (태그 없음 — verbose/tb 설정 확인)")
            ax.legend(); continue
        ax.plot(s, v, lw=2.0)
        ax.set_title(tag); ax.grid(alpha=0.3); ax.legend(loc="best")
    axes[-1].set_xlabel("timesteps")
    fig.tight_layout(); fig.savefig(output, dpi=130)
    print("saved:", output)


if __name__ == "__main__":
    main()

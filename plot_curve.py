from datetime import datetime, timezone
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")               # headless 저장용 (SSH에서도 됨)
import matplotlib.pyplot as plt

from crazyflie_rl.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = load_config(PROJECT_ROOT / "configs" / "residual_train.yaml")
MODE = CONFIG.control_mode
ARTIFACT_ROOT = CONFIG.resolve_path("artifact_root")
SEED = CONFIG.seed
SEED_TAG = "unset" if SEED is None else f"{SEED:04d}"
CONDITION = CONFIG.condition


def _artifact_plot_path(filename):
    source = Path(filename)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source.stem).strip("-._") or "plot"
    condition = re.sub(r"[^A-Za-z0-9._-]+", "-", CONDITION).strip("-._") or "unspecified"
    utc_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = source.suffix or ".png"
    output_dir = ARTIFACT_ROOT / "runs" / "diagnostics" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (
        f"{stem}__mode-{MODE}__condition-{condition}"
        f"__seed-{SEED_TAG}__date-{utc_date}{suffix}"
    )


def main():
    steps = [20,40,60,80,100,120,140,160,180,200,220,240,260,280,300]
    steps = [s*1000 for s in steps]
    resid = [0.0278,0.0275,0.0271,0.0267,0.0269,0.0292,0.0310,0.0323,
             0.0392,0.0562,0.0612,0.0819,0.1046,0.1155,0.1037]
    floor = 0.0277

    plt.figure(figsize=(8,5))
    plt.plot(steps, resid, "o-", label="PID + residual")
    plt.axhline(floor, ls="--", color="gray", label=f"PID floor ({floor:.4f})")
    plt.axvspan(40000, 100000, alpha=0.12, color="green")   # WIN 구간
    plt.annotate("best ~80k", (80000, 0.0267), textcoords="offset points",
                 xytext=(0,-25), ha="center", arrowprops=dict(arrowstyle="->"))
    plt.xlabel("timesteps"); plt.ylabel("steady-state pos_err [m] (lower is best)")
    plt.title("10g residual: WIN then collapse")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    output = _artifact_plot_path("learning_curve.png")
    plt.savefig(output, dpi=130)
    print("saved", output)


if __name__ == "__main__":
    main()

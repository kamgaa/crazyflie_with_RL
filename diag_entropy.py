"""
diag_entropy_kl.py

330k 붕괴가 PPO 탐색-수렴 동역학 때문인지 확인:
  train/entropy_loss, train/approx_kl, train/std, train/clip_fraction 을
  residual 붕괴 시점(~330k)과 같은 x축에 겹쳐 본다.

전제: train_ppo_02.py 에서 PPO(..., tensorboard_log=LOGDIR, verbose=1) 로 학습했어야 함.
  -> TB 로그가 없으면 아래 [대안] 참고 (330k 만 재현하며 콜백으로 수집).
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ===== 채울 것: train_ppo_02.py 의 tensorboard_log 경로 (PPO_1 등 하위 폴더까지) =====
LOGDIR = "/home/mrl_6534/gwpark/crazyflie_RL/tb"     # <-- 실제 경로로 수정
#COLLAPSE = 330000
OUT = "/home/mrl_6534/gwpark/crazyflie_RL/diag_entropy_kl.png"


def find_run(root):
    """PPO_* 하위에 event 파일이 있으면 그 폴더를, 없으면 root 자체를 반환."""
    if glob.glob(os.path.join(root, "events.out.*")):
        return root
    subs = sorted(glob.glob(os.path.join(root, "*")))
    for s in subs:
        if glob.glob(os.path.join(s, "events.out.*")):
            return s
    return root


def main():
    run = find_run(LOGDIR)
    print("event dir:", run)
    ea = EventAccumulator(run, size_guidance={"scalars": 0})
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
    fig.tight_layout(); fig.savefig(OUT, dpi=130)
    print("saved:", OUT)


if __name__ == "__main__":
    main()

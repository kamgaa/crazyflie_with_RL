"""
diag_iterm_sat.py

가설 검정용 계측:  "residual 은 PID velocity I-term 이 anti-windup clip(±2) 에
포화해 손대지 못하는 영역을 대신 메운다" 를 채널 단위로 가른다.

방법:
  - case-1 payload(m_w=30g, offset=+x 30mm) 를 건 상태에서 hover 고정.
  - 초기 섭동 제거(pos_perturb=0) -> 순수 payload transient + steady state.
  - floor(policy=None, residual=0) 와 residual(PPO) 를 *같은 seed* 로 rollout.
  - 지표:  z sag,  roll/pitch,  i_vel(3축) + 포화율,  action(4채널; 특히 δFz=0.3*a[3]).

model-based 예측(반증 기준):
  floor    -> i_vel_z 포화율 ~100%,  z sag ~299mm(z≈0.70m)
  residual -> a[3] ≈ +0.98(δFz≈+0.29N),  i_vel_z 포화 해소,  z sag ~0
  가설이 z 채널에 국한되는지 attitude(x/y)까지 걸치는지는 i_vel_x/y 포화율로 판정.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from crazyflie_residual_env import CrazyflieResidualEnv

# ===== 학습/데모와 동일 =====
XML    = "/home/mrl_6534/ros2_ws/src/mujoco_crazyflie/plant/data/cf21B_500.xml"
MODEL  = "/home/mrl_6534/gwpark/crazyflie_RL/model/ppo_best"
RESIDUAL_SCALE = (0.006, 0.006, 0.0001, 0.3)
SEED   = 42
OUTDIR = "/home/mrl_6534/gwpark/crazyflie_RL"

# ===== 테스트 조건: case-1 (in-distribution) =====
M_W, R, THETA = 0.030, 0.03, 0.0
OFF   = (R * math.cos(THETA), R * math.sin(THETA))   # (x,y)[m]
HOVER = np.array([0.0, 0.0, 1.0])
T_SEC = 8.0
ICLIP = 2.0        # CascadePID 의 _i_vel anti-windup clip
SAT_EPS = 0.02     # |i_vel| > ICLIP - SAT_EPS 이면 '포화'로 계수


def rpy_deg(q):
    w, x, y, z = q
    roll  = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.degrees(math.asin(sp))
    return roll, pitch


def rollout(policy, label):
    env = CrazyflieResidualEnv(
        XML, mode="residual",
        com_bias_randomize=False,
        com_bias_mass=M_W, com_bias_offset=OFF,
        residual_scale=RESIDUAL_SCALE,
        episode_sec=T_SEC + 1.0,
        pos_perturb=0.0,             # 초기 위치 섭동 제거 -> 순수 payload 응답만
    )
    obs, _ = env.reset(seed=SEED)
    env.pos_des = HOVER.copy()

    m_now = float(env.model.body_mass[env.drone_bid])
    print(f"[{label}] payload check: mass={m_now:.5f}kg "
          f"(nominal {env._m0:.5f}, Δ={m_now - env._m0:+.5f})  "
          f"ipos={np.round(env.model.body_ipos[env.drone_bid], 4)}")

    dt = env.dt_phys * env.substeps
    n = int(round(T_SEC / dt))
    rows = []
    for k in range(n):
        act = policy.predict(obs, deterministic=True)[0] if policy else np.zeros(4)
        obs, _, term, trunc, _ = env.step(act)   # term/trunc 무시하고 끝까지 기록
        pos = obs[0:3] + env.pos_des
        roll, pitch = rpy_deg(obs[6:10])
        ivx, ivy, ivz = env.pid._i_vel           # step() 내부 마지막 substep 값
        rows.append([k * dt, pos[2], roll, pitch, ivx, ivy, ivz, *act])
    return np.array(rows)


def summarize(A, label):
    t = A[:, 0]
    ss = t >= (T_SEC - 2.0)                       # 마지막 2s = 정상상태 창
    z, roll, pitch = A[:, 1], A[:, 2], A[:, 3]
    ivx, ivy, ivz = A[:, 4], A[:, 5], A[:, 6]
    a = A[:, 7:11]

    def sat(v):
        return 100.0 * float(np.mean(np.abs(v) > ICLIP - SAT_EPS))

    print(f"\n=== [{label}] 정상상태(마지막 2s) 요약 ===")
    print(f"  z            = {z[ss].mean():.4f} m   (sag {1.0 - z[ss].mean():+.4f} m)")
    print(f"  roll / pitch = {roll[ss].mean():+.2f} / {pitch[ss].mean():+.2f} deg")
    print(f"  i_vel[x,y,z] = [{ivx[ss].mean():+.3f}, {ivy[ss].mean():+.3f}, {ivz[ss].mean():+.3f}]")
    print(f"  i_vel 포화율(전구간) x/y/z = {sat(ivx):.0f}% / {sat(ivy):.0f}% / {sat(ivz):.0f}%")
    print(f"  action[x,y,z,Fz] = [{a[ss,0].mean():+.3f}, {a[ss,1].mean():+.3f}, "
          f"{a[ss,2].mean():+.3f}, {a[ss,3].mean():+.3f}]")
    print(f"    -> δτ=[{0.006*a[ss,0].mean():+.4f},{0.006*a[ss,1].mean():+.4f},"
          f"{0.0001*a[ss,2].mean():+.5f}] N·m,  δFz={0.3*a[ss,3].mean():+.4f} N "
          f"(payload 무게 {M_W*9.81:.4f} N)")


def make_plot(F, Rd):
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax[0].axhline(1.0, ls=":", c="k", lw=1.2, label="ref z=1.0")
    ax[0].plot(F[:, 0], F[:, 1], lw=2.5, label="floor")
    ax[0].plot(Rd[:, 0], Rd[:, 1], lw=2.5, label="residual")
    ax[0].set_ylabel("z [m]"); ax[0].set_title("z position (payload sag)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(F[:, 0], F[:, 3], lw=2.5, label="floor")
    ax[1].plot(Rd[:, 0], Rd[:, 3], lw=2.5, label="residual")
    ax[1].axhline(0, ls="--", c="gray", lw=1)
    ax[1].set_ylabel("pitch [deg]"); ax[1].set_title("pitch (case-1 = pitch 외란)")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    ax[2].axhline(+ICLIP, ls=":", c="r", lw=1.5, label="clip ±2")
    ax[2].axhline(-ICLIP, ls=":", c="r", lw=1.5)
    ax[2].plot(F[:, 0], F[:, 6], lw=2.5, label="floor  i_vel_z")
    ax[2].plot(Rd[:, 0], Rd[:, 6], lw=2.5, label="residual i_vel_z")
    ax[2].set_ylabel("i_vel_z"); ax[2].set_title("velocity I-term (z) vs anti-windup clip")
    ax[2].legend(); ax[2].grid(alpha=0.3)

    ax[3].plot(Rd[:, 0], 0.3 * Rd[:, 10], lw=2.5, c="tab:green", label="residual δFz [N]")
    ax[3].axhline(M_W * 9.81, ls="--", c="k", lw=1.2, label=f"payload weight {M_W*9.81:.3f} N")
    ax[3].set_ylabel("δFz [N]"); ax[3].set_xlabel("time [s]")
    ax[3].set_title("residual δFz vs payload weight")
    ax[3].legend(); ax[3].grid(alpha=0.3)

    fig.tight_layout()
    png = os.path.join(OUTDIR, "diag_iterm_sat_case1.png")
    fig.savefig(png, dpi=130); plt.close(fig)
    print(f"\nsaved: {png}")


if __name__ == "__main__":
    print(f"case-1: m_w={M_W*1000:.0f}g  off=({OFF[0]*1000:+.0f},{OFF[1]*1000:+.0f})mm  "
          f"|τ_g|={R*M_W*9.81*1e3:.2f} mN·m")

    F = rollout(None, "floor")
    summarize(F, "floor")

    model = PPO.load(MODEL)
    Rd = rollout(model, "residual")
    summarize(Rd, "residual")

    make_plot(F, Rd)
    print("\n[해석 가이드]")
    print("  floor i_vel_z 포화율 ~100% & z sag 크면  -> I-term 포화 확인")
    print("  residual δFz ≈ payload weight & i_vel_z 포화 해소 & sag↓  -> 가설(z채널) 지지")
    print("  i_vel_x/y 포화율까지 높고 residual δτ_x/y 가 이를 상쇄하면 -> 가설이 attitude 로도 확장")

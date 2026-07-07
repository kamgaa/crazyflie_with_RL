import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time, math
import numpy as np
import mujoco, mujoco.viewer
import matplotlib
matplotlib.use("Agg")                     # headless 저장 (VNC/SSH 무관하게 PNG 생성)
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from crazyflie_residual_env import CrazyflieResidualEnv

# ===== 학습과 *동일하게* 맞출 것 =====
XML   = "/home/mrl_6534/ros2_ws/src/mujoco_crazyflie/plant/data/cf21B_500.xml"
MODEL = "/home/mrl_6534/gwpark/crazyflie_RL/model/ppo_best"

RESIDUAL_SCALE = (0.006, 0.006, 0.0001, 0.3)   # ← train_ppo.py 와 동일하게!
SEED  = 42
OUTDIR = "/home/mrl_6534/gwpark/crazyflie_RL"
LW = 3.0                                   # 선 굵기 (MATLAB 기준 ~3)


ARM = 0.035355
R = 2.3 * ARM
#m_c, m_e = 0.0293, 0.0117
m_c, m_e = 0.03, 0.029

# 3개 대표 조건: (반지름 비율, 각도) — 결정론적 지정
#TEST_CONDITIONS = [
#    ("center-heavy", 0.15*R, 0.0),           # 중심 근방: 고무게, 저토크 (z 지배)
#    ("mid",          0.55*R, np.pi/2),        # 중간: 무게-토크 균형
#    ("edge-light",   0.95*R, np.pi),          # 가장자리: 저무게, 고토크 (rotational 지배)
#]


TEST_CONDITIONS = [
    #  이름            r (거리)        theta        m_w (무게, kg)
    ("my-case-1",    0.030,          0.0,         0.030),   # 30mm, roll축, 20g
    ("my-case-2",    0.060,          np.pi/2,     0.035),   # 40mm, pitch축, 15g
    ("my-case-3",    0.090,          np.pi,       0.029),   # 20mm, -roll축, 29g
]

def quat_to_euler_deg(q):
    """q=[w,x,y,z] -> (roll, pitch, yaw) in degrees."""
    w, x, y, z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sp = max(-1.0, min(1.0, 2*(w*y - z*x)))
    pitch = math.asin(sp)
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.degrees([roll, pitch, yaw])


def save_plot(tag, t, pos, rpy, png_path):
    """pos: (N,3) 실제 위치[m], rpy: (N,3) [deg]. 한 장에 위치+자세 두 subplot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- 위치 ---
    ax1.plot(t, pos[:, 0], lw=LW, label="x")
    ax1.plot(t, pos[:, 1], lw=LW, label="y")
    ax1.plot(t, pos[:, 2], lw=LW, label="z")
    ax1.axhline(0.0, ls="--", color="gray", lw=1.5)   # x,y setpoint
    ax1.axhline(1.0, ls=":",  color="gray", lw=1.5)   # z setpoint (hover 1.0m)
    ax1.set_ylabel("position [m]")
    ax1.set_title(f"{tag} — position (x,y,z)")
    ax1.legend(loc="best"); ax1.grid(alpha=0.3)

    # --- 자세 ---
    ax2.plot(t, rpy[:, 0], lw=LW, label="roll")
    ax2.plot(t, rpy[:, 1], lw=LW, label="pitch")
    ax2.plot(t, rpy[:, 2], lw=LW, label="yaw")
    ax2.axhline(0.0, ls="--", color="gray", lw=1.5)
    ax2.set_ylabel("attitude [deg]"); ax2.set_xlabel("time [s]")
    ax2.set_title(f"{tag} — attitude (roll,pitch,yaw)")
    ax2.legend(loc="best"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"    saved: {png_path}")

def view(policy, tag, r_bias, theta, png_name):
    T, P_hist, RPY_hist = [], [], []                 # 자료구조: 명확한 접미사
    env = CrazyflieResidualEnv(XML, mode="residual",
                               com_bias_randomize=False,
                               residual_scale=RESIDUAL_SCALE)   # ← 반드시 전달
    env.dist_torque_body = np.zeros(3)
    dt = env.dt_phys * env.substeps
    m_w = m_c - (m_c - m_e) * (r_bias / R)            # R: 전역 반지름 상수 (불변)
    off = np.array([r_bias*np.cos(theta), r_bias*np.sin(theta)])
    obs, _ = env.reset(seed=SEED)
    env._set_com_bias(m_w, off)
    mujoco.mj_forward(env.model, env.data)
    print(f"\n>>> [{tag}] 시작")
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.distance = 3.7
        done = False; k = 0
        while v.is_running() and not done:
            t0 = time.time()
            act = policy.predict(obs, deterministic=True)[0] if policy else np.zeros(4)
            obs, rew, term, trunc, _ = env.step(act)    # r → rew (물리 반지름과 분리)
            P_hist.append(obs[0:3] + env.pos_des)
            RPY_hist.append(quat_to_euler_deg(obs[6:10]))
            T.append(k * dt); k += 1
            v.sync()
            done = term or trunc
            slack = dt - (time.time() - t0)
            if slack > 0: time.sleep(slack)
    ss = slice(int(len(P_hist)*0.7), None)              # 정착 구간 (후반 30%)
    ss_err = np.mean([np.linalg.norm(np.array(P_hist[i]) - env.pos_des) for i in range(len(P_hist))][int(len(P_hist)*0.7):])
    print(f"<<< [{tag}] 정착 오차(후반30%) = {ss_err:.4f} m")
    save_plot(tag, np.array(T), np.array(P_hist), np.array(RPY_hist),
              os.path.join(OUTDIR, png_name))

if __name__ == "__main__":
    model = PPO.load(MODEL)
    for name, r, theta , m_w in TEST_CONDITIONS:
        # 동일 조건에서 floor / residual 쌍 비교
        view(None,  f"{name} | floor (PID)",    r, theta, f"traj_{name}_floor.png")
        view(model, f"{name} | residual (PID+RL)", r, theta, f"traj_{name}_residual.png")
    print("완료: 3개 조건 × 2정책 궤적 저장")

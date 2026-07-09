import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time, math, sys
import numpy as np
import mujoco, mujoco.viewer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from crazyflie_residual_env import CrazyflieResidualEnv

# ================= 학습과 *동일하게* =================
XML   = "/home/mrl_6534/ros2_ws/src/mujoco_crazyflie/plant/data/cf21B_500.xml"
MODEL = "/home/mrl_6534/gwpark/crazyflie_RL/model/ppo_best"

RESIDUAL_SCALE = (0.006, 0.006, 0.0001, 0.3)   # ← train_ppo.py 와 동일하게!
SEED   = 42
OUTDIR = "/home/mrl_6534/gwpark/crazyflie_RL"
LW     = 3.0

CAM_TRACK = True        # 카메라가 드론 추종. 'nocam' 인자로 끄면 원인 격리 테스트.

# ---- CoM 외란: 이 데모는 0g (무바이어스) — 순수 원 추종/OOD 관찰용 ----
# 외란을 걸어 비교하고 싶으면 run() 의 com_bias_mass=0.0 을 원하는 값으로 바꾸고
# reset 뒤 env._set_com_bias(m_w, (x,y)) 를 명시 호출할 것.

# ================= 임무 파라미터 =================
HOVER_Z   = 1.0          # 목표 호버 고도 [m]
GOTO_XY   = (1.0, 0.0)   # 원 진입 전 이동 목표 (x,y)

# 상태기계 구간 길이 [s]
T_TAKEOFF = 4.0          # z: 0 -> HOVER_Z (cosine ease)
T_SETTLE1 = 2.0          # 이륙 후 정착
T_GOTO    = 4.0          # (0,0) -> GOTO_XY (cosine ease)
T_SETTLE2 = 2.0          # 이동 후 정착
T_RAMP    = 2.0          # 원 각속도 0 -> omega (cosine ease-in)
N_LAPS    = 2.0          # 정속 원 바퀴 수

# 원 프리셋: (반경 rho[m], 주기 period[s])
CIRCLE_PRESETS = {
    "1": ("radius 0.5m, period 10s (slow/safe)", 0.5, 10.0),
    "2": ("radius 1.0m, period  8s (medium)",    1.0,  8.0),
    "3": ("radius 0.5m, period  5s (fast/aggressive)", 0.5, 5.0),
}

# ================= CoM 외란 테스트 조건 =================
# 각 조건마다 payload(m_w)를 CoM 에서 극좌표 (r, theta) 위치에 물리적으로 부착한다.
#   - env._set_com_bias 가 body_mass / body_ipos / body_inertia 를 편집해 주입
#     (외력 근사가 아니라 실제 관성 재계산 → training(reset) 과 *동일* 메커니즘)
#   - theta 는 offset 의 *방향각* 이며, 유발되는 중력토크 축은 이와 90도 어긋남:
#       offset +x (theta=0)    -> pitch(y) 토크
#       offset +y (theta=pi/2) -> roll(x)  토크   (tau_g = m_w g (r sin th, -r cos th, 0))
#   - 안전/분포 참고: training disk 반경 R = 2.3*ARM = 81.3mm, r_crit ≈ 91.4mm,
#     PID max_tau = 20 mN*m. |tau_g| = r*m_w*g 가 이를 넘으면 정적 균형이 경계/불가.
#
#  이름            r [m]        theta [rad]   m_w [kg]
TEST_CONDITIONS = [
    #("my-case-1",  0.03,        0.0,          0.030),   # |tau|=8.8mNm(44%), r=33%R : in-dist, 안전
    #("my-case-2", 0.060,       np.pi/2,      0.035),   # |tau|=20.6mNm(>max_tau) : 경계
    ("my-case-3", 0.120,       np.pi,        0.010),   # r≈r_crit(98%) & OOD(>81mm) : 발산 예상
]


def quat_to_euler_deg(q):
    w, x, y, z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sp = max(-1.0, min(1.0, 2*(w*y - z*x)))
    pitch = math.asin(sp)
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.degrees([roll, pitch, yaw])


def ease(a):
    """cosine ease-in-out: a in [0,1] -> [0,1], C1 연속 (양 끝 속도 0)."""
    a = min(1.0, max(0.0, a))
    return 0.5 * (1.0 - math.cos(math.pi * a))


def build_reference(rho, period):
    """
    상태기계 reference generator.
    반환: ref(t) -> (pos_des[3], phase_tag)
    모든 전이는 위치·속도 연속 (cosine ease).
    """
    omega = 2.0 * math.pi / period          # 정속 각속도 [rad/s]
    cx, cy = GOTO_XY[0] - rho, GOTO_XY[1]    # 원 중심: t=0 위상에서 (1,0)이 되도록
    T_CIRCLE = T_RAMP + N_LAPS * period      # 원 구간 총 길이

    # 구간 경계 (누적 시각)
    t0 = 0.0
    t1 = t0 + T_TAKEOFF          # 이륙 끝
    t2 = t1 + T_SETTLE1          # 정착1 끝
    t3 = t2 + T_GOTO             # 이동 끝
    t4 = t3 + T_SETTLE2          # 정착2 끝
    t5 = t4 + T_CIRCLE           # 원 끝
    T_TOTAL = t5 + 2.0           # 마지막 여유

    def ref(t):
        # --- TAKEOFF: z 0 -> HOVER_Z ---
        if t < t1:
            z = HOVER_Z * ease((t - t0) / T_TAKEOFF)
            return np.array([0.0, 0.0, z]), "TAKEOFF"
        # --- SETTLE1 ---
        if t < t2:
            return np.array([0.0, 0.0, HOVER_Z]), "SETTLE1"
        # --- GOTO(1,0,z) ---
        if t < t3:
            s = ease((t - t2) / T_GOTO)
            x = GOTO_XY[0] * s
            y = GOTO_XY[1] * s
            return np.array([x, y, HOVER_Z]), "GOTO"
        # --- SETTLE2 ---
        if t < t4:
            return np.array([GOTO_XY[0], GOTO_XY[1], HOVER_Z]), "SETTLE2"
        # --- CIRCLE ---
        if t < t5:
            tc = t - t4
            if tc < T_RAMP:
                # 각속도 0 -> omega 로 ease-in.
                # 위상 phi(tc) = omega * ∫_0^tc ease(s/T_RAMP) ds
                # ease의 해석적 적분: ∫ 0.5(1-cos(pi*s/T))ds = 0.5*s - (T/2pi)sin(pi*s/T)
                Tp = T_RAMP
                phi = omega * (0.5 * tc - (Tp / (2*math.pi)) * math.sin(math.pi * tc / Tp))
            else:
                # ramp 종료 시 누적 위상 = omega*T_RAMP/2 (ease 평균 0.5)
                phi = omega * (0.5 * T_RAMP) + omega * (tc - T_RAMP)
            x = cx + rho * math.cos(phi)
            y = cy + rho * math.sin(phi)
            return np.array([x, y, HOVER_Z]), "CIRCLE"
        # --- HOLD (원 종료 후 시작점 유지) ---
        return np.array([GOTO_XY[0], GOTO_XY[1], HOVER_Z]), "HOLD"

    return ref, T_TOTAL


def force_floor_start(env):
    """reset 후 드론을 바닥 근처(z≈0.02)에 강제 배치. free-joint qpos 레이아웃 가정."""
    try:
        # 자유관절: qpos[0:3]=위치, qpos[3:7]=quat(w,x,y,z)
        env.data.qpos[0:3] = np.array([0.0, 0.0, 0.02])
        env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        env.data.qvel[:6]  = 0.0
        mujoco.mj_forward(env.model, env.data)
        return True
    except Exception as e:
        print(f"    [warn] force_floor_start 실패 ({e}); reset 기본 위치 사용")
        return False


def save_plot(tag, t, pos, rpy, ref_pos, png_path, err=None):
    fig = plt.figure(figsize=(11, 13))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 3, sharex=ax1)     # |pos_err| (OOD 계측)
    ax3 = fig.add_subplot(4, 1, 4)                  # XY (독립 축)

    # 위치 (실측 실선 + reference 점선)
    for i, c, lbl in [(0, "tab:blue", "x"), (1, "tab:orange", "y"), (2, "tab:green", "z")]:
        ax1.plot(t, pos[:, i], lw=LW, color=c, label=lbl)
        ax1.plot(t, ref_pos[:, i], ls=":", lw=1.8, color=c, alpha=0.7)
    ax1.set_ylabel("position [m]")
    ax1.set_title(f"{tag} — position (solid=actual, dotted=ref)")
    ax1.legend(loc="best"); ax1.grid(alpha=0.3)

    # 자세
    ax2.plot(t, rpy[:, 0], lw=LW, label="roll")
    ax2.plot(t, rpy[:, 1], lw=LW, label="pitch")
    ax2.plot(t, rpy[:, 2], lw=LW, label="yaw")
    ax2.axhline(0.0, ls="--", color="gray", lw=1.5)
    ax2.set_ylabel("attitude [deg]")
    ax2.set_title(f"{tag} — attitude"); ax2.legend(loc="best"); ax2.grid(alpha=0.3)

    # |pos_err| — OOD 계측 (학습 분포 경계 15cm 표시)
    if err is not None:
        ax4.plot(t, err, lw=LW, color="tab:red", label="|pos_err|")
        ax4.axhline(0.15, ls="--", color="k", lw=1.5,
                    label="train dist. boundary (0.15m)")
        ax4.fill_between(t, 0.15, err, where=(err > 0.15),
                         color="tab:red", alpha=0.15)
        ax4.set_ylabel("|pos_err| [m]"); ax4.set_xlabel("time [s]")
        ax4.set_title(f"{tag} — OOD indicator (red = outside training dist.)")
        ax4.legend(loc="best"); ax4.grid(alpha=0.3)

    # XY 궤적 (원 추종 품질)
    ax3.plot(pos[:, 0], pos[:, 1], lw=LW, color="tab:blue", label="actual")
    ax3.plot(ref_pos[:, 0], ref_pos[:, 1], ls=":", lw=1.8, color="k", label="ref")
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]")
    ax3.set_title(f"{tag} — XY path"); ax3.axis("equal")
    ax3.legend(loc="best"); ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"    saved: {png_path}")


def run_headless(policy, tag, rho, period, png_name,
                 m_w=0.0, com_offset=(0.0, 0.0)):
    """
    뷰어 없이 순수 물리만 완주하는 진단 버전.
    - v.is_running() / v.sync() / v.cam / time.sleep 전부 제거
    - 종료를 일으킬 수 있는 건 오직 (1) 정해진 스텝 수 소진, (2) NaN 폭주뿐
    - |e| 가 0.15(학습분포), 1.5(env guard) 를 언제 처음 넘는지 명시적으로 잡음
    → "뷰어 탓 vs 물리 발산 탓" 을 완전히 분리한다.
    """
    env = CrazyflieResidualEnv(XML, mode="residual",
                               com_bias_randomize=False,
                               com_bias_mass=m_w,           # ← 테스트 조건 payload 질량
                               com_bias_offset=com_offset,  # ← (x,y)[m], reset()이 _set_com_bias로 주입
                               residual_scale=RESIDUAL_SCALE)
    env.dist_torque_body = np.zeros(3)
    dt = env.dt_phys * env.substeps

    obs, _ = env.reset(seed=SEED)
    force_floor_start(env)
    ref, T_TOTAL = build_reference(rho, period)
    n_steps = int(round(T_TOTAL / dt))

    T, P_hist, RPY_hist, REF_hist, ERR_hist, PHASE_hist = [], [], [], [], [], []
    last_phase = None
    t_cross_015 = None      # |e| 가 0.15 처음 넘은 시각
    t_cross_150 = None      # |e| 가 1.5  처음 넘은 시각 (env guard 임계)
    diverged_at = None      # NaN/폭주 시각

    print(f"\n>>> [HEADLESS: {tag}] 시작  n_steps={n_steps}, T_total≈{T_TOTAL:.1f}s")

    for k in range(n_steps):
        t_now = k * dt
        pos_des, phase = ref(t_now)
        env.pos_des = pos_des.copy()
        if phase != last_phase:
            cur = f"{ERR_hist[-1]:.3f}m" if ERR_hist else "n/a"
            print(f"    t={t_now:6.2f}s  phase -> {phase}  (|e| 지금 {cur})")
            last_phase = phase

        act = policy.predict(obs, deterministic=True)[0] if policy else np.zeros(4)
        obs, rew, term, trunc, _ = env.step(act)

        # NaN 이면 물리가 완전히 터진 것 → 여기서만 중단
        if not np.all(np.isfinite(obs)):
            diverged_at = t_now
            print(f"    [NaN] 물리 발산 @ t={t_now:.2f}s phase={phase} → 중단")
            break

        pos_err = obs[0:3]
        err_norm = float(np.linalg.norm(pos_err))
        actual_pos = pos_err + env.pos_des
        rpy_now = quat_to_euler_deg(obs[6:10])

        # 임계 통과 시각 기록 (끊지 않고 계속 진행)
        if t_cross_015 is None and err_norm > 0.15:
            t_cross_015 = t_now
            print(f"    [>0.15] |e| 학습분포 이탈 @ t={t_now:.2f}s phase={phase} "
                  f"|e|={err_norm:.3f}")
        if t_cross_150 is None and err_norm > 1.5:
            t_cross_150 = t_now
            print(f"    [>1.5 ] |e| env-guard 임계 돌파 @ t={t_now:.2f}s phase={phase} "
                  f"|e|={err_norm:.3f}  rpy=({rpy_now[0]:.1f},{rpy_now[1]:.1f}) "
                  f"act={np.round(act,3)}")

        # phase 별 상세 (200ms 간격) — 발산 궤적을 프레임으로
        if k % 20 == 0:
            print(f"      t={t_now:5.2f} {phase:8s} |e|={err_norm:.3f} "
                  f"pos={np.round(actual_pos,2)} rpy=({rpy_now[0]:.1f},{rpy_now[1]:.1f}) "
                  f"act={np.round(act,3)}")

        T.append(t_now); P_hist.append(actual_pos.copy())
        RPY_hist.append(rpy_now); REF_hist.append(env.pos_des.copy())
        ERR_hist.append(err_norm); PHASE_hist.append(phase)

    # --- 요약 ---
    ERR = np.array(ERR_hist); PH = np.array(PHASE_hist)
    print(f"<<< [HEADLESS: {tag}] 요약")
    for ph in ["TAKEOFF", "SETTLE1", "GOTO", "SETTLE2", "CIRCLE", "HOLD"]:
        m = PH == ph
        if m.any():
            print(f"      {ph:8s}: max={ERR[m].max():.3f}  mean={ERR[m].mean():.3f}  "
                  f"end={ERR[m][-1]:.3f}")
    print(f"      완주 스텝: {len(ERR)}/{n_steps}  "
          f"({'전구간 완주' if len(ERR)==n_steps else 'NaN 중단'})")
    print(f"      |e|>0.15 최초: {t_cross_015}   |e|>1.5 최초: {t_cross_150}   "
          f"NaN: {diverged_at}")

    if len(P_hist) >= 5:
        save_plot(tag, np.array(T), np.array(P_hist), np.array(RPY_hist),
                  np.array(REF_hist), os.path.join(OUTDIR, png_name), err=ERR)


def run(policy, tag, rho, period, png_name, realtime=True,
        m_w=0.0, com_offset=(0.0, 0.0)):
    # CoM bias 없음(0g): env 기본 10g 를 무력화하려면 com_bias_mass=0.0 을 명시해야 함.
    # (reset() 이 else 가지에서 _set_com_bias(com_bias_mass, ...) 를 자동 호출하는데,
    #  m_w=0 이면 질량·CoM·관성 모두 원본으로 복원됨 → 순수 원 추종만 검증)
    #env = CrazyflieResidualEnv(XML, mode="residual",
    #                           com_bias_randomize=False,
    #                           com_bias_mass=0.0,          # ← 0g 강제 (교락 제거)
    #                           residual_scale=RESIDUAL_SCALE)
    #env.dist_torque_body = np.zeros(3)
    #dt = env.dt_phys * env.substeps

    #obs, _ = env.reset(seed=SEED)
    #force_floor_start(env)

   # ref, T_TOTAL = build_reference(rho, period)
    ref, T_TOTAL = build_reference(rho, period)

    env = CrazyflieResidualEnv(
        XML,
        mode="residual",
        com_bias_randomize=False,
        com_bias_mass=m_w,                 # ← 테스트 조건 payload 질량
        com_bias_offset=com_offset,        # ← (x,y)[m], reset()이 _set_com_bias로 주입
        residual_scale=RESIDUAL_SCALE,
        episode_sec=T_TOTAL + 2.0,   # 핵심 수정
    )

    env.dist_torque_body = np.zeros(3)
    dt = env.dt_phys * env.substeps

    obs, _ = env.reset(seed=SEED)
    force_floor_start(env)

    # reset 직후 관측을 새 pos_des 기준으로 다시 만들어야 함 (아래 루프에서 매 스텝 갱신)
    T, P_hist, RPY_hist, REF_hist = [], [], [], []
    ERR_hist, PHASE_hist = [], []       # OOD 계측: |pos_err|, phase 태그
    first_crash_t = None                # env guard 최초 발동 시각 (None=미발동)
    if hasattr(run, "_circ_t0"):
        del run._circ_t0                # CIRCLE 로깅 타이머 리셋 (run 간 격리)
    print(f"\n>>> [{tag}] 시작  (rho={rho}m, period={period}s, T_total≈{T_TOTAL:.1f}s)")
    last_phase = None

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        with v.lock():
            v.cam.distance = 4.5
            v.cam.azimuth = 135
            v.cam.elevation = -25
            v.cam.lookat[:] = np.array([0.5, 0.0, 1.0])   # 초기: 임무 영역 중심 근처
        k = 0
        while v.is_running():
            t0 = time.time()
            t_now = k * dt
            if t_now > T_TOTAL:
                break

            # --- reference 주입: env.pos_des 를 매 스텝 갱신 ---
            pos_des, phase = ref(t_now)
            env.pos_des = pos_des.copy()
            if phase != last_phase:
                print(f"    t={t_now:6.2f}s  phase -> {phase}")
                last_phase = phase

            act = policy.predict(obs, deterministic=True)[0] if policy else np.zeros(4)
            obs, rew, term, trunc, _ = env.step(act)

            pos_err = obs[0:3]                       # obs[0:3] = pos - pos_des
            err_norm = float(np.linalg.norm(pos_err))
            actual_pos = pos_err + env.pos_des

            # --- 발산/NaN 감지: "조용히 사라지는" 현상의 정체를 명시적으로 잡는다 ---
            rpy_now = quat_to_euler_deg(obs[6:10])
            if not np.all(np.isfinite(obs)):
                print(f"    [DIVERGE] NaN/Inf 관측 @ t={t_now:.2f}s phase={phase} "
                      f"→ 물리 발산. 원인: residual OOD 폭주 가능성. 중단.")
                break
            if actual_pos[2] > 5.0 or err_norm > 5.0 or abs(rpy_now[0]) > 80 or abs(rpy_now[1]) > 80:
                print(f"    [DIVERGE] 상태 폭주 @ t={t_now:.2f}s phase={phase}  "
                      f"z={actual_pos[2]:.2f} |e|={err_norm:.2f} "
                      f"roll={rpy_now[0]:.1f} pitch={rpy_now[1]:.1f} "
                      f"act={np.round(act,3)} → 드론이 날아감(카메라 밖). 중단.")
                break

            # --- CIRCLE 진입 직후 상세 로깅 (원인 특정용, 처음 진입 후 ~1s) ---
            if phase == "CIRCLE" and last_phase == "CIRCLE":
                if not hasattr(run, "_circ_t0"):
                    run._circ_t0 = t_now
                if t_now - run._circ_t0 < 1.0 and k % 5 == 0:   # 50ms 간격
                    print(f"      [CIRC] t={t_now:.2f} ref={np.round(env.pos_des,3)} "
                          f"|e|={err_norm:.3f} rpy=({rpy_now[0]:.1f},{rpy_now[1]:.1f}) "
                          f"act={np.round(act,3)}")

            P_hist.append(actual_pos.copy())
            RPY_hist.append(rpy_now)
            REF_hist.append(env.pos_des.copy())
            ERR_hist.append(err_norm)
            PHASE_hist.append(phase)
            T.append(t_now)
            k += 1

            # --- OOD stress test: env 의 crashed(term) 를 죽음으로 처리하지 않음 ---
            # env 는 hover-정착 가정의 guard(pos[2]<0.2, |e_pos|>1.5) 로 term=True 를
            # 반환하지만, 물리(mj_step)는 계속 돌므로 궤적은 유효하다. 첫 crash 순간과
            # phase 만 기록하고 진행 → "정책이 OOD 를 지나 분포로 복귀하는가" 관찰.
            if term and (first_crash_t is None):
                first_crash_t = t_now
                print(f"    [OOD] env guard 발동 (term=True) @ t={t_now:.2f}s "
                      f"phase={phase}  |e|={err_norm:.3f}m  z={actual_pos[2]:.3f}m "
                      f"→ 무시하고 진행")

            # --- 카메라가 드론을 따라감 (화면 밖 이탈 방지) ---
            # 중요: passive viewer 에서 v.cam 수정은 반드시 v.lock() 안에서.
            # 렌더 스레드가 동시에 cam 을 읽으면 뷰어가 죽을 수 있음(GOTO 진입 시 재현).
            if CAM_TRACK:
                with v.lock():
                    v.cam.lookat[:] = 0.9 * np.asarray(v.cam.lookat) + 0.1 * actual_pos

            v.sync()
            if trunc:
                print(
                    f"    [STOP] env truncated=True @ t={t_now:.2f}s "
                    f"phase={phase}, env._step={env._step}, max_steps={env.max_steps}. "
                    f"episode_sec가 mission T_TOTAL={T_TOTAL:.2f}s보다 짧음."
                )# 시간초과만 진짜 종료로 취급
                break
            if realtime:
                slack = dt - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)

    if len(P_hist) < 5:
        print(f"<<< [{tag}] 데이터 부족, 플롯 생략")
        return

    # --- OOD 요약: phase 별 최대/평균 위치오차 ---
    ERR = np.array(ERR_hist); PH = np.array(PHASE_hist)
    print(f"<<< [{tag}] OOD 요약 (phase별 |pos_err| [m])")
    for ph in ["TAKEOFF", "SETTLE1", "GOTO", "SETTLE2", "CIRCLE", "HOLD"]:
        m = PH == ph
        if m.any():
            print(f"      {ph:8s}: max={ERR[m].max():.3f}  mean={ERR[m].mean():.3f}")
    if first_crash_t is not None:
        print(f"      [env guard 최초 발동 @ t={first_crash_t:.2f}s]")
    else:
        print(f"      [env guard 미발동 — 전 구간 hover 가정 내 유지]")

    save_plot(tag, np.array(T), np.array(P_hist), np.array(RPY_hist),
              np.array(REF_hist), os.path.join(OUTDIR, png_name), err=ERR)


def choose_mode():
    if len(sys.argv) > 1 and sys.argv[1] in CIRCLE_PRESETS:
        return sys.argv[1]
    print("\n=== 원 프리셋 선택 ===")
    for k, (desc, _, _) in CIRCLE_PRESETS.items():
        print(f"  [{k}] {desc}")
    while True:
        sel = input("mode 번호 입력 (1/2/3): ").strip()
        if sel in CIRCLE_PRESETS:
            return sel
        print("  잘못된 입력. 1, 2, 3 중 선택.")


if __name__ == "__main__":
    mode = choose_mode()
    desc, rho, period = CIRCLE_PRESETS[mode]
    print(f"\n선택: mode {mode} — {desc}")

    # 2번째 인자로 어떤 정책을 돌릴지 선택: floor / residual / both (기본 both)
    which = "both"
    headless = False
    realtime = True
    for a in sys.argv[1:]:
        if a in ("floor", "residual", "both"):
            which = a
        if a == "headless":
            headless = True
        if a == "norealtime":          # time.sleep 제거 → 뷰어 stall 가설 검증
            realtime = False
        if a == "nocam":               # 카메라 추종 끔 → 카메라 코드가 범인인지 격리
            CAM_TRACK = False
    print(f"정책 실행 대상: {which}   headless={headless}   realtime={realtime}   cam_track={CAM_TRACK}")

    tagbase = f"mission-mode{mode}"

    def go(policy, tag, png, m_w=0.0, com_offset=(0.0, 0.0)):
        if headless:
            run_headless(policy, tag, rho, period, png,
                         m_w=m_w, com_offset=com_offset)
        else:
            run(policy, tag, rho, period, png, realtime=realtime,
                m_w=m_w, com_offset=com_offset)

    # residual 정책은 여러 조건에서 재사용하므로 루프 밖에서 한 번만 로드
    model = PPO.load(MODEL) if which in ("residual", "both") else None

    for (name, r, theta, m_w) in TEST_CONDITIONS:
        # 극좌표 (r, theta) -> body xy offset (env._set_com_bias 는 (x,y) 를 받음)
        com_offset = (r * math.cos(theta), r * math.sin(theta))
        tau_g = r * m_w * 9.81                          # 유발 중력토크 크기 [N·m]
        cond = (f"{name} | r={r*1000:.0f}mm  θ={math.degrees(theta):.0f}°  "
                f"m_w={m_w*1000:.0f}g  off=({com_offset[0]*1000:+.0f},{com_offset[1]*1000:+.0f})mm  "
                f"|τ_g|={tau_g*1e3:.1f}mNm")
        print(f"\n{'='*74}\n[TEST CONDITION] {cond}\n{'='*74}")

        if which in ("floor", "both"):
            # baseline: PID 만으로 payload 를 버티며 미션 완주하는지 (모델 불필요)
            go(None, f"{tagbase} | {name} | floor (PID)",
               f"{tagbase}_{name}_floor.png", m_w=m_w, com_offset=com_offset)

        if which in ("residual", "both"):
            go(model, f"{tagbase} | {name} | residual (PID+RL)",
               f"{tagbase}_{name}_residual.png", m_w=m_w, com_offset=com_offset)

    print(f"\n완료: mode {mode} ({which}, headless={headless}, realtime={realtime}), "
          f"conditions={len(TEST_CONDITIONS)}")

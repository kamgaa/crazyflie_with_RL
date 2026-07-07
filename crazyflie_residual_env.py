"""
crazyflie_residual_env.py

SEOSUK/mujoco_crazyflie (just_flight) 의 plant.py MuJoCo 코어를 ROS2에서 분리한
동기식 Gymnasium 환경 + cascade PID 포팅 (residual RL 용).

설계 합의 사항 반영:
  - action  = residual wrench [d_tau_x, d_tau_y, d_tau_z, d_Fz]  (PID 출력 위에 더함)
  - baseline= cascade PID (physics rate 500Hz 로 동작), residual 은 policy rate 로 held
  - obs(13) = [pos_err(3), vel_W(3), quat_wxyz(4, q_w>=0 normalize), omega_B(3)]  (Markov)
  - 물리/allocation 은 plant.py 와 *비트 동일* (같은 B, 같은 timestep, 같은 apply_control)
  - CoM bias 는 body_ipos/body_mass 를 런타임 편집해 물리적으로 주입

주의: xml_path 는 본인 레포의 plant/data/cf21B_500.xml 절대경로로 지정.
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


# ============================================================
# 물리 상수 (레포 params / MJCF 에서 그대로)
# ============================================================
MASS = 0.04338                       # kg  (cf21B_500, MJCF inertial 과 일치)
GRAV = 9.81
ARM = 0.035355                       # m   (plant.py arm_xy)
K_TAU = 0.00594                      # N·m/N (plant.py k_tau)
MOTOR_DIR = np.array([1.0, -1.0, 1.0, -1.0])
THRUST_MIN = 0.0
THRUST_MAX = 0.20                    # N per motor
J_DIAG = np.array([2.3951e-5, 2.3951e-5, 3.2347e-5])   # MJCF diaginertia
PHYSICS_HZ = 500.0


def _build_B():
    """plant.py __init__ 의 allocation B 와 동일하게 구성."""
    a = ARM
    x = np.array([+a, -a, -a, +a])
    y = np.array([-a, -a, +a, +a])
    d = MOTOR_DIR
    k = K_TAU
    B = np.vstack([
        y,            # tau_x
        -x,           # tau_y
        d * k,        # tau_z (reaction torque)
        np.ones(4),   # Fz
    ]).astype(float)
    return B, np.linalg.pinv(B)


def quat_normalize_wxyz(q):
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def rotmat_from_quat_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),    2*(y*z - w*x)],
        [    2*(x*z - w*y),   2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ])


# ============================================================
# Cascade PID  (PID_Cascade.cpp 구조 포팅: pos -> vel -> att -> rate)
#   * 단일 객체. reset() 으로 적분기 초기화, __call__ 로 wrench 산출.
#   * 게인은 43g 기체 안정화용 초기값 — 최종 비교 전 C++ 소스 게인과 reconcile 필요.
# ============================================================
class CascadePID:
    def __init__(self, dt):
        self.dt = dt
        # position -> velocity setpoint
        self.kp_pos = 4.0
        self.v_max = 1.5
        # velocity PID -> accel setpoint
        self.kp_vel = 4.0
        self.ki_vel = 1.0           # (작은 I — "있으나 없으나" 가설 검증용)
        self.kd_vel = 0.0
        # attitude P -> rate setpoint
        self.kp_att = 12.0
        # rate PID -> torque
        self.kp_rate = np.array([0.0008, 0.0008, 0.0006])
        self.ki_rate = np.array([0.0, 0.0, 0.0])
        self.kd_rate = np.array([0.0, 0.0, 0.0])
        # limits (C++ 소스 기본값)
        self.max_tilt = np.deg2rad(35.0)
        self.max_tau = 0.02
        self.max_Fz = 1.0
        self.reset()
        self.dist_torque_body = np.zeros(3)

    def reset(self):
        self._i_vel = np.zeros(3)
        self._i_rate = np.zeros(3)
        #self.data.xfrc_applied[:] = 0.0

    def __call__(self, pos_W, quat_wxyz, vel_W, omega_B, pos_des, yaw_des=0.0):
        R = rotmat_from_quat_wxyz(quat_wxyz)         # body->world
        e_z = np.array([0.0, 0.0, 1.0])

        # 1) position -> velocity setpoint
        v_sp = self.kp_pos * (pos_des - pos_W)
        sp_norm = np.linalg.norm(v_sp)
        if sp_norm > self.v_max:
            v_sp *= self.v_max / sp_norm

        # 2) velocity PID -> desired accel (world)
        e_v = v_sp - vel_W
        self._i_vel += e_v * self.dt
        self._i_vel = np.clip(self._i_vel, -2.0, 2.0)     # anti-windup
        a_sp = self.kp_vel * e_v + self.ki_vel * self._i_vel

        # 3) desired force (world) & collective thrust
        F_des = MASS * (a_sp + GRAV * e_z)
        Fz = float(F_des @ (R @ e_z))                     # body z 투영
        Fz = np.clip(Fz, 0.0, self.max_Fz)

        # 4) desired attitude: b3 = F_des 방향, yaw = yaw_des
        nF = np.linalg.norm(F_des)
        b3 = F_des / nF if nF > 1e-6 else e_z
        # tilt 제한
        cos_t = np.clip(b3 @ e_z, -1.0, 1.0)
        tilt = np.arccos(cos_t)
        if tilt > self.max_tilt:
            # b3 를 e_z 쪽으로 끌어와 tilt 한계로 클램프
            axis = np.cross(e_z, b3)
            an = np.linalg.norm(axis)
            if an > 1e-6:
                axis /= an
                b3 = (np.cos(self.max_tilt) * e_z +
                      np.sin(self.max_tilt) * np.cross(axis, e_z))
                b3 /= np.linalg.norm(b3)
        c_yaw = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        b2 = np.cross(b3, c_yaw)
        b2 /= max(np.linalg.norm(b2), 1e-6)
        b1 = np.cross(b2, b3)
        R_des = np.column_stack([b1, b2, b3])

        # 5) attitude error -> rate setpoint  (e_R = 0.5*vee(R_des^T R - R^T R_des))
        M = R_des.T @ R - R.T @ R_des
        e_R = 0.5 * np.array([M[2, 1], M[0, 2], M[1, 0]])
        rate_sp = -self.kp_att * e_R

        # 6) rate PID -> body torque
        e_w = rate_sp - omega_B
        self._i_rate += e_w * self.dt
        tau = self.kp_rate * e_w + self.ki_rate * self._i_rate
        # gyroscopic feedforward
        tau += np.cross(omega_B, J_DIAG * omega_B)
        tau = np.clip(tau, -self.max_tau, self.max_tau)

        return np.array([tau[0], tau[1], tau[2], Fz])


# ============================================================
# Gymnasium 환경
# ============================================================
class CrazyflieResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 xml_path,
                 policy_hz=100.0,
                 episode_sec=8.0,
                 residual_scale=(0.006, 0.006, 0.0001, 0.3),  # [tau_x,tau_y,tau_z,Fz] 권한 x3 bigger then 4th
                 #residual_scale=(0.004, 0.004, 0.003, 0.05),  # [tau_x,tau_y,tau_z,Fz] 권한
                 #  ^ 검증된 값: tau ~20% of max_tau, Fz ~12% of hover thrust.
                 #    이보다 크면 미숙련 정책이 PID floor 를 파괴함(실측 확인).
                 mode="residual",          # "residual" | "absolute"
                 com_bias_mass=0.010,       # kg (10g)
                 com_bias_offset=(0.035, 0.0),  # (x,y) m, 추 위치
                 com_bias_randomize=False,  # 에피소드 간 랜덤화 여부
                 pos_perturb=0.15,          # reset 시 위치 섭동 [m]
                 seed=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 1.0 / PHYSICS_HZ
        self.dt_phys = 1.0 / PHYSICS_HZ
        self.substeps = int(round(PHYSICS_HZ / policy_hz))
        self.max_steps = int(round(episode_sec * policy_hz))

        self.B, self.B_pinv = _build_B()
        self.drone_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.gyro_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.act_force = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_force") for i in range(4)]
        self.act_torque = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}_torque") for i in range(4)]

        # 원본 inertial 백업 (CoM bias 재설정용)
        self._m0 = float(self.model.body_mass[self.drone_bid])
        self._ipos0 = self.model.body_ipos[self.drone_bid].copy()
        self._J0 = self.model.body_inertia[self.drone_bid].copy()

        self.pid = CascadePID(self.dt_phys)
        self.mode = mode
        self.residual_scale = np.array(residual_scale)
        self.com_bias_mass = com_bias_mass
        self.com_bias_offset = np.array(com_bias_offset)
        self.com_bias_randomize = com_bias_randomize
        self.pos_perturb = pos_perturb
        self.pos_des = np.array([0.0, 0.0, 1.0])     # 목표 hover 위치
        self.yaw_des = 0.0
        self.dist_torque_body = np.zeros(3)


        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        high = np.full(13, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._prev_action = np.zeros(4)        # ← 추가: action 변화율용
        self.w_dact = 0.02                     # ← 추가: 변화율 페널티 가중치 (시작값)

    # ---------- CoM bias 주입 (body_ipos/body_mass 편집) ----------
    def _set_com_bias(self, m_w, off_xy):
        r_w = np.array([off_xy[0], off_xy[1], 0.0])
        M = self._m0 + m_w
        new_ipos = (self._m0 * self._ipos0 + m_w * r_w) / M
        self.model.body_mass[self.drone_bid] = M
        self.model.body_ipos[self.drone_bid] = new_ipos
         # 관성 갱신 (환산질량 mu 로 편심 기여, 대각 근사)
        mu = (self._m0 * m_w) / M
        x, y = r_w[0], r_w[1]
        self.model.body_inertia[self.drone_bid] = self._J0 + mu * np.array([y*y, x*x, x*x + y*y])


    # ---------- allocation (plant.py apply_control 과 동일) ----------
    def _apply_control(self, wrench):
        f = self.B_pinv @ wrench
        f = np.clip(f, THRUST_MIN, THRUST_MAX)
        self._last_f = f.copy()
        for i in range(4):
            self.data.ctrl[self.act_force[i]] = float(f[i])
        tau_m = MOTOR_DIR * K_TAU * f
        for i in range(4):
            self.data.ctrl[self.act_torque[i]] = float(tau_m[i])

    # ---------- state read (plant.py read_state 와 정합) ----------
    def _read_state(self):
        pos = self.data.qpos[0:3].copy()
        quat = quat_normalize_wxyz(self.data.qpos[3:7].copy())
        if quat[0] < 0:                       # q_w >= 0 반구 정규화
            quat = -quat
        vel = self.data.qvel[0:3].copy()      # world linear vel
        omega_B = self.data.sensordata[
            self.model.sensor_adr[self.gyro_sid]:
            self.model.sensor_adr[self.gyro_sid] + 3].copy()   # body gyro (PID 가 쓰는 신호)
        return pos, quat, vel, omega_B

    def _obs(self, pos, quat, vel, omega_B):
        return np.concatenate([pos - self.pos_des, vel, quat, omega_B]).astype(np.float32)

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0

        # CoM bias 설정 (고정 or 에피소드 간 랜덤)
        #if self.com_bias_randomize:
        #    m_w = self._rng.uniform(0.0, self.com_bias_mass)
         #   ang = self._rng.uniform(0, 2 * np.pi)
         #   rad = self._rng.uniform(0.0, np.linalg.norm(self.com_bias_offset))
         #   off = np.array([rad * np.cos(ang), rad * np.sin(ang)])
        #    self._set_com_bias(m_w, off)
        #else:
        #    self._set_com_bias(self.com_bias_mass, self.com_bias_offset)

        if self.com_bias_randomize:
            R = 2.3 * ARM
            m_c, m_e = 0.03, 0.029
            theta = self._rng.uniform(0, 2*np.pi)
            r = R * np.sqrt(self._rng.uniform(0, 1))
            m_w = m_c - (m_c - m_e) * (r / R)
            off = np.array([r*np.cos(theta), r*np.sin(theta)])
            self._set_com_bias(m_w, off)
        else:
            self._set_com_bias(self.com_bias_mass, self.com_bias_offset)

        # 초기 상태: 목표 근방 + 위치 섭동, level, 정지
        self.data.qpos[0:3] = self.pos_des + self._rng.uniform(-self.pos_perturb, self.pos_perturb, 3)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.pid.reset()
        self._step = 0
        self._prev_action = np.zeros(4)        # ← 추가
        pos, quat, vel, omega_B = self._read_state()
        return self._obs(pos, quat, vel, omega_B), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        residual = self.residual_scale * action     # 물리 단위 환산

        # substep 루프: PID 는 매 physics step(500Hz), residual 은 held
        for _ in range(self.substeps):
            pos, quat, vel, omega_B = self._read_state()
            if self.mode == "residual":
                u_pid = self.pid(pos, quat, vel, omega_B, self.pos_des, self.yaw_des)
                u = u_pid + residual
            else:  # absolute: 중력보상 bias 만 깔고 정책이 전체 산출
                u = residual + np.array([0, 0, 0, MASS * GRAV])
            self._apply_control(u)
            R = rotmat_from_quat_wxyz(quat)
            self.data.xfrc_applied[self.drone_bid, 3:6] = R @ self.dist_torque_body
            mujoco.mj_step(self.model, self.data)

        pos, quat, vel, omega_B = self._read_state()
        obs = self._obs(pos, quat, vel, omega_B)

        # ---- reward: quadratic cost (평가지표와 동일 형태) ----
        e_pos = pos - self.pos_des
        d_pos = np.linalg.norm(e_pos)               # added
        e_tilt = 2.0 * (quat[1]**2 + quat[2]**2)      # = 1 - cos(theta), q=[w,x,y,z]
        d_action = action - self._prev_action          # ← action 변화율
        cost = (3.0 * e_pos @ e_pos
                #+ 0.5 * d_pos # ← 추가: 1차 항 (작은 오차에서 gradient 유지)
                + 0.1 * vel @ vel
                + 3.0 * e_tilt
                + 0.001 * omega_B @ omega_B
                + 0.001 * (action @ action)
                + self.w_dact * (d_action @ d_action))   # ← 추가: 부드러운 제어 보상
        reward = -cost
        self._prev_action = action.copy()              # ← prev 갱신 (reward 계산 후)

        self._step += 1
        # 종료: tilt/고도 이탈 or 시간초과
        tilt = np.arccos(np.clip(1 - 2*(quat[1]**2 + quat[2]**2), -1, 1))
        crashed = (pos[2] < 0.2) or (pos[2] > 2.5) or (tilt > np.deg2rad(60)) \
                  or (np.linalg.norm(e_pos) > 1.5)
        truncated = self._step >= self.max_steps
        if crashed:
            reward -= 10.0
        return obs, float(reward), bool(crashed), bool(truncated), {}


if __name__ == "__main__":
    # ---- 자체 검증: PID-only(residual=0) 로 hover 수렴하는지 ----
    xml = "/home/mrl_6534/ros2_ws/src/mujoco_crazyflie/plant/data/cf21B_500.xml"
    env = CrazyflieResidualEnv(xml, mode="residual", com_bias_mass=0.0)  # bias 없이 floor 확인
    obs, _ = env.reset(seed=0)
    print(f"obs dim = {obs.shape}, substeps = {env.substeps}, max_steps = {env.max_steps}")
    errs = []
    for t in range(env.max_steps):
        obs, r, term, trunc, _ = env.step(np.zeros(4))   # residual=0 -> 순수 PID
        errs.append(np.linalg.norm(obs[0:3]))            # 위치 오차 norm
        if term:
            print(f"  CRASHED at step {t}")
            break
    errs = np.array(errs)
    print(f"PID-only hover: pos_err  start={errs[0]:.3f}  end={errs[-1]:.4f}  min={errs.min():.4f}")


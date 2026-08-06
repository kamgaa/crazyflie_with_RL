import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from crazyflie_residual_env import CrazyflieResidualEnv

XML = "/home/mrl_6534/ros2_ws/src/mujoco_crazyflie/plant/data/cf21B_500.xml"
#BIAS = 0.020

def make_env():
    return CrazyflieResidualEnv(XML, mode="e2e",
                                com_bias_randomize=False, com_bias_mass=0.0,
                                att_perturb_deg=0.0, pos_perturb=0.05)

def eval_policy(model, n=30, tilt_limit=30.0):
    env = make_env()
    errs, disq, lens = [], 0, []
    for ep in range(n):
        obs, _ = env.reset(seed=1000 + ep)
        ee, tilts, done = [], [], False
        while not done:
            a = model.predict(obs, deterministic=True)[0] if model else np.zeros(4)
            obs, r, term, trunc, _ = env.step(a)
            ee.append(np.linalg.norm(obs[0:3]))
            q = obs[6:10]
            tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0*(q[1]**2 + q[2]**2), -1.0, 1.0)))
            tilts.append(tilt)
            done = term or trunc
        lens.append(len(ee))                          # 생존 스텝
        tail = max(1, int(len(ee)*0.3))
        errs.append(np.mean(ee[-tail:]))
        if np.max(tilts[-tail:]) > tilt_limit:
            disq += 1
    return float(np.mean(errs)), disq, float(np.mean(lens))   # 3개 반환


class EvalCB(BaseCallback):
    def __init__(self, every, save_path):          # floor 인자 제거 (매번 재계산하니 불필요)
        super().__init__()
        self.every = every; self.last = 0
        self.save_path = save_path; self.best = float('inf')

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            cur,   cur_disq, cur_len = eval_policy(self.model)
            floor, _,        flr_len = eval_policy(None)
            #cur,   cur_disq = eval_policy(self.model)   # ← 튜플 언팩
            #floor, _ = eval_policy(None)         # ← floor 의 실격은 무시 (아래 설명)
            #cur   = eval_policy(self.model)   # 같은 seed 집합 → 같은 ξ_k
            #floor = eval_policy(None)         # 동일 ξ_k 에서 PID floor
            if cur < self.best and cur_disq == 0:
                self.best = cur
                self.model.save(self.save_path)
                tag = "WIN★(saved)"
            else:
                tag = "WIN" if cur < floor else "..."
            print(f"  step {self.num_timesteps:>7d} | residual={cur:.4f} "
                  f"| floor={floor:.4f} | surv={cur_len:.0f} | disq={cur_disq}/30 | {tag} "
                  f"({100*(floor-cur)/floor:+.1f}%)")

        return True

venv = DummyVecEnv([make_env])
"""
model = PPO("MlpPolicy", venv, verbose=0, n_steps=2048, batch_size=256,
            learning_rate=3e-4, ent_coef=0.0, clip_range=0.1,
            policy_kwargs=dict(log_std_init=-3.0, net_arch=[64, 64]))
"""
model = PPO("MlpPolicy", venv, verbose=0, n_steps=2048, batch_size=256,
            learning_rate=3e-4, ent_coef=0.003, clip_range=0.1,
            target_kl=0.03,
            tensorboard_log="/home/mrl_6534/gwpark/crazyflie_RL/tb",
            policy_kwargs=dict(log_std_init=-1.5, net_arch=[64, 64])) 


floor, _,flr_len = eval_policy(None)               # PID-only, 10 seed 평균
#print(f"[floor]  PID-only = {floor:.4f} m")
#print(f"[floor]  PID-only (randomized disk, n=10 avg) = {floor:.4f} m")
print(f"[floor]  PID-only (n=30 avg) = {floor:.4f} m, surv={flr_len:.0f}")
#model.learn(total_timesteps=500_000, callback=EvalCB(every=20_000, floor=floor, save_path="/home/mrl_6534/gwpark/crazyflie_RL/model/ppo_best"))
model.learn(total_timesteps=1_000_000,callback=EvalCB(every=20_000, save_path="/home/mrl_6534/gwpark/crazyflie_RL/model/ppo_best"))
model.save("/home/mrl_6534/gwpark/crazyflie_RL/model/ppo_residual_cf")
print("done.")

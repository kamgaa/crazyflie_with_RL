"""
train_ppo.py — residual RL 파이프라인 smoke test (SB3 PPO).

목적: env + cascade PID(baseline) + residual + RL 전 구간이 도는지 *빠르게* 검증.
본 비교(REINFORCE vs A2C vs PPO)는 본인 PyTorch 구현을 같은 Gym API 로 꽂으면 됨
(이 env 는 표준 gymnasium.Env 이므로 별도 수정 불필요).

실행:  python train_ppo.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from crazyflie_residual_env import CrazyflieResidualEnv

XML = "/home/mrl_6534/ros2_ws/src/mujoco_crazyflie/plant/data/cf21B_500.xml" 

def make_env(bias=0.010, randomize=False):
    def _f():
        return CrazyflieResidualEnv(
            XML, mode="residual",
            com_bias_mass=bias,
            com_bias_randomize=randomize,   # True 면 에피소드 간 bias 샘플링(과적합 방지)
        )
    return _f


def eval_policy(model, env, n_ep=5):
    """평가: 정상상태 위치오차(=비교 yardstick)."""
    errs = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=100 + ep)
        ep_err = []
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True) if model else (np.zeros(4), None)
            obs, r, term, trunc, _ = env.step(act)
            ep_err.append(np.linalg.norm(obs[0:3]))
            done = term or trunc
        errs.append(np.mean(ep_err[-int(len(ep_err) * 0.3):]))  # 후반 30% 평균
    return float(np.mean(errs))


if __name__ == "__main__":
    bias = 0.010
    venv = DummyVecEnv([make_env(bias=bias)])

    # residual 출력층을 작게 시작(≈PID 성능에서 출발)하도록 log_std 초기값 낮춤
    model = PPO("MlpPolicy", venv, verbose=1,
                n_steps=2048, batch_size=256, gae_lambda=0.95, gamma=0.99,
                learning_rate=3e-4, ent_coef=0.0,
                policy_kwargs=dict(log_std_init=-2.0, net_arch=[64, 64]))

    eval_env = CrazyflieResidualEnv(XML, mode="residual", com_bias_mass=bias)

    base = eval_policy(None, eval_env)        # residual=0 → 순수 PID floor
    print(f"\n[floor] PID-only ({int(bias*1000)}g bias) steady-state pos_err = {base:.4f} m")

    model.learn(total_timesteps=30_000, progress_bar=False)

    learned = eval_policy(model, eval_env)
    print(f"[learned] PID+residual steady-state pos_err = {learned:.4f} m")
    print(f"[delta] residual 이 줄인 오차: {base - learned:+.4f} m "
          f"({100*(base-learned)/max(base,1e-9):+.1f}%)")
    model.save("/home/mrl_6534/gwpark/crazyflie_RL/model/ppo_residual_cf")


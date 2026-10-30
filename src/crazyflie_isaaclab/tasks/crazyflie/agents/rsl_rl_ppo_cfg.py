"""Phase-3 fixed PPO configuration; no observation or reward normalization."""
import math
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class CrazyfliePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = 'cuda:0'
    num_steps_per_env = 32
    max_iterations = 50
    init_at_random_ep_len = False
    save_interval = 50
    experiment_name = 'crazyflie_phase3'
    obs_groups = {'actor': ['policy'], 'critic': ['policy']}
    clip_actions = 1.0
    logger = 'tensorboard'
    actor = RslRlMLPModelCfg(
        hidden_dims=[64,64], activation='tanh', obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=math.exp(-1.5),std_type='log'))
    critic = RslRlMLPModelCfg(hidden_dims=[64,64],activation='tanh',obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.003, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3e-4, schedule='fixed', gamma=0.99, lam=0.95,
        desired_kl=None, max_grad_norm=1.0, optimizer='adam',
        normalize_advantage_per_mini_batch=False)

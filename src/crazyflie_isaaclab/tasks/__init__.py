import gymnasium as gym

gym.register(
    id="MRL-Crazyflie-Hover-E2E-v0",
    entry_point="crazyflie_isaaclab.tasks.crazyflie.crazyflie_env:CrazyflieEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "crazyflie_isaaclab.tasks.crazyflie.crazyflie_env_cfg:CrazyflieEnvCfg",
            "rsl_rl_cfg_entry_point": "crazyflie_isaaclab.tasks.crazyflie.agents.rsl_rl_ppo_cfg:CrazyfliePPORunnerCfg",
            "default_agent": "rsl_rl"},
)

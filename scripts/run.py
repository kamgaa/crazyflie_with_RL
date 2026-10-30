"""Launch the registered task without an optimizer. --validate runs physical checks."""
import argparse
from pathlib import Path
import sys
import warp as wp
wp.config.enable_backward = False
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from isaaclab.app import add_launcher_args, launch_simulation
parser=argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('--task',default='MRL-Crazyflie-Hover-E2E-v0')
parser.add_argument('--physics',choices=['isaacsim_physx'],default='isaacsim_physx')
parser.add_argument('--num_envs',type=int,default=1)
parser.add_argument('--steps',type=int,default=1000)
parser.add_argument('--reward',choices=['A','B'],default='A')
parser.add_argument('--profile',action='store_true')
parser.add_argument('--validate',action='store_true')
parser.add_argument('--output',default='results/latest')
add_launcher_args(parser)
parser.add_argument('--headless',action='store_true',help='Alias for --visualizer none')
args=parser.parse_args()
if args.headless:
    args.visualizer = None
    args.visualizer_explicit = True
import gymnasium as gym
import torch
import crazyflie_isaaclab.tasks
from isaaclab_tasks.utils import parse_env_cfg
cfg=parse_env_cfg(args.task,device=args.device,num_envs=args.num_envs,overrides=['physics=isaacsim_physx'])
cfg.reward_variant=args.reward
with launch_simulation(cfg=cfg,launcher_args=args) as launched_cfg:
    cfg.sim.physics = launched_cfg
    env=gym.make(args.task,cfg=cfg).unwrapped
    try:
        env.reset(seed=42)
        print('ACTUAL_BACKEND',env.sim.physics_manager.__module__,env.sim.physics_manager.__name__,flush=True)
        if args.profile:
            from profile_gpu import profile
            profile(env,Path(args.output))
        elif args.validate:
            from validate_physics import validate
            validate(env,Path(args.output))
        else:
            action=torch.zeros((env.num_envs,4),device=env.device)
            for _ in range(args.steps):
                if not env.sim.is_headless_or_exist_active_visualizer():
                    break
                env.step(action)
            print('NONLEARNING_RUN_COMPLETE',env.total_policy_calls,env.total_motor_calls,flush=True)
    finally:
        env.close()

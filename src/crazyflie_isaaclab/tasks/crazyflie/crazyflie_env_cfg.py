from pathlib import Path
import gymnasium as gym
import numpy as np
from isaaclab_physx.physics import PhysxCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_tasks.utils import PresetCfg

@configclass
class CrazyfliePhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg()
    default = isaacsim_physx

@configclass
class CrazyflieEnvCfg(DirectRLEnvCfg):
    decimation = 5
    episode_length_s = 8.0
    action_space = gym.spaces.Box(-1.0,1.0,(4,),dtype=np.float32)
    observation_space = 15
    state_space = 0
    seed = 42
    reward_variant: str = 'A'
    reset_position_range: float = 0.0
    target: tuple = (0.0,0.0,1.0)
    yaw_target: float = 0.0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=.002, render_interval=5, gravity=(0,0,-9.81), physics=CrazyfliePhysicsCfg())
    robot_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/CF21B',
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).resolve().parents[2]/'assets/cf21b.usda'),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.0, angular_damping=0.0, max_linear_velocity=1000.0,
                max_angular_velocity=1000.0, enable_gyroscopic_forces=True,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=8, solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,1),rot=(0,0,0,1)))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True)

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(.35,-.45,1.3),lookat=(0,0,1))

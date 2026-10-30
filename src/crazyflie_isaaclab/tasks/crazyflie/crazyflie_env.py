"""CF21B DirectRLEnv on explicit Isaac Sim PhysX, E2E wrench control."""
import torch
import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from .constants import ACTION_SCALE, BODY_MASS, GRAVITY, THRUST_MAX, ROTOR_POS, COM
from .allocation import allocation_inverse, rotor_wrench
from .actuator_model import MotorModel
from .observations_rewards import state_observation, failures, reward

class CrazyflieEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.reward_variant not in ('A','B'):
            raise ValueError('reward_variant must be A or B')
        super().__init__(cfg, render_mode, **kwargs)
        if self.sim.physics_manager.__module__ != 'isaaclab_physx.physics.physx_manager':
            raise RuntimeError('This task requires isaacsim_physx')
        if self._physics_handles_decimation:
            raise RuntimeError('Motor state requires external 500 Hz substeps')
        self.motors = MotorModel(self.num_envs, self.device)
        self.b_inv = allocation_inverse(self.device)
        self.rotor_arms = torch.tensor(ROTOR_POS,device=self.device)-torch.tensor(COM,device=self.device)
        self.action_scale = torch.tensor(ACTION_SCALE, device=self.device)
        self.target = torch.tensor(cfg.target, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.held_wrench = torch.zeros_like(self.actions)
        self.force_b = torch.zeros((self.num_envs,1,3),device=self.device)
        self.torque_b = torch.zeros_like(self.force_b)
        self.motor_updates = torch.zeros(self.num_envs,dtype=torch.int64,device=self.device)
        self.total_motor_calls = 0
        self.total_policy_calls = 0

    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot_cfg)
        sim_utils.spawn_ground_plane('/World/ground', sim_utils.GroundPlaneCfg())
        pos = cloner.grid_transforms(self.scene.num_envs,self.scene.cfg.env_spacing,device=self.device)[0]
        plan = cloner.clone_plan_from_env_0('/World/envs/env_0','/World/envs/env_{}',
            self.scene.num_envs,self.device,pos,global_paths=('/World/ground',))
        cloner.replicate(plan,stage=self.scene.stage)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects['crazyflie'] = self.robot
        light = sim_utils.DistantLightCfg(intensity=2500.0,color=(1.0,1.0,1.0))
        light.func('/World/Light',light)

    def _pre_physics_step(self, actions):
        if actions.shape != (self.num_envs,4):
            raise ValueError(f'Expected {(self.num_envs,4)}, got {actions.shape}')
        torch._assert_async(torch.isfinite(actions).all(), 'Non-finite Crazyflie action')
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1,1))
        self.held_wrench.copy_(self.actions*self.action_scale)
        self.held_wrench[:,3] += BODY_MASS*GRAVITY
        # Allocation and inverse ESC are constant over the held policy command.
        self.motors.set_command((self.held_wrench @ self.b_inv.T).clamp(0,THRUST_MAX))
        self.total_policy_calls += 1

    def _apply_action(self):
        self.motors.advance()
        self.set_rotor_forces(self.motors.force,self.motors.reaction)
        self.motor_updates += 1
        self.total_motor_calls += 1

    def set_rotor_forces(self, force, reaction):
        """Apply actual rotor forces, summed about COM; also used by physics diagnostics."""
        f,t = rotor_wrench(force,reaction,self.rotor_arms)
        self.force_b.copy_(f)
        self.torque_b.copy_(t)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self.force_b,torques=self.torque_b,is_global=False)

    def policy_observation(self):
        d = self.robot.data
        return state_observation(d.root_link_pos_w.torch,d.root_link_lin_vel_w.torch,
            d.root_link_quat_w.torch,d.root_link_ang_vel_w.torch,
            self.scene.env_origins,self.target,self.cfg.yaw_target)

    def _get_observations(self):
        return {'policy': self.policy_observation()}

    def _get_dones(self):
        obs = self.policy_observation()
        altitude = self.robot.data.root_link_pos_w.torch[:,2]-self.scene.env_origins[:,2]
        return failures(obs,altitude), self.episode_length_buf >= self.max_episode_length

    def _get_rewards(self):
        return reward(self.policy_observation(),self.reset_terminated,self.cfg.reward_variant)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:,:3] = self.target + self.scene.env_origins[env_ids]
        if self.cfg.reset_position_range:
            pose[:,:3] += (2*torch.rand_like(pose[:,:3])-1)*self.cfg.reset_position_range
        velocity = torch.zeros((len(env_ids),6),device=self.device)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose,env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=velocity,env_ids=env_ids)
        airborne = pose[:,2]-self.scene.env_origins[env_ids,2] > .1
        f = airborne[:,None].float().expand(-1,4)*(BODY_MASS*GRAVITY/4)
        self.motors.reset(env_ids,f)
        self.actions[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.held_wrench[env_ids] = 0
        self.held_wrench[env_ids,3] = BODY_MASS*GRAVITY
        self.motor_updates[env_ids] = 0
        f,t = rotor_wrench(self.motors.force[env_ids],self.motors.reaction[env_ids],self.rotor_arms)
        self.force_b[env_ids],self.torque_b[env_ids] = f,t
        # super reset clears both composer inputs. Replace selected buffers with
        # fresh airborne equilibrium force, exactly as the reference reset does.
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=f,torques=t,env_ids=env_ids,is_global=False)

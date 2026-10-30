"""Focused actual PhysX validation for off-center rigid payload; no policy updates."""
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json
from phase15_payload import install,read_engine
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(conflict_handler='resolve');p.add_argument('--run-dir',required=True);p.add_argument('--physics',default='isaacsim_physx');add_launcher_args(p);p.add_argument('--headless',action='store_true');a=p.parse_args();a.visualizer=None;a.visualizer_explicit=True
r=Path(a.run_dir);cases=json.loads((r/'experiment_plan.json').read_text())['cases']
import torch,gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import crazyflie_isaaclab.tasks
from crazyflie_isaaclab.tasks.crazyflie.constants import *
from crazyflie_isaaclab.tasks.crazyflie.allocation import rotor_wrench
cfg=parse_env_cfg(TASK,device=a.device,num_envs=len(cases),overrides=['physics=isaacsim_physx']);cfg.seed=42;cfg.episode_length_s=30;cfg.reset_position_range=0
checks=[]
def check(name,actual,expected,tol,unit=''):
 x=np.asarray(actual,dtype=float);y=np.asarray(expected,dtype=float);error=float(np.max(np.abs(x-y)))
 checks.append(dict(name=name,max_abs_error=error,tolerance=tol,unit=unit,passed=bool(np.isfinite(x).all() and error<=tol)))
 assert checks[-1]['passed'],checks[-1]
with launch_simulation(cfg=cfg,launcher_args=a) as pc:
 cfg.sim.physics=pc;raw=gym.make(TASK,cfg=cfg).unwrapped
 try:
  evidence=install(raw,cases);dump_json(r/'payload_properties.json',evidence);E=evidence['expected'];N=len(cases);device=raw.device
  def t(x):return torch.tensor(x,dtype=torch.float32,device=device)
  def cpu(x):return x.detach().cpu().numpy().copy()
  def reset(q=None):
   raw.reset(seed=42)
   if q is not None:
    pose=raw.robot.data.root_link_pose_w.torch.clone();pose[:,3:]=t(q);raw.robot.write_root_link_pose_to_sim_index(root_pose=pose)
   raw.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros((N,6),device=device))
  def step():raw.scene.write_data_to_sim();raw.sim.step(render=False);raw.scene.update(DT)
  def forces(f):raw.set_rotor_forces(f,f*t(MOTOR_DIR)*TORQUE_RATIO)
  masses=np.array([z['total_mass_kg'] for z in E]);J=np.array([z['inertia_body_kg_m2'] for z in E]);com=np.array([z['com_body_m'] for z in E])
  reset();check('initial_link_position_error',cpu(raw.policy_observation()[:,:3]),0,2e-6,'m');check('nominal_initial_motor_same_for_all',cpu(raw.motors.omega),1581.521674476,.002,'rad/s')
  forces(torch.zeros((N,4),device=device));step();check('gravity_only_COM_acceleration',cpu(raw.robot.data.root_com_lin_vel_w.torch)/DT,[0,0,-GRAVITY],5e-5,'m/s2')
  reset();f=torch.full((N,4),BODY_MASS*GRAVITY/4,device=device);forces(f)
  tau=cpu(raw.torque_b[:,0]);expected_alpha=np.linalg.solve(J,tau[...,None])[...,0]
  check('payload_moment_sign',tau[:,1],com[:,0]*BODY_MASS*GRAVITY,2e-8,'Nm')
  step();check('nominal_rotors_loaded_angular_acceleration',cpu(raw.robot.data.root_link_ang_vel_w.torch)/DT,expected_alpha,.002,'rad/s2')
  check('nominal_rotors_loaded_COM_acceleration',cpu(raw.robot.data.root_com_lin_vel_w.torch)[:,2]/DT,BODY_MASS*GRAVITY/masses-GRAVITY,5e-5,'m/s2')
  # Rotate body 90deg about world Z: body pitch torque becomes world -X torque.
  reset([0,0,2**-.5,2**-.5]);forces(f);step();R=np.array([[0,-1,0],[1,0,0],[0,0,1]])
  check('rotated_mass_axes_and_body_wrench',cpu(raw.robot.data.root_link_ang_vel_w.torch)/DT,expected_alpha@R.T,.003,'rad/s2')
  # Direct physical hover forces are a diagnostic only, never fed to the evaluated actor.
  reset();eq=np.array([z['equilibrium_rotor_thrust_N'] for z in E]);feas=np.array([z['static_level_rotor_feasible'] for z in E]);eq[~feas]=BODY_MASS*GRAVITY/4;forces(t(eq));step()
  check('physically_feasible_equilibrium_COM_acceleration',cpu(raw.robot.data.root_com_lin_vel_w.torch)[feas]/DT,0,5e-5,'m/s2')
  check('physically_feasible_equilibrium_angular_acceleration',cpu(raw.robot.data.root_link_ang_vel_w.torch)[feas]/DT,0,.002,'rad/s2')
  reset();vel=torch.zeros((N,6),device=device);vel[:,4]=1.;raw.robot.write_root_link_velocity_to_sim_index(root_velocity=vel)
  check('observation_keeps_link_velocity',cpu(raw.policy_observation()[:,3:6]),0,2e-7,'m/s')
  check('COM_link_velocity_difference',cpu(raw.robot.data.root_com_lin_vel_w.torch),np.cross(np.broadcast_to([0,1,0],com.shape),com),2e-7,'m/s')
  reset();m0=raw.total_motor_calls;s0=raw.sim.get_physics_step_count();raw.step(torch.zeros((N,4),device=device));check('motor_updates_per_policy',raw.total_motor_calls-m0,5,0);check('physics_steps_per_policy',raw.sim.get_physics_step_count()-s0,5,0)
  check('gravity_compensation_unchanged',cpu(raw.held_wrench[:,3]),BODY_MASS*GRAVITY,1e-7,'N')
  check('finite_observations',bool(torch.isfinite(raw.policy_observation()).all()),True,0)
  keep=torch.arange(1,N,2,device=device);ids=torch.arange(0,N,2,device=device)
  old=[raw.policy_observation()[keep].clone(),raw.motors.omega[keep].clone(),raw.force_b[keep].clone(),raw.torque_b[keep].clone()]
  raw._reset_idx(ids)
  for key,oldval,newval in zip(['observations','rpm','force','torque'],old,[raw.policy_observation()[keep],raw.motors.omega[keep],raw.force_b[keep],raw.torque_b[keep]]):check('partial_reset_preserves_'+key,cpu(newval),cpu(oldval),0)
  ff,tt=rotor_wrench(raw.motors.force[ids],raw.motors.reaction[ids],raw._payload_arms[ids]);check('partial_reset_uses_payload_COM_torque',cpu(raw.torque_b[ids]),cpu(tt),0)
  mass2,com2,_,J2=read_engine(raw);check('reset_preserves_mass',mass2,evidence['readback']['masses_kg'],0,'kg');check('reset_preserves_COM',com2[:,:3],com,1e-8,'m');check('reset_preserves_inertia',J2,J,2e-10,'kg m2')
  dump_json(r/'physics_verification.json',dict(success=all(x['passed'] for x in checks),checks=checks,optimizer_updates=0,actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__,cases=N))
  print('PAYLOAD_PHYSICS_VERIFIED',len(checks),flush=True)
 except BaseException:
  dump_json(r/'physics_verification_failure.json',dict(error=traceback.format_exc(),checks=checks));raise
 finally:raw.close()

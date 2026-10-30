"""Focused actual PhysX fault timing, scaling, mapping and reset tests; no optimizer."""
import argparse,json,sys,traceback
from pathlib import Path
import numpy as np
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(conflict_handler='resolve');p.add_argument('--run-dir',required=True);p.add_argument('--physics',default='isaacsim_physx');add_launcher_args(p);p.add_argument('--headless',action='store_true');a=p.parse_args();a.visualizer=None;a.visualizer_explicit=True
import torch,gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import crazyflie_isaaclab.tasks
from crazyflie_isaaclab.tasks.crazyflie.constants import *
from crazyflie_isaaclab.tasks.crazyflie.actuator_model import thrust_from_omega
from phase17_motor_fault import Fault,metadata
from phase15_payload import read_engine
r=Path(a.run_dir);plan=json.loads((r/'experiment_plan.json').read_text());cases=plan['cases'];N=len(cases)
cfg=parse_env_cfg(TASK,device=a.device,num_envs=N,overrides=['physics=isaacsim_physx']);cfg.seed=42;cfg.episode_length_s=30;cfg.reset_position_range=0
checks=[]
def cpu(x):return x.detach().cpu().numpy().copy()
def check(name,actual,expected,tol,unit=''):
 x=np.asarray(actual,dtype=float);y=np.asarray(expected,dtype=float);err=float(np.max(np.abs(x-y)));ok=bool(np.isfinite(x).all() and err<=tol);checks.append(dict(name=name,error=err,tolerance=tol,unit=unit,passed=ok));assert ok,checks[-1]
with launch_simulation(cfg=cfg,launcher_args=a) as pc:
 cfg.sim.physics=pc;raw=gym.make(TASK,cfg=cfg).unwrapped
 try:
  fault=Fault(raw,cases,5);raw.reset(seed=42);mass,com,_,J=read_engine(raw)
  check('nominal_mass',mass,TOTAL_MASS,5e-9,'kg');check('nominal_COM',com[:,:3],COM,1e-12,'m');check('nominal_inertia',J,np.diag(INERTIA),5e-12,'kg m2')
  for i,c in enumerate(cases):
   expected=np.ones(4)
   if c['motor_user']:expected[4-c['motor_user']]=c['efficiency'];check('user_motor_mapping_'+c['name'],cpu(fault.desired[i]),expected,3e-8)
  check('reset_efficiency_healthy',cpu(fault.efficiency),1,0);check('reset_motor_nominal_RPM',cpu(raw.motors.omega),1581.521674476,.002,'rad/s')
  original_omega=raw.motors.omega.clone();fault.steps.fill_(2500)
  raw._pre_physics_step(torch.zeros((N,4),device=raw.device));raw._apply_action()
  check('fault_does_not_change_RPM',cpu(raw.motors.omega),cpu(original_omega),.001,'rad/s')
  check('actual_thrust_once',cpu(raw.motors.force),cpu(fault.nominal_thrust*fault.desired),0,'N');check('reaction_once',cpu(raw.motors.reaction),cpu(fault.nominal_reaction*fault.desired),0,'Nm')
  tau=cpu(raw.torque_b[:,0]);alpha=np.linalg.solve(J,tau[...,None])[...,0];az=cpu(raw.force_b[:,0,2])/mass-GRAVITY
  raw.scene.write_data_to_sim();raw.sim.step(render=False);raw.scene.update(DT)
  check('engine_angular_acceleration',cpu(raw.robot.data.root_link_ang_vel_w.torch)/DT,alpha,.003,'rad/s2');check('engine_COM_acceleration',cpu(raw.robot.data.root_com_lin_vel_w.torch)[:,2]/DT,az,5e-5,'m/s2')
  # Integration at the true5s boundary, using original zero action and motors.
  raw.reset(seed=42);s0=raw.sim.get_physics_step_count();m0=raw.total_motor_calls
  for i in range(500):raw.step(torch.zeros((N,4),device=raw.device))
  check('500_policy_steps_is2500_substeps',cpu(fault.steps),2500,0);check('healthy_through5s',cpu(fault.efficiency),1,0);check('no_early_fault',cpu(fault.first_fault_substep),-1,0)
  raw.step(torch.zeros((N,4),device=raw.device));check('first_fault_substep',cpu(fault.first_fault_substep)[1:],2500,0);check('post5s_efficiency',cpu(fault.efficiency),cpu(fault.desired),0)
  check('steps_per_policy',raw.sim.get_physics_step_count()-s0,501*5,0);check('motor_updates_per_policy',raw.total_motor_calls-m0,501*5,0)
  check('nominal_polynomial_unmodified',cpu(fault.nominal_thrust),cpu(thrust_from_omega(raw.motors.omega)),0);check('no_compounding_scale',cpu(raw.motors.force),cpu(fault.nominal_thrust*fault.desired),0)
  ids=torch.arange(1,N,2,device=raw.device);keep=torch.arange(0,N,2,device=raw.device);saved=[x[keep].clone() for x in [fault.steps,fault.efficiency,raw.motors.omega,raw.policy_observation()]];raw._reset_idx(ids)
  for key,old,new in zip(['timer','efficiency','RPM','observation'],saved,[fault.steps[keep],fault.efficiency[keep],raw.motors.omega[keep],raw.policy_observation()[keep]]):check('partial_reset_preserves_'+key,cpu(new),cpu(old),0)
  check('partial_reset_restarts_fault_clock',cpu(fault.steps[ids]),0,0);check('partial_reset_starts_healthy',cpu(fault.efficiency[ids]),1,0)
  check('finite_state',bool(torch.isfinite(raw.policy_observation()).all()),True,0)
  dump_json(r/'physics_verification.json',dict(success=True,checks=checks,cases=metadata(cases),engine_properties=dict(masses=mass.tolist(),com_pose_xyzw=com.tolist(),inertia_body=J.tolist()),optimizer_updates=0,actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__))
  print('FAULT_PHYSICS_VERIFIED',len(checks),flush=True)
 except BaseException:dump_json(r/'physics_validation_failure.json',dict(error=traceback.format_exc(),checks=checks));raise
 finally:raw.close()

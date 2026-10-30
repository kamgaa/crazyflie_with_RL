"""Evaluation-only plant motor effectiveness. Original controller and RPM model unchanged."""
import numpy as np
import torch
from crazyflie_isaaclab.tasks.crazyflie.constants import USER_TO_NATIVE,ROTOR_POS,MOTOR_DIR,TORQUE_RATIO,TOTAL_MASS,GRAVITY,THRUST_MAX,ACTION_SCALE,BODY_MASS,ALLOC_ARM,DT

def metadata(cases):
 B=np.array([[-ALLOC_ARM,-ALLOC_ARM,ALLOC_ARM,ALLOC_ARM],[-ALLOC_ARM,ALLOC_ARM,ALLOC_ARM,-ALLOC_ARM],np.array(MOTOR_DIR)*TORQUE_RATIO,[1,1,1,1]])
 out=[]
 for c in cases:
  eta=np.ones(4)
  if c['motor_user'] is not None:eta[USER_TO_NATIVE[c['motor_user']-1]]=c['efficiency']
  nominal=np.full(4,TOTAL_MASS*GRAVITY/4)/eta;w=B@nominal;a=(w-[0,0,0,BODY_MASS*GRAVITY])/ACTION_SCALE
  out.append(dict(**c,efficiency_native=eta.tolist(),efficiency_user=eta[::-1].tolist(),faulty_rotor_position_body_m=None if c['native_index'] is None else list(ROTOR_POS[c['native_index']]),reaction_sign_body_z=None if c['native_index'] is None else MOTOR_DIR[c['native_index']],effective_max_thrust_native_N=(eta*THRUST_MAX).tolist(),static_hover_nominal_motor_commands_N=nominal.tolist(),static_hover_normalized_action=a.tolist(),static_hover_thrust_feasible=bool(np.all(nominal<=THRUST_MAX)),static_hover_action_feasible=bool(np.all(nominal<=THRUST_MAX) and np.all(np.abs(a)<=1))))
 return out

class Fault:
 def __init__(self,raw,cases,onset_s):
  self.raw=raw;self.details=metadata(cases);self.onset_steps=round(onset_s/DT);assert abs(self.onset_steps*DT-onset_s)<1e-12
  self.desired=torch.tensor([c['efficiency_native'] for c in self.details],device=raw.device,dtype=torch.float32)
  self.steps=torch.zeros(raw.num_envs,device=raw.device,dtype=torch.int64);self.efficiency=torch.ones_like(self.desired)
  self.nominal_thrust=torch.zeros_like(self.desired);self.nominal_reaction=torch.zeros_like(self.desired)
  self.first_fault_substep=torch.full((raw.num_envs,),-1,device=raw.device,dtype=torch.int64)
  self.original_advance=raw.motors.advance;self.original_reset=raw.motors.reset
  raw.motors.advance=self.advance;raw.motors.reset=self.reset
 def advance(self):
  self.original_advance()
  self.nominal_thrust.copy_(self.raw.motors.force);self.nominal_reaction.copy_(self.raw.motors.reaction)
  active=self.steps>=self.onset_steps
  self.efficiency.copy_(torch.where(active[:,None],self.desired,torch.ones_like(self.desired)))
  actual_fault=active & (self.desired<1).any(-1)
  self.first_fault_substep.copy_(torch.where(actual_fault & (self.first_fault_substep<0),self.steps,self.first_fault_substep))
  self.raw.motors.force.mul_(self.efficiency);self.raw.motors.reaction.mul_(self.efficiency)
  self.steps+=1
 def reset(self,ids,force):
  self.original_reset(ids,force);self.steps[ids]=0;self.first_fault_substep[ids]=-1
  self.efficiency[ids]=self.desired[ids] if self.onset_steps==0 else 1.
  self.nominal_thrust[ids]=self.raw.motors.force[ids];self.nominal_reaction[ids]=self.raw.motors.reaction[ids]
  self.raw.motors.force[ids]*=self.efficiency[ids];self.raw.motors.reaction[ids]*=self.efficiency[ids]

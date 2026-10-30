from pathlib import Path
import sys,json,math
import numpy as np
import torch
from checks import Checks
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'migration_phase1_20261008/source'
sys.path.insert(0,str(SOURCE))
from crazyflie_rl.actuators import Cf21bFirstOrderActuatorModel
from crazyflie_rl.controllers import build_allocation_matrix
from crazyflie_rl.config import load_config
from crazyflie_rl.environment import CrazyflieResidualEnv
from crazyflie_isaaclab.tasks.crazyflie.constants import *
from crazyflie_isaaclab.tasks.crazyflie.actuator_model import MotorModel,inverse_thrust,thrust_from_omega
from crazyflie_isaaclab.tasks.crazyflie.allocation import allocation_matrix,allocation_inverse
from crazyflie_isaaclab.tasks.crazyflie.frames import xyzw_to_wxyz,wxyz_to_xyzw,rotate,inverse_rotate
from crazyflie_isaaclab.tasks.crazyflie.observations_rewards import reward,failures
c=Checks(); device='cuda:0'
ref=Cf21bFirstOrderActuatorModel(dt=DT,motor_direction=MOTOR_DIR,thrust_max=THRUST_MAX,positive_branch_min_ratio=BRANCH_MIN)
b,bi=build_allocation_matrix(motor_direction=MOTOR_DIR)
c.close('allocation_matrix',allocation_matrix(device,torch.float64),b,1e-15)
c.close('allocation_pseudoinverse',allocation_inverse(device,torch.float64),bi,1e-11)
rng=np.random.default_rng(42)
a=rng.uniform(-2,2,(256,4));w=np.clip(a,-1,1)*ACTION_SCALE;w[:,3]+=BODY_MASS*GRAVITY
expected=np.clip(w@bi.T,0,THRUST_MAX)
measured=(torch.tensor(w,device=device,dtype=torch.float32)@allocation_inverse(device).T).clamp(0,THRUST_MAX)
c.close('allocation_256_commands',measured,expected,1e-7,'N')
forces=np.r_[[-1,0,1e-7,1.19707567e-6,1.3e-6,.10638945,.289,1],rng.uniform(0,.289,248)].reshape(-1,4)
omega=np.stack([ref.inverse_thrust(x) for x in forces])
target=inverse_thrust(torch.tensor(forces,device=device,dtype=torch.float32))
c.close('inverse_vs_numpy',target,omega,.002,'rad/s')
omegatest=np.array([[-50,0,BRANCH_MIN*GAIN,GAIN],[GAIN+100,500,1500,2300]],float)
c.close('forward_clipping_vs_numpy',thrust_from_omega(torch.tensor(omegatest,device=device)),np.stack([ref.thrust_from_omega(x) for x in omegatest]),1e-12,'N')
c.close('forward_inverse_roundtrip',thrust_from_omega(target),np.where(np.clip(forces,0,.289)>1.19707567e-6,np.clip(forces,0,.289),0),1.3e-6,'N',note='Includes original low-force dead zone')
m=MotorModel(1,device);m.set_command(torch.full((1,4),.289,device=device))
ref.reset()
for i in range(25):
    m.advance();o=ref.apply(np.full(4,.289))
    if i in (0,4,24):
        c.close(f'rpm_step_{i+1}',m.omega[0],o.omega,.002,'rad/s')
        c.close(f'thrust_step_{i+1}',m.force[0],o.f_actual,3e-7,'N')
        c.close(f'reaction_step_{i+1}',m.reaction[0],o.q_actual,2e-9,'Nm')
c.close('tau_50ms',m.omega/GAIN,1-math.exp(-1),1e-6,'fraction')
m=MotorModel(4,device)
refs=[Cf21bFirstOrderActuatorModel(dt=DT,motor_direction=MOTOR_DIR,thrust_max=THRUST_MAX,positive_branch_min_ratio=BRANCH_MIN) for _ in range(4)]
max_w=max_f=max_q=0
for _ in range(100):
    f=rng.uniform(-.1,.4,(4,4));m.set_command(torch.tensor(f,device=device,dtype=torch.float32));m.advance()
    outputs=[r.apply(fi) for r,fi in zip(refs,f)]
    max_w=max(max_w,float(np.max(abs(m.omega.cpu().numpy()-np.stack([x.omega for x in outputs])))))
    max_f=max(max_f,float(np.max(abs(m.force.cpu().numpy()-np.stack([x.f_actual for x in outputs])))))
    max_q=max(max_q,float(np.max(abs(m.reaction.cpu().numpy()-np.stack([x.q_actual for x in outputs])))))
c.close('random_motor_numpy_omega',max_w,0,.003,'rad/s')
c.close('random_motor_numpy_force',max_f,0,8e-7,'N')
c.close('random_motor_numpy_reaction',max_q,0,5e-9,'Nm')
q=torch.tensor([[1,0,0,0],[2**-.5,2**-.5,0,0],[2**-.5,0,2**-.5,0],[2**-.5,0,0,2**-.5]],device=device)
c.close('quaternion_identity_xyz90_roundtrip',xyzw_to_wxyz(wxyz_to_xyzw(q)),q,2e-7)
v=torch.tensor([[1.,2,3]]*4,device=device)
c.close('quaternion_identity_xyz90_rotation',rotate(q,v),[[1,2,3],[1,-3,2],[3,2,-1],[-2,1,3]],1e-6)
c.close('quaternion_inverse_rotation',inverse_rotate(q,rotate(q,v)),v,2e-6)
c.close('quaternion_canonical_sign',xyzw_to_wxyz(-wxyz_to_xyzw(q)),q,2e-7)
for variant in ('A','B'):
    cfg=load_config(SOURCE/f'configs/e2e_train_velocity_ab_{variant.lower()}.yaml')
    env=CrazyflieResidualEnv(config=cfg,seed=42)
    env.reset(seed=42)
    obs,r,term,trunc,_=env.step(np.zeros(4,dtype=np.float32))
    got=reward(torch.tensor(obs[None],device=device),torch.tensor([term],device=device),variant)
    c.close(f'reward_{variant}_vs_mujoco',got,r,1e-7,'reward/policy step')
    env.close()
passed=c.save(ROOT/'results/math',dict(device=device,reference=str(SOURCE),no_optimizer=True))
raise SystemExit(0 if passed else 1)

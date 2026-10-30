"""A/B contracts. Reward is per policy step: no dt multiplier."""
import torch
from .frames import xyzw_to_wxyz, inverse_rotate

def state_observation(pos, vel, quat_xyzw, ang_vel_w, origins, target, yaw_target=0.0):
    q = xyzw_to_wxyz(quat_xyzw)
    e = pos - origins - target
    omega = inverse_rotate(q, ang_vel_w)
    w,x,y,z = q.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)) - yaw_target
    yaw_error = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return torch.cat((e, vel, q, omega, torch.sin(yaw_error)[:,None], torch.cos(yaw_error)[:,None]), -1)

def failures(obs, altitude):
    tilt_cos = (1-2*(obs[:,7]**2+obs[:,8]**2)).clamp(-1,1)
    return ((altitude < .2) | (altitude > 2.5) | (torch.acos(tilt_cos) > torch.pi/3)
            | (torch.linalg.vector_norm(obs[:,:3],dim=-1) > 1.5))

def reward(obs, crashed, variant='A'):
    e, v, omega = obs[:,:3], obs[:,3:6], obs[:,10:13]
    if variant == 'B':
        desired = -4*e
        desired = desired * (1.5 / torch.linalg.vector_norm(desired,dim=-1,keepdim=True).clamp_min(1.5))
        v = v - desired
    yaw = torch.atan2(obs[:,13],obs[:,14])
    return -(10*(e[:,:2]**2).sum(-1)+6*e[:,2]**2+.005*(v*v).sum(-1)
             +6*(obs[:,7]**2+obs[:,8]**2)+.001*(omega*omega).sum(-1)+yaw*yaw+10*crashed.float())

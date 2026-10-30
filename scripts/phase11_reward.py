"""Opt-in axis reward; policy observation and physical control path unchanged."""
import torch
from phase9_reward import components as legacy_components
TERMS=('position_x','position_y','position_z','velocity_x','velocity_y','velocity_z','roll','pitch','yaw','omega_x','omega_y','omega_z','failure')
def desired_velocity(e,cfg=None):
    gain=4. if cfg is None else cfg['return_gain_s_inv']
    limit=1.5 if cfg is None else cfg['return_speed_limit_m_s']
    x=-gain*e
    return x*(limit/x.norm(dim=-1,keepdim=True).clamp_min(limit))
def euler_zyx(q):
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    w,x,y,z=q.unbind(-1)
    roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=torch.asin((2*(w*y-z*x)).clamp(-1,1))
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return torch.stack((roll,pitch,yaw),-1)
def components(o,failed,cfg,velocity_mode=None):
    e=o[:,:3];v=o[:,3:6]-desired_velocity(e,cfg);angles=euler_zyx(o[:,6:10]);om=o[:,10:13]
    p=cfg['position'];wt=cfg['axis_weights'];yaw=torch.atan2(o[:,13],o[:,14])
    return torch.stack([p[a]['weight']*torch.expm1(-e[:,i]**2/p[a]['sigma_m']**2) for i,a in enumerate('xyz')]+[-wt['velocity'][i]*v[:,i]**2 for i in range(3)]+[-wt['attitude'][0]*angles[:,0]**2,-wt['attitude'][1]*angles[:,1]**2,-wt['attitude'][2]*yaw**2]+[-wt['angular_velocity'][i]*om[:,i]**2 for i in range(3)]+[-cfg['failure_weight']*failed.float()],-1)
def apply_reward(raw,cfg):
    assert cfg['reward_id']=='phase11_axis' and not cfg['dt_multiplier']
    raw._phase9_reward_config=cfg
    def adjusted():
        o=raw.policy_observation();f=raw.reset_terminated
        raw._phase9_common_stop=legacy_components(o,f,cfg,'stop').sum(-1)
        raw._phase9_common_return=legacy_components(o,f,cfg,'return').sum(-1)
        return components(o,f,cfg).sum(-1)
    raw._get_rewards=adjusted

"""Explicit phase8 reward; unchanged phase3-7 task defaults."""
import torch
TERMS=('xy','z','velocity','tilt','angular_velocity','yaw','termination')
def components(obs,failed,cfg):
    e,v,w=obs[:,:3],obs[:,3:6],obs[:,10:13]
    xy2=(e[:,:2]**2).sum(-1);z2=e[:,2]**2
    sx=cfg['sigma_xy_m'];sz=cfg['sigma_z_m'];weights=cfg['weights']
    if cfg['position_mode']=='quadratic':
        xy=-weights['xy']*xy2;z=-weights['z']*z2
    elif cfg['position_mode']=='exponential':
        xy=cfg['exponential_weight_xy']*torch.expm1(-xy2/(sx*sx))
        z=cfg['exponential_weight_z']*torch.expm1(-z2/(sz*sz))
    else:raise ValueError(cfg['position_mode'])
    yaw=torch.atan2(obs[:,13],obs[:,14])
    return torch.stack((xy,z,-weights['velocity']*(v*v).sum(-1),
        -weights['tilt']*(obs[:,7]**2+obs[:,8]**2),
        -weights['angular_velocity']*(w*w).sum(-1),-weights['yaw']*yaw*yaw,
        -weights['termination']*failed.float()),-1)
def apply_reward(raw,cfg):
    assert raw.cfg.reward_variant=='A' and not cfg['dt_multiplier']
    raw._phase8_reward_config=cfg
    original=raw._get_rewards
    def adjusted():
        common=original()
        raw._phase8_common_reward=common
        if cfg['position_mode']=='quadratic' and cfg['weights']['angular_velocity']==.001:
            return common # Preserve original arithmetic bit-for-bit for Q.
        return components(raw.policy_observation(),raw.reset_terminated,cfg).sum(-1)
    raw._get_rewards=adjusted


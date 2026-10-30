"""Phase9 exponential position reward; velocity reference changes only in reward."""
import torch
TERMS=('xy','z','velocity','tilt','angular_velocity','yaw','termination')
def desired_velocity(error):
    desired=-4.0*error
    return desired*(1.5/torch.linalg.vector_norm(desired,dim=-1,keepdim=True).clamp_min(1.5))
def components(obs,failed,cfg,velocity_mode=None):
    e,v,w=obs[:,:3],obs[:,3:6],obs[:,10:13]
    mode=velocity_mode or cfg['velocity_mode']
    vr=v if mode=='stop' else v-desired_velocity(e)
    if mode not in ('stop','return'):raise ValueError(mode)
    yaw=torch.atan2(obs[:,13],obs[:,14]);wt=cfg['weights']
    return torch.stack((.100*torch.expm1(-(e[:,:2]**2).sum(-1)/(.10**2)),
        .015*torch.expm1(-e[:,2]**2/(.05**2)),-wt['velocity']*(vr*vr).sum(-1),
        -wt['tilt']*(obs[:,7]**2+obs[:,8]**2),-wt['angular_velocity']*(w*w).sum(-1),
        -wt['yaw']*yaw*yaw,-wt['termination']*failed.float()),-1)
def apply_reward(raw,cfg):
    assert raw.cfg.reward_variant=='A' and cfg['position_mode']=='exponential' and not cfg['dt_multiplier']
    raw._phase9_reward_config=cfg
    def adjusted():
        o=raw.policy_observation();failed=raw.reset_terminated
        stop=components(o,failed,cfg,'stop');ret=components(o,failed,cfg,'return')
        raw._phase9_common_stop=stop.sum(-1);raw._phase9_common_return=ret.sum(-1)
        return raw._phase9_common_stop if cfg['velocity_mode']=='stop' else raw._phase9_common_return
    raw._get_rewards=adjusted

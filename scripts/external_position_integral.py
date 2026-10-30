"""Opt-in batched observation-only virtual-target integral; no plant/reward edits."""
import torch

class PositionIntegral:
    def __init__(self,num_envs,device,enabled=False,gain=(.2,.2,.2),dt=.01,xy_limit=.4,z_limit=.15):
        if dt<=0 or xy_limit<=0 or z_limit<=0 or any(x<0 for x in gain):raise ValueError('invalid integral configuration')
        self.enabled=bool(enabled);self.dt=float(dt);self.xy_limit=float(xy_limit);self.z_limit=float(z_limit)
        self.xi=torch.zeros((num_envs,3),device=device,dtype=torch.float32)
        self.gain=torch.tensor(gain,device=device,dtype=self.xi.dtype)
        self.generation=torch.zeros(num_envs,device=device,dtype=torch.int64)
        self.pending=None;self.completed_intervals=0;self._original_reset=None

    def bind_reset(self,raw):
        if self._original_reset is not None:raise RuntimeError('already bound')
        self._original_reset=raw._reset_idx
        def reset(env_ids):
            self._original_reset(env_ids)
            self.reset(env_ids)
        raw._reset_idx=reset

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.xi[ids]=0;self.generation[ids]+=1
        # Keep pending context: a reset during env.step must invalidate only these rows.

    def prepare(self,observation):
        if self.pending is not None:raise RuntimeError('previous interval not finished')
        if observation.shape!=(len(self.xi),15):raise ValueError('expected [N,15] raw observations')
        torch._assert_async(torch.isfinite(observation).all(),'nonfinite observation')
        used=self.xi.clone() if self.enabled else torch.zeros_like(self.xi)
        corrected=observation.clone();corrected[:,:3]=observation[:,:3]-used
        self.pending=dict(e_actual=observation[:,:3].clone(),xi_used=used,e_policy=corrected[:,:3].clone(),generation=self.generation.clone())
        return corrected

    def finish(self,allocation_before_clip,done=None,paused=False):
        if self.pending is None:raise RuntimeError('missing prepared observation')
        if allocation_before_clip.shape!=(len(self.xi),4):raise ValueError('expected rotor commands [N,4]')
        p=self.pending;self.pending=None
        clipped=((allocation_before_clip<0)|(allocation_before_clip>.289)).any(-1)
        reset_during=self.generation!=p['generation']
        ended=torch.zeros_like(clipped) if done is None else done.bool()
        candidate=p['xi_used']-self.dt*self.gain*p['e_actual']
        norm=candidate[:,:2].norm(dim=-1,keepdim=True)
        projected=candidate.clone();projected[:,:2]*=self.xy_limit/norm.clamp_min(self.xy_limit)
        projected[:,2].clamp_(-self.z_limit,self.z_limit)
        interval_allowed=(~clipped)&self.enabled&(not paused)
        allowed=interval_allowed&(~reset_during)&(~ended)
        interval_next=torch.where(interval_allowed[:,None],projected,p['xi_used'])
        # Actual next-episode state stays zero. Never restore a stale previous-episode xi.
        self.xi.copy_(torch.where((reset_during|ended)[:,None],torch.zeros_like(self.xi),interval_next))
        if not self.enabled:self.xi.zero_()
        if not paused:self.completed_intervals+=1
        out={k:v for k,v in p.items() if k!='generation'}
        out.update(xi_candidate=candidate,xi_projected=projected,xi_next=self.xi.clone(),
          xi_interval_end=interval_next,allocator_clipped=clipped,
          integral_frozen=clipped&self.enabled&(~reset_during)&(~ended)&(not paused),
          integral_update_allowed=allowed,reset_during_interval=reset_during,episode_ended=ended,
          xy_projected=allowed&(norm[:,0]>self.xy_limit),z_projected=allowed&(candidate[:,2].abs()>self.z_limit),
          xy_at_limit=interval_next[:,:2].norm(dim=-1)>=self.xy_limit-1e-6,
          z_at_limit=interval_next[:,2].abs()>=self.z_limit-1e-6)
        return out

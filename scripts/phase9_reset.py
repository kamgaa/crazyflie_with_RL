"""Run-local XYZ-box reset sampler with its own GPU RNG; original defaults preserved."""
import torch
class BoxReset:
    def __init__(self,raw,seed=42):
        assert raw.cfg.reset_position_range==.05
        self.raw=raw;self.generator=torch.Generator(device=raw.device);self.generator.manual_seed(seed)
        self.initial_generator_state=self.generator.get_state().clone()
        self.original=raw._reset_idx;raw._reset_idx=self.reset
        self.count=torch.zeros((),device=raw.device,dtype=torch.int64)
        self.sum=torch.zeros(3,device=raw.device,dtype=torch.float64);self.sum2=self.sum.clone()
        self.minimum=torch.full((3,),float('inf'),device=raw.device);self.maximum=-self.minimum.clone()
        self.per_env=torch.zeros(raw.num_envs,device=raw.device,dtype=torch.int64)
        self.first=None;self.telemetry=None
    def reset(self,env_ids):
        raw=self.raw
        if env_ids is None:env_ids=raw.robot._ALL_INDICES
        n=len(env_ids)
        # Reuse original reset for velocities, motors, histories, composer and episode state.
        # All poses are airborne, so its nominal motor initialisation remains identical.
        raw.cfg.reset_position_range=0.
        try:self.original(env_ids)
        finally:raw.cfg.reset_position_range=.05
        delta=(2*torch.rand((n,3),device=raw.device,generator=self.generator)-1)*.05
        pose=raw.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:,:3]=raw.scene.env_origins[env_ids]+raw.target+delta
        raw.robot.write_root_link_pose_to_sim_index(root_pose=pose,env_ids=env_ids)
        if self.telemetry is not None:
            self.telemetry.ep[env_ids]=0;self.telemetry.common_ep[env_ids]=0
        self.count+=n;self.sum+=delta.double().sum(0);self.sum2+=(delta.double()**2).sum(0)
        if n:
            self.minimum=torch.minimum(self.minimum,delta.min(0).values);self.maximum=torch.maximum(self.maximum,delta.max(0).values)
        self.per_env[env_ids]+=1
        if self.first is None and n==raw.num_envs:self.first=delta.clone()
    def stats(self):
        mean=self.sum/self.count.clamp_min(1);std=(self.sum2/self.count.clamp_min(1)-mean**2).clamp_min(0).sqrt()
        values=torch.cat((self.count[None].double(),self.minimum.double(),self.maximum.double(),mean,std,self.per_env.min()[None].double(),self.per_env.max()[None].double())).cpu().tolist()
        result={'Reset/sample_count':values[0]}
        for j,label in enumerate(['min','max','mean','std']):
            for i,axis in enumerate('xyz'):result[f'Reset/{axis}_{label}_m']=values[1+j*3+i]
        result['Reset/min_count_per_env']=values[-2];result['Reset/max_count_per_env']=values[-1]
        return result

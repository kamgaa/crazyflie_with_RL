"""Read-only policy-step telemetry. No task or optimizer arithmetic is replaced."""
import ast, inspect, textwrap, types
import torch

TERMS=('xy','z','velocity','tilt','angular_velocity','yaw','termination')
REASONS=('min_altitude','max_altitude','max_tilt','max_position_error','timeout','failure')

def reward_terms(obs, terminated):
    e,v,w=obs[:,:3],obs[:,3:6],obs[:,10:13]
    yaw=torch.atan2(obs[:,13],obs[:,14])
    return torch.stack((-10*(e[:,:2]**2).sum(-1),-6*e[:,2]**2,-.005*(v*v).sum(-1),
        -6*(obs[:,7]**2+obs[:,8]**2),-.001*(w*w).sum(-1),-yaw*yaw,-10*terminated.float()),-1)

def snapshot(raw, raw_action, returned_reward):
    o=raw.policy_observation().detach()
    terms=reward_terms(o,raw.reset_terminated)
    terms=terms*o.new_tensor([raw._phase7_position_scale,raw._phase7_position_scale,1,1,1,1,1])
    tilt=torch.acos((1-2*(o[:,7]**2+o[:,8]**2)).clamp(-1,1))*180/torch.pi
    pos=torch.linalg.vector_norm(o[:,:3],dim=-1)
    alloc=raw.held_wrench @ raw.b_inv.T
    reasons=torch.stack((o[:,2]+1<.2,o[:,2]+1>2.5,tilt>60,pos>1.5,raw.reset_time_outs,raw.reset_terminated),-1)
    return dict(common_reward_A=raw._phase7_common_reward.detach(),obs=o,terms=terms,reward=returned_reward.detach(),tilt=tilt,pos=pos,
        raw_action=raw_action,action=raw.actions,allocation=alloc,command=raw.motors.command,
        thrust=raw.motors.force,omega=raw.motors.omega,request=raw.held_wrench,
        actual=torch.cat((raw.torque_b[:,0],raw.force_b[:,0,2:3]),-1),
        reasons=reasons,terminated=raw.reset_terminated,truncated=raw.reset_time_outs)

class Telemetry:
    def __init__(self,raw,detail=False):
        self.raw=raw;self.detail=detail;self.raw_action=torch.zeros_like(raw.actions)
        self.ep=torch.zeros((raw.num_envs,9),device=raw.device,dtype=torch.float64)
        self.common_ep=torch.zeros(raw.num_envs,device=raw.device,dtype=torch.float64)
        self.latest=None;self.rows=[];self.clear_rollout()
        self.original_reward=raw._get_rewards
        raw._get_rewards=self.reward_hook
    def clear_rollout(self):
        self.sums=None;self.maximum=None;self.count=0
        self.common_done=torch.zeros((),device=self.raw.device,dtype=torch.float64)
        self.ep_done=torch.zeros(10,device=self.raw.device,dtype=torch.float64)
        self.causes=torch.zeros(6,device=self.raw.device,dtype=torch.float64)
        self.max_reward_residual=torch.zeros((),device=self.raw.device)
    def reward_hook(self):
        r=self.original_reward()
        s=snapshot(self.raw,self.raw_action,r)
        terms=s['terms'];o=s['obs'];a=s['raw_action'];alloc=s['allocation'];thrust=s['thrust']
        self.max_reward_residual=torch.maximum(self.max_reward_residual,(terms.sum(-1)-r).abs().max())
        values={**{f'RewardStep/{k}':terms[:,i] for i,k in enumerate(TERMS)},'RewardStep/total':r,'Monitor/common_reward_A':s['common_reward_A'],
            'Flight/xy_error_m':torch.linalg.vector_norm(o[:,:2],dim=-1),
            'Flight/z_error_m':o[:,2], 'Flight/abs_z_error_m':o[:,2].abs(),
            'Flight/position_error_m':s['pos'],'Flight/position_squared_error_m2':s['pos']**2,
            'Flight/tilt_deg':s['tilt'],'Flight/body_angular_velocity_rad_s':o[:,10:13].norm(dim=-1),
            'Flight/speed_m_s':o[:,3:6].norm(dim=-1),'Actuator/action_l2':a.norm(dim=-1),
            'Actuator/action_clipping_fraction':(a.abs()>1).float().mean(-1),
            'Actuator/allocation_clipping_fraction':((alloc<0)|(alloc>.289)).float().mean(-1),
            'Actuator/command_lower_fraction':(s['command']<=1e-7).float().mean(-1),
            'Actuator/command_upper_fraction':(s['command']>=.289-1e-7).float().mean(-1),
            'Actuator/actual_upper_fraction':(thrust>=.289-1e-6).float().mean(-1),
            'Flight/wrench_force_residual_N':(s['request'][:,3]-s['actual'][:,3]).abs(),
            'Flight/wrench_torque_residual_Nm':(s['request'][:,:3]-s['actual'][:,:3]).norm(dim=-1)}
        for i in range(4):values[f'Actuator/thrust_native_{i}_N']=thrust[:,i]
        for i in range(4):values[f'Diversity/action_axis_{i}_std']=a[:,i].std(unbiased=False).expand(self.raw.num_envs)
        values['Diversity/position_spread_m']=o[:,:3].var(dim=0,unbiased=False).sum().sqrt().expand(self.raw.num_envs)
        vals=torch.stack(list(values.values()),-1).double()
        self.keys=list(values)
        if self.sums is None:self.sums=vals.sum(0);self.maximum=vals.max(0).values
        else:self.sums+=vals.sum(0);self.maximum=torch.maximum(self.maximum,vals.max(0).values)
        self.count+=self.raw.num_envs
        self.ep[:,:7]+=terms.double();self.ep[:,7]+=r.double();self.ep[:,8]+=1
        done=s['terminated']|s['truncated'];mask=done.double()
        self.common_ep+=s['common_reward_A'].double()
        self.common_done+=(self.common_ep*mask).sum();self.common_ep*=~done
        self.ep_done[:9]+=(self.ep*mask[:,None]).sum(0);self.ep_done[9]+=mask.sum()
        self.causes+=s['reasons'].sum(0)
        self.ep*=~done[:,None]
        if self.detail:self.latest={k:v.clone() for k,v in s.items()}
        return r
    def wrap(self,wrapped):
        original=wrapped.step
        def step(a):
            self.raw_action=a.detach()
            return original(a)
        wrapped.step=step
    def finish_rollout(self):
        arr=torch.cat((self.sums/self.count,self.maximum,self.ep_done,self.causes,self.common_done[None],self.max_reward_residual[None])).cpu().tolist()
        n=len(self.keys);data=dict(zip(self.keys,arr[:n]))
        maxima=dict(zip(self.keys,arr[n:2*n]));ep=arr[2*n:2*n+10];causes=arr[2*n+10:2*n+16]
        data['Flight/position_rmse_m']=data['Flight/position_squared_error_m2']**.5
        for k in ['position_error_m','tilt_deg','body_angular_velocity_rad_s']:data['Flight/'+k+'_max']=maxima['Flight/'+k]
        data['Episode/completed_count']=ep[9]
        for k,v in zip(REASONS,causes):data['Episode/end_'+k+'_count']=v
        if ep[9]:
            data.update({f'RewardEpisode/{k}':ep[i]/ep[9] for i,k in enumerate(TERMS)})
            data['RewardEpisode/total']=data['Episode/return']=ep[7]/ep[9]
            data['MonitorEpisode/common_reward_A']=arr[-2]/ep[9]
            data['Episode/length_steps']=ep[8]/ep[9];data['Episode/length_s']=ep[8]/ep[9]*.01
        data['Validation/reward_sum_max_abs_error']=arr[-1]
        self.clear_rollout()
        return data


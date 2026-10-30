"""Read-only policy-step telemetry. No task or optimizer arithmetic is replaced."""
import ast, inspect, textwrap, types
import torch

TERMS=('xy','z','velocity','tilt','angular_velocity','yaw','termination')
REASONS=('min_altitude','max_altitude','max_tilt','max_position_error','timeout','failure')

from phase9_reward import components,desired_velocity

def snapshot(raw, raw_action, returned_reward):
    o=raw.policy_observation().detach()
    terms=components(o,raw.reset_terminated,raw._phase9_reward_config)
    tilt=torch.acos((1-2*(o[:,7]**2+o[:,8]**2)).clamp(-1,1))*180/torch.pi
    pos=torch.linalg.vector_norm(o[:,:3],dim=-1)
    alloc=raw.held_wrench @ raw.b_inv.T
    reasons=torch.stack((o[:,2]+1<.2,o[:,2]+1>2.5,tilt>60,pos>1.5,raw.reset_time_outs,raw.reset_terminated),-1)
    return dict(common_reward_stop=raw._phase9_common_stop.detach(),common_reward_return=raw._phase9_common_return.detach(),obs=o,terms=terms,reward=returned_reward.detach(),tilt=tilt,pos=pos,
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
        values={**{f'Reward/step/{k}':terms[:,i] for i,k in enumerate(TERMS)},'Reward/step/total':r,'Reward/common_stop':s['common_reward_stop'],'Reward/common_return':s['common_reward_return'],
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
        cfg=self.raw._phase9_reward_config
        xy2=(o[:,:2]**2).sum(-1);z2=o[:,2]**2
        values.update({'Physical/xy_error_squared_m2':xy2,'Physical/z_error_squared_m2':z2,
            'Physical/velocity_squared_m2_s2':(o[:,3:6]**2).sum(-1),
            'Physical/body_omega_squared_rad2_s2':(o[:,10:13]**2).sum(-1),
            'Physical/tilt_quaternion_cost':o[:,7]**2+o[:,8]**2,
            'Physical/wrapped_yaw_squared_rad2':torch.atan2(o[:,13],o[:,14])**2,
            'Physical/failure':s['terminated'].float(),
            'Kernel/xy':torch.exp(-xy2/cfg['sigma_xy_m']**2),
            'Kernel/z':torch.exp(-z2/cfg['sigma_z_m']**2)})
        vd=desired_velocity(o[:,:3]);ve=o[:,3:6]-vd
        values.update({'Velocity/desired_speed_m_s':vd.norm(dim=-1),'Velocity/error_m_s':ve.norm(dim=-1),
            'Velocity/error_squared_m2_s2':(ve*ve).sum(-1),'Velocity/speed_squared_m2_s2':(o[:,3:6]**2).sum(-1)})
        for i,axis in enumerate('xyz'):
            values[f'Velocity/desired_{axis}_m_s']=vd[:,i]
            values[f'Velocity/actual_{axis}_m_s']=o[:,3+i]
            values[f'Velocity/error_{axis}_m_s']=ve[:,i]
        for i in range(3):
            values[f'Physical/body_omega_axis_{i}_squared_rad2_s2']=o[:,10+i]**2
        for i in range(4):
            values[f'Actuator/thrust_native_{i}_N']=thrust[:,i]
            values[f'Actuator/motor_native_{i}_rad_s']=s['omega'][:,i]
            values[f'Actuator/action_axis_{i}_clipping_fraction']=(a[:,i].abs()>1).float()
            values[f'Actuator/allocation_native_{i}_clipping_fraction']=((alloc[:,i]<0)|(alloc[:,i]>.289)).float()
            values[f'Actuator/raw_action_axis_{i}']=a[:,i]
            values[f'Actuator/applied_action_axis_{i}']=s['action'][:,i]

        for i in range(4):values[f'Diversity/action_axis_{i}_std']=a[:,i].std(unbiased=False).expand(self.raw.num_envs)
        values['Diversity/position_spread_m']=o[:,:3].var(dim=0,unbiased=False).sum().sqrt().expand(self.raw.num_envs)
        vals=torch.stack(list(values.values()),-1).double()
        self.keys=list(values)
        if self.sums is None:self.sums=vals.sum(0);self.maximum=vals.max(0).values
        else:self.sums+=vals.sum(0);self.maximum=torch.maximum(self.maximum,vals.max(0).values)
        self.count+=self.raw.num_envs
        self.ep[:,:7]+=terms.double();self.ep[:,7]+=r.double();self.ep[:,8]+=1
        done=s['terminated']|s['truncated'];mask=done.double()
        self.common_ep+=s['common_reward_stop'].double()
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
            data.update({f'Reward/episode/{k}':ep[i]/ep[9] for i,k in enumerate(TERMS)})
            data['Reward/episode/total']=data['Episode/return']=ep[7]/ep[9]
            data['Reward/episode_common_stop']=arr[-2]/ep[9]
            data['Episode/length_steps']=ep[8]/ep[9];data['Episode/length_s']=ep[8]/ep[9]*.01
        cfg=self.raw._phase9_reward_config
        data.update({'RewardConfig/sigma_xy_m':cfg['sigma_xy_m'],'RewardConfig/sigma_z_m':cfg['sigma_z_m'],
            'RewardConfig/exponential_weight_xy':cfg['exponential_weight_xy'],
            'RewardConfig/exponential_weight_z':cfg['exponential_weight_z'],
            'RewardConfig/omega_coefficient':cfg['weights']['angular_velocity'],
            'RewardConfig/exponential_active':float(cfg['position_mode']=='exponential'),
            'Flight/body_omega_rms_rad_s':data['Physical/body_omega_squared_rad2_s2']**.5})
        data['Validation/reward_sum_max_abs_error']=arr[-1]
        data['Velocity/error_rms_m_s']=data['Velocity/error_squared_m2_s2']**.5
        data['Velocity/speed_rms_m_s']=data['Velocity/speed_squared_m2_s2']**.5
        self.clear_rollout()
        return data


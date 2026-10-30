"""Targeted phase8 math/config/checkpoint checks; zero optimizer updates."""
import argparse,json,copy,types,sys
from pathlib import Path
import torch,numpy as np
from ppo_common import ROOT,dump_json,sha256
from phase8_reward import components,apply_reward
from phase6_common import differences
from phase8_evalmetrics import spectral
from crazyflie_isaaclab.tasks.crazyflie.observations_rewards import reward,state_observation
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir)
plan=json.loads((r/'experiment_plan.json').read_text());cfgs={n:json.loads((r/n/'reward_config.json').read_text()) for n in plan['order']}
checks={};measured={}
def ck(name,value,**evidence):checks[name]=bool(value);measured[name]=evidence;assert value,(name,evidence)
parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)
ck('parent_actual_counters',(parent['iter'],parent['infos']['completed_iterations'],parent['infos']['collected_transitions'],parent['infos']['optimizer_updates'])==(1279,1280,41943040,102400))
ck('parent_actual_adam_steps',len(parent['optimizer_state_dict']['state'])==13 and {int(v['step']) for v in parent['optimizer_state_dict']['state'].values()}=={102400})
ck('parent_rng_available',set(parent['rng_state'])=={'python','numpy','torch_cpu','torch_cuda'})
torch.manual_seed(1729);o=torch.randn((1024,15));o[:,6:10]/=o[:,6:10].norm(dim=-1,keepdim=True);done=torch.arange(1024)%3==0
base=reward(o,done,'A')
for name,cfg in cfgs.items():
    class Fake:
        def __init__(self):self.cfg=types.SimpleNamespace(reward_variant='A');self.reset_terminated=done
        def policy_observation(self):return o
        def _get_rewards(self):return reward(o,done,'A')
    fake=Fake();apply_reward(fake,cfg);got=fake._get_rewards();terms=components(o,done,cfg)
    err=float((got-terms.sum(-1)).abs().max());ck(name+'_sum',err<2e-5,max_error=err)
    ck(name+'_finite',torch.isfinite(got).all())
    ck(name+'_common_A_exact',torch.equal(base,fake._phase8_common_reward))
    if name=='Q_base':ck('Q_exact_original',torch.equal(base,got))
cfg=cfgs['E_position']
zero=torch.zeros((1,15),dtype=torch.float64);zero[:,6]=1;zero[:,14]=1
ck('target_zero',torch.equal(components(zero,torch.tensor([False]),cfg),torch.zeros((1,7),dtype=torch.float64)))
x=torch.zeros(3,dtype=torch.float64,requires_grad=True)
def posf(e):
    obs=zero.clone();obs[0,:3]=e
    return components(obs,torch.tensor([False]),cfg)[0,:2].sum()
hess=torch.autograd.functional.hessian(posf,x)
ck('near_target_hessian',torch.allclose(hess,torch.diag(torch.tensor([-20.,-20.,-12.],dtype=torch.float64)),atol=1e-12,rtol=0),hessian=hess.tolist())
large=zero.repeat(2,1);large[0,:3]=1000;large[1,:3]=-1000
t=components(large,torch.tensor([False,False]),cfg)[:,:2]
ck('saturation_bounds',torch.allclose(t,torch.tensor([[-.1,-.015]]*2,dtype=torch.float64)),values=t.tolist())
small=zero.clone();small[0,:3]=1e-6
ratio=components(small,torch.tensor([False]),cfg)[0,:2]/torch.tensor([-2e-11,-6e-12],dtype=torch.float64)
ck('near_target_quadratic_ratio',torch.allclose(ratio,torch.ones(2,dtype=torch.float64),atol=1e-8,rtol=0),ratio=ratio.tolist())
delta=components(o,done,cfgs['E_omega'])-components(o,done,cfgs['E_position'])
expected=torch.zeros_like(delta);expected[:,4]=-.009*(o[:,10:13]**2).sum(-1)
ck('omega_only_delta',torch.allclose(delta,expected,atol=3e-8,rtol=1e-6),max_error=float((delta-expected).abs().max()))
for n in [1,64,256,1024]:
    origins=torch.zeros(n,3);origins[:,0]=torch.arange(n)*3.;origins[:,1]=torch.arange(n)%16*3.
    e=torch.tensor([.125,-.25,.125]);target=torch.tensor([0.,0.,1.]);q=torch.tensor([0.,0.,0.,1.]).expand(n,4)
    obs=state_observation(origins+target+e,torch.zeros(n,3),q,torch.zeros(n,3),origins,target)
    ck(f'origin_independence_{n}',torch.equal(obs[:,:3],e.expand(n,3)))
    for name,c in cfgs.items():
        full=components(obs,torch.zeros(n,dtype=torch.bool),c)
        ck(f'batch_{name}_{n}',torch.equal(full,full[:1].expand_as(full)))
diff={}
baseagent=json.loads((r/'Q_base/agent_config.json').read_text())
for name in plan['order']:
    ad=differences(baseagent,json.loads((r/name/'agent_config.json').read_text()))
    rd=differences(cfgs['Q_base'],cfgs[name]);diff[name]=dict(agent=ad,reward=rd)
    ck(name+'_agent_only_metadata',{x['path'] for x in ad}<={'run_name'})
ck('E_omega_only_coefficient',[x['path'] for x in differences(cfgs['E_position'],cfgs['E_omega'])]==['weights.angular_velocity'])
ck('E_position_only_position_mode',[x['path'] for x in differences(cfgs['Q_base'],cfgs['E_position'])]==['position_mode'])
times=np.arange(800)/100.
sp=spectral(.02*np.sin(2*np.pi*14*times))
ck('spectral_14Hz',abs(sp['peak_frequency_hz']-14)<1e-12 and abs(sp['band_10_20_power']-.0002)<1e-6,peak=sp['peak_frequency_hz'],band_power=sp['band_10_20_power'])
sp=spectral(.001*np.sin(2*np.pi*3.5*times))
ck('spectral_3_5Hz',abs(sp['peak_frequency_hz']-3.5)<1e-12,peak=sp['peak_frequency_hz'])
dump_json(r/'condition_diff.json',dict(from_Q_base=diff,E_position_to_E_omega=differences(cfgs['E_position'],cfgs['E_omega'])))
dump_json(r/'reward_verification.json',dict(success=all(checks.values()),checks=checks,measured=measured,optimizer_updates=0,device='CPU math; separate CUDA PhysX preflight'))
dump_json(r/'reward_ready.json',dict(success=True,optimizer_updates=0,parent_sha256=sha256(Path(plan['checkpoint']))))
print('REWARD_VERIFIED',len(checks),flush=True)


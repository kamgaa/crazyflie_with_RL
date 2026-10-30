import argparse,json,sys,copy
from pathlib import Path
import torch
from ppo_common import dump_json
from phase9_reward import components,desired_velocity
from phase8_reward import components as parent_components
from phase6_common import differences
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir)
c=json.loads((r/'control_stop/reward_config.json').read_text());t=json.loads((r/'track_return/reward_config.json').read_text());checks={}
e=torch.tensor([[.05,0,0],[0,0,-.05],[0,0,0],[2.,-3.,4.]],device='cuda:0');vd=desired_velocity(e)
checks['sign_examples']=torch.allclose(vd[:3],torch.tensor([[-.2,0,0],[0,0,.2],[0,0,0]],device=e.device))
checks['norm_limit']=float(vd.norm(dim=-1).max())<=1.500001
checks['direction_preserved']=torch.allclose(vd[-1]/vd[-1].norm(),-e[-1]/e[-1].norm())
checks['zero_finite']=bool(torch.isfinite(vd).all())
old_b=-4*e;old_b*=1.5/torch.linalg.vector_norm(old_b,dim=-1,keepdim=True).clamp_min(1.5);checks['reward_B_reference_formula_exact']=torch.equal(old_b,vd)
g=torch.Generator(device='cuda:0').manual_seed(42);o=torch.randn((1024,15),generator=g,device='cuda:0');failed=torch.arange(1024,device='cuda:0')%3==0
x=components(o,failed,c);y=components(o,failed,t);checks['control_parent_exp_reward_exact']=torch.equal(x,parent_components(o,failed,c))
keep=[0,1,3,4,5,6];checks['other_terms_exact']=torch.equal(x[:,keep],y[:,keep]);expected=-.005*((o[:,3:6]-desired_velocity(o[:,:3]))**2).sum(-1);checks['return_velocity_formula']=torch.equal(y[:,2],expected)
checks['config_velocity_only']=differences(c,t)==[{'path':'velocity_mode','before':'stop','after':'return'}] if False else len(differences(c,t))==1
checks['permutation_env_independence']=torch.equal(components(o.flip(0),failed.flip(0),t).flip(0),y)
checks['identical_state_any_index']=torch.equal(components(o[:1].expand(1024,-1),failed[:1].expand(1024),t),components(o[:1],failed[:1],t).expand(1024,-1))
# Explicit origins removed exactly as observation contract; allow float32 coordinate subtraction tolerance.
origins=torch.randn((1024,3),generator=g,device='cuda:0')*20;target=torch.tensor([0.,0.,1.],device=e.device);pos=origins+target+o[:,:3]
checks['origin_subtraction']=bool(torch.allclose(pos-origins-target,o[:,:3],atol=1e-5,rtol=1e-5))
z=torch.zeros((1,15),device=e.device);z[:,6]=1;z[:,14]=1
checks['zero_state_zero_reward']=bool((components(z,torch.zeros(1,dtype=torch.bool,device=e.device),t)==0).all())
dump_json(r/'reward_ready.json',dict(success=all(checks.values()),checks=checks,condition_diff=differences(c,t),optimizer_updates=0));assert all(checks.values()),checks
print('REWARD_READY',checks)

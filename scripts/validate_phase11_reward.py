import json,sys,math
from pathlib import Path
import torch
from phase11_reward import *
from phase9_reward import components as old
from ppo_common import dump_json
r=Path(sys.argv[1]);c=json.loads((r/'reward_config.json').read_text());dev='cuda:0'
g=torch.Generator(device=dev).manual_seed(123)
o=torch.randn((1024,15),generator=g,device=dev)*.2;o[:,6:10]=o[:,6:10]/o[:,6:10].norm(dim=-1,keepdim=True);o[:,13]=.2;o[:,14]=.98
f=torch.zeros(1024,dtype=torch.bool,device=dev);t=components(o,f,c)
z=torch.zeros((1,15),device=dev);z[:,6]=z[:,14]=1
checks={}
checks['zero_reward']=bool((components(z,f[:1],c)==0).all())
checks['failure_minus10_only']=torch.allclose(components(o,~f,c)-t,torch.tensor([0.]*12+[-10.],device=dev).expand_as(t),atol=1e-6)
b=o.clone();b[:,1]+=10
checks['x_independent_of_y']=torch.equal(components(b,f,c)[:,0],t[:,0])
checks['permutation_independent']=torch.equal(components(o.flip(0),f.flip(0),c).flip(0),t)
legacy=old(o,f,c,'return')
checks['velocity_sum_equal']=torch.allclose(t[:,3:6].sum(-1),legacy[:,2],atol=1e-7)
checks['omega_sum_equal']=torch.allclose(t[:,9:12].sum(-1),legacy[:,4],atol=1e-7)
q=torch.tensor([[math.cos(math.pi/4),math.sin(math.pi/4),0,0],[math.cos(math.pi/4),0,math.sin(math.pi/4),0],[math.cos(math.pi/4),0,0,math.sin(math.pi/4)]],device=dev,dtype=torch.float64)
checks['zyx_axis90_wxyz_radians']=torch.allclose(euler_zyx(q),torch.eye(3,device=dev,dtype=torch.float64)*math.pi/2,atol=1e-7)
checks['quaternion_normalization']=torch.allclose(euler_zyx(q*3),euler_zyx(q),atol=1e-7)
e=torch.tensor([[0.,0,0],[.05,0,0],[0,0,-.05],[3,-4,5]],device=dev);v=desired_velocity(e,c)
checks['vdes_sign_zero']=torch.allclose(v[:3],torch.tensor([[0.,0,0],[-.2,0,0],[0,0,.2]],device=dev))
checks['vdes_norm_limit']=bool((v.norm(dim=-1)<=1.500001).all())
checks['finite_negative']=bool(torch.isfinite(t).all() and (t<=0).all())
checks['exact_13_terms_no_tilt']=len(TERMS)==13 and 'tilt' not in TERMS and 'xy' not in TERMS
checks={k:bool(v) for k,v in checks.items()};dump_json(r/'reward_ready.json',dict(success=all(checks.values()),checks=checks,optimizer_updates=0));assert all(checks.values()),checks
print('REWARD_READY',checks)

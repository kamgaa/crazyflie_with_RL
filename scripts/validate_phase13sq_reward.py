"""Focused CUDA formula tests, no optimizer."""
import json,sys,math
from pathlib import Path
import torch
from phase13sq_reward import *
from ppo_common import dump_json
r=Path(sys.argv[1]);c=json.loads((r/'reward_config.json').read_text());dev='cuda:0'
g=torch.Generator(device=dev).manual_seed(123)
o=torch.randn((1024,15),generator=g,device=dev)*.2
o[:,6:10]=o[:,6:10]/o[:,6:10].norm(dim=-1,keepdim=True)
o[:,13]=.2;o[:,14]=.98;f=torch.zeros(1024,dtype=torch.bool,device=dev)
t=components(o,f,c);old=baseline_components(o,f,c);keep=[i for i in range(13) if i!=5]
z=torch.zeros((1,15),device=dev);z[:,6]=z[:,14]=1
checks={}
checks['zero_reward']=bool((components(z,f[:1],c)==0).all())
checks['failure_minus10_only']=torch.allclose(components(o,~f,c)-t,torch.tensor([0.]*12+[-10.],device=dev).expand_as(t),atol=1e-6)
checks['only_velocity_z_changed']=torch.equal(t[:,keep],old[:,keep])
checks['absolute_vz_squared_exact']=torch.equal(t[:,5],-.005*o[:,5].square())
b=o.clone();b[:,:3]+=torch.tensor([.5,-.7,1.2],device=dev)
checks['vertical_term_independent_of_position_and_vdes']=torch.equal(components(b,f,c)[:,5],t[:,5])
b=o.clone();b[:,5]*=-1
checks['abs_squared_equals_squared']=torch.equal(t[:,5],-.005*o[:,5].abs().square())
checks['even_vz_sign']=torch.equal(components(b,f,c)[:,5],t[:,5])
b=o.clone();b[:,5]*=2
checks['doubling_vz_multiplies_penalty4']=torch.allclose(components(b,f,c)[:,5],t[:,5]*4,atol=1e-7)
b=o.clone();b[:,1]+=10
checks['x_position_independent_y']=torch.equal(components(b,f,c)[:,0],t[:,0])
checks['environment_permutation_independent']=torch.equal(components(o.flip(0),f.flip(0),c).flip(0),t)
checks['finite_nonpositive']=bool(torch.isfinite(t).all() and (t<=0).all())
checks['13_terms_no_duplicate_tilt']=len(TERMS)==13 and 'tilt' not in TERMS
v=torch.tensor([0.,.2,-.2,.5,1.,2.],device=dev)
s=z.repeat(len(v),1);s[:,5]=v;s[:,2]=.05
checks['numeric_vz_examples']=torch.allclose(components(s,torch.zeros(len(v),dtype=torch.bool,device=dev),c)[:,5],torch.tensor([0.,-.0002,-.0002,-.00125,-.005,-.02],device=dev),atol=1e-8)
vd=desired_velocity(torch.tensor([[0.,0,0],[.05,0,0],[0,0,-.05],[3,-4,5]],device=dev),c)
checks['unchanged_vdes_3d_norm']=torch.allclose(vd[:3],torch.tensor([[0.,0,0],[-.2,0,0],[0,0,.2]],device=dev)) and bool((vd.norm(dim=-1)<=1.500001).all())
checks['baseline_weights']=c['axis_weights']['velocity']==[.005]*3 and c['axis_weights']['attitude']==[.075,.075,1.]
checks={k:bool(v) for k,v in checks.items()}
dump_json(r/'reward_ready.json',dict(success=all(checks.values()),checks=checks,optimizer_updates=0,
formula='r_velocity_z=-0.005*(world_vz_m_s)**2; desired_z not subtracted',
numeric_vz=v.cpu().tolist(),numeric_reward=(-.005*v.square()).cpu().tolist(),
other_terms_max_error=float((t[:,keep]-old[:,keep]).abs().max())))
assert all(checks.values()),checks
print('REWARD_READY',checks,flush=True)

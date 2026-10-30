import sys,json,copy,shutil,socket,datetime,subprocess,hashlib
from pathlib import Path
import torch
from ppo_common import ROOT,dump_json,sha256
from phase6_common import differences
from phase9_reward import components,desired_velocity
assert socket.gethostname()=='MRL-server'
r=ROOT/'runs'/('phase10_track_return_continue_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
r.mkdir()
for n in ['evidence','evaluation','plots','tensorboard','source_snapshot']:(r/n).mkdir()
p=ROOT/'runs/phase9_hover_return_velocity_20261013_171849/track_return';control=p.parent/'control_stop'
c=torch.load(p/'final.pt',weights_only=False,map_location='cpu');info=c['infos']
assert (info['completed_iterations'],info['collected_transitions'],info['optimizer_updates'],c['iter'])==(1664,54525952,225280,1663)
assert sorted(set(int(s['step']) for s in c['optimizer_state_dict']['state'].values()))==[225280]
assert set(c['rng_state'])=={'python','numpy','torch_cpu','torch_cuda'} and 'reset_generator_state' in c
agent=json.loads((p/'agent_config.json').read_text());reward=json.loads((p/'reward_config.json').read_text())
assert agent['algorithm']==dict(class_name='PPO',num_learning_epochs=5,num_mini_batches=64,learning_rate=.0003,schedule='fixed',gamma=.99,lam=.95,entropy_coef=.003,desired_kl=None,max_grad_norm=1.,optimizer='adam',value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,normalize_advantage_per_mini_batch=False,share_cnn_encoders=False,rnd_cfg=None,symmetry_cfg=None)
assert reward==dict(variant='A',position_mode='exponential',sigma_xy_m=.1,sigma_z_m=.05,exponential_weight_xy=.1,exponential_weight_z=.015,weights=dict(xy=10.,z=6.,velocity=.005,tilt=6.,angular_velocity=.001,yaw=1.,termination=10.),dt_multiplier=False,velocity_mode='return',return_gain_s_inv=4.,return_speed_limit_m_s=1.5)
plan=dict(experiment=str(r),parent=str(p),checkpoint=str(p/'final.pt'),parent_sha256=sha256(p/'final.pt'),control_checkpoint=str(control/'final.pt'),control_sha256=sha256(control/'final.pt'),branch='track_return_continue',seed=42,num_envs=1024,rollout=32,epochs=5,minibatches=64,updates=163840,position_mode='exponential',velocity_mode='return',base_iterations=1664,base_transitions=54525952,base_optimizer_updates=225280,additional_iterations=512,additional_transitions=16777216,final_iterations=2176,final_transitions=71303168,final_optimizer_updates=389120,evaluation_additional_iterations=list(range(0,513,64)),evaluation_horizon_s=30,best_rule=['maximize fixed7 completed count','if below7 maximize mean duration','if7 minimize mean3D RMSE','earliest additional iteration'],random20_not_for_selection=True,reset_rng='restore parent generator then draw new independent XYZ reset; physics/episodes/rollout not restored')
dump_json(r/'experiment_plan.json',plan)
a=copy.deepcopy(agent);a.update(max_iterations=2176,load_run=str(p),load_checkpoint='final.pt',experiment_name='crazyflie_phase10',run_name='track_return_continue')
dump_json(r/'agent_config.json',a);dump_json(r/'reward_config.json',reward)
dump_json(r/'parent_info.json',dict(parent=str(p/'final.pt'),parent_sha256=sha256(p/'final.pt'),control=str(control/'final.pt'),control_sha256=sha256(control/'final.pt'),parent_infos=info,parent_native_iter=c['iter'],parent_keys=list(c),rng_keys=list(c['rng_state']),adam_parameter_states=len(c['optimizer_state_dict']['state']),adam_steps=[225280],environment_restored=False,rollout_restored=False))
dump_json(r/'configuration_plan_diff.json',dict(agent=differences(agent,a),reward=[],environment=[],physical=[]))
for src,dst in [(p/'final.pt',r/'initial.pt'),(p/'final_reference.pt',r/'initial_reference.pt'),(p.parent/'evaluation/initial_positions.json',r/'evaluation/initial_positions.json'),(p.parent/'REPORT_phase9_ko.txt',r/'evidence/REPORT_phase9_ko.txt'),(p.parent/'COMMANDS.txt',r/'evidence/COMMANDS_phase9.txt')]:shutil.copy2(src,dst)
pos=json.loads((r/'evaluation/initial_positions.json').read_text());assert len(pos['fixed'])==7 and len(pos['random20'])==20
expected=[[0,0,0],[.05,0,0],[-.05,0,0],[0,.05,0],[0,-.05,0],[0,0,.05],[0,0,-.05]];assert [x['offset'] for x in pos['fixed']]==expected
dump_json(r/'evaluation/positions_provenance.json',dict(source=str(p.parent/'evaluation/initial_positions.json'),sha256=sha256(r/'evaluation/initial_positions.json'),previously_evaluated=True,OOD=False))
files=[q for d in ['scripts','src','assets'] if (ROOT/d).exists() for q in (ROOT/d).rglob('*') if q.is_file() and '__pycache__' not in q.parts and 'phase10' not in q.name]
dump_json(r/'evidence/project_before_sha256.json',{str(q.relative_to(ROOT)):sha256(q) for q in files})
protected=[p/'final.pt',p/'final_reference.pt',control/'final.pt',control/'final_reference.pt',p.parent/'evaluation/initial_positions.json']
dump_json(r/'evidence/parent_files_sha256.json',{str(q):sha256(q) for q in protected})
for name,cmd in [('git_before.txt',['git','status','--short','--branch']),('git_diff_before.patch',['git','diff']),('processes_before.txt',['ps','-eo','pid,etime,args'])]:
 z=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);(r/'evidence'/name).write_text(z.stdout+z.stderr)
# Focused reward contract checks on GPU, no optimizer.
e=torch.tensor([[0.,0,0],[.05,0,0],[0,0,-.05],[2,-3,4]],device='cuda:0');vd=desired_velocity(e)
o=torch.zeros((4,15),device='cuda:0');o[:,6]=1;o[:,14]=1;o[:,:3]=e
f=torch.zeros(4,dtype=torch.bool,device='cuda:0');x=components(o,f,reward);y=components(o,~f,reward)
checks=dict(zero_reward=bool((x[0]==0).all()),failure_minus10=torch.equal(y-x,torch.tensor([0,0,0,0,0,0,-10.],device='cuda:0').expand(4,-1)),sign_examples=torch.allclose(vd[:3],torch.tensor([[0.,0,0],[-.2,0,0],[0,0,.2]],device='cuda:0')),norm_limit=bool((vd.norm(dim=-1)<=1.500001).all()),direction=torch.allclose(vd[-1]/vd[-1].norm(),-e[-1]/e[-1].norm()),negative_contributions=bool((x<=0).all()),velocity_formula=torch.equal(x[:,2],-.005*(vd*vd).sum(-1)),xy_formula=torch.equal(x[:,0],.1*torch.expm1(-(e[:,:2]**2).sum(-1)/.1**2)),z_formula=torch.equal(x[:,1],.015*torch.expm1(-e[:,2]**2/.05**2)),batch_permutation=torch.equal(components(o.flip(0),f.flip(0),reward).flip(0),x),finite=bool(torch.isfinite(x).all()))
checks={k:bool(v) for k,v in checks.items()};assert all(checks.values()),checks
dump_json(r/'reward_ready.json',dict(success=True,checks=checks,optimizer_updates=0))
(ROOT/'runs/.phase10_current_path').write_text(str(r))
print(r)

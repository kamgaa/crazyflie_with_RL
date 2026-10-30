"""Targeted CUDA unit checks and actual PhysX integration checks, optimizer0."""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json,sha256,model_hash
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(conflict_handler='resolve');p.add_argument('--run-dir',required=True);p.add_argument('--physics',default='isaacsim_physx');add_launcher_args(p);p.add_argument('--headless',action='store_true');a=p.parse_args()
if a.headless:a.visualizer=None;a.visualizer_explicit=True
import torch,gymnasium as gym
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from external_position_integral import PositionIntegral
from phase14z_reward import apply_reward,components
from phase14z_metrics import Telemetry
from phase6_common import jsonable
import crazyflie_isaaclab.tasks
r=Path(a.run_dir);plan=json.loads((r/'experiment_plan.json').read_text());checks={};details={}
def check(name,truth):
 checks[name]=bool(truth)
 if not checks[name]:raise AssertionError(name)
def close(x,y,tol=1e-6):return torch.allclose(x,torch.as_tensor(y,device=x.device,dtype=x.dtype).expand_as(x),atol=tol,rtol=0)
def obs(n=4):
 o=torch.zeros((n,15),device=a.device);o[:,6]=1;o[:,14]=1;return o
try:
 o=obs();o[:,:3]=torch.tensor([[.05,0,0],[0,0,-.05],[0,0,0],[.01,.02,.03]],device=a.device)
 for enabled,gain,label in [(False,(.2,.2,.2),'off'),(True,(0,0,0),'zero_gain')]:
  w=PositionIntegral(4,a.device,enabled,gain);q=w.prepare(o)
  check(label+'_exact_observation',torch.equal(o,q));w.finish(torch.zeros((4,4),device=a.device));check(label+'_zero_state',close(w.xi,0))
 w=PositionIntegral(4,a.device,True);original=o.clone();q=w.prepare(o);rec=w.finish(torch.full((4,4),.1,device=a.device))
 check('source_observation_not_mutated',torch.equal(o,original));check('first_action_uses_xi_k',torch.equal(q,o))
 check('below_target_increases_virtual_z',w.xi[1,2]>0);check('single_update_formula',close(w.xi,-.002*o[:,:3]))
 q=w.prepare(o);check('only_position_channels_changed',torch.equal(q[:,3:],o[:,3:]));check('policy_error_sign',close(q[:,:3],o[:,:3]-w.xi));w.finish(torch.full((4,4),.1,device=a.device))
 w=PositionIntegral(4,a.device,True)
 for _ in range(100):w.prepare(o);w.finish(torch.full((4,4),.1,device=a.device))
 check('one_second_integral',close(w.xi,-.2*o[:,:3]));details['one_second_xi']=w.xi.cpu().tolist()
 old=w.xi.clone();w.prepare(obs());w.finish(torch.zeros((4,4),device=a.device));check('zero_error_keeps_state',torch.equal(w.xi,old))
 o2=obs();o2[:,:3]=torch.tensor([-1000.,-1000.,-1000.],device=a.device);w.reset();w.prepare(o2);w.finish(torch.zeros((4,4),device=a.device))
 check('xy_norm_projection',close(w.xi[:,:2].norm(dim=-1),.4));check('xy_direction_preserved',close(w.xi[:,0],w.xi[:,1]));check('z_limit',close(w.xi[:,2],.15))
 w.reset();alloc=torch.full((4,4),.1,device=a.device);alloc[1,2]=-.01;alloc[2,3]=.30;alloc[3]=torch.tensor([0,.289,0,.289],device=a.device)
 w.prepare(o);rec=w.finish(alloc);check('any_rotor_freezes_all3_only_selected',close(w.xi[1:3],0) and close(w.xi[[0,3]],-.002*o[[0,3],:3]));check('exact_rotor_bounds_not_clipping',not rec['allocator_clipped'][3])
 w.xi.copy_(torch.tensor([.1,.2,.03],device=a.device));old=w.xi.clone();w.reset(torch.tensor([1,3],device=a.device));check('partial_reset_preserves_other_xi',torch.equal(w.xi[[0,2]],old[[0,2]]) and close(w.xi[[1,3]],0))
 w.prepare(o);w.reset(torch.tensor([0],device=a.device));w.finish(torch.zeros((4,4),device=a.device));check('stale_interval_cannot_overwrite_reset',close(w.xi[0],0))
 old=w.xi.clone();count=w.completed_intervals;w.prepare(o);w.finish(torch.zeros((4,4),device=a.device),paused=True);check('pause_no_update',torch.equal(w.xi,old) and w.completed_intervals==count)
 cfg=parse_env_cfg(TASK,device=a.device,num_envs=4,overrides=['physics=isaacsim_physx']);cfg.seed=42;cfg.reset_position_range=0;cfg.episode_length_s=30
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 with launch_simulation(cfg=cfg,launcher_args=a) as pc:
  cfg.sim.physics=pc;raw=gym.make(TASK,cfg=cfg).unwrapped
  try:
   env=RslRlVecEnvWrapper(raw,clip_actions=1.);agent=json.loads((r/'agent_config.json').read_text());runner=OnPolicyRunner(env,copy.deepcopy(agent),log_dir=None,device=a.device)
   cp=Path(plan['checkpoint']);check('checkpoint_sha',sha256(cp)==plan['checkpoint_sha256']);infos=runner.load(str(cp),strict=True,map_location=a.device);policy=runner.get_inference_policy(device=a.device)
   policy.requires_grad_(False);policy.eval();before=model_hash(policy);updates=[0];hook=runner.alg.optimizer.register_step_post_hook(lambda *_:updates.__setitem__(0,updates[0]+1))
   ref=torch.load(plan['reference'],weights_only=True,map_location=a.device);x=ref['observations'];call=lambda z:policy(TensorDict({'policy':z},batch_size=[len(z)]))
   with torch.inference_mode():
    pred=call(x);err=float((pred-ref['deterministic_actions']).abs().max());check('fixed128_reload',err<=1e-6);details['max_reload_action_error']=err
    for enabled,gain,label in [(False,(.2,.2,.2),'off'),(True,(0,0,0),'zero_gain')]:
     z=PositionIntegral(len(x),a.device,enabled,gain).prepare(x);check(label+'_policy_output_exact',torch.equal(call(z),pred))
   rc=json.loads((r/'reward_config.json').read_text());apply_reward(raw,rc);tel=Telemetry(raw,detail=True);tel.wrap(env)
   w=PositionIntegral(4,a.device,True);w.bind_reset(raw);raw.reset(seed=42);raw.obs_buf=raw._get_observations();o=env.get_observations()['policy']
   w.xi.copy_(torch.tensor([.1,.05,.02],device=a.device));target=raw.target.clone();e=raw.policy_observation().clone();rew=raw._get_rewards().clone();fail=raw._get_dones()[0].clone()
   q=w.prepare(o)
   check('virtual_change_keeps_env_obs_target',torch.equal(raw.policy_observation(),e) and torch.equal(raw.target,target))
   check('virtual_change_keeps_reward_failure',torch.equal(raw._get_rewards(),rew) and torch.equal(raw._get_dones()[0],fail))
   w.finish(torch.zeros((4,4),device=a.device),paused=True)
   old=w.xi.clone();raw.target.add_(torch.tensor([.01,-.01,0],device=a.device));check('target_change_keeps_xi',torch.equal(old,w.xi));raw.target.copy_(target)
   original_pre=raw._pre_physics_step;stepno=[0]
   def moving_pre(action):
    original_pre(action);stepno[0]+=1;raw.target.copy_(target+torch.tensor([.001*stepno[0],0,0],device=a.device))
   raw._pre_physics_step=moving_pre
   raw.reset(seed=42);raw.obs_buf=raw._get_observations();o=env.get_observations()['policy'];n0=raw.total_motor_calls;p0=raw.sim.get_physics_step_count();formula_error=0
   for k in range(20):
    with torch.inference_mode():
     q=w.prepare(o);action=call(q);nxt,_,_,_=env.step(action);snap=tel.latest;rec=w.finish(snap['allocation'],snap['terminated']|snap['truncated']);o=nxt['policy']
    expected=rec['xi_used']-.002*rec['e_actual'];allowed=rec['integral_update_allowed']
    if bool(allowed.any()):formula_error=max(formula_error,float((rec['xi_next'][allowed]-expected[allowed]).abs().max()))
   check('actual_physx_5_substeps_one_integral',raw.total_motor_calls-n0==100 and raw.sim.get_physics_step_count()-p0==100 and w.completed_intervals==20)
   check('moving_targets_retain_accumulation',bool(w.xi.abs().max()>1e-5) and formula_error<1e-7);details['actual_integral_formula_max_error']=formula_error
   state=raw.robot.data.root_link_state_w.torch.clone();rpm=raw.motors.omega.clone();old=w.xi.clone();raw._reset_idx(torch.tensor([1],device=a.device))
   check('actual_partial_reset',close(w.xi[1],0) and torch.equal(w.xi[[0,2,3]],old[[0,2,3]]) and torch.equal(raw.motors.omega[[0,2,3]],rpm[[0,2,3]]) and torch.equal(raw.robot.data.root_link_state_w.torch[[0,2,3]],state[[0,2,3]]))
   raw._pre_physics_step=original_pre;raw.target.copy_(target);raw.reset(seed=42);raw.obs_buf=raw._get_observations();o=env.get_observations()['policy']
   w.prepare(o);large=torch.zeros((4,4),device=a.device);large[:,3]=2
   with torch.inference_mode():env.step(large)
   snap=tel.latest;rec=w.finish(snap['allocation'],snap['terminated']|snap['truncated'])
   check('action_clipping_alone_does_not_freeze',bool((snap['raw_action'][:,3]>1).all()) and bool(rec['integral_update_allowed'].all()) and not bool(rec['allocator_clipped'].any()))
   raw.reset(seed=42);raw.obs_buf=raw._get_observations();o=env.get_observations()['policy'];w.xi[:]=.02
   raw.episode_length_buf[0]=raw.max_episode_length-1
   q=w.prepare(o)
   with torch.inference_mode():nxt,_,_,_=env.step(call(q))
   snap=tel.latest;rec=w.finish(snap['allocation'],snap['terminated']|snap['truncated'])
   check('actual_timeout_autoreset_wins',bool(snap['truncated'][0]) and bool(rec['reset_during_interval'][0]) and close(w.xi[0],0))
   check('nonreset_environment_integral_preserved',bool(w.xi[1:].abs().max()>0))
   # Force a physical-failure state; capture is before automatic reset.
   raw.reset(seed=42);pose=raw.robot.data.default_root_pose.torch.clone();pose[:,:3]=raw.target+raw.scene.env_origins;pose[0,2]=raw.scene.env_origins[0,2]+.1;raw.robot.write_root_link_pose_to_sim_index(root_pose=pose)
   raw.obs_buf=raw._get_observations();o=env.get_observations()['policy'];w.xi[:]=.03;w.prepare(o)
   with torch.inference_mode():env.step(torch.zeros((4,4),device=a.device))
   snap=tel.latest;rec=w.finish(snap['allocation'],snap['terminated']|snap['truncated'])
   check('physical_failure_terminal_snapshot_before_reset',bool(snap['terminated'][0]) and snap['obs'][0,2]<-.7 and raw.policy_observation()[0,2].abs()<1e-4)
   check('physical_failure_integral_zero_after_reset',close(w.xi[0],0))
   check('frozen_actor_distribution_and_optimizer0',model_hash(policy)==before and updates[0]==0)
   details['actual_backend']=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__;details['num_envs']=4
   dump_json(r/'functional_verification.json',dict(success=all(checks.values()),checks=checks,details=details,optimizer_updates=0));print('FUNCTIONAL_OK',len(checks),flush=True)
   hook.remove()
  finally:raw.close()
 dump_json(r/'functional_verification.json',dict(success=all(checks.values()),checks=checks,details=details,optimizer_updates=0));print('FUNCTIONAL_OK',len(checks),flush=True)
except BaseException:
 dump_json(r/'functional_verification.json',dict(success=False,checks=checks,details=details,error=traceback.format_exc(),optimizer_updates=0));raise

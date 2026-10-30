"""Phase12 attitude-weight best, unchanged two-lap position-only circle protocol."""
import time
BOOT=time.perf_counter()
import argparse,copy,csv,json,math,os,shlex,sys,traceback
from pathlib import Path
import numpy as np
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json,sha256,model_hash,protected_hashes
from circle_eval_common import reference,load_pid,wrench_to_action,trajectory_metrics
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(conflict_handler='resolve')
p.add_argument('--run-dir',required=True);p.add_argument('--controller',choices=['phase12_best'],required=True)
p.add_argument('--period',type=float,choices=[10.,5.],required=True);p.add_argument('--output')
p.add_argument('--physics',choices=['isaacsim_physx'],default='isaacsim_physx')
add_launcher_args(p);p.add_argument('--headless',action='store_true');args=p.parse_args()
if args.headless:args.visualizer=None;args.visualizer_explicit=True
assert args.device=='cuda:0'
r=Path(args.run_dir).resolve();plan=json.loads((r/'experiment_plan.json').read_text())
out=Path(args.output).resolve() if args.output else r/'cases'/f'{args.controller}_T{args.period:g}'
out.mkdir(parents=True,exist_ok=False)
import torch,gymnasium as gym
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.visualizers import VisualizerCfg
from phase11_metrics import Telemetry,TERMS,REASONS
from phase11_reward import apply_reward
from phase9_evalmetrics import augment_row
from phase6_common import jsonable
import crazyflie_isaaclab.tasks

torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
horizon=2*args.period;steps=round(horizon*100)
refs,vrefs,arefs=reference(np.arange(steps+1)*.01,args.period)
cfg=parse_env_cfg(TASK,device=args.device,num_envs=1,overrides=['physics=isaacsim_physx'])
cfg.seed=42;cfg.episode_length_s=horizon;cfg.target=(1.,0.,1.);cfg.yaw_target=0.;cfg.reset_position_range=0.
cfg.sim.default_visualizer_cfg=VisualizerCfg(eye=(3.,-4.,3.),lookat=(0.,0.,1.))
reward_name='phase12_best'
reward_cfg=json.loads((r/'models'/reward_name/'reward_config.json').read_text())
updates=[0];hook=None;rows=[];engine_dts=[]
with launch_simulation(cfg=cfg,launcher_args=args) as pc:
 cfg.sim.physics=pc;raw=gym.make(TASK,cfg=cfg).unwrapped
 try:
  env=RslRlVecEnvWrapper(raw,clip_actions=1.)
  if args.controller!='PID':
   folder=r/'models'/args.controller;cp=folder/'best.pt'
   agent=json.loads((folder/'agent_config.json').read_text())
   runner=OnPolicyRunner(env,copy.deepcopy(agent),log_dir=None,device=args.device)
   infos=runner.load(str(cp),strict=True,map_location=args.device);policy=runner.get_inference_policy(device=args.device)
   fixed=torch.load(folder/'best_reference.pt',weights_only=True,map_location=args.device)
   with torch.inference_mode():actual=policy(TensorDict({'policy':fixed['observations']},batch_size=[len(fixed['observations'])]))
   error=float((actual-fixed['deterministic_actions']).abs().max());assert error<=1e-6
   before_model=model_hash(policy)
   hook=runner.alg.optimizer.register_step_post_hook(lambda *_:updates.__setitem__(0,updates[0]+1))
   dump_json(out/'reload.json',dict(checkpoint=str(cp),source=plan['models'][args.controller]['source'],sha256=sha256(cp),infos=infos,
    fixed_observations=len(fixed['observations']),max_action_error=error,bitwise_equal=torch.equal(actual,fixed['deterministic_actions']),model_sha256=before_model))
  else:pid=load_pid(r,plan);pid.reset()
  apply_reward(raw,reward_cfg)
  telemetry=Telemetry(raw,detail=True);telemetry.wrap(env)
  original_reward=raw._get_rewards
  def capture_reward():
   reward=original_reward()
   telemetry.latest['link_position_local']=(raw.robot.data.root_link_pos_w.torch-raw.scene.env_origins).clone()
   return reward
  raw._get_rewards=capture_reward
  targets=torch.tensor(refs,dtype=torch.float32,device=args.device)
  raw.target.copy_(targets[0]);raw.reset(seed=42)
  telemetry.ep.zero_();telemetry.common_ep.zero_();telemetry.clear_rollout()
  raw.obs_buf=raw._get_observations();obs=env.get_observations()
  initial=dict(observation=obs['policy'][0].cpu().tolist(),position=raw.robot.data.root_link_pos_w.torch[0].cpu().tolist(),
   env_origin=raw.scene.env_origins[0].cpu().tolist(),motor_omega=raw.motors.omega[0].cpu().tolist())
  assert torch.allclose(obs['policy'][0,:6],torch.zeros(6,device=args.device),atol=1e-7)
  original_pre=raw._pre_physics_step;schedule=[0]
  def moving_pre(action):
   original_pre(action)
   schedule[0]+=1
   raw.target.copy_(targets[schedule[0]])
  raw._pre_physics_step=moving_pre
  target_marker=None
  if not args.headless and args.visualizer:
   from pxr import UsdGeom,Gf,Vt
   stage=raw.scene.stage
   ring=UsdGeom.BasisCurves.Define(stage,'/World/CircleReference')
   ring.CreateTypeAttr('linear');ring.CreateWrapAttr('periodic')
   ringp=reference(np.linspace(0,args.period,180,endpoint=False),args.period)[0]
   ring.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float,x)) for x in ringp]));ring.CreateCurveVertexCountsAttr([len(ringp)])
   ring.CreateWidthsAttr([.005]);ring.CreateDisplayColorAttr([(1.,.55,.05)])
   marker=UsdGeom.Sphere.Define(stage,'/World/MovingTarget');marker.CreateRadiusAttr(.02);marker.CreateDisplayColorAttr([(1.,.1,.1)])
   target_marker=UsdGeom.Xformable(marker).AddTranslateOp()
  import omni.physx
  subscription=omni.physx.get_physx_interface().subscribe_physics_on_step_events(lambda dt:engine_dts.append(float(dt)),pre_step=False,order=0)
  motor_before=raw.total_motor_calls;physics_before=raw.sim.get_physics_step_count();policy_before=raw.total_policy_calls
  init_time=time.perf_counter()-BOOT;wall=time.perf_counter();max_observation_error=0.;max_action_path_error=0.
  for step in range(1,steps+1):
   tick=time.perf_counter()
   # t=(step-1)*dt. PID and actor receive the same current position target; no reference velocity feedforward.
   with torch.inference_mode():
    if args.controller=='PID':
     o=obs['policy'][0].cpu().numpy();pos=o[:3]+refs[step-1]
     pid_wrench=pid(pos,o[6:10],o[3:6],o[10:13],refs[step-1],0.)
     action=torch.tensor(wrench_to_action(pid_wrench),device=args.device,dtype=torch.float32)[None]
    else:action=policy(obs)
    obs,rew,done,extras=env.step(action)
   snap={k:v[0].detach().cpu().tolist() for k,v in telemetry.latest.items()}
   o=snap['obs'];pos=np.asarray(snap['link_position_local']);ref=refs[step];vref=vrefs[step]
   max_observation_error=max(max_observation_error,float(np.max(np.abs(pos-ref-np.asarray(o[:3])))))
   row=dict(step=step,time_s=step*.01,control_reference_time_s=(step-1)*.01,x=float(pos[0]),y=float(pos[1]),z=float(pos[2]),
    target_x=float(ref[0]),target_y=float(ref[1]),target_z=float(ref[2]),target_vx=float(vref[0]),target_vy=float(vref[1]),target_vz=float(vref[2]),
    reference_phase_rad=2*math.pi*step*.01/args.period,position_error_m=snap['pos'],tilt_deg=snap['tilt'],reward=snap['reward'],common_reward_stop=snap['common_reward_stop'],common_reward_return=snap['common_reward_return'],
    terminated=bool(snap['terminated']),truncated=bool(snap['truncated']))
   row.update(zip(('ex','ey','ez','vx','vy','vz','qw','qx','qy','qz','wx','wy','wz','sin_yaw','cos_yaw'),o))
   for key in ['raw_action','action','allocation','command','thrust','omega','request','actual']:
    row.update({f'{key}_{i}':v for i,v in enumerate(snap[key])})
   row.update({f'reward_{k}':snap['terms'][i] for i,k in enumerate(TERMS)})
   row.update({f'end_{k}':bool(v) for k,v in zip(REASONS,snap['reasons'])})
   row['action_clipping_fraction']=sum(abs(v)>1 for v in snap['raw_action'])/4
   row['allocation_clipping_fraction']=sum(v<0 or v>.289 for v in snap['allocation'])/4
   row['radial_error_m']=float(np.linalg.norm(pos[:2])-1)
   theta=row['reference_phase_rad'];row['tangential_error_m']=-math.sin(theta)*o[0]+math.cos(theta)*o[1]
   row['phase_lag_deg']=math.degrees(math.atan2(math.sin(theta-math.atan2(pos[1],pos[0])),math.cos(theta-math.atan2(pos[1],pos[0]))))
   for k,v in zip(['vex','vey','vez'],np.asarray(o[3:6])-vref):row[k]=float(v)
   augment_row(row);rows.append(row)
   expected=np.clip(np.asarray(snap['raw_action']),-1,1)*[.0075,.0075,.001,.5]+[0,0,0,.04338*9.81]
   max_action_path_error=max(max_action_path_error,float(np.abs(expected-np.asarray(snap['request'])).max()))
   if target_marker is not None:target_marker.Set(Gf.Vec3d(*map(float,ref)))
   assert all(math.isfinite(float(v)) for v in row.values())
   if row['terminated'] or row['truncated']:break
   if not args.headless and args.visualizer:time.sleep(max(0,.01-(time.perf_counter()-tick)))
  eval_wall=time.perf_counter()-wall
  motor_calls=raw.total_motor_calls-motor_before;physics_calls=raw.sim.get_physics_step_count()-physics_before
  subscription.unsubscribe()
  metrics=trajectory_metrics(rows,args.period)
  checks=dict(finite_trajectory=all(math.isfinite(float(v)) for row in rows for v in row.values()),
   no_optimizer_updates=updates[0]==0,policy_calls=raw.total_policy_calls-policy_before==len(rows),
   motor_calls_5_per_policy=motor_calls==len(rows)*5,physics_calls_5_per_policy=physics_calls==len(rows)*5,
   engine_events=len(engine_dts)==len(rows)*5,engine_time=abs(sum(engine_dts)-len(rows)*.01)<1e-5,
   target_observation_error=max_observation_error<2e-6,action_to_wrench_error=max_action_path_error<2e-7,
   reward_sum=max(abs(sum(row['reward_'+k] for k in TERMS)-row['reward']) for row in rows)<1e-4,
   first_failure_stop=not any(row['terminated'] or row['truncated'] for row in rows[:-1]),
   checkpoint_model_unchanged=True if args.controller=='PID' else model_hash(policy)==before_model,
   protected_physics_unchanged=protected_hashes()==json.loads((r/'evidence/protected_physics_sha256.json').read_text()))
  result=dict(controller=args.controller,period_s=args.period,radius_m=1.,requested_laps=2,initialization_s=init_time,evaluation_wall_s=eval_wall,
   optimizer_updates=updates[0],initial=initial,metrics=metrics,checks=checks,success=all(checks.values()),
   physics_steps=physics_calls,motor_updates=motor_calls,engine_time_s=sum(engine_dts),engine_dt_range_s=[min(engine_dts),max(engine_dts)],
   observation_error_max_m=max_observation_error,action_to_wrench_error_max_mixed_units=max_action_path_error,
   reward_definition=reward_cfg,reward_sum=sum(row['reward'] for row in rows),common_reward_stop_sum=sum(row['common_reward_stop'] for row in rows),common_reward_return_sum=sum(row['common_reward_return'] for row in rows),
   actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__,command=shlex.join([sys.executable,*sys.argv]),pid=os.getpid())
  dump_json(out/'result.json',result);dump_json(out/'trajectory.json',dict(result=result,trajectory=rows))
  with (out/'trajectory.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  dump_json(out/'resolved_config.json',jsonable(cfg.to_dict()))
  print('CIRCLE_RESULT',json.dumps(result),flush=True)
  assert result['success'],checks
 except Exception:
  dump_json(out/'failure.json',dict(error=traceback.format_exc(),valid_steps=len(rows),optimizer_updates=updates[0]));dump_json(out/'partial_trajectory.json',rows)
  raise
 finally:
  if hook is not None:hook.remove()
  raw.close()

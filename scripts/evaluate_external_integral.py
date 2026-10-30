"""Same frozen phase14 policy: opt-in observation integral, independent task process."""
import time
BOOT=time.perf_counter()
import argparse,copy,csv,json,math,os,sys,shlex,traceback
from pathlib import Path
import numpy as np
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json,sha256,model_hash
from circle_eval_common import reference
from external_integral_metrics import summarize
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(conflict_handler='resolve');p.add_argument('--run-dir',required=True);p.add_argument('--integral',choices=['off','on'],default='off');p.add_argument('--task',choices=['hover','T10','T5'],required=True);p.add_argument('--output');p.add_argument('--physics',choices=['isaacsim_physx'],default='isaacsim_physx');add_launcher_args(p);p.add_argument('--headless',action='store_true');args=p.parse_args()
if args.headless:args.visualizer=None;args.visualizer_explicit=True
r=Path(args.run_dir).resolve();plan=json.loads((r/'experiment_plan.json').read_text());assert json.loads((r/'functional_verification.json').read_text())['success'];assert args.device=='cuda:0'
task=next(t for t in plan['tasks'] if t['name']==args.task);period=task['period_s'];horizon=task['horizon_s'];steps=round(horizon*100)
out=Path(args.output).resolve() if args.output else r/'evaluation'/args.integral/args.task;out.mkdir(parents=True,exist_ok=False)
refs=np.tile([0.,0.,1.],(steps+1,1)) if period is None else reference(np.arange(steps+1)*.01,period)[0]
import torch,gymnasium as gym
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.visualizers import VisualizerCfg
from phase14z_reward import apply_reward,components
from phase14z_metrics import Telemetry,TERMS,REASONS
from phase9_evalmetrics import augment_row
from phase6_common import jsonable
from phase15_payload import read_engine
from external_position_integral import PositionIntegral
import crazyflie_isaaclab.tasks

torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.manual_seed(42);np.random.seed(42)
cfg=parse_env_cfg(TASK,device=args.device,num_envs=1,overrides=['physics=isaacsim_physx']);cfg.seed=42;cfg.episode_length_s=horizon;cfg.target=tuple(map(float,refs[0]));cfg.yaw_target=0.;cfg.reset_position_range=0.
cfg.sim.default_visualizer_cfg=VisualizerCfg(eye=(3,-4,3),lookat=(0,0,1))
rc=json.loads((r/'reward_config.json').read_text());agent=json.loads((r/'agent_config.json').read_text());assert not agent['actor']['obs_normalization']
updates=[0];snaps=[];records=[];controls=[];engine_dts=[];rows=[];hook=None
with launch_simulation(cfg=cfg,launcher_args=args) as pc:
 cfg.sim.physics=pc;raw=gym.make(TASK,cfg=cfg).unwrapped
 try:
  env=RslRlVecEnvWrapper(raw,clip_actions=1.);runner=OnPolicyRunner(env,copy.deepcopy(agent),log_dir=None,device=args.device)
  cp=Path(plan['checkpoint']);assert sha256(cp)==plan['checkpoint_sha256'];infos=runner.load(str(cp),strict=True,map_location=args.device);policy=runner.get_inference_policy(device=args.device);policy.requires_grad_(False);policy.eval()
  ref=torch.load(plan['reference'],weights_only=True,map_location=args.device)
  with torch.inference_mode():act=policy(TensorDict({'policy':ref['observations']},batch_size=[128]))
  err=float((act-ref['deterministic_actions']).abs().max());assert err<=1e-6
  before=model_hash(policy);hook=runner.alg.optimizer.register_step_post_hook(lambda *_:updates.__setitem__(0,updates[0]+1))
  dump_json(out/'reload.json',dict(checkpoint=str(cp),sha256=sha256(cp),selected_iteration=448,infos=infos,reference_count=128,max_action_error=err,policy_state_sha256=before,all_parameters_frozen=all(not q.requires_grad for q in policy.parameters())))
  apply_reward(raw,rc);tel=Telemetry(raw,detail=True);tel.wrap(env)
  orig_reward=raw._get_rewards
  def capture():
   rew=orig_reward();tel.latest['link_position_local']=(raw.robot.data.root_link_pos_w.torch-raw.scene.env_origins).clone();return rew
  raw._get_rewards=capture
  integ=PositionIntegral(1,args.device,enabled=args.integral=='on');integ.bind_reset(raw)
  targets=torch.tensor(refs,dtype=torch.float32,device=args.device);raw.target.copy_(targets[0]);raw.reset(seed=42)
  tel.ep.zero_();tel.common_ep.zero_();tel.clear_rollout();raw.obs_buf=raw._get_observations();obs=env.get_observations()
  mass,com,_,J=read_engine(raw);dump_json(out/'engine_properties.json',dict(masses=mass.tolist(),com_pose_xyzw=com.tolist(),inertia_body=J.tolist()))
  dump_json(out/'initial_state.json',dict(observation=obs['policy'].cpu().tolist(),world_position=raw.robot.data.root_link_pos_w.torch.cpu().tolist(),env_origin=raw.scene.env_origins.cpu().tolist(),motor_omega=raw.motors.omega.cpu().tolist(),target=refs[0].tolist(),xi=integ.xi.cpu().tolist()))
  assert bool((obs['policy'][:,:6].abs()<1e-6).all())
  original_pre=raw._pre_physics_step;counter=[0]
  def moving_pre(action):
   original_pre(action);counter[0]+=1;raw.target.copy_(targets[counter[0]])
  raw._pre_physics_step=moving_pre
  marker_op=None
  if not args.headless and args.visualizer:
   from pxr import UsdGeom,Gf,Vt
   stage=raw.scene.stage
   if period:
    ring=UsdGeom.BasisCurves.Define(stage,'/World/CircleReference');ring.CreateTypeAttr('linear');ring.CreateWrapAttr('periodic')
    ringp=reference(np.linspace(0,period,180,endpoint=False),period)[0];ring.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float,x)) for x in ringp]));ring.CreateCurveVertexCountsAttr([len(ringp)]);ring.CreateWidthsAttr([.005]);ring.CreateDisplayColorAttr([(1,.55,.05)])
   marker=UsdGeom.Sphere.Define(stage,'/World/ActualTarget');marker.CreateRadiusAttr(.02);marker.CreateDisplayColorAttr([(1,.1,.1)]);marker_op=UsdGeom.Xformable(marker).AddTranslateOp()
  import omni.physx
  sub=omni.physx.get_physx_interface().subscribe_physics_on_step_events(lambda dt:engine_dts.append(float(dt)),pre_step=False,order=0)
  motor0=raw.total_motor_calls;physics0=raw.sim.get_physics_step_count();policy0=raw.total_policy_calls;init_s=time.perf_counter()-BOOT;wall=time.perf_counter();step=0
  while step<steps:
   if not raw.sim.is_playing():
    # No policy/physics interval is issued while the GUI timeline is paused.
    raw.sim.render();time.sleep(.01);continue
   tick=time.perf_counter()
   with torch.inference_mode():
    controls.append(obs['policy'].clone());corrected=obs.clone();corrected['policy']=integ.prepare(obs['policy']);action=policy(corrected)
    obs,_,_,_=env.step(action)
    snap={k:v.clone() for k,v in tel.latest.items()};rec=integ.finish(snap['allocation'],snap['terminated']|snap['truncated'])
   snaps.append(snap);records.append({k:v.clone() for k,v in rec.items()});step+=1
   if marker_op is not None:marker_op.Set(Gf.Vec3d(*map(float,refs[step])))
   if bool(snap['terminated'][0]|snap['truncated'][0]):break
   if not args.headless and args.visualizer:time.sleep(max(0,.01-(time.perf_counter()-tick)))
  sim_s=time.perf_counter()-wall;sub.unsubscribe()
  data={k:torch.stack([s[k] for s in snaps]).cpu().numpy()[:,0] for k in snaps[0]};recs={k:torch.stack([s[k] for s in records]).cpu().numpy()[:,0] for k in records[0]};ctrl=torch.stack(controls).cpu().numpy()[:,0]
  for si in range(len(snaps)):
   s={k:v[si] for k,v in data.items()};rec={k:v[si] for k,v in recs.items()};o=s['obs'];pos=s['link_position_local'];endref=refs[si+1];startref=refs[si]
   row=dict(step=si+1,time_s=(si+1)*.01,control_time_s=si*.01,position_error_m=float(s['pos']),tilt_deg=float(s['tilt']),reward=float(s['reward']),terminated=bool(s['terminated']),truncated=bool(s['truncated']))
   for key,vals in [('position',pos),('target',endref),('control_target',startref),('control_position',ctrl[si,:3]+startref),('virtual_target',startref+rec['xi_used'])]:
    row.update({f'{key}_{ax}':float(vals[i]) for i,ax in enumerate('xyz')})
   row.update(zip(('x','y','z'),map(float,pos)));row.update(zip(('ex','ey','ez','vx','vy','vz','qw','qx','qy','qz','wx','wy','wz','sin_yaw','cos_yaw'),map(float,o)))
   row.update({'world_'+ax:float(s['world_position'][i]) for i,ax in enumerate('xyz')})
   for key in ['e_actual','e_policy','xi_used','xi_candidate','xi_projected','xi_next','xi_interval_end']:
    row.update({key+'_'+ax:float(rec[key][i]) for i,ax in enumerate('xyz')})
   for key in ['allocator_clipped','integral_frozen','integral_update_allowed','reset_during_interval','episode_ended','xy_projected','z_projected','xy_at_limit','z_at_limit']:row[key]=bool(rec[key])
   for key in ['raw_action','action','allocation','command','thrust','omega','request','actual']:row.update({f'{key}_{i}':float(v) for i,v in enumerate(s[key])})
   row.update({'reward_'+k:float(s['terms'][i]) for i,k in enumerate(TERMS)});row.update({'end_'+k:bool(v) for k,v in zip(REASONS,s['reasons'])})
   row['action_clipping_fraction']=float(np.mean(np.abs(s['raw_action'])>1));row['allocation_clipping_fraction']=float(np.mean((s['allocation']<0)|(s['allocation']>.289)))
   if period:
    theta=2*math.pi*row['time_s']/period;row['reference_phase_rad']=theta;row['radial_error_m']=float(np.linalg.norm(pos[:2])-1);row['phase_lag_deg']=math.degrees(math.atan2(math.sin(theta-math.atan2(pos[1],pos[0])),math.cos(theta-math.atan2(pos[1],pos[0]))))
   augment_row(row);rows.append(row)
  metrics=summarize(rows,args.task,horizon,period)
  expected=ctrl[:,:3]-recs['xi_used'];obs_error=float(np.abs(expected-recs['e_policy']).max());enderr=float(np.max(np.abs(data['link_position_local']-refs[1:len(rows)+1]-data['obs'][:,:3])))
  expected_req=np.clip(data['raw_action'],-1,1)*[.0075,.0075,.001,.5]+[0,0,0,.04338*9.81];wrench_err=float(np.abs(expected_req-data['request']).max())
  checks=dict(optimizer0=updates[0]==0,frozen_policy_distribution=model_hash(policy)==before,frozen_checkpoint=sha256(cp)==plan['checkpoint_sha256'],
    corrected_observation=obs_error<1e-7,actual_target_alignment=enderr<2e-6,action_path=wrench_err<2e-7,
    integral_once_per_policy=integ.completed_intervals==len(rows),policy_calls=raw.total_policy_calls-policy0==len(rows),
    motor_calls=raw.total_motor_calls-motor0==len(rows)*5,physics_calls=raw.sim.get_physics_step_count()-physics0==len(rows)*5,engine_events=len(engine_dts)==len(rows)*5,engine_time=abs(sum(engine_dts)-len(rows)*.01)<1e-5,
    first_terminal_only=not any(z['terminated'] or z['truncated'] for z in rows[:-1]),finite=all(math.isfinite(float(v)) for z in rows for v in z.values()),
    reward_sum=max(abs(sum(z['reward_'+k] for k in TERMS)-z['reward']) for z in rows)<1e-5,
    reset_final_zero=bool(torch.equal(integ.xi,torch.zeros_like(integ.xi))) if rows[-1]['episode_ended'] else True)
  result=dict(task=args.task,integral=args.integral,label='PPO 단독' if args.integral=='off' else 'PPO + 외부 적분',checkpoint=str(cp),checkpoint_sha256=sha256(cp),selected_iteration=448,metrics=metrics,
    success=all(checks.values()),checks=checks,initialization_s=init_s,simulation_wall_s=sim_s,total_wall_s=time.perf_counter()-BOOT,optimizer_updates=updates[0],physics_steps=len(engine_dts),engine_time_s=sum(engine_dts),
    errors=dict(policy_observation=obs_error,actual_target_alignment_m=enderr,action_to_wrench=wrench_err),actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__,command=shlex.join([sys.executable,*sys.argv]))
  with (out/'trajectory.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  dump_json(out/'trajectory.json',rows);dump_json(out/'result.json',result);dump_json(out/'resolved_config.json',jsonable(cfg.to_dict()))
  writer=SummaryWriter(str(out/'tensorboard_time'));scalar_rows=[]
  tags=['position_error_m','ex','ey','ez','vx','vy','vz','tilt_deg','wx','wy','wz','reward','action_clipping_fraction','allocation_clipping_fraction','integral_frozen','xy_at_limit','z_at_limit']+[f'xi_used_{x}' for x in 'xyz']+[f'thrust_{i}' for i in range(4)]
  for z in rows:
   if z['step']%10==0 or z is rows[-1]:
    for key in tags:
     val=float(np.float32(z[key]));writer.add_scalar('Eval/'+key,val,z['step']);scalar_rows.append(dict(tag='Eval/'+key,step=z['step'],time_s=z['time_s'],value=val))
  writer.flush();writer.close();dump_json(out/'tensorboard_scalars.json',scalar_rows)
  assert result['success'],checks
  print('EVAL_COMPLETE',args.integral,args.task,metrics['completed'],metrics['full']['position_rmse_m'],flush=True)
 except BaseException:
  dump_json(out/'failure.json',dict(error=traceback.format_exc(),valid_steps=len(snaps),optimizer_updates=updates[0]));raise
 finally:
  if hook is not None:hook.remove()
  raw.close()

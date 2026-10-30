"""Frozen baseline with rigid off-center payload sweep; first-terminal snapshots."""
import argparse,copy,csv,json,math,os,sys,time,traceback,shlex
from pathlib import Path
import numpy as np
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json,sha256,model_hash
from phase15_payload import install,read_engine
from circle_eval_common import reference,trajectory_metrics
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(conflict_handler='resolve');p.add_argument('--run-dir',required=True);p.add_argument('--suite',choices=['hover','circle_T10','circle_T5'],required=True);p.add_argument('--case');p.add_argument('--output');p.add_argument('--physics',default='isaacsim_physx');add_launcher_args(p);p.add_argument('--headless',action='store_true');a=p.parse_args()
if a.headless:a.visualizer=None;a.visualizer_explicit=True
r=Path(a.run_dir);plan=json.loads((r/'experiment_plan.json').read_text());assert json.loads((r/'physics_verification.json').read_text())['success'];base=Path(plan['baseline']);cases=plan['cases']
if a.case:cases=[c for c in cases if c['name']==a.case]
assert cases and a.device=='cuda:0';N=len(cases);out=Path(a.output) if a.output else r/'evaluation'/a.suite;out.mkdir(parents=True,exist_ok=False)
period=None if a.suite=='hover' else (10 if a.suite=='circle_T10' else 5);horizon=30 if period is None else period*2;steps=horizon*100
refs=np.tile([0.,0.,1.],(steps+1,1)) if period is None else reference(np.arange(steps+1)*.01,period)[0]
vrefs=np.zeros_like(refs) if period is None else reference(np.arange(steps+1)*.01,period)[1]
import torch,gymnasium as gym
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.visualizers import VisualizerCfg
from phase14z_reward import apply_reward
from phase14z_metrics import Telemetry,TERMS,REASONS
from phase11_evalmetrics import augment_row,summarize,extra_metrics
from phase6_common import jsonable
import crazyflie_isaaclab.tasks

torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=parse_env_cfg(TASK,device=a.device,num_envs=N,overrides=['physics=isaacsim_physx']);cfg.seed=42;cfg.episode_length_s=horizon;cfg.reset_position_range=0;cfg.target=tuple(map(float,refs[0]));cfg.yaw_target=0
cfg.sim.default_visualizer_cfg=VisualizerCfg(eye=(3,-4,3),lookat=(0,0,1))
BOOT=time.perf_counter();updates=[0];snaps=[];results=[];engine_dts=[]
with launch_simulation(cfg=cfg,launcher_args=a) as pc:
 cfg.sim.physics=pc;raw=gym.make(TASK,cfg=cfg).unwrapped
 try:
  engine=install(raw,cases);dump_json(out/'engine_properties.json',engine)
  env=RslRlVecEnvWrapper(raw,clip_actions=1.);agent=json.loads((base/'agent_config.json').read_text());runner=OnPolicyRunner(env,copy.deepcopy(agent),log_dir=None,device=a.device)
  cp=base/'best.pt';assert sha256(cp)==plan['checkpoint_sha256'];infos=runner.load(str(cp),strict=True,map_location=a.device);policy=runner.get_inference_policy(device=a.device)
  ref=torch.load(base/'best_reference.pt',weights_only=True,map_location=a.device)
  with torch.inference_mode():acts=policy(TensorDict({'policy':ref['observations']},batch_size=[128]))
  error=float((acts-ref['deterministic_actions']).abs().max());assert error<=1e-6
  before=model_hash(policy);hook=runner.alg.optimizer.register_step_post_hook(lambda *_:updates.__setitem__(0,updates[0]+1))
  dump_json(out/'reload.json',dict(checkpoint=str(cp),sha256=sha256(cp),reference_count=128,max_action_error=error,infos=infos,optimizer_updates=0))
  rc=json.loads((base/'reward_config.json').read_text());apply_reward(raw,rc);telemetry=Telemetry(raw,detail=True);telemetry.wrap(env)
  orig_reward=raw._get_rewards
  def capture():
   rew=orig_reward();telemetry.latest['link_position_local']=(raw.robot.data.root_link_pos_w.torch-raw.scene.env_origins).clone();telemetry.latest['com_velocity']=raw.robot.data.root_com_lin_vel_w.torch.clone();return rew
  raw._get_rewards=capture
  targets=torch.tensor(refs,device=a.device,dtype=torch.float32);raw.target.copy_(targets[0]);raw.reset(seed=42);telemetry.ep.zero_();telemetry.common_ep.zero_();telemetry.clear_rollout();raw.obs_buf=raw._get_observations();obs=env.get_observations()
  assert obs['policy'].dtype==torch.float32
  initial=dict(observations=obs['policy'].cpu().tolist(),motor_omega=raw.motors.omega.cpu().tolist(),env_origins=raw.scene.env_origins.cpu().tolist(),target=refs[0].tolist())
  assert bool((obs['policy'][:,:6].abs()<5e-6).all());dump_json(out/'initial_state.json',initial)
  if period is not None:
   orig_pre=raw._pre_physics_step;counter=[0]
   def moving_pre(action):orig_pre(action);counter[0]+=1;raw.target.copy_(targets[counter[0]])
   raw._pre_physics_step=moving_pre
  if not a.headless and a.visualizer:
   from pxr import UsdGeom,Gf,Vt
   stage=raw.scene.stage
   for i,c in enumerate(cases):
    if not c['mass_g']:continue
    path=f'/World/envs/env_{i}/CF21B/payload_visual'
    rod=UsdGeom.BasisCurves.Define(stage,path+'/rod');rod.CreateTypeAttr('linear');rod.CreateCurveVertexCountsAttr([2]);rod.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0,0,0),Gf.Vec3f(c['x_cm']/100,0,0)]));rod.CreateWidthsAttr([.002]);rod.CreateDisplayColorAttr([(1,.6,.1)])
    dot=UsdGeom.Sphere.Define(stage,path+'/point_mass');dot.CreateRadiusAttr(.008);dot.CreateDisplayColorAttr([(1,.1,.1)]);UsdGeom.Xformable(dot).AddTranslateOp().Set(Gf.Vec3d(c['x_cm']/100,0,0))
   if period:
    ring=UsdGeom.BasisCurves.Define(stage,'/World/CircleReference');ring.CreateTypeAttr('linear');ring.CreateWrapAttr('periodic');points=reference(np.linspace(0,period,180,endpoint=False),period)[0];ring.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float,z)) for z in points]));ring.CreateCurveVertexCountsAttr([180]);ring.CreateWidthsAttr([.004])
  import omni.physx
  sub=omni.physx.get_physx_interface().subscribe_physics_on_step_events(lambda dt:engine_dts.append(float(dt)),pre_step=False,order=0)
  motor0=raw.total_motor_calls;phys0=raw.sim.get_physics_step_count();init_s=time.perf_counter()-BOOT;start=time.perf_counter();alive=torch.ones(N,device=a.device,dtype=torch.bool)
  for step in range(1,steps+1):
   tick=time.perf_counter()
   with torch.inference_mode():action=policy(obs);action=torch.where(alive[:,None],action,torch.zeros_like(action));obs,reward,done,extra=env.step(action)
   snaps.append({k:v.clone() for k,v in telemetry.latest.items()});alive &= ~(telemetry.latest['terminated']|telemetry.latest['truncated'])
   if not bool(alive.any()):break
   if not a.headless and a.visualizer:time.sleep(max(0,.01-(time.perf_counter()-tick)))
  sim_s=time.perf_counter()-start;sub.unsubscribe();data={k:torch.stack([s[k] for s in snaps]).cpu().numpy() for k in snaps[0]}
  writer=SummaryWriter(str(out/'tensorboard_time'));scalar_rows=[]
  for j,case in enumerate(cases):
   rows=[]
   for si in range(len(snaps)):
    s={k:v[si,j] for k,v in data.items()};o=s['obs'].tolist();local=s['link_position_local'];refp=refs[si+1];vr=vrefs[si+1]
    row=dict(step=si+1,time_s=(si+1)*.01,x=float(local[0]),y=float(local[1]),z=float(local[2]),target_x=float(refp[0]),target_y=float(refp[1]),target_z=float(refp[2]),target_vx=float(vr[0]),target_vy=float(vr[1]),target_vz=float(vr[2]),position_error_m=float(s['pos']),tilt_deg=float(s['tilt']),reward=float(s['reward']),terminated=bool(s['terminated']),truncated=bool(s['truncated']))
    row.update(zip(('ex','ey','ez','vx','vy','vz','qw','qx','qy','qz','wx','wy','wz','sin_yaw','cos_yaw'),o))
    row.update({'world_'+axis:float(s['world_position'][i]) for i,axis in enumerate('xyz')})
    for key in ['raw_action','action','allocation','command','thrust','omega','request','actual','com_velocity']:row.update({f'{key}_{i}':float(v) for i,v in enumerate(s[key])})
    row.update({'reward_'+k:float(s['terms'][i]) for i,k in enumerate(TERMS)});row.update({'end_'+k:bool(v) for k,v in zip(REASONS,s['reasons'])})
    row['action_clipping_fraction']=float(np.mean(np.abs(s['raw_action'])>1));row['allocation_clipping_fraction']=float(np.mean((s['allocation']<0)|(s['allocation']>.289)))
    if period:
     theta=2*math.pi*row['time_s']/period;row['reference_phase_rad']=theta;row['radial_error_m']=float(np.linalg.norm(local[:2])-1);row['tangential_error_m']=-math.sin(theta)*o[0]+math.cos(theta)*o[1];row['phase_lag_deg']=math.degrees(math.atan2(math.sin(theta-math.atan2(local[1],local[0])),math.cos(theta-math.atan2(local[1],local[0]))))
     row.update(zip(['vex','vey','vez'],map(float,np.array(o[3:6])-vr)))
    augment_row(row);rows.append(row)
    if row['terminated'] or row['truncated']:break
   metrics,spectra=summarize(rows,horizon);metrics.update(extra_metrics(rows,initial['observations'][j][:3],metrics['completed']))
   last=rows[-1];cause=[k for k in REASONS[:-1] if last['end_'+k]]
   result=dict(**case,suite=a.suite,checkpoint=str(cp),checkpoint_sha256=sha256(cp),**metrics,terminated=last['terminated'],truncated=last['truncated'],reasons=cause,physical=engine['expected'][j],episode_return=sum(z['reward'] for z in rows),reward_sum_max_error=max(abs(sum(z['reward_'+k] for k in TERMS)-z['reward']) for z in rows))
   if period:result['circle']=trajectory_metrics(rows,period)
   assert result['reward_sum_max_error']<1e-4 and all(math.isfinite(float(v)) for z in rows for v in z.values());assert not any(z['terminated'] or z['truncated'] for z in rows[:-1])
   dest=out/case['name'];dest.mkdir();dump_json(dest/'result.json',result);dump_json(dest/'spectra.json',spectra)
   with (dest/'trajectory.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   for z in rows:
    if z['step']%10==0 or z is last:
     for k in ['position_error_m','ex','ey','ez','vx','vy','vz','tilt_deg','wx','wy','wz','action_clipping_fraction','allocation_clipping_fraction']:
      tag=case['name']+'/'+k;v=float(np.float32(z[k]));writer.add_scalar(tag,v,z['step']);scalar_rows.append(dict(tag=tag,step=z['step'],value=v,time_s=z['time_s']))
   results.append(result);print('EVAL',a.suite,case['name'],metrics['completed'],metrics['duration_s'],metrics['position_rmse_m'],flush=True)
  writer.flush();writer.close();dump_json(out/'tensorboard_scalars.json',scalar_rows);dump_json(out/'results.json',results)
  checks=dict(frozen_checkpoint=sha256(cp)==plan['checkpoint_sha256'],frozen_actor=model_hash(policy)==before,no_optimizer_updates=updates[0]==0,physics_calls=raw.sim.get_physics_step_count()-phys0==len(snaps)*5,motor_calls=raw.total_motor_calls-motor0==len(snaps)*5,engine_dt_count=len(engine_dts)==len(snaps)*5,engine_time=abs(sum(engine_dts)-len(snaps)*.01)<1e-5,first_terminal_only=True,finite=True,reward_sums=all(x['reward_sum_max_error']<1e-4 for x in results))
  assert all(checks.values()),checks
  dump_json(out/'execution.json',dict(success=True,checks=checks,initialization_s=init_s,simulation_wall_s=sim_s,total_wall_s=time.perf_counter()-BOOT,optimizer_updates=updates[0],policy_batch_steps=len(snaps),physics_batch_steps=len(engine_dts),scored_policy_samples=sum(round(z['duration_s']*100) for z in results),env_count=N,actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__,env_config=jsonable(cfg.to_dict()),command=shlex.join([sys.executable,*sys.argv])))
  hook.remove();print('EVALUATION_COMPLETE',a.suite,flush=True)
 except BaseException:
  dump_json(out/'failure.json',dict(error=traceback.format_exc(),valid_policy_batch_steps=len(snaps),optimizer_updates=updates[0]));raise
 finally:raw.close()

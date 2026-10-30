"""Isolated deterministic evaluation; terminal snapshots captured before autoreset."""
import time
BOOT=time.perf_counter()
import argparse,copy,csv,json,os,shlex,sys,math,datetime
from pathlib import Path
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json,sha256,model_hash

from phase11_evalmetrics import augment_row,summarize,extra_metrics
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(conflict_handler='resolve');p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--positions',required=True);p.add_argument('--suite',choices=['fixed','random20'],default='fixed');p.add_argument('--case');p.add_argument('--horizon',type=int,choices=[8,30],default=8);p.add_argument('--physics',choices=['isaacsim_physx'],default='isaacsim_physx');add_launcher_args(p);p.add_argument('--headless',action='store_true');a=p.parse_args()
if a.headless:a.visualizer=None;a.visualizer_explicit=True
assert a.device=='cuda:0'
run=Path(a.run_dir).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
cases=json.loads(Path(a.positions).read_text())[a.suite]
if a.case:cases=[x for x in cases if x['name']==a.case]
assert cases
import torch,gymnasium as gym
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import crazyflie_isaaclab.tasks
from phase6_common import jsonable
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=parse_env_cfg(TASK,device=a.device,num_envs=len(cases),overrides=['physics=isaacsim_physx']);cfg.seed=42;cfg.episode_length_s=float(a.horizon);cfg.reset_position_range=0.
agent=json.loads((run/'agent_config.json').read_text());rc=json.loads((run/'reward_config.json').read_text());results=[];checks=[]
if rc.get('reward_id')=='phase14z_axis_vz_squared':
 from phase14z_metrics import Telemetry,TERMS,REASONS
 from phase14z_reward import apply_reward
else:
 import phase9_metrics as old_metrics
 from phase9_metrics import Telemetry,TERMS,REASONS
 from phase9_reward import apply_reward
 original_snapshot=old_metrics.snapshot
 def extended_snapshot(raw,action,reward):
  s=original_snapshot(raw,action,reward);s['world_position']=raw.robot.data.root_link_pos_w.torch.detach();return s
 old_metrics.snapshot=extended_snapshot

with launch_simulation(cfg=cfg,launcher_args=a) as pc:
 cfg.sim.physics=pc;raw=gym.make(TASK,cfg=cfg).unwrapped
 try:
  env=RslRlVecEnvWrapper(raw,clip_actions=1.);runner=OnPolicyRunner(env,copy.deepcopy(agent),log_dir=None,device=a.device)
  apply_reward(raw,rc);telemetry=Telemetry(raw,detail=True);telemetry.wrap(env);updates=[0];hook=runner.alg.optimizer.register_step_post_hook(lambda *_:updates.__setitem__(0,updates[0]+1))
  init_s=time.perf_counter()-BOOT;eval_start=time.perf_counter()
  for cps in a.checkpoints:
   cp=Path(cps).resolve(strict=True);info=runner.load(str(cp),strict=True,map_location=a.device);policy=runner.get_inference_policy(device=a.device);dest=out/cp.stem;dest.mkdir()
   ref=torch.load(cp.with_name(cp.stem+'_reference.pt'),weights_only=True,map_location=a.device)
   with torch.inference_mode():act=policy(TensorDict({'policy':ref['observations']},batch_size=[len(ref['observations'])]))
   err=float((act-ref['deterministic_actions']).abs().max());assert err<=1e-6
   ck=dict(checkpoint=str(cp),sha256=sha256(cp),reference_count=len(ref['observations']),max_action_error=err,bitwise_equal=torch.equal(act,ref['deterministic_actions']),loaded_infos=info,pid=os.getpid(),optimizer_updates_in_process=updates[0]);dump_json(dest/'reload.json',ck);checks.append(ck)
   raw.reset(seed=42);telemetry.ep.zero_();telemetry.common_ep.zero_();telemetry.clear_rollout()
   offsets=torch.tensor([x['offset'] for x in cases],device=a.device);pose=raw.robot.data.default_root_pose.torch.clone();pose[:,:3]=raw.target+raw.scene.env_origins+offsets;raw.robot.write_root_link_pose_to_sim_index(root_pose=pose)
   raw.obs_buf=raw._get_observations();obs=env.get_observations();initial=obs['policy'].cpu().tolist()
   assert (obs['policy'][:,:3]-offsets).abs().max()<1e-5
   snapshots=[];alive=torch.ones(len(cases),device=a.device,dtype=torch.bool);wall=time.perf_counter()
   for step in range(1,a.horizon*100+1):
    tick=time.perf_counter()
    with torch.inference_mode():action=policy(obs);action=torch.where(alive[:,None],action,torch.zeros_like(action));obs,reward,done,extra=env.step(action)
    snapshots.append({k:v.clone() for k,v in telemetry.latest.items()})
    alive &= ~(telemetry.latest['terminated']|telemetry.latest['truncated'])
    if not bool(alive.any()):break
    if not a.headless and a.visualizer:time.sleep(max(0,.01-(time.perf_counter()-tick)))
   data={k:torch.stack([s[k] for s in snapshots]).cpu().tolist() for k in snapshots[0]};simulation_wall=time.perf_counter()-wall
   writer=SummaryWriter(str(dest/'tensorboard_time'))
   for j,case in enumerate(cases):
    rows=[]
    for si in range(len(snapshots)):
     s={k:v[si][j] for k,v in data.items()};o=s['obs'];row=dict(step=si+1,time_s=(si+1)*.01,x=o[0],y=o[1],z=o[2]+1,position_error_m=s['pos'],tilt_deg=s['tilt'],reward=s['reward'],common_reward_stop=s['common_reward_stop'],common_reward_return=s['common_reward_return'],terminated=s['terminated'],truncated=s['truncated'])
     row.update({'world_'+axis:s['world_position'][i] for i,axis in enumerate('xyz')})
     row.update(zip(('ex','ey','ez','vx','vy','vz','qw','qx','qy','qz','wx','wy','wz','sin_yaw','cos_yaw'),o))
     for key in ['raw_action','action','allocation','command','thrust','omega','request','actual']:row.update({f'{key}_{i}':v for i,v in enumerate(s[key])})
     row.update({'reward_'+k:s['terms'][i] for i,k in enumerate(TERMS)});row.update({'end_'+k:bool(v) for k,v in zip(REASONS,s['reasons'])})
     row['action_clipping_fraction']=sum(abs(v)>1 for v in s['raw_action'])/4;row['allocation_clipping_fraction']=sum(v<0 or v>.289 for v in s['allocation'])/4
     augment_row(row);rows.append(row)
     if row['terminated'] or row['truncated']:break
    measured,spectra=summarize(rows,a.horizon);measured.update(extra_metrics(rows,initial[j][:3],measured['completed']));last=rows[-1]
    result=dict(checkpoint=str(cp),checkpoint_name=cp.stem,iteration=info['completed_iterations'],transitions=info['collected_transitions'],condition=case['name'],offset=case['offset'],terminated=last['terminated'],truncated=last['truncated'],reasons=[k for k in REASONS[:-1] if last['end_'+k]],episode_return=sum(x['reward'] for x in rows),common_reward_stop_return=sum(x['common_reward_stop'] for x in rows),common_reward_return_return=sum(x['common_reward_return'] for x in rows),reward_terms={k:sum(x['reward_'+k] for x in rows) for k in TERMS},reward_sum_max_abs_error=max(abs(sum(x['reward_'+k] for k in TERMS)-x['reward']) for x in rows),**measured)
    assert result['reward_sum_max_abs_error']<1e-4 and all(math.isfinite(float(v)) for row in rows for v in row.values())
    assert not any(row['terminated'] or row['truncated'] for row in rows[:-1])
    name=case['name'];dump_json(dest/(name+'.json'),dict(initial_observation=initial[j],env_origin=raw.scene.env_origins[j].cpu().tolist(),target_local=raw.target.cpu().tolist(),position_source='root_link_pos_w float32; terminal snapshot before reset',result=result));dump_json(dest/(name+'_spectra.json'),spectra)
    with (dest/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for row in rows:
     if row['step']%10==0 or row is rows[-1]:
      for k in ['time_s','position_error_m','tilt_deg','vx','vy','vz','vdes_x','vdes_y','vdes_z','roll_rad','pitch_rad','ez','wx']:writer.add_scalar(name+'/'+k,row[k],row['step'])
    results.append(result);print('EVAL',cp.stem,name,result['completed'],result['duration_s'],result['position_rmse_m'],flush=True)
   writer.flush();writer.close();dump_json(dest/'batch_execution.json',dict(simulation_wall_s=simulation_wall,num_envs=len(cases),first_terminal_capture=True,post_reset_samples_excluded=True))
  hook.remove();assert updates[0]==0;dump_json(out/'results.json',results);dump_json(out/'reload_checks.json',checks)
  dump_json(out/'execution.json',dict(pid=os.getpid(),optimizer_updates=0,initialization_s=init_s,evaluation_s=time.perf_counter()-eval_start,total_wall_s=time.perf_counter()-BOOT,horizon_s=a.horizon,num_envs=len(cases),cases=cases,actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__,env_config=jsonable(cfg.to_dict()),command=shlex.join([sys.executable,*sys.argv])))
 finally:raw.close()

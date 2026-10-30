"""Explicit checkpoint reload, deterministic equivalence, then one nominal hover episode."""
import time
BOOT=time.perf_counter()
import argparse,copy,csv,json,math,os,shlex,socket,sys,datetime
from pathlib import Path
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json,protected_hashes,model_hash,sha256
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
parser=argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--output')
parser.add_argument('--physics',choices=['isaacsim_physx'],default='isaacsim_physx')
add_launcher_args(parser)
parser.add_argument('--headless',action='store_true')
args=parser.parse_args()
if args.headless:args.visualizer=None;args.visualizer_explicit=True
if args.device!='cuda:0':parser.error('Use cuda:0 for phase-3 replay')
checkpoint=Path(args.checkpoint).resolve(strict=True);run=checkpoint.parent
out=Path(args.output).resolve() if args.output else run/('evaluation_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir(parents=True,exist_ok=False)
training=json.loads((run/'training_result.json').read_text())
agent_cfg=json.loads((run/'agent_config.json').read_text())
assert sha256(checkpoint)==training['checkpoint_sha256'],'Checkpoint does not match the recorded final model'
assert protected_hashes()==json.loads((run/'phase2_protected_sha256.json').read_text())
import gymnasium as gym
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import crazyflie_isaaclab.tasks

torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=parse_env_cfg(TASK,device=args.device,num_envs=1,overrides=['physics=isaacsim_physx'])
cfg.seed=42
# This only captures terminal observations before the existing automatic reset.
cfg.compute_final_obs=True
assert cfg.reward_variant=='A' and cfg.reset_position_range==0
with launch_simulation(cfg=cfg,launcher_args=args) as physics_cfg:
    cfg.sim.physics=physics_cfg
    raw=gym.make(TASK,cfg=cfg).unwrapped
    try:
        wrapped=RslRlVecEnvWrapper(raw,clip_actions=agent_cfg['clip_actions'])
        runner=OnPolicyRunner(wrapped,copy.deepcopy(agent_cfg),log_dir=None,device=args.device)
        updates=[0]
        hook=runner.alg.optimizer.register_step_post_hook(lambda *_:updates.__setitem__(0,updates[0]+1))
        loaded_infos=runner.load(str(checkpoint),strict=True,map_location=args.device)
        policy=runner.get_inference_policy(device=args.device)
        ref=torch.load(run/'deterministic_reference.pt',weights_only=True,map_location=args.device)
        with torch.inference_mode():
            reloaded=policy(TensorDict({'policy':ref['observations']},batch_size=[len(ref['observations'])]))
        max_error=float((reloaded-ref['deterministic_actions']).abs().max())
        clipped_error=float((reloaded.clamp(-1,1)-ref['clipped_actions']).abs().max())
        reload_checks=dict(new_process=os.getpid()!=training['pid'],strict_load_succeeded=True,
            identical_actor_hash=model_hash(policy)==training['actor_final_sha256'],
            deterministic_action_match=max_error<=1e-6,clipped_action_match=clipped_error<=1e-6,
            finite_actions=bool(torch.isfinite(reloaded).all()))
        reload_result=dict(success=all(reload_checks.values()),checks=reload_checks,pid=os.getpid(),training_pid=training['pid'],
            checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),reference_observations=list(ref['observations'].shape),
            max_abs_action_error=max_error,max_abs_clipped_action_error=clipped_error,tolerance=1e-6,
            bitwise_equal=bool(torch.equal(reloaded,ref['deterministic_actions'])),loaded_infos=loaded_infos,
            optimizer_updates_in_reload_process=updates[0])
        dump_json(out/'reload_result.json',reload_result)
        assert all(reload_checks.values()),reload_checks
        obs,_=raw.reset(seed=42)
        initial=obs['policy'][0].detach().cpu().tolist()
        trajectory=[];all_obs=[initial];episode_return=0.0;steps=0;terminated=False;truncated=False;closed=False
        init_s=time.perf_counter()-BOOT
        import omni.physx
        dts=[]
        subscription=omni.physx.get_physx_interface().subscribe_physics_on_step_events(lambda dt:dts.append(float(dt)),pre_step=False,order=0)
        start=time.perf_counter()
        for k in range(800):
            tick=time.perf_counter()
            if not raw.sim.is_headless_or_exist_active_visualizer():closed=True;break
            with torch.inference_mode():
                action=policy(TensorDict(obs,batch_size=[1]))
                obs,reward,term,trunc,extras=raw.step(action)
            steps=k+1;terminated=bool(term[0]);truncated=bool(trunc[0])
            measured=extras['final_obs']['policy'][0] if (terminated or truncated) else obs['policy'][0]
            data=measured.detach().cpu().tolist();all_obs.append(data)
            roll_pitch_cos=max(-1.0,min(1.0,1-2*(data[7]**2+data[8]**2)))
            row=dict(step=steps,time_s=steps*raw.step_dt,x=data[0],y=data[1],z=data[2]+1,
                position_error_m=math.sqrt(sum(x*x for x in data[:3])),tilt_deg=math.degrees(math.acos(roll_pitch_cos)),
                vx=data[3],vy=data[4],vz=data[5],qw=data[6],qx=data[7],qy=data[8],qz=data[9],
                wx=data[10],wy=data[11],wz=data[12],reward=float(reward[0]),terminated=terminated,truncated=truncated,
                terminal_observation_used=terminated or truncated)
            for i,value in enumerate(action[0].detach().cpu().tolist()):
                row[f'action_raw_{i}']=value;row[f'action_applied_{i}']=max(-1,min(1,value))
            trajectory.append(row);episode_return+=row['reward']
            if terminated or truncated:break
            if not args.headless and args.visualizer:
                time.sleep(max(0,raw.step_dt-(time.perf_counter()-tick)))
        wall_s=time.perf_counter()-start;subscription.unsubscribe();hook.remove()
        last=trajectory[-1] if trajectory else None
        reasons=[]
        if last:
            if last['z']<.2:reasons.append('min_altitude')
            if last['z']>2.5:reasons.append('max_altitude')
            if last['tilt_deg']>60:reasons.append('max_tilt')
            if last['position_error_m']>1.5:reasons.append('max_position_error')
        if truncated:reasons.append('time_limit_8s')
        if closed:reasons.append('visualizer_closed')
        with (out/'trajectory.csv').open('w',newline='') as f:
            if trajectory:
                writer=csv.DictWriter(f,fieldnames=list(trajectory[0]));writer.writeheader();writer.writerows(trajectory)
        dump_json(out/'trajectory.json',dict(initial_observation=initial,steps=trajectory))
        result=dict(pipeline_success=reload_result['success'] and updates[0]==0,
            hover_8s_survived=steps==800 and not terminated and not closed,
            checkpoint=str(checkpoint),pid=os.getpid(),hostname=socket.gethostname(),num_envs=1,seed=42,
            reward_variant=cfg.reward_variant,nominal_start=dict(position=[0,0,1],quaternion_wxyz=[1,0,0,0],reset_motor_mass_kg=.04338),
            policy_steps=steps,duration_s=steps*raw.step_dt,actual_physx_step_events=len(dts),actual_physx_elapsed_s=sum(dts),
            terminated=terminated,truncated=truncated,reasons=reasons,episode_return=episode_return,
            initial_observation=initial,terminal_state=last,
            position_rmse_m=math.sqrt(sum(sum(v*v for v in o[:3]) for o in all_obs)/len(all_obs)),
            max_position_error_m=max((r['position_error_m'] for r in trajectory),default=0),
            max_tilt_deg=max((r['tilt_deg'] for r in trajectory),default=0),
            initialization_and_reload_s=init_s,evaluation_wall_s=wall_s,optimizer_updates=updates[0],
            phase2_files_unchanged=protected_hashes()==json.loads((run/'phase2_protected_sha256.json').read_text()),
            selected_backend='isaacsim_physx',actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__)
        dump_json(out/'evaluation_result.json',result);dump_json(out/'env_config.json',cfg.to_dict())
        dump_json(out/'command.json',dict(argv=[sys.executable,*sys.argv],shell_command=shlex.join([sys.executable,*sys.argv]),cwd=str(Path.cwd())))
        print('RELOAD_RESULT',json.dumps(reload_result),flush=True)
        print('EVALUATION_RESULT',json.dumps(result),flush=True)
    finally:raw.close()

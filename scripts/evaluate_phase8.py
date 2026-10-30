"""Phase8 isolated deterministic evaluation with branch-specific reward telemetry only."""
import time
BOOT=time.perf_counter()
import argparse,copy,csv,datetime,json,os,shlex,sys,math
from pathlib import Path
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json,sha256,model_hash,protected_hashes
from phase8_common import Telemetry,TERMS,REASONS,apply_reward
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(conflict_handler='resolve')
p.add_argument('--checkpoints',nargs='+',required=True)
p.add_argument('--run-dir',required=True)
p.add_argument('--output')
p.add_argument('--horizon',type=int,choices=[8,30],default=8)
p.add_argument('--physics',choices=['isaacsim_physx'],default='isaacsim_physx')
add_launcher_args(p);p.add_argument('--headless',action='store_true');args=p.parse_args()
if args.headless:args.visualizer=None;args.visualizer_explicit=True
assert args.device=='cuda:0'
run=Path(args.run_dir).resolve();out=Path(args.output) if args.output else run/'evaluation'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
out.mkdir(parents=True,exist_ok=False)
import torch,gymnasium as gym
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import crazyflie_isaaclab.tasks

torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=parse_env_cfg(TASK,device=args.device,num_envs=1,overrides=['physics=isaacsim_physx']);cfg.seed=42;cfg.episode_length_s=float(args.horizon)
agent=json.loads((run/'agent_config.json').read_text())
reward_cfg=json.loads((run/'reward_config.json').read_text())
checks=[];results=[]
initialization_s=None
with launch_simulation(cfg=cfg,launcher_args=args) as pc:
    cfg.sim.physics=pc;raw=gym.make(TASK,cfg=cfg).unwrapped
    try:
        env=RslRlVecEnvWrapper(raw,clip_actions=1.0)
        runner=OnPolicyRunner(env,copy.deepcopy(agent),log_dir=None,device=args.device)
        apply_reward(raw,reward_cfg)
        telemetry=Telemetry(raw,detail=True);telemetry.wrap(env)
        updates=[0];hook=runner.alg.optimizer.register_step_post_hook(lambda *_:updates.__setitem__(0,updates[0]+1))
        initialization_s=time.perf_counter()-BOOT
        for checkpoint_s in args.checkpoints:
            cp=Path(checkpoint_s).resolve(strict=True)
            info=runner.load(str(cp),strict=True,map_location=args.device)
            policy=runner.get_inference_policy(device=args.device)
            name=cp.stem;dest=out/name;dest.mkdir()
            refpath=cp.with_name(cp.stem+'_reference.pt')
            if not refpath.exists():refpath=cp.parent/'deterministic_reference.pt'
            ref=torch.load(refpath,weights_only=True,map_location=args.device)
            with torch.inference_mode():act=policy(TensorDict({'policy':ref['observations']},batch_size=[len(ref['observations'])]))
            err=float((act-ref['deterministic_actions']).abs().max())
            ck=dict(checkpoint=str(cp),sha256=sha256(cp),pid=os.getpid(),loaded_infos=info,
                reference_count=len(ref['observations']),max_action_error=err,bitwise_equal=torch.equal(act,ref['deterministic_actions']),
                actor_sha256=model_hash(policy),optimizer_updates_in_process=updates[0])
            assert err<=1e-6;checks.append(ck);dump_json(dest/'reload.json',ck)
            for label,offset in [('nominal',(0,0,0)),('x_plus',(.05,0,0)),('x_minus',(-.05,0,0)),('y_plus',(0,.05,0)),('y_minus',(0,-.05,0))]:
                raw.reset(seed=42);telemetry.ep.zero_();telemetry.common_ep.zero_();telemetry.clear_rollout()
                pose=raw.robot.data.default_root_pose.torch.clone()
                pose[:,:3]=raw.target+raw.scene.env_origins+torch.tensor(offset,device=args.device)
                raw.robot.write_root_link_pose_to_sim_index(root_pose=pose)
                raw.obs_buf=raw._get_observations();obs=env.get_observations()
                initial=obs['policy'][0].cpu().tolist();rows=[];wall=time.perf_counter()
                for step in range(1,args.horizon*100+1):
                    tick=time.perf_counter()
                    with torch.inference_mode():
                        action=policy(obs);obs,reward,done,extras=env.step(action)
                    # Read-only observer captured end-of-step state BEFORE autoreset.
                    snap={k:v[0].detach().cpu().tolist() for k,v in telemetry.latest.items()}
                    o=snap['obs'];row=dict(step=step,time_s=step*.01,x=o[0],y=o[1],z=o[2]+1,
                        position_error_m=snap['pos'],tilt_deg=snap['tilt'],reward=snap['reward'],common_reward_A=snap['common_reward_A'],
                        terminated=snap['terminated'],truncated=snap['truncated'])
                    for names,vals in [(('ex','ey','ez','vx','vy','vz','qw','qx','qy','qz','wx','wy','wz','sin_yaw','cos_yaw'),o)]:row.update(zip(names,vals))
                    for key in ['raw_action','action','allocation','command','thrust','omega','request','actual']:
                        row.update({f'{key}_{i}':v for i,v in enumerate(snap[key])})
                    row.update({'reward_'+k:snap['terms'][i] for i,k in enumerate(TERMS)})
                    row.update({'end_'+k:bool(v) for k,v in zip(REASONS,snap['reasons'])})
                    row['action_clipping_fraction']=sum(abs(v)>1 for v in snap['raw_action'])/4
                    row['allocation_clipping_fraction']=sum(v<0 or v>.289 for v in snap['allocation'])/4
                    row['command_lower_fraction']=sum(v<=1e-7 for v in snap['command'])/4
                    row['command_upper_fraction']=sum(v>=.289-1e-7 for v in snap['command'])/4
                    from phase8_evalmetrics import augment_row
                    augment_row(row)
                    rows.append(row)
                    if snap['terminated'] or snap['truncated']:break
                    if not args.headless and args.visualizer:time.sleep(max(0,.01-(time.perf_counter()-tick)))
                from phase8_evalmetrics import summarize
                measured,spectra=summarize(rows,args.horizon)
                last=rows[-1]
                result=dict(checkpoint=str(cp),checkpoint_name=name,iteration=info['completed_iterations'],
                    transitions=info['collected_transitions'],condition=label,offset=list(offset),
                    terminated=last['terminated'],truncated=last['truncated'],
                    reasons=[k for k in REASONS[:-1] if last['end_'+k]],
                    episode_return=sum(r['reward'] for r in rows),
                    common_reward_A_return=sum(r['common_reward_A'] for r in rows),
                    reward_definition=reward_cfg,evaluation_wall_s=time.perf_counter()-wall,
                    reward_terms={k:sum(r['reward_'+k] for r in rows) for k in TERMS},
                    reward_sum_max_abs_error=max(abs(sum(r['reward_'+k] for k in TERMS)-r['reward']) for r in rows),
                    **measured)
                assert result['reward_sum_max_abs_error']<1e-4
                assert all(math.isfinite(float(v)) for row in rows for v in row.values())
                dump_json(dest/(label+'_spectra.json'),spectra)
                with (dest/(label+'.csv')).open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                dump_json(dest/(label+'.json'),dict(initial_observation=initial,result=result,trajectory=rows))
                results.append(result);print('EVAL',json.dumps(result),flush=True)
        hook.remove()
        dump_json(out/'results.json',results);dump_json(out/'reload_checks.json',checks)
        dump_json(out/'execution.json',dict(pid=os.getpid(),optimizer_updates=updates[0],initialization_and_evaluation_wall_s=time.perf_counter()-BOOT,horizon_s=args.horizon,
            initialization_s=initialization_s,env_episode_length_s=cfg.episode_length_s,actual_max_episode_steps=raw.max_episode_length,
            config=__import__('phase6_common').jsonable(cfg.to_dict()),
            actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__,command=shlex.join([sys.executable,*sys.argv]),protected_hashes=protected_hashes()))
        assert updates[0]==0
    finally:raw.close()

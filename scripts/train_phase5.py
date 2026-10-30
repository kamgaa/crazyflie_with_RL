"""Fresh shared-policy PPO; exact data budget, common initialization, bounded variants."""
import time
BOOT=time.perf_counter()
import argparse,copy,csv,fcntl,json,os,shlex,sys,traceback
from pathlib import Path
import numpy as np
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json,sha256,model_hash,protected_hashes
from phase5_common import *
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(conflict_handler='resolve')
p.add_argument('--experiment',required=True);p.add_argument('--num-envs',type=int,choices=[64,256,1024],required=True)
p.add_argument('--preflight',action='store_true');p.add_argument('--create-common',action='store_true')
p.add_argument('--physics',choices=['isaacsim_physx'],default='isaacsim_physx')
add_launcher_args(p);p.add_argument('--headless',action='store_true');args=p.parse_args()
assert args.device=='cuda:0' and args.headless
args.visualizer=None;args.visualizer_explicit=True
root=Path(args.experiment).resolve();run=root/f'N{args.num_envs}';n=args.num_envs;iterations=COUNTS[n]
assert not args.create_common or (args.preflight and n==64)
lock=open(ROOT/'runs/.phase4_training.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if not args.preflight and (run/'training_started.json').exists():raise RuntimeError('Already started; refusing any unbudgeted retry')
import gymnasium as gym
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import crazyflie_isaaclab.tasks

torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=parse_env_cfg(TASK,device=args.device,num_envs=n,overrides=['physics=isaacsim_physx']);cfg.seed=42
agent=json.loads((run/'agent_config.json').read_text());assert agent['algorithm']['clip_param']==.2 and agent['num_steps_per_env']==32
assert cfg.reward_variant=='A' and cfg.reset_position_range==0 and cfg.decimation==5 and cfg.sim.dt==.002
before=protected_hashes();dump_json(run/('preflight_command.json' if args.preflight else 'command.json'),dict(command=shlex.join([sys.executable,*sys.argv]),cwd=str(ROOT)))
common=root/'common/initial.pt';writer=None;raw=None;updates=[0];completed=0;steps=0
try:
    with launch_simulation(cfg=cfg,launcher_args=args) as pc:
        cfg.sim.physics=pc;raw=gym.make(TASK,cfg=cfg).unwrapped
        try:
            env=RslRlVecEnvWrapper(raw,clip_actions=1.0)
            seed_all(42) # Network init independent of environment-creation RNG use.
            runner=OnPolicyRunner(env,copy.deepcopy(agent),log_dir=None,device=args.device)
            if args.create_common:
                assert not common.exists()
                s=runner.alg.save();s.update(iter=-1,infos=dict(completed_iterations=0,collected_transitions=0,optimizer_updates=0,num_envs=0,creation_pid=os.getpid()))
                assert not s['optimizer_state_dict']['state']
                torch.save(s,common);fixed=fixed_observations().to(args.device)
                policy=runner.get_inference_policy(device=args.device)
                with torch.inference_mode():act=policy(TensorDict({'policy':fixed},batch_size=[len(fixed)]))
                torch.save(dict(observations=fixed.cpu(),deterministic_actions=act.cpu()),root/'common/initial_reference.pt')
                dump_json(root/'common/agent_config.json',agent)
                dump_json(root/'common/initialization.json',dict(seed=42,actor_sha256=model_hash(runner.alg.actor),critic_sha256=model_hash(runner.alg.critic),
                    actor_std_parameter=runner.alg.actor.state_dict()['distribution.log_std_param'].cpu().tolist(),optimizer_state_count=0,initial_sha256=sha256(common)))
            s=torch.load(common,map_location=args.device,weights_only=False)
            runner.alg.load(s,{'actor':True,'critic':True,'optimizer':False,'iteration':False},strict=True)
            assert equal_state(runner.alg.actor.state_dict(),s['actor_state_dict']) and equal_state(runner.alg.critic.state_dict(),s['critic_state_dict'])
            assert not runner.alg.optimizer.state
            raw.reset(seed=42);seed_all(42)
            initcheck=dict(actor_sha256=model_hash(runner.alg.actor),critic_sha256=model_hash(runner.alg.critic),optimizer_state_count=len(runner.alg.optimizer.state),
                common_sha256=sha256(common),same_actor=True,same_critic=True,new_optimizer=True,seed=42,initialization_s=time.perf_counter()-BOOT,pid=os.getpid())
            fixed=torch.load(root/'common/initial_reference.pt',weights_only=True,map_location=args.device)['observations']
            telemetry=Telemetry(raw);telemetry.wrap(env)
            if args.preflight:
                with torch.inference_mode():
                    for _ in range(8):
                        action=torch.randn((n,4),device=args.device)*.1
                        obs,reward,done,extra=env.step(action)
                    idx=torch.arange(0,n,2,device=args.device);other=torch.arange(1,n,2,device=args.device)
                    state_before=raw.policy_observation()[other].clone();rpm_before=raw.motors.omega[other].clone()
                    raw._reset_idx(idx)
                    assert torch.equal(state_before,raw.policy_observation()[other]) and torch.equal(rpm_before,raw.motors.omega[other])
                    for _ in range(8):obs,reward,done,extra=env.step(torch.zeros((n,4),device=args.device))
                torch.cuda.synchronize()
                assert obs['policy'].shape==(n,15) and torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                assert raw.total_policy_calls==16 and raw.total_motor_calls==80
                dump_json(run/'preflight.json',dict(success=True,optimizer_updates=0,policy_steps=16,transitions_outside_training=n*16,
                    partial_reset_preserves_other_envs=True,**initcheck,memory=memory(),actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__))
                print('PREFLIGHT_OK',n,flush=True)
            else:
                assert json.loads((run/'preflight.json').read_text())['success']
                dump_json(run/'initialization.json',initcheck);dump_json(run/'env_config.json',cfg.to_dict())
                dump_json(run/'optimizer_config.json',dict(defaults=runner.alg.optimizer.defaults))
                install_probe(runner.alg,run/'evidence')
                hook=runner.alg.optimizer.register_step_post_hook(lambda *_:updates.__setitem__(0,updates[0]+1))
                writer=SummaryWriter(str(run/'tensorboard'),flush_secs=10)
                writer.add_custom_scalars({'Reward':{'terms':['Multiline',[f'RewardStep/{k}' for k in (*TERMS,'total')]]},
                    'PPO':{'loss':['Multiline',['Loss/policy','Loss/value_contribution','Loss/entropy_contribution','Loss/ppo_total']],
                           'clipping':['Multiline',['PPO/clip_fraction','PPO/approx_kl','PPO/clip_epsilon']]}})
                dump_json(run/'training_started.json',dict(pid=os.getpid(),num_envs=n,iterations=iterations,planned_transitions=BUDGET,planned_optimizer_updates=iterations*20))
                obs=env.get_observations();runner.alg.train_mode();rows=[];totals=dict(reward=0.,completed_reward=0.,completed_steps=0,completed_episodes=0)
                initial_mem=memory();peak_device=initial_mem['cuda_device_used_MiB'];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                def save(name):
                    policy=runner.get_inference_policy(device=args.device)
                    with torch.inference_mode():det=policy(TensorDict({'policy':fixed},batch_size=[len(fixed)]))
                    torch.save(dict(observations=fixed.cpu(),deterministic_actions=det.cpu()),run/(name+'_reference.pt'))
                    state=runner.alg.save();state.update(iter=completed-1,infos=dict(completed_iterations=completed,optimizer_updates=updates[0],
                        collected_transitions=steps*n,num_envs=n,training_pid=os.getpid(),training_elapsed_s=time.perf_counter()-start,
                        actor_sha256=model_hash(policy)),rng_state=rng_state())
                    torch.save(state,run/(name+'.pt'));runner.alg.train_mode()
                for it in range(iterations):
                    tick=time.perf_counter()
                    with torch.inference_mode():
                        for _ in range(32):
                            action=runner.alg.act(obs);obs,reward,done,extras=env.step(action);steps+=1
                            torch._assert_async(torch.isfinite(obs['policy']).all() & torch.isfinite(reward).all(),'Non-finite rollout')
                            runner.alg.process_env_step(obs,reward,done,extras)
                        collect_s=time.perf_counter()-tick;u0=time.perf_counter();runner.alg.compute_returns(obs)
                    runner.alg.update();update_s=time.perf_counter()-u0;completed=it+1;transitions=steps*n
                    ep=telemetry.ep_done.cpu().tolist();metrics=telemetry.finish_rollout();metrics.update(loss_metrics(runner.alg))
                    std=runner.alg.actor.output_std.detach().mean(0).cpu().tolist()
                    for j,v in enumerate(std):metrics[f'Policy/std_axis_{j}']=v
                    metrics['Policy/std_mean']=sum(std)/4;metrics['PPO/learning_rate']=runner.alg.learning_rate;metrics['PPO/clip_epsilon']=runner.alg.clip_param
                    count=metrics['Episode/completed_count']
                    if count:
                        for cause in REASONS:metrics[f'Episode/end_{cause}_fraction']=metrics[f'Episode/end_{cause}_count']/count
                    elapsed=time.perf_counter()-start;mem=memory();peak_device=max(peak_device,mem['cuda_device_used_MiB'])
                    metrics.update({'Performance/cumulative_transitions':transitions,'Performance/completed_iterations':completed,'Performance/optimizer_updates':updates[0],
                        'Performance/physics_batch_steps':raw.total_motor_calls,'Performance/physics_env_steps':transitions*5,
                        'Performance/rollout_s':collect_s,'Performance/update_s':update_s,'Performance/training_elapsed_s':elapsed,
                        'Performance/transitions_per_s':transitions/elapsed,'Performance/initialization_s':initcheck['initialization_s'],
                        **{'Performance/'+k:v for k,v in mem.items()}})
                    totals['reward']+=metrics['RewardStep/total']*n*32;totals['completed_reward']+=ep[7];totals['completed_steps']+=ep[8];totals['completed_episodes']+=int(ep[9])
                    assert updates[0]==completed*20
                    assert metrics['Validation/reward_sum_max_abs_error']<1e-4 and metrics['Validation/ppo_loss_residual']<1e-6
                    metrics={k:float(np.float32(v)) for k,v in metrics.items()};assert all(np.isfinite(v) for v in metrics.values())
                    for k,v in metrics.items():writer.add_scalar(k,v,transitions)
                    row=dict(global_step=transitions,iteration=completed,**metrics);rows.append(row)
                    with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                    if completed%32==0:writer.flush();print('PROGRESS',n,completed,transitions,round(elapsed,2),flush=True)
                    if transitions%INTERVAL==0:save('final' if transitions==BUDGET else f'checkpoint_{transitions}')
                torch.cuda.synchronize();wall=time.perf_counter()-start;hook.remove();writer.flush();writer.close();writer=None
                active=telemetry.ep.sum(0).cpu().tolist();accounting=dict(reward_residual=totals['reward']-totals['completed_reward']-active[7],step_residual=BUDGET-totals['completed_steps']-active[8],**totals)
                assert abs(accounting['reward_residual'])<1e-3 and accounting['step_residual']==0
                adam=sorted({int(x['step']) for x in runner.alg.optimizer.state.values()})
                assert adam==[iterations*20] and steps*n==BUDGET and raw.total_motor_calls==steps*5 and before==protected_hashes()
                keys=list(dict.fromkeys(k for x in rows for k in x))
                with (run/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
                result=dict(success=True,num_envs=n,rollout_steps=32,iterations=completed,transitions=steps*n,optimizer_updates=updates[0],adam_steps=adam,
                    minibatch_size=n*8,policy_batch_steps=steps,physics_batch_steps=raw.total_motor_calls,physics_env_steps=steps*n*5,
                    initialization_s=initcheck['initialization_s'],training_wall_s=wall,transitions_per_s=BUDGET/wall,
                    memory_initial=initial_mem,memory_final=memory(),cuda_device_used_peak_sampled_MiB=peak_device,
                    memory_sampling='CUDA mem_get_info per iteration (all processes); torch allocator peaks; NVML unavailable',
                    episode_accounting=accounting,protected_files_unchanged=True,pid=os.getpid(),final_sha256=sha256(run/'final.pt'),
                    actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__)
                dump_json(run/'training_result.json',result);print('TRAINING_RESULT',json.dumps(result),flush=True)
        finally:
            if writer:writer.flush();writer.close()
            raw.close();raw=None
except BaseException as e:
    record=dict(success=False,num_envs=n,preflight=args.preflight,error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),
        collected_transitions=steps*n,completed_iterations=completed,actual_optimizer_updates=updates[0],
        saved_checkpoints=sorted(str(p) for p in run.glob('*.pt')))
    dump_json(run/('preflight_failure.json' if args.preflight else 'failure.json'),record)
    raise

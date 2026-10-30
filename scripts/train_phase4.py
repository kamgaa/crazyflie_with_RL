"""Bounded full-state continuation: +450 iterations, one shared policy, 64 envs."""
import time
BOOT=time.perf_counter()
import argparse,copy,csv,fcntl,hashlib,json,os,random,shlex,sys
from pathlib import Path
import numpy as np
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json,sha256,model_hash,protected_hashes,GpuSampler
from phase4_metrics import Telemetry,install_ppo_probe,loss_metrics,TERMS
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(conflict_handler='resolve')
p.add_argument('--run-dir',required=True);p.add_argument('--resume',required=True)
p.add_argument('--additional-iterations',type=int,choices=[450],default=450)
p.add_argument('--physics',choices=['isaacsim_physx'],default='isaacsim_physx')
add_launcher_args(p);p.add_argument('--headless',action='store_true');args=p.parse_args()
assert args.device=='cuda:0' and args.headless
args.visualizer=None;args.visualizer_explicit=True
run=Path(args.run_dir).resolve();cp=Path(args.resume).resolve(strict=True)
lock=open(ROOT/'runs/.phase4_training.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (run/'training_started.json').exists():raise RuntimeError('Run already started: no unbudgeted retry')
import torch,gymnasium as gym
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import crazyflie_isaaclab.tasks

torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=parse_env_cfg(TASK,device=args.device,num_envs=64,overrides=['physics=isaacsim_physx']);cfg.seed=42
agent=json.loads((run/'agent_config.json').read_text())
assert cfg.reward_variant=='A' and cfg.reset_position_range==0 and agent['num_steps_per_env']==32
assert cfg.decimation==5 and cfg.sim.dt==.002 and agent['algorithm']['rnd_cfg'] is None and agent['algorithm']['symmetry_cfg'] is None
before=protected_hashes();assert before==json.loads((run/'evidence/phase3_phase2_protected_sha256.json').read_text())
dump_json(run/'command.json',dict(command=shlex.join([sys.executable,*sys.argv]),cwd=str(ROOT)))

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch_cpu'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in s['torch_cuda']])

with launch_simulation(cfg=cfg,launcher_args=args) as pc:
    cfg.sim.physics=pc;raw=gym.make(TASK,cfg=cfg).unwrapped
    writer=None;sampler=None
    try:
        env=RslRlVecEnvWrapper(raw,clip_actions=agent['clip_actions'])
        runner=OnPolicyRunner(env,copy.deepcopy(agent),log_dir=None,device=args.device)
        saved=torch.load(cp,map_location=args.device,weights_only=False)
        infos=runner.load(str(cp),strict=True,map_location=args.device)
        restored={k:equal(runner.alg.save()[k],saved[k]) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict')}
        assert all(restored.values()) and infos['completed_iterations']==50 and infos['optimizer_updates']==1000
        # Native iter is zero-based last-completed. The next index is 50, not 49.
        runner.current_learning_iteration=infos['completed_iterations']
        if 'rng_state' in saved:restore_rng(saved['rng_state'])
        else:random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        dump_json(run/'resume.json',dict(checkpoint=str(cp),sha256=sha256(cp),restored=restored,native_saved_iteration=saved['iter'],
            start_completed_iterations=50,start_transitions=102400,start_optimizer_updates=1000,
            rng_restored='rng_state' in saved,rng_fallback_seed=42,environment_state_restored=False,
            normalizers='actor/critic Identity; no observation/value running statistics',pid=os.getpid()))
        dump_json(run/'env_config.json',cfg.to_dict())
        dump_json(run/'optimizer_config.json',dict(defaults=runner.alg.optimizer.defaults,param_groups=[{k:v for k,v in g.items() if k!='params'} for g in runner.alg.optimizer.param_groups]))
        telemetry=Telemetry(raw);telemetry.wrap(env)
        install_ppo_probe(runner.alg,run/'evidence')
        updates=[1000];hook=runner.alg.optimizer.register_step_post_hook(lambda *_:updates.__setitem__(0,updates[0]+1))
        obs=env.get_observations().to(args.device)
        fixed=torch.cat((obs['policy'].clone(),torch.load(cp.parent/'deterministic_reference.pt',weights_only=True,map_location=args.device)['observations']),0)
        dump_json(run/'initial_state.json',dict(observation=obs['policy'][0].cpu().tolist(),max_cross_env_observation_difference=float((obs['policy']-obs['policy'][0]).abs().max()),
            motor_omega=raw.motors.omega[0].cpu().tolist(),position_range=cfg.reset_position_range,action_sampling='independent diagonal Gaussian draws per env and dimension; shared weights',initial_episode_lengths=raw.episode_length_buf.cpu().tolist()))
        writer=SummaryWriter(str(run/'tensorboard'),flush_secs=10)
        writer.add_custom_scalars({
            'Reward':{'Signed contributions':['Multiline',[f'RewardStep/{k}' for k in (*TERMS,'total')]],'Episode':['Multiline',['Episode/return','Episode/length_s']]},
            'PPO':{'Objective':['Multiline',['Loss/policy','Loss/value_contribution','Loss/entropy_contribution','Loss/ppo_total']],
                'Exploration':['Multiline',['Policy/entropy','Policy/std_mean','PPO/kl','PPO/clip_fraction']]},
            'Flight':{'Errors':['Multiline',['Flight/position_rmse_m','Flight/tilt_deg']],
                'Saturation':['Multiline',['Actuator/action_clipping_fraction','Actuator/allocation_clipping_fraction']]}})
        writer.add_scalar('Performance/cumulative_transitions',102400,102400)
        writer.add_scalar('Performance/completed_iterations',50,102400)
        writer.add_scalar('Performance/optimizer_updates',1000,102400)
        rows=[];totals=dict(reward=0.0,completed_reward=0.0,completed_episodes=0,completed_steps=0)
        sampler=GpuSampler();init_s=time.perf_counter()-BOOT
        dump_json(run/'training_started.json',dict(pid=os.getpid(),planned_additional_iterations=450,planned_additional_transitions=921600,planned_final_iterations=500,planned_final_transitions=1024000,planned_final_updates=10000))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();sampler.start();start=time.perf_counter()
        runner.alg.train_mode()
        for it in range(50,500):
            tick=time.perf_counter()
            with torch.inference_mode():
                for _ in range(32):
                    action=runner.alg.act(obs)
                    obs,reward,done,extras=env.step(action)
                    torch._assert_async(torch.isfinite(obs['policy']).all() & torch.isfinite(reward).all(),'Non-finite rollout')
                    runner.alg.process_env_step(obs,reward,done,extras)
                collection=time.perf_counter()-tick;update_start=time.perf_counter()
                runner.alg.compute_returns(obs)
            loss=runner.alg.update()
            update_s=time.perf_counter()-update_start
            completed=it+1;transitions=completed*2048
            episode_sums=telemetry.ep_done.detach().cpu().tolist()
            metrics=telemetry.finish_rollout();metrics.update(loss_metrics(runner.alg))
            std=runner.alg.actor.output_std.detach().mean(0).cpu().tolist()
            metrics['Policy/std_mean']=sum(std)/len(std)
            for j,v in enumerate(std):metrics[f'Policy/std_axis_{j}']=v
            metrics['PPO/learning_rate']=runner.alg.learning_rate
            totals['reward']+=metrics['RewardStep/total']*2048
            totals['completed_reward']+=episode_sums[7];totals['completed_steps']+=episode_sums[8];totals['completed_episodes']+=int(episode_sums[9])
            elapsed=time.perf_counter()-start
            metrics.update({'Performance/cumulative_transitions':transitions,'Performance/completed_iterations':completed,
                'Performance/optimizer_updates':updates[0],'Performance/rollout_s':collection,'Performance/update_s':update_s,
                'Performance/training_elapsed_s':elapsed,'Performance/transitions_per_s':(completed-50)*2048/elapsed,
                'Performance/torch_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20})
            if sampler.samples and 'process_mib' in sampler.samples[-1]:metrics['Performance/gpu_process_sampled_MiB']=sampler.samples[-1]['process_mib']
            assert updates[0]==completed*20
            assert metrics['Validation/reward_sum_max_abs_error']<1e-4 and metrics['Validation/ppo_loss_residual']<1e-6
            # Preserve the same float32 values in events and CSV/JSON.
            metrics={k:float(np.float32(v)) for k,v in metrics.items()}
            assert all(np.isfinite(v) for v in metrics.values())
            for k,v in metrics.items():writer.add_scalar(k,v,transitions)
            row=dict(global_step=transitions,iteration=completed,**metrics);rows.append(row)
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if completed%10==0:writer.flush();print('PHASE4_PROGRESS',completed,transitions,updates[0],metrics.get('Episode/length_s'),flush=True)
            if completed in (100,200,300,400,500):
                runner.current_learning_iteration=it
                policy=runner.get_inference_policy(device=args.device)
                with torch.inference_mode():det=policy(TensorDict({'policy':fixed},batch_size=[len(fixed)]))
                name='final' if completed==500 else f'checkpoint_{completed}'
                torch.save(dict(observations=fixed.cpu(),deterministic_actions=det.cpu()),run/(name+'_reference.pt'))
                state=runner.alg.save();state.update(iter=it,infos=dict(completed_iterations=completed,optimizer_updates=updates[0],collected_transitions=transitions,
                    actor_sha256=model_hash(policy),training_pid=os.getpid()),rng_state=rng_state())
                torch.save(state,run/(name+'.pt'));runner.alg.train_mode()
        torch.cuda.synchronize();train_s=time.perf_counter()-start
        sampler.stop();writer.flush();writer.close();writer=None;hook.remove()
        active=telemetry.ep.sum(0).cpu().tolist()
        accounting=dict(reward_residual=totals['reward']-totals['completed_reward']-active[7],
            step_residual=921600-totals['completed_steps']-active[8],**totals,active_episode_sums=active)
        assert abs(accounting['reward_residual'])<1e-4 and accounting['step_residual']==0
        adam_steps=sorted(set(int(v['step'].item()) for v in runner.alg.optimizer.state.values()))
        result=dict(success=True,pid=os.getpid(),initialization_s=init_s,training_wall_s=train_s,transitions_per_s=921600/train_s,
            additional_iterations=450,total_iterations=500,additional_transitions=921600,total_transitions=1024000,
            additional_optimizer_updates=updates[0]-1000,total_optimizer_updates=updates[0],adam_steps=adam_steps,
            policy_batch_steps=raw.total_policy_calls,motor_batch_updates=raw.total_motor_calls,
            gpu_process_peak_sampled_MiB=max(s.get('process_mib',0) for s in sampler.samples),gpu_sample_period_s=1,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            protected_files_unchanged=before==protected_hashes(),episode_accounting=accounting,
            actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__,final_sha256=sha256(run/'final.pt'))
        assert adam_steps==[10000] and raw.total_policy_calls==14400 and raw.total_motor_calls==72000 and result['protected_files_unchanged']
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (run/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
        dump_json(run/'training_result.json',result);dump_json(run/'gpu_samples.json',sampler.samples)
        print('PHASE4_TRAINING_RESULT',json.dumps(result),flush=True)
    finally:
        if sampler:sampler.stop()
        if writer:writer.flush();writer.close()
        raw.close()

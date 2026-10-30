"""Exactly one bounded phase-3 PPO run, with optimizer and checkpoint evidence."""
import time
BOOT=time.perf_counter()
import argparse,copy,csv,json,os,shlex,socket,sys
from pathlib import Path
import importlib.metadata as md
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json,protected_hashes,model_hash,sha256,gpu_snapshot,GpuSampler
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
parser=argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('--run-dir',required=True)
parser.add_argument('--num_envs',type=int,choices=[64],default=64)
parser.add_argument('--rollout-steps',type=int,choices=[32],default=32)
parser.add_argument('--iterations',type=int,choices=[50],default=50)
parser.add_argument('--seed',type=int,choices=[42],default=42)
parser.add_argument('--physics',choices=['isaacsim_physx'],default='isaacsim_physx')
parser.add_argument('--initialize-only',action='store_true')
add_launcher_args(parser)
parser.add_argument('--headless',action='store_true')
args=parser.parse_args()
if args.headless:args.visualizer=None;args.visualizer_explicit=True
if args.device!='cuda:0':parser.error('Phase 3 is fixed to cuda:0')
run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
if (run/'training_started.json').exists():raise RuntimeError('This run already started; refusing extra training')
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
import crazyflie_isaaclab.tasks

torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
env_cfg=parse_env_cfg(TASK,device=args.device,num_envs=64,overrides=['physics=isaacsim_physx'])
env_cfg.seed=42
agent_cfg=handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(TASK,'rsl_rl_cfg_entry_point'),md.version('rsl-rl-lib'))
agent_dict=agent_cfg.to_dict()
# These deprecated placeholders are not consumed by RSL-RL 5.4.1.
agent_dict.pop('policy',None);agent_dict.pop('empirical_normalization',None)
assert env_cfg.reward_variant=='A' and env_cfg.reset_position_range==0
assert agent_dict['max_iterations']==50 and agent_dict['num_steps_per_env']==32
before=protected_hashes()
if (run/'phase2_protected_sha256.json').exists():
    assert before==json.loads((run/'phase2_protected_sha256.json').read_text())
else:dump_json(run/'phase2_protected_sha256.json',before)
dump_json(run/'agent_config.json',agent_dict)
dump_json(run/'command.json',dict(argv=[sys.executable,*sys.argv],shell_command=shlex.join([sys.executable,*sys.argv]),cwd=str(Path.cwd())))
setup_import_s=time.perf_counter()-BOOT
sim_start=time.perf_counter()
with launch_simulation(cfg=env_cfg,launcher_args=args) as physics_cfg:
    env_cfg.sim.physics=physics_cfg
    raw=gym.make(TASK,cfg=env_cfg).unwrapped
    try:
        env=RslRlVecEnvWrapper(raw,clip_actions=agent_dict['clip_actions'])
        sim_env_s=time.perf_counter()-sim_start
        runner_start=time.perf_counter()
        runner=OnPolicyRunner(env,copy.deepcopy(agent_dict),log_dir=str(run/'tensorboard'),device=args.device)
        runner_setup_s=time.perf_counter()-runner_start
        dump_json(run/'env_config.json',env_cfg.to_dict())
        dump_json(run/'optimizer_config.json',dict(optimizer=type(runner.alg.optimizer).__name__,defaults=runner.alg.optimizer.defaults))
        initial={k:v.detach().cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
        torch.save(initial,run/'initial_actor.pt')
        initial_hash=model_hash(runner.alg.actor)
        initial_obs=env.get_observations()['policy'].detach().clone()
        if args.initialize_only:
            dump_json(run/'initialization_check.json',dict(success=True,optimizer_updates=0,observation_shape=list(initial_obs.shape),
                actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__))
            print('INITIALIZATION_ONLY_OK',flush=True)
        else:
            updates=[0];iterations=[]
            def count_update(opt,arguments,kwargs):updates[0]+=1
            hook=runner.alg.optimizer.register_step_post_hook(count_update)
            original_log=runner.logger.log
            def log_iteration(**kw):
                original_log(**kw)
                row=dict(iteration=kw['it']+1,optimizer_updates=updates[0],
                    transitions=(kw['it']+1)*64*32,collect_s=kw['collect_time'],update_s=kw['learn_time'],
                    learning_rate=kw['learning_rate'],**{f'loss_{k}':float(v) for k,v in kw['loss_dict'].items()})
                iterations.append(row)
                with (run/'iterations.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                print('PHASE3_PROGRESS',json.dumps(row),flush=True)
            runner.logger.log=log_iteration
            step_calls=[0];ended=torch.zeros(3,device=args.device,dtype=torch.int64)
            original_step=env.step
            def counted_step(action):
                result=original_step(action);step_calls[0]+=1
                ended[0]+=raw.reset_terminated.sum();ended[1]+=raw.reset_time_outs.sum()
                ended[2]+=(raw.reset_terminated & raw.reset_time_outs).sum()
                return result
            env.step=counted_step
            init_total_s=time.perf_counter()-BOOT
            gpu_after_init=gpu_snapshot();sampler=GpuSampler()
            dump_json(run/'training_started.json',dict(pid=os.getpid(),started=time.time(),planned_transitions=102400,
                iterations=50,planned_optimizer_updates=1000,seed=42))
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            sampler.start();start=time.perf_counter()
            runner.learn(num_learning_iterations=50,init_at_random_ep_len=False)
            torch.cuda.synchronize();train_s=time.perf_counter()-start
            sampler.stop();hook.remove()
            final_hash=model_hash(runner.alg.actor)
            differences={k:float((v.detach().cpu()-initial[k]).double().norm()) for k,v in runner.alg.actor.state_dict().items()}
            total_l2=sum(v*v for v in differences.values())**.5
            mlp_l2=sum(v*v for k,v in differences.items() if k.startswith('mlp.'))**.5
            adam_steps=sorted(set(int(s['step'].item()) for s in runner.alg.optimizer.state.values() if 'step' in s))
            policy=runner.get_inference_policy(device=args.device)
            final_obs=env.get_observations()['policy'].detach().clone()
            observations=torch.cat((initial_obs,final_obs),0)
            with torch.inference_mode():actions=policy(TensorDict({'policy':observations},batch_size=[len(observations)]))
            torch.save(dict(observations=observations.cpu(),deterministic_actions=actions.cpu(),
                clipped_actions=actions.clamp(-1,1).cpu()),run/'deterministic_reference.pt')
            infos=dict(completed_iterations=50,optimizer_updates=updates[0],collected_transitions=step_calls[0]*64,actor_sha256=final_hash)
            runner.save(str(run/'final.pt'),infos=infos)
            with (run/'iterations.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(iterations[0]));writer.writeheader();writer.writerows(iterations)
            checks=dict(iterations_50=len(iterations)==50,optimizer_updates_1000=updates[0]==1000,
                adam_steps_1000=adam_steps==[1000],transitions_102400=step_calls[0]*64==102400,
                actor_changed=initial_hash!=final_hash and mlp_l2>0,actor_finite=all(torch.isfinite(p).all().item() for p in policy.parameters()),
                phase2_files_unchanged=before==protected_hashes())
            good_samples=[s for s in sampler.samples if 'process_mib' in s]
            result=dict(success=all(checks.values()),checks=checks,hostname=socket.gethostname(),pid=os.getpid(),
                task=TASK,selected_backend='isaacsim_physx',actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__,
                seed=42,num_envs=64,rollout_steps=32,completed_iterations=len(iterations),last_rsl_iteration_index=runner.current_learning_iteration,
                optimizer_updates=updates[0],adam_parameter_steps=adam_steps,collected_transitions=step_calls[0]*64,
                policy_batch_steps=step_calls[0],motor_batch_updates=raw.total_motor_calls,
                initialization_s=init_total_s,imports_config_s=setup_import_s,simulator_env_s=sim_env_s,runner_setup_s=runner_setup_s,
                training_wall_s=train_s,policy_transitions_per_s=step_calls[0]*64/train_s,
                rollout_time_sum_s=sum(r['collect_s'] for r in iterations),update_time_sum_s=sum(r['update_s'] for r in iterations),
                gpu_after_initialization=gpu_after_init,gpu_process_peak_sampled_mib=max((s['process_mib'] for s in good_samples),default=None),
                gpu_device_peak_sampled_mib=max((s['device_used_mib'] for s in good_samples),default=None),gpu_sample_period_s=1,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                actor_initial_sha256=initial_hash,actor_final_sha256=final_hash,actor_delta_l2=total_l2,actor_mlp_delta_l2=mlp_l2,
                actor_parameter_delta_l2=differences,training_terminations=int(ended[0]),training_timeouts=int(ended[1]),both_done=int(ended[2]),
                checkpoint=str(run/'final.pt'),checkpoint_sha256=sha256(run/'final.pt'),
                versions={k:md.version(k) for k in ['rsl-rl-lib','isaaclab','isaacsim','torch','newton']})
            dump_json(run/'gpu_samples.json',sampler.samples);dump_json(run/'training_result.json',result)
            print('TRAINING_RESULT',json.dumps(result),flush=True)
            assert all(checks.values()),checks
    finally:raw.close()

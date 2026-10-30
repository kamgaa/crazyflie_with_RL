"""Phase 9: exactly 256 additional iterations per independent shared-parent branch."""
import time
BOOT=time.perf_counter()
import argparse,copy,csv,fcntl,json,os,shlex,sys,traceback
from pathlib import Path
import numpy as np
import warp as wp
wp.config.enable_backward=False
from ppo_common import ROOT,TASK,dump_json,sha256,model_hash,protected_hashes
from phase9_common import *
sys.path.insert(0,str(ROOT/'src'))
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser(conflict_handler='resolve')
p.add_argument('--run-dir',required=True)
p.add_argument('--additional-iterations',type=int,choices=[ADD_IT],default=ADD_IT,
    help='Additional iterations fixed to 256; final cumulative count is 1664.')
p.add_argument('--preflight',action='store_true')
p.add_argument('--physics',choices=['isaacsim_physx'],default='isaacsim_physx')
add_launcher_args(p);p.add_argument('--headless',action='store_true');args=p.parse_args()
assert args.device=='cuda:0' and args.headless
args.visualizer=None;args.visualizer_explicit=True
run=Path(args.run_dir).resolve();plan=json.loads((run/'experiment_plan.json').read_text())
parent_path=Path(plan['checkpoint']);parent_root=Path(plan['parent'])
assert args.additional_iterations==ADD_IT and plan['additional_iterations']==ADD_IT
assert parent_path==parent_root/'best.pt'
lock=open(ROOT/'runs/.phase4_training.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if not args.preflight and (run/'training_started.json').exists():
    raise RuntimeError('Run already started; refusing duplicate or unbudgeted retry.')
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import crazyflie_isaaclab.tasks
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=parse_env_cfg(TASK,device=args.device,num_envs=NUM_ENVS,overrides=['physics=isaacsim_physx']);cfg.seed=42;cfg.reset_position_range=.05
agent=json.loads((run/'agent_config.json').read_text())
parent_resolved=json.loads((parent_root/'resolved_config.json').read_text())
parent_agent=parent_resolved['agent']
assert sha256(parent_path)==plan['parent_sha256']
expected_algorithm=copy.deepcopy(parent_agent['algorithm']);expected_algorithm['num_mini_batches']=plan['minibatches']
assert agent['algorithm']==expected_algorithm
UPDATES_PER_IT=plan['minibatches']*5
ADD_UPDATES=ADD_IT*UPDATES_PER_IT
FINAL_UPDATES=BASE_UPDATES+ADD_UPDATES
assert ADD_UPDATES==plan['updates']
experiment=Path(plan['experiment'])
assert json.loads((experiment/'reward_ready.json').read_text())['success']
parent_total_training_s=torch.load(parent_path,map_location='cpu',weights_only=False)['infos']['cumulative_training_elapsed_s']
assert agent['num_steps_per_env']==ROLLOUT and agent['clip_actions']==1.
assert cfg.reward_variant=='A' and cfg.reset_position_range==.05 and cfg.decimation==5 and cfg.sim.dt==.002
before=protected_hashes()
dump_json(run/('preflight_command.json' if args.preflight else 'command.json'),
    dict(command=shlex.join([sys.executable,*sys.argv]),cwd=str(ROOT),cli_semantics='additional_iterations=256; final cumulative=1664'))
writer=None;raw=None;updates=[0];completed=0;steps=0
try:
    with launch_simulation(cfg=cfg,launcher_args=args) as physics_cfg:
        cfg.sim.physics=physics_cfg;raw=gym.make(TASK,cfg=cfg).unwrapped
        try:
            env=RslRlVecEnvWrapper(raw,clip_actions=1.)
            seed_all(42)
            runner=OnPolicyRunner(env,copy.deepcopy(agent),log_dir=None,device=args.device)
            parent=torch.load(parent_path,map_location='cpu',weights_only=False)
            loaded_info=runner.load(str(parent_path),load_cfg={'actor':True,'critic':True,'optimizer':True,'iteration':True,'rnd':True},
                strict=True,map_location='cpu')
            assert loaded_info==parent['infos']
            from phase9_reset import BoxReset
            reset_tracker=BoxReset(raw)
            raw.reset(seed=42)
            raw.obs_buf=raw._get_observations()
            reference=torch.load(parent_path.with_name('best_reference.pt'),map_location='cpu',weights_only=True)
            verification=verify_resume(runner,parent,reference,agent,parent_agent,
                run/('preflight_resume_verification.json' if args.preflight else 'resume_verification.json'),plan)
            actual_cfg=jsonable(cfg.to_dict())
            env_diff=differences(parent_resolved['env'],actual_cfg)
            expected_env=copy.deepcopy(parent_resolved['env']);expected_env['reset_position_range']=.05
            assert not differences(expected_env,actual_cfg),env_diff
            config_diff=dict(agent=differences(parent_agent,agent),environment=env_diff,
                optimizer=differences(parent_resolved['optimizer'],jsonable(runner.alg.optimizer.defaults)))
            assert not config_diff['optimizer']
            apply_reward(raw,json.loads((run/'reward_config.json').read_text()))
            telemetry=Telemetry(raw);telemetry.wrap(env);reset_tracker.telemetry=telemetry
            delta=reset_tracker.first
            initial_reset=dict(min=delta.min(0).values.cpu().tolist(),max=delta.max(0).values.cpu().tolist(),mean=delta.mean(0).cpu().tolist(),std=delta.std(0,unbiased=False).cpu().tolist(),unique_rows=len(torch.unique(delta,dim=0)),target=raw.target.cpu().tolist(),actual_error_max=float((raw.policy_observation()[:,:3]-delta).abs().max()))
            assert initial_reset['unique_rows']==NUM_ENVS and initial_reset['actual_error_max']<1e-5
            torch.save(dict(delta=delta.cpu(),observations=raw.policy_observation().cpu(),generator_state=reset_tracker.generator.get_state()),run/('preflight_initial_reset.pt' if args.preflight else 'initial_reset.pt'))
            dump_json(run/('preflight_initial_reset.json' if args.preflight else 'initial_reset.json'),initial_reset)
            fixed=reference['observations'].to(args.device)
            restore_rng(parent['rng_state'])
            assert rng_fingerprints(rng_state())==rng_fingerprints(parent['rng_state'])
            verification.update(rng_restored=True,rng_fingerprints=rng_fingerprints(rng_state()),parent_rng_fingerprints=rng_fingerprints(parent['rng_state']))
            if args.preflight:dump_json(run/'preflight_resume_verification.json',verification)
            if args.preflight:
                policy=runner.get_inference_policy(device=args.device)
                obs=env.get_observations()
                with torch.inference_mode():
                    for _ in range(8):obs,reward,done,extra=env.step(policy(obs))
                    selected=torch.arange(0,NUM_ENVS,2,device=args.device)
                    others=torch.arange(1,NUM_ENVS,2,device=args.device)
                    original_obs=raw.policy_observation()[others].clone()
                    original_rpm=raw.motors.omega[others].clone()
                    original_ep=telemetry.ep[others].clone()
                    raw._reset_idx(selected)
                    assert torch.equal(original_obs,raw.policy_observation()[others])
                    assert torch.equal(original_rpm,raw.motors.omega[others])
                    assert torch.equal(original_ep,telemetry.ep[others])
                    assert not telemetry.ep[selected].any()
                    for _ in range(8):obs,reward,done,extra=env.step(torch.zeros((NUM_ENVS,4),device=args.device))
                torch.cuda.synchronize()
                assert obs['policy'].shape==(NUM_ENVS,15)
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                assert raw.total_policy_calls==16 and raw.total_motor_calls==80
                assert adam_steps(runner.alg.optimizer)==[BASE_UPDATES]
                assert equal_state(runner.alg.optimizer.state_dict(),parent['optimizer_state_dict'])
                preflight_metrics=telemetry.finish_rollout()
                assert preflight_metrics['Validation/reward_sum_max_abs_error']<1e-4
                dump_json(run/'preflight_metrics.json',preflight_metrics)
                dump_json(run/'preflight.json',dict(success=True,optimizer_updates=0,parent_adam_steps=adam_steps(runner.alg.optimizer),
                    policy_batch_steps=16,physics_batch_steps=80,transitions_outside_training=16*NUM_ENVS,
                    partial_reset_preserves_other_envs=True,memory=memory(),environment_diff=env_diff,
                    actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__))
                print('PREFLIGHT_OK',flush=True)
            else:
                assert json.loads((run/'preflight.json').read_text())['success']
                assert json.loads((experiment/'comparison_ready.json').read_text())['success']
                dump_json(run/'env_config.json',actual_cfg)
                dump_json(run/'optimizer_config.json',dict(defaults=runner.alg.optimizer.defaults,
                    restored_param_groups=runner.alg.optimizer.state_dict()['param_groups']))
                dump_json(run/'config_diff.json',config_diff)
                dump_json(run/'resolved_config.json',dict(agent=agent,env=actual_cfg,
                    optimizer=jsonable(runner.alg.optimizer.defaults),reward=json.loads((run/'reward_config.json').read_text()),plan=plan))
                install_probe(runner.alg,run/'evidence')
                def counted_step(*_):
                    updates[0]+=1
                    if updates[0]==1:
                        current=runner.alg.optimizer.state_dict()
                        proof=dict(additional_optimizer_updates=1,cumulative_optimizer_updates=BASE_UPDATES+1,
                            actual_adam_steps=adam_steps(runner.alg.optimizer),
                            actor_changed=model_hash(runner.alg.actor)!=verification['actor_sha256'],
                            moments_changed=not equal_state(current,parent['optimizer_state_dict']))
                        assert proof['actual_adam_steps']==[BASE_UPDATES+1] and proof['actor_changed'] and proof['moments_changed']
                        dump_json(run/'first_optimizer_step_verification.json',proof)
                hook=runner.alg.optimizer.register_step_post_hook(counted_step)
                writer=SummaryWriter(str(run/'tensorboard'),flush_secs=10)
                writer.add_custom_scalars({'Reward':{'signed terms':['Multiline',[f'Reward/step/{k}' for k in (*TERMS,'total')]]},
                    'PPO':{'objective':['Multiline',['Loss/policy','Loss/value_contribution','Loss/entropy_contribution','Loss/ppo_total']],
                        'clipping':['Multiline',['PPO/clip_fraction','PPO/approx_kl','PPO/clip_epsilon']]},
                    'Flight':{'error':['Multiline',['Flight/position_rmse_m','Flight/tilt_deg']]}})
                obs=env.get_observations();runner.alg.train_mode()
                assert getattr(runner.alg.storage,'step',0)==0
                restore_rng(parent['rng_state'])
                fingerprints=rng_fingerprints(rng_state())
                verification.update(rng_restored=True,rng_fingerprints=fingerprints,parent_rng_fingerprints=rng_fingerprints(parent['rng_state']),
                    initial_observations_identical=bool(torch.equal(obs['policy'],obs['policy'][:1].expand_as(obs['policy']))),
                    initial_obs_shape=list(obs['policy'].shape),rollout_storage_step=getattr(runner.alg.storage,'step',None),
                    initialization_s=time.perf_counter()-BOOT,env_diff=env_diff)
                assert fingerprints==verification['parent_rng_fingerprints']
                dump_json(run/'resume_verification.json',verification)
                dump_json(run/'training_started.json',dict(pid=os.getpid(),base_iterations=BASE_IT,additional_iterations=ADD_IT,
                    final_iterations=FINAL_IT,base_transitions=BASE_TRANSITIONS,planned_additional_transitions=ADD_TRANSITIONS,
                    parent_checkpoint=str(parent_path),parent_sha256=sha256(parent_path),start_time=time.time()))
                initial_mem=memory();peak_device=initial_mem['cuda_device_used_MiB']
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                rows=[];totals=dict(reward=0.,completed_reward=0.,completed_steps=0.,completed_episodes=0)
                def save(name):
                    policy=runner.get_inference_policy(device=args.device)
                    with torch.inference_mode():actions=policy(TensorDict({'policy':fixed},batch_size=[len(fixed)]))
                    torch.save(dict(observations=fixed.cpu(),deterministic_actions=actions.cpu()),run/(name+'_reference.pt'))
                    elapsed=time.perf_counter()-start
                    state=runner.alg.save()
                    state.update(iter=BASE_IT+completed-1,infos=dict(completed_iterations=BASE_IT+completed,
                        additional_completed_iterations=completed,optimizer_updates=BASE_UPDATES+updates[0],
                        additional_optimizer_updates=updates[0],collected_transitions=BASE_TRANSITIONS+steps*NUM_ENVS,
                        additional_collected_transitions=steps*NUM_ENVS,num_envs=NUM_ENVS,training_pid=os.getpid(),
                        training_elapsed_s=elapsed,cumulative_training_elapsed_s=parent_total_training_s+elapsed,
                        parent_checkpoint=str(parent_path),parent_sha256=sha256(parent_path),actor_sha256=model_hash(policy),
                        branch=plan['branch'],position_mode=plan['position_mode'],minibatches=plan['minibatches'],
                        environment_state_saved=False,velocity_mode=plan['velocity_mode']),rng_state=rng_state(),reset_generator_state=reset_tracker.generator.get_state())
                    torch.save(state,run/(name+'.pt'));runner.alg.train_mode()
                first_trace=[]
                for additional_index in range(ADD_IT):
                    telemetry.detail=additional_index==0
                    runner.current_learning_iteration=BASE_IT+additional_index
                    tick=time.perf_counter()
                    with torch.inference_mode():
                        for _ in range(ROLLOUT):
                            if additional_index==0: input_obs=obs['policy'].clone()
                            action=runner.alg.act(obs)
                            obs,reward,done,extras=env.step(action);steps+=1
                            if additional_index==0:
                                first_trace.append(dict(input_obs=input_obs,sampled_action=action.clone(),next_obs=obs['policy'].clone(),**{k:v.clone() for k,v in telemetry.latest.items()}))
                            torch._assert_async(torch.isfinite(obs['policy']).all() & torch.isfinite(reward).all(),'Non-finite rollout')
                            runner.alg.process_env_step(obs,reward,done,extras)
                        collect_s=time.perf_counter()-tick;u0=time.perf_counter()
                        runner.alg.compute_returns(obs)
                    if additional_index==0:
                        trace={k:torch.stack([x[k] for x in first_trace]).cpu() for k in first_trace[0]}
                        torch.save(trace,run/'first_rollout.pt')
                        if plan['branch']=='track_return':
                            control=torch.load(experiment/'control_stop/first_rollout.pt',weights_only=True)
                            same={k:torch.equal(v,control[k]) for k,v in trace.items() if k not in ['terms','reward']}
                            assert all(same.values()),same
                            delta=trace['terms']-control['terms'];mask=torch.ones(7,dtype=torch.bool);mask[2]=False
                            assert not delta[:,:,mask].any()
                            dump_json(run/'first_rollout_comparison.json',dict(success=True,checks=same,velocity_only=True,steps=32,transitions=32768,max_reward_difference=float((trace['reward']-control['reward']).abs().max())))
                        first_trace.clear();telemetry.detail=False
                    runner.alg.update();update_s=time.perf_counter()-u0
                    completed=additional_index+1;transitions=BASE_TRANSITIONS+steps*NUM_ENVS
                    ep=telemetry.ep_done.cpu().tolist()
                    metrics=telemetry.finish_rollout();metrics.update(loss_metrics(runner.alg))
                    std=runner.alg.actor.output_std.detach().mean(0).cpu().tolist()
                    for j,v in enumerate(std):metrics[f'Policy/std_axis_{j}']=v
                    logstd=runner.alg.actor.state_dict()['distribution.log_std_param'].detach().cpu().tolist()
                    for j,v in enumerate(logstd):metrics[f'Policy/log_std_axis_{j}']=v
                    metrics.update(reset_tracker.stats())
                    metrics['PPO/num_minibatches']=plan['minibatches']
                    metrics['PPO/minibatch_size']=BATCH//plan['minibatches']
                    metrics['Policy/std_mean']=sum(std)/4
                    metrics['PPO/learning_rate']=runner.alg.learning_rate
                    metrics['PPO/clip_epsilon']=runner.alg.clip_param
                    count=metrics['Episode/completed_count']
                    if count: metrics['Episode/failure_fraction']=metrics['Episode/end_failure_count']/count
                    if count:
                        for cause in REASONS:metrics[f'Episode/end_{cause}_fraction']=metrics[f'Episode/end_{cause}_count']/count
                    elapsed=time.perf_counter()-start;mem=memory();peak_device=max(peak_device,mem['cuda_device_used_MiB'])
                    metrics.update({'Performance/cumulative_transitions':transitions,
                        'Performance/additional_transitions':steps*NUM_ENVS,
                        'Performance/completed_iterations':BASE_IT+completed,'Performance/additional_iterations':completed,
                        'Performance/optimizer_updates':BASE_UPDATES+updates[0],'Performance/additional_optimizer_updates':updates[0],
                        'Performance/physics_batch_steps':BASE_IT*ROLLOUT*5+raw.total_motor_calls,
                        'Performance/additional_physics_batch_steps':raw.total_motor_calls,
                        'Performance/physics_env_steps':transitions*5,
                        'Performance/rollout_s':collect_s,'Performance/update_s':update_s,
                        'Performance/training_elapsed_s':elapsed,
                        'Performance/cumulative_training_elapsed_s':parent_total_training_s+elapsed,
                        'Performance/transitions_per_s':steps*NUM_ENVS/elapsed,
                        'Performance/initialization_s':verification['initialization_s'],
                        **{'Performance/'+k:v for k,v in mem.items()}})
                    totals['reward']+=metrics['Reward/step/total']*BATCH
                    totals['completed_reward']+=ep[7];totals['completed_steps']+=ep[8];totals['completed_episodes']+=int(ep[9])
                    assert updates[0]==completed*UPDATES_PER_IT and transitions==BASE_TRANSITIONS+completed*BATCH
                    assert metrics['Validation/reward_sum_max_abs_error']<1e-4 and metrics['Validation/ppo_loss_residual']<1e-6
                    metrics={k:float(np.float32(v)) for k,v in metrics.items()}
                    assert all(np.isfinite(v) for v in metrics.values())
                    for k,v in metrics.items():writer.add_scalar(k,v,steps*NUM_ENVS)
                    row=dict(global_step=steps*NUM_ENVS,iteration=BASE_IT+completed,additional_iteration=completed,**metrics)
                    rows.append(row)
                    with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                    if completed%16==0:
                        writer.flush();print('PROGRESS',completed,BASE_IT+completed,transitions,round(elapsed,2),flush=True)
                    if completed%64==0:save(f'checkpoint_{completed:03d}')
                    if completed==ADD_IT:
                        import shutil
                        shutil.copy2(run/'checkpoint_256.pt',run/'final.pt');shutil.copy2(run/'checkpoint_256_reference.pt',run/'final_reference.pt')
                torch.cuda.synchronize();wall=time.perf_counter()-start;hook.remove()
                writer.flush();writer.close();writer=None
                active=telemetry.ep.sum(0).cpu().tolist()
                accounting=dict(reward_residual=totals['reward']-totals['completed_reward']-active[7],
                    step_residual=ADD_TRANSITIONS-totals['completed_steps']-active[8],**totals)
                assert abs(accounting['reward_residual'])<1e-3 and accounting['step_residual']==0
                assert completed==ADD_IT and updates[0]==ADD_UPDATES and adam_steps(runner.alg.optimizer)==[FINAL_UPDATES]
                assert steps*NUM_ENVS==ADD_TRANSITIONS and raw.total_motor_calls==steps*5
                assert runner.current_learning_iteration==FINAL_IT-1
                assert before==protected_hashes()
                project_check=preserved(experiment/'evidence/project_before_sha256.json',ROOT)
                parent_check=dict(success=sha256(parent_path)==plan['parent_sha256'],path=str(parent_path))
                assert project_check['success'] and parent_check['success']
                dump_json(run/'preservation.json',dict(project=project_check,parent=parent_check))
                fields=list(dict.fromkeys(k for row in rows for k in row))
                with (run/'iterations.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
                result=dict(success=True,pid=os.getpid(),num_envs=NUM_ENVS,rollout_steps=ROLLOUT,minibatch_size=BATCH//plan['minibatches'],branch=plan['branch'],position_mode=plan['position_mode'],
                    base_iterations=BASE_IT,additional_iterations=completed,iterations=BASE_IT+completed,
                    framework_iteration=runner.current_learning_iteration,
                    base_transitions=BASE_TRANSITIONS,additional_transitions=steps*NUM_ENVS,transitions=BASE_TRANSITIONS+steps*NUM_ENVS,
                    base_optimizer_updates=BASE_UPDATES,additional_optimizer_updates=updates[0],optimizer_updates=BASE_UPDATES+updates[0],
                    adam_steps=adam_steps(runner.alg.optimizer),additional_policy_batch_steps=steps,
                    additional_physics_batch_steps=raw.total_motor_calls,physics_env_steps=FINAL_TRANSITIONS*5,
                    initialization_s=verification['initialization_s'],training_wall_s=wall,transitions_per_s=ADD_TRANSITIONS/wall,
                    memory_initial=initial_mem,memory_final=memory(),cuda_device_used_peak_sampled_MiB=peak_device,
                    episode_accounting=accounting,environment_state_restored=False,rng_state_restored=True,
                    parent_checkpoint=str(parent_path),parent_sha256=sha256(parent_path),final_sha256=sha256(run/'final.pt'),
                    actual_backend=raw.sim.physics_manager.__module__+'.'+raw.sim.physics_manager.__name__)
                dump_json(run/'training_result.json',result);print('TRAINING_RESULT',json.dumps(result),flush=True)
        finally:
            if writer:writer.flush();writer.close()
            raw.close();raw=None
except BaseException as e:
    dump_json(run/('preflight_failure.json' if args.preflight else 'failure.json'),
        dict(success=False,error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),
            additional_collected_transitions=steps*NUM_ENVS,additional_completed_iterations=completed,
            actual_additional_optimizer_updates=updates[0],saved_checkpoints=sorted(str(p) for p in run.glob('*.pt'))))
    raise


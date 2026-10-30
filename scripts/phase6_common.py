"""Strict phase-6 continuation helpers. Preserved phase-5 task/PPO telemetry is reused."""
import copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from phase5_common import seed_all,rng_state,equal_state,memory,Telemetry,TERMS,REASONS,install_probe,loss_metrics
from ppo_common import dump_json,model_hash,sha256
BASE_IT=256
ADD_IT=768
FINAL_IT=1024
NUM_ENVS=1024
ROLLOUT=32
BATCH=32768
BASE_TRANSITIONS=8388608
ADD_TRANSITIONS=25165824
FINAL_TRANSITIONS=33554432
BASE_UPDATES=5120
ADD_UPDATES=15360
FINAL_UPDATES=20480
INTERVAL=2097152

def jsonable(x):
    return json.loads(json.dumps(x,default=str))

def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [v for k in sorted(a.keys()|b.keys()) for v in differences(a.get(k),b.get(k),path+'.'+k if path else k)]
    return [] if a==b else [dict(path=path,before=a,after=b)]

def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def restore_rng(saved):
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch_cpu'].cpu())
    assert len(saved['torch_cuda'])==torch.cuda.device_count()
    torch.cuda.set_rng_state_all([x.cpu() for x in saved['torch_cuda']])

def rng_fingerprints(state):
    n=state['numpy']
    return dict(python=hashlib.sha256(repr(state['python']).encode()).hexdigest(),
        numpy=hashlib.sha256(n[0].encode()+n[1].tobytes()+repr(n[2:]).encode()).hexdigest(),
        torch_cpu=tensor_sha(state['torch_cpu']),torch_cuda=[tensor_sha(x) for x in state['torch_cuda']])

def adam_steps(optimizer):
    return sorted({int(x['step']) for x in optimizer.state.values()})

def verify_resume(runner,parent,reference,agent,parent_agent,path):
    alg=runner.alg
    state=alg.save()
    checks={
        'actor_exact':equal_state(state['actor_state_dict'],parent['actor_state_dict']),
        'critic_exact':equal_state(state['critic_state_dict'],parent['critic_state_dict']),
        'distribution_exact':equal_state(state['actor_state_dict']['distribution.log_std_param'],parent['actor_state_dict']['distribution.log_std_param']),
        'adam_all_state_exact':equal_state(state['optimizer_state_dict'],parent['optimizer_state_dict']),
        'adam_steps_5120':adam_steps(alg.optimizer)==[BASE_UPDATES],
        'restored_parameter_count':len(alg.optimizer.state)==len(parent['optimizer_state_dict']['state'])==13,
        'ppo_config_equal':agent['algorithm']==parent_agent['algorithm'],
        'actor_config_equal':agent['actor']==parent_agent['actor'],
        'critic_config_equal':agent['critic']==parent_agent['critic'],
        'learning_rate_equal':alg.learning_rate==parent_agent['algorithm']['learning_rate']==.0003,
        'fixed_schedule':alg.schedule=='fixed',
        'framework_iteration_255':runner.current_learning_iteration==BASE_IT-1==parent['iter'],
        'parent_completed_iterations_256':parent['infos']['completed_iterations']==BASE_IT,
        'parent_transitions_8388608':parent['infos']['collected_transitions']==BASE_TRANSITIONS,
        'parent_updates_5120':parent['infos']['optimizer_updates']==BASE_UPDATES,
    }
    policy=runner.get_inference_policy(device='cuda:0')
    with torch.inference_mode():
        predicted=policy(TensorDict({'policy':reference['observations'].to('cuda:0')},batch_size=[len(reference['observations'])]))
    expected=reference['deterministic_actions'].to('cuda:0')
    error=float((predicted-expected).abs().max())
    checks['deterministic_action_equal']=torch.equal(predicted,expected)
    checks['deterministic_action_error_ok']=error<=1e-6
    names={id(v):'actor.'+k for k,v in alg.actor.named_parameters()}
    names.update({id(v):'critic.'+k for k,v in alg.critic.named_parameters()})
    parameter_records=[]
    for actual_group,saved_group in zip(alg.optimizer.param_groups,parent['optimizer_state_dict']['param_groups']):
        for param,index in zip(actual_group['params'],saved_group['params']):
            actual=alg.optimizer.state[param];saved=parent['optimizer_state_dict']['state'][index]
            rec=dict(name=names[id(param)],parameter_id=index,shape=list(param.shape),step=int(actual['step']),
                step_exact=equal_state(actual['step'],saved['step']),
                exp_avg_exact=equal_state(actual['exp_avg'],saved['exp_avg']),
                exp_avg_sq_exact=equal_state(actual['exp_avg_sq'],saved['exp_avg_sq']),
                exp_avg_sha256=tensor_sha(actual['exp_avg']),exp_avg_sq_sha256=tensor_sha(actual['exp_avg_sq']),
                exp_avg_nonzero=int(torch.count_nonzero(actual['exp_avg'])),
                exp_avg_sq_nonzero=int(torch.count_nonzero(actual['exp_avg_sq'])))
            parameter_records.append(rec)
    checks['all_parameter_moments_exact']=all(x['step_exact'] and x['exp_avg_exact'] and x['exp_avg_sq_exact'] for x in parameter_records)
    result=dict(success=all(checks.values()),checks=checks,deterministic_action_max_abs_error=error,
        reference_count=len(reference['observations']),actor_sha256=model_hash(alg.actor),critic_sha256=model_hash(alg.critic),
        log_std=alg.actor.state_dict()['distribution.log_std_param'].detach().cpu().tolist(),
        std=alg.actor.state_dict()['distribution.log_std_param'].exp().detach().cpu().tolist(),
        optimizer_class=type(alg.optimizer).__name__,optimizer_parameter_states=parameter_records,
        optimizer_param_groups=jsonable(state['optimizer_state_dict']['param_groups']),algorithm_config=agent['algorithm'],
        normalizers={who:{name:type(module).__name__ for name,module in model.named_modules() if 'normaliz' in name}
            for who,model in [('actor',alg.actor),('critic',alg.critic)]},
        normalizer_state_note='obs_normalization=false; full actor/critic state_dict including all buffers strictly restored',
        environment_state_restored=False,environment_reset_seed=42,rollout_buffer_restored=False,
        episode_accumulators_restored=False,additional_optimizer_updates_at_verification=0)
    dump_json(path,result)
    assert result['success'],result
    alg.train_mode()
    return result

def preserved(path,root):
    prior=json.loads(Path(path).read_text())
    changed=[name for name,want in prior.items() if not (root/name).is_file() or sha256(root/name)!=want]
    return dict(success=not changed,checked=len(prior),changed=changed)


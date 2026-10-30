"""Phase9 strict shared-parent continuation and reward-only intervention."""
import copy,json,types
from pathlib import Path
import torch
from tensordict import TensorDict
from ppo_common import dump_json,model_hash,sha256
from phase5_common import seed_all,rng_state,equal_state,memory,TERMS,REASONS,install_probe,loss_metrics
from phase6_common import jsonable,differences,tensor_sha,restore_rng,rng_fingerprints,adam_steps,preserved
from phase9_metrics import Telemetry
BASE_IT=1408
ADD_IT=256
FINAL_IT=1664
NUM_ENVS=1024
ROLLOUT=32
BATCH=32768
BASE_TRANSITIONS=46137344
ADD_TRANSITIONS=8388608
FINAL_TRANSITIONS=54525952
BASE_UPDATES=143360
INTERVAL=2097152

from phase9_reward import apply_reward

def verify_resume(runner,parent,reference,agent,parent_agent,path,plan):
    alg=runner.alg;state=alg.save()
    expected=copy.deepcopy(parent_agent);expected['algorithm']['num_mini_batches']=plan['minibatches']
    checks=dict(actor_exact=equal_state(state['actor_state_dict'],parent['actor_state_dict']),
        critic_exact=equal_state(state['critic_state_dict'],parent['critic_state_dict']),
        distribution_exact=equal_state(state['actor_state_dict']['distribution.log_std_param'],parent['actor_state_dict']['distribution.log_std_param']),
        adam_all_state_exact=equal_state(state['optimizer_state_dict'],parent['optimizer_state_dict']),
        adam_steps_143360=adam_steps(alg.optimizer)==[BASE_UPDATES],
        parameter_states_13=len(alg.optimizer.state)==13,
        ppo_config_expected=agent['algorithm']==expected['algorithm'],
        actor_config_equal=agent['actor']==parent_agent['actor'],critic_config_equal=agent['critic']==parent_agent['critic'],
        actual_minibatches=alg.num_mini_batches==plan['minibatches'],
        learning_rate_fixed=alg.learning_rate==.0003 and alg.schedule=='fixed',
        framework_iteration_1407=runner.current_learning_iteration==parent['iter']==1407,
        parent_counters=parent['infos']['completed_iterations']==BASE_IT and parent['infos']['collected_transitions']==BASE_TRANSITIONS and parent['infos']['optimizer_updates']==BASE_UPDATES)
    obs=TensorDict({'policy':reference['observations'].to('cuda:0')},batch_size=[len(reference['observations'])])
    alg.eval_mode()
    with torch.inference_mode():
        mean=alg.actor(obs).cpu();critic=alg.critic(obs).cpu()
        logstd=alg.actor.state_dict()['distribution.log_std_param'].cpu();std=logstd.exp().expand_as(mean).clone()
    output=dict(observations=reference['observations'].cpu(),actor_mean=mean,actor_std=std,critic=critic,log_std=logstd)
    # Read-only output collection never samples the stochastic distribution.
    common=Path(plan['experiment'])/'common/initial_outputs.pt'
    if common.exists():
        saved=torch.load(common,map_location='cpu',weights_only=True)
        checks['shared_initial_outputs_exact']=equal_state(output,saved)
    else:
        assert plan['branch']=='control_stop'
        torch.save(output,common);checks['shared_initial_outputs_exact']=True
    checks['parent_reference_action_exact']=torch.equal(mean,reference['deterministic_actions'].cpu())
    records=[]
    current=alg.optimizer.state_dict()
    for key,x in current['state'].items():
        y=parent['optimizer_state_dict']['state'][key]
        records.append(dict(parameter_id=key,step=int(x['step']),step_exact=equal_state(x['step'],y['step']),
            exp_avg_exact=equal_state(x['exp_avg'],y['exp_avg']),exp_avg_sq_exact=equal_state(x['exp_avg_sq'],y['exp_avg_sq']),
            exp_avg_sha256=tensor_sha(x['exp_avg']),exp_avg_sq_sha256=tensor_sha(x['exp_avg_sq'])))
    result=dict(success=all(checks.values()),checks=checks,parent_sha256=plan['parent_sha256'],branch=plan['branch'],
        actor_sha256=model_hash(alg.actor),critic_sha256=model_hash(alg.critic),
        mean_sha256=tensor_sha(mean),std_sha256=tensor_sha(std),critic_output_sha256=tensor_sha(critic),
        max_action_error=float((mean-reference['deterministic_actions']).abs().max()),std=logstd.exp().tolist(),log_std=logstd.tolist(),
        optimizer_parameters=records,normalizers={who:{name:type(m).__name__ for name,m in model.named_modules() if 'normaliz' in name}
        for who,model in [('actor',alg.actor),('critic',alg.critic)]},
        environment_state_restored=False,rollout_restored=False,normalization_note='Identity; full state_dict strict load; fixed LR no scheduler state')
    dump_json(path,result);assert result['success'],checks
    alg.train_mode()
    return result


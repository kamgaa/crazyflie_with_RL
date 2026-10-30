"""No-update check of installed timeout handling and checkpoint optimizer/RNG state."""
import argparse,json,torch,types
from pathlib import Path
from rsl_rl.algorithms import PPO
from ppo_common import dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir)
captured={}
model=types.SimpleNamespace(update_normalization=lambda x:None,reset=lambda d:None)
transition=types.SimpleNamespace(values=torch.tensor([[2.],[3.]],device='cuda:0'),clear=lambda:None)
storage=types.SimpleNamespace(add_transition=lambda t:captured.update(rewards=t.rewards.clone(),dones=t.dones.clone()))
alg=types.SimpleNamespace(actor=model,critic=model,rnd=None,transition=transition,storage=storage,gamma=.99,device='cuda:0')
rewards=torch.tensor([-10.,-1.],device='cuda:0');dones=torch.tensor([1,1],device='cuda:0')
PPO.process_env_step(alg,None,rewards,dones,{'time_outs':torch.tensor([False,True],device='cuda:0')})
assert torch.allclose(captured['rewards'],torch.tensor([-10.,1.97],device='cuda:0'))
assert torch.equal(rewards,torch.tensor([-10.,-1.],device='cuda:0'))
result=dict(optimizer_updates=0,stock_timeout_bootstrap_verified=True,original_reward_unchanged=True,bootstrapped_rewards=captured['rewards'].cpu().tolist(),
    note='Installed stock RSL uses stored current-state value for timeout correction, not terminal-state value. Training encountered 0 timeouts; semantics retained.')
for name in ['checkpoint_100','checkpoint_200','checkpoint_300','checkpoint_400','final','best']:
    cp=torch.load(r/(name+'.pt'),map_location='cpu',weights_only=False)
    assert 'rng_state' in cp and set(cp['rng_state'])=={'python','numpy','torch_cpu','torch_cuda'}
    steps=sorted({int(v['step']) for v in cp['optimizer_state_dict']['state'].values()})
    assert steps==[cp['infos']['optimizer_updates']]
    result[name]=dict(iteration=cp['infos']['completed_iterations'],adam_steps=steps,rng_saved=True)
dump_json(r/'evidence/timeout_checkpoint_checks.json',result);print(json.dumps(result))

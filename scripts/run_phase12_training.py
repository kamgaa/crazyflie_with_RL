"""Sequential, fixed-budget fresh training; never retries an attempted training run."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();plan=json.loads((r/'experiment_plan.json').read_text());records=[]
try:
 for branch,path in plan['branches'].items():
  d=Path(path)
  for script,extra,flag,logname in [('validate_phase11_reward.py',[], 'reward_ready.json','reward_validation.log'),('train_phase11.py',['--preflight'], 'preflight.json','preflight.log')]:
   if (d/flag).exists():assert json.loads((d/flag).read_text())['success'];continue
   command=[sys.executable,str(ROOT/'scripts'/script)]
   command += [str(d)] if script.startswith('validate') else ['--run-dir',str(d),'--iterations','512','--device','cuda:0','--headless','--physics','isaacsim_physx',*extra]
   with (d/logname).open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
 # Fresh initial actor/critic, initial reset and RNG must agree before any learning.
 import torch
 proofs={b:json.loads((Path(path)/'preflight_fresh_verification.json').read_text()) for b,path in plan['branches'].items()}
 keys=['actor_sha256','critic_sha256','adam_state_count','log_std','rng_fingerprints']
 assert all(proofs['phase12-1'][k]==proofs['phase12-2'][k] for k in keys)
 resets=[torch.load(Path(path)/'preflight_initial_reset.pt',weights_only=True,map_location='cpu') for path in plan['branches'].values()]
 assert all(torch.equal(resets[0][k],resets[1][k]) for k in resets[0])
 base=json.loads((Path(plan['baseline'])/'fresh_verification.json').read_text())
 baseline_same={k:proofs['phase12-1'][k]==base[k] for k in ['actor_sha256','critic_sha256','adam_state_count','log_std','rng_fingerprints']}
 assert all(baseline_same.values()),baseline_same
 dump_json(r/'common_initialization_verification.json',dict(success=True,branch_models_and_rng_identical=True,branch_reset_states_identical=True,baseline_phase11_matches=baseline_same,optimizer_updates=0,proofs=proofs))
 for branch,path in plan['branches'].items():
  d=Path(path);assert not (d/'training_started.json').exists()
  cmd=[sys.executable,str(ROOT/'scripts/train_phase11.py'),'--run-dir',str(d),'--iterations','512','--device','cuda:0','--headless','--physics','isaacsim_physx']
  start=time.time();print('START_TRAIN',branch,flush=True)
  with (d/'training.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  result=json.loads((d/'training_result.json').read_text());assert result['success']
  records.append(dict(branch=branch,command=cmd,wall_s=time.time()-start,result=str(d/'training_result.json')))
  dump_json(r/'training_sequence.json',dict(success=len(records)==2,completed=records));print('DONE_TRAIN',branch,flush=True)
except BaseException:
 dump_json(r/'training_sequence_failure.json',dict(error=traceback.format_exc(),completed=records));raise

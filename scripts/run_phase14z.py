"""One fresh run, fixed 512 iterations; never retry training automatically."""
import argparse,json,subprocess,sys,time,traceback,shutil
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
plan=json.loads((r/'experiment_plan.json').read_text());base=Path(plan['configuration_source']);records=[]
def stage(script,extra=(),log=None):
 cmd=[sys.executable,str(ROOT/'scripts'/script),'--run-dir',str(r),*extra]
 tick=time.perf_counter()
 with (r/(log or script.replace('.py','.log'))).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 records.append(dict(script=script,command=cmd,wall_s=time.perf_counter()-tick))
 dump_json(r/'sequence.json',dict(success=False,completed=records));print('DONE',script,*extra,flush=True)
try:
 assert json.loads((r/'reward_ready.json').read_text())['success']
 stage('train_phase14z.py',['--preflight','--iterations','512','--device','cuda:0','--headless','--physics','isaacsim_physx'],'preflight.log')
 import torch
 proof=json.loads((r/'preflight_fresh_verification.json').read_text())
 bp=json.loads((base/'fresh_verification.json').read_text())
 checks={k:proof[k]==bp[k] for k in ['actor_sha256','critic_sha256','adam_state_count','log_std','rng_fingerprints']}
 x=torch.load(r/'preflight_initial_reset.pt',weights_only=True,map_location='cpu')
 y=torch.load(base/'initial_reset.pt',weights_only=True,map_location='cpu')
 checks['identical_initial_reset']=all(torch.equal(x[k],y[k]) for k in x)
 assert all(checks.values()),checks
 dump_json(r/'baseline_initialization_verification.json',dict(success=True,checks=checks,baseline=str(base),optimizer_updates=0))
 snapshot=r/'source_snapshot'
 shutil.copytree(ROOT/'scripts',snapshot/'scripts',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
 shutil.copytree(ROOT/'src',snapshot/'src',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
 dump_json(r/'runtime_source_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'scripts').glob('*phase14z*.py')})
 stage('train_phase14z.py',['--iterations','512','--device','cuda:0','--headless','--physics','isaacsim_physx'],'training.log')
 tr=json.loads((r/'training_result.json').read_text());assert tr['success']
 assert tr['transitions']==16777216 and tr['optimizer_updates']==163840 and tr['iterations']==512
 x=torch.load(r/'first_rollout.pt',weights_only=True,map_location='cpu');y=torch.load(base/'first_rollout.pt',weights_only=True,map_location='cpu')
 checks={k:torch.equal(x[k],y[k]) for k in x if k not in ['terms','reward']}
 keep=[i for i in range(13) if i!=5]
 checks['other_reward_terms_identical']=torch.equal(x['terms'][...,keep],y['terms'][...,keep])
 checks['only_z_difference_in_total']=torch.allclose(x['reward']-y['reward'],x['terms'][...,5]-y['terms'][...,5],atol=2e-6)
 checks['squared_formula']=torch.allclose(x['terms'][...,5],-.015*x['obs'][...,5].square(),atol=1e-7)
 assert all(checks.values()),checks
 dump_json(r/'first_rollout_comparison.json',dict(success=True,checks=checks,baseline=str(base),before_first_optimizer_update=True))
 stage('finalize_phase14z.py')
 stage('audit_phase14z.py')
 dump_json(r/'sequence.json',dict(success=True,completed=records));print('TRAIN_AND_HOVER_DONE',flush=True)
except BaseException:
 dump_json(r/'sequence_failure.json',dict(error=traceback.format_exc(),completed=records));raise

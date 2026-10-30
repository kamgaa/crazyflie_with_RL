"""Select final best from hover checkpoints, then explicit independent reload evaluations."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();plan=json.loads((r/'experiment_plan.json').read_text());tr=json.loads((r/'training_result.json').read_text());assert tr['success']
assert not (r/'best_selection.json').exists()
candidates=[]
for k in range(0,513,64):
 folder=r/'evaluation'/f'validation_{k:03d}';data=json.loads((folder/'results.json').read_text());assert len(data)==7
 n=sum(x['completed'] for x in data);cp=r/('initial.pt' if k==0 else f'checkpoint_{k:03d}.pt')
 candidates.append(dict(additional_iteration=k,checkpoint=str(cp),sha256=sha256(cp),completed=n,mean_duration_s=sum(x['duration_s'] for x in data)/7,mean_position_rmse_m=sum(x['position_rmse_m'] for x in data)/7 if n==7 else None,evaluation=str(folder/'results.json')))
def score(x):return (-x['completed'],-x['mean_duration_s'] if x['completed']<7 else x['mean_position_rmse_m'],x['additional_iteration'])
best=min(candidates,key=score);cp=Path(best['checkpoint']);shutil.copy2(cp,r/'best.pt');shutil.copy2(cp.with_name(cp.stem+'_reference.pt'),r/'best_reference.pt')
dump_json(r/'best_selection.json',dict(rule=plan['best_rule'],selected=best,initial_untrained_selected=best['additional_iteration']==0,candidates=candidates,best_sha256=sha256(r/'best.pt'),final_sha256=sha256(r/'final.pt'),best_final_identical=sha256(r/'best.pt')==sha256(r/'final.pt'),circle_used_for_selection=False))
base=Path(plan['configuration_source']);baseline=[]
for suite in ['fixed','random20']:
 source=base/'evaluation'/f'best_{suite}_30s';dest=r/'evaluation'/f'phase12_baseline_{suite}_30s';shutil.copytree(source,dest)
 baseline.append(dict(suite=suite,source=str(source),copy=str(dest),results_sha256=sha256(source/'results.json'),checkpoint=str(base/'best.pt'),checkpoint_sha256=sha256(base/'best.pt'),historical_reused=True,new_evaluation=False))
dump_json(r/'baseline_evaluation_provenance.json',baseline)
records=[]
for label in ['best','final']:
 cp=r/(label+'.pt')
 for suite in ['fixed','random20']:
  out=r/'evaluation'/f'{label}_{suite}_30s'
  command=[sys.executable,str(ROOT/'scripts/evaluate_phase13.py'),'--checkpoints',str(cp),'--run-dir',str(r),'--output',str(out),'--positions',str(r/'evaluation/initial_positions.json'),'--suite',suite,'--horizon','30','--device','cuda:0','--headless','--physics','isaacsim_physx']
  tick=time.perf_counter()
  with out.with_suffix('.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  records.append(dict(model=label,suite=suite,checkpoint=str(cp),sha256=sha256(cp),output=str(out),wall_s=time.perf_counter()-tick,command=command));dump_json(r/'final_evaluation_execution.json',records);print('FINAL_EVAL_DONE',label,suite,flush=True)

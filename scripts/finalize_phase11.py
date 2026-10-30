"""Fixed predeclared 30 s selection followed by independent explicit reload evaluations."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
plan=json.loads((r/'experiment_plan.json').read_text());tr=json.loads((r/'training_result.json').read_text());assert tr['success'] and tr['additional_iterations']==512
candidates=[]
for k in range(0,513,64):
 d=r/'evaluation'/f'validation_{k:03d}';results=json.loads((d/'results.json').read_text());assert len(results)==7
 n=sum(x['completed'] for x in results);duration=sum(x['duration_s'] for x in results)/7
 rmse=sum(x['position_rmse_m'] for x in results)/7 if n==7 else None
 cp=r/('initial.pt' if k==0 else f'checkpoint_{k:03d}.pt')
 candidates.append(dict(additional_iteration=k,checkpoint=str(cp),sha256=sha256(cp),completed=n,mean_duration_s=duration,mean_position_rmse_m=rmse,evaluation=str(d/'results.json')))
def score(x):return (-x['completed'],-x['mean_duration_s'] if x['completed']<7 else x['mean_position_rmse_m'],x['additional_iteration'])
best=min(candidates,key=score);source=Path(best['checkpoint'])
shutil.copy2(source,r/'best.pt');shutil.copy2(source.with_name(source.stem+'_reference.pt'),r/'best_reference.pt')
dump_json(r/'best_selection.json',dict(rule=plan['best_rule'],selected=best,initial_untrained_selected=best['additional_iteration']==0,candidates=candidates,best_sha256=sha256(r/'best.pt'),final_sha256=sha256(r/'final.pt'),best_final_identical=sha256(r/'best.pt')==sha256(r/'final.pt')))
models={'phase10':(Path(plan['comparison_checkpoint']),Path(plan['comparison_checkpoint']).parent),'best':(r/'best.pt',r),'final':(r/'final.pt',r)}
records=[]
for label,(cp,cfg) in models.items():
 for suite in ['fixed','random20']:
  out=r/'evaluation'/f'{label}_{suite}_30s'
  command=[sys.executable,str(ROOT/'scripts/evaluate_phase11.py'),'--checkpoints',str(cp),'--run-dir',str(cfg),'--output',str(out),'--positions',str(r/'evaluation/initial_positions.json'),'--suite',suite,'--horizon','30','--device','cuda:0','--headless','--physics','isaacsim_physx']
  tick=time.perf_counter()
  with out.with_suffix('.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  records.append(dict(model=label,suite=suite,checkpoint=str(cp),sha256=sha256(cp),output=str(out),wall_s=time.perf_counter()-tick,command=command))
  dump_json(r/'final_evaluation_execution.json',records)
  print('FINAL_EVAL_DONE',label,suite,flush=True)
print('FINALIZATION_DONE',flush=True)

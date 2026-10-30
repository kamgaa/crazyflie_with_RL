"""Serial fixed-budget workflow. Stops on integrity errors; never retries training."""
import argparse,json,subprocess,sys,time,shutil
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();commands=[]
def run(script,args,log):
 cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)];commands.append(cmd);dump_json(r/'actual_commands.json',commands)
 start=time.perf_counter()
 with Path(log).open('w') as f:ret=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 dump_json(r/'workflow_status.json',dict(script=script,args=list(map(str,args)),returncode=ret.returncode,wall_s=time.perf_counter()-start,time=time.time()))
 if ret.returncode:raise RuntimeError(f'{script} failed; inspect {log}; no automatic retry')
def evaluate(cp,rd,out,horizon,suite='fixed'):
 run('evaluate_phase9.py',['--checkpoints',cp,'--run-dir',rd,'--output',out,'--positions',r/'evaluation/initial_positions.json','--suite',suite,'--horizon',horizon,'--device','cuda:0','--headless','--physics','isaacsim_physx'],str(out)+'.log')
assert json.loads((r/'comparison_ready.json').read_text())['success']
for b in ['control_stop','track_return']:
 run('train_phase9.py',['--run-dir',r/b,'--additional-iterations',256,'--device','cuda:0','--headless','--physics','isaacsim_physx'],r/b/'training.log')
 assert json.loads((r/b/'training_result.json').read_text())['success']
for b in ['control_stop','track_return']:
 candidates=[dict(additional_iteration=0,path=str(r/b/'initial.pt'),results=json.loads((r/'evaluation/parent_preflight_8/results.json').read_text()))]
 for it in [64,128,192,256]:
  cp=r/b/f'checkpoint_{it:03d}.pt';out=r/b/'evaluation'/f'validation_{it:03d}_8s';evaluate(cp,r/b,out,8)
  candidates.append(dict(additional_iteration=it,path=str(cp),results=json.loads((out/'results.json').read_text())))
 def score(c):
  res=c['results'];n=sum(x['completed'] for x in res);duration=sum(x['duration_s'] for x in res)/7
  return (-n,-duration,sum(x['position_rmse_m'] for x in res)/7 if n==7 else 0,sum(x['tail_speed_mean_m_s'] for x in res)/7 if n==7 else 0,c['additional_iteration'])
 best=min(candidates,key=score);src=Path(best['path']);shutil.copy2(src,r/b/'best.pt');shutil.copy2(src.with_name(src.stem+'_reference.pt'),r/b/'best_reference.pt')
 dump_json(r/b/'best_selection.json',dict(source=str(src),additional_iteration=best['additional_iteration'],parent_selected=best['additional_iteration']==0,sha256=sha256(r/b/'best.pt'),score=list(score(best)),candidates=[dict(additional_iteration=c['additional_iteration'],path=c['path'],score=list(score(c)),results=c['results']) for c in candidates],selection_uses='7 fixed initial conditions, 8 seconds only; 30s and random20 excluded'))
models=[('parent',r/'common/initial.pt',r/'common')]+[(b+'_'+kind,r/b/(kind+'.pt'),r/b) for b in ['control_stop','track_return'] for kind in ['best','final']]
for label,cp,rd in models:
 for horizon,suite in [(8,'fixed'),(30,'fixed'),(30,'random20')]:
  evaluate(cp,rd,r/'evaluation'/f'{label}_{suite}_{horizon}s',horizon,suite)
dump_json(r/'execution_complete.json',dict(success=True,time=time.time(),total_additional_transitions=16777216,total_additional_optimizer_updates=163840,models=[x[0] for x in models],no_extra_training=True))
print('PHASE9_TRAIN_EVAL_COMPLETE',flush=True)

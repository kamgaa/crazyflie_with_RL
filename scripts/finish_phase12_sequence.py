"""Wait for exactly the two authorized training runs; evaluation/postprocessing only afterward."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();plan=json.loads((r/'experiment_plan.json').read_text());records=[]
def call(script,branch,extra=None):
 d=Path(branch);command=[sys.executable,str(ROOT/'scripts'/script),'--run-dir',str(d),*(extra or [])];start=time.time()
 log=d/(script.replace('.py','')+('_T'+str(extra[extra.index('--period')+1]) if extra and '--period' in extra else '')+'.log')
 with log.open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
 records.append(dict(script=script,run=str(d),command=command,wall_s=time.time()-start,log=str(log)));dump_json(r/'postprocessing_sequence.json',dict(success=False,completed=records));print('DONE',script,d.name,flush=True)
try:
 while True:
  if (r/'training_sequence_failure.json').exists():raise RuntimeError((r/'training_sequence_failure.json').read_text())
  if (r/'training_sequence.json').exists() and json.loads((r/'training_sequence.json').read_text())['success']:break
  time.sleep(5)
 for branch,path in plan['branches'].items():
  d=Path(path)
  for script in ['finalize_phase12.py','audit_phase12.py','analyze_phase12.py','plot_phase12_comparison.py','prepare_circle_phase12.py']:call(script,d)
  for T in [10,5]:call('evaluate_circle_phase12.py',d/'circle',['--controller','phase12_best','--period',str(T),'--device','cuda:0','--headless','--physics','isaacsim_physx'])
  call('analyze_phase12_circle_attitude.py',d/'circle')
 dump_json(r/'postprocessing_sequence.json',dict(success=True,completed=records));print('ALL_EVALUATIONS_AND_BRANCH_PLOTS_DONE',flush=True)
except BaseException:
 dump_json(r/'postprocessing_sequence_failure.json',dict(error=traceback.format_exc(),completed=records));raise

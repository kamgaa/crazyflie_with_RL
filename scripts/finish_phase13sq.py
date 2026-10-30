"""Sequential postprocessing after the single fixed-budget training; no learning retries."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir);records=[]
def stage(name,target=None,extra=()):
 cmd=[sys.executable,str(ROOT/'scripts'/name),'--run-dir',str(target or r),*extra]
 log=r/(name.replace('.py','')+('_T'+extra[extra.index('--period')+1] if '--period' in extra else '')+'.log')
 tick=time.perf_counter()
 with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 records.append(dict(script=name,command=cmd,wall_s=time.perf_counter()-tick))
 dump_json(r/'postprocessing_sequence.json',dict(success=False,completed=records));print('DONE',name,*extra,flush=True)
try:
 while True:
  assert not (r/'sequence_failure.json').exists(),(r/'sequence_failure.json').read_text() if (r/'sequence_failure.json').exists() else ''
  if (r/'sequence.json').exists() and json.loads((r/'sequence.json').read_text())['success']:break
  time.sleep(10)
 stage('analyze_phase13sq.py')
 stage('plot_phase13sq_comparison.py')
 stage('prepare_circle_phase13sq.py')
 for period in ['10','5']:
  stage('evaluate_circle_phase13sq.py',r/'circle',['--controller','phase13sq_best','--period',period,'--device','cuda:0','--headless','--physics','isaacsim_physx'])
 stage('analyze_phase13sq_circle_attitude.py',r/'circle')
 stage('report_phase13sq.py')
 stage('package_phase13sq.py')
 dump_json(r/'postprocessing_sequence.json',dict(success=True,completed=records));print('PHASE13_DELIVERY_READY',flush=True)
except BaseException:
 dump_json(r/'postprocessing_failure.json',dict(error=traceback.format_exc(),completed=records));raise

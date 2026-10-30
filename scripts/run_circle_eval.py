"""Six sequential, independent evaluation processes; no training or tuning."""
import argparse,json,subprocess,sys,time,os
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
assert json.loads((r/'preflight_verification.json').read_text())['success']
jobs=[]
for period in [10,5]:
 for name in ['PID','Q_base','E_position']:
  key=f'{name}_T{period}';out=r/'cases'/key
  assert not out.exists(),f'Refusing to rerun {out}'
  cmd=[sys.executable,str(ROOT/'scripts/evaluate_circle.py'),'--run-dir',str(r),'--controller',name,'--period',str(period),'--device','cuda:0','--headless','--physics','isaacsim_physx']
  t=time.perf_counter();print('START',key,flush=True)
  with (r/'cases'/(key+'.log')).open('w') as f:
   result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1'})
  job=dict(case=key,command=cmd,exit_code=result.returncode,wall_s=time.perf_counter()-t);jobs.append(job);dump_json(r/'evaluation_jobs.json',jobs)
  print('FINISH',key,result.returncode,job['wall_s'],flush=True)
  if (out/'result.json').exists():
   z=json.loads((out/'result.json').read_text());print('SUMMARY',key,json.dumps(z['metrics']),flush=True)
dump_json(r/'evaluation_finished.json',dict(success=all(x['exit_code']==0 for x in jobs),jobs=len(jobs),optimizer_updates=0))
print('CIRCLE_EVALUATION_FINISHED',flush=True)

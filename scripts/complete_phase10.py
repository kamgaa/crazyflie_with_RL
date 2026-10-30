"""Finish authorized evaluation/report only; never restarts or extends training."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
state=json.loads((r/'training_started.json').read_text());pid=state['pid']
while True:
 try:os.kill(pid,0)
 except ProcessLookupError:break
 if (r/'failure.json').exists():raise RuntimeError((r/'failure.json').read_text())
 time.sleep(5)
assert json.loads((r/'training_result.json').read_text())['success']
records=[]
for script in ['finalize_phase10.py','analyze_phase10.py','report_phase10.py','package_phase10.py']:
 start=time.perf_counter();command=[sys.executable,str(ROOT/'scripts'/script),'--run-dir',str(r)]
 with (r/(script.replace('.py','.log'))).open('w') as out:
  subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True)
 records.append(dict(script=script,wall_s=time.perf_counter()-start,command=command));print('DONE',script,flush=True)
 # Keep this mutable orchestration log outside the archived run.
print('PHASE10_COMPLETE',flush=True)

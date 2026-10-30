"""Only postprocessing; no optimizer, simulator, or training retry."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();done=[]
try:
 while True:
  for f in ['training_sequence_failure.json','postprocessing_sequence_failure.json']:
   if (r/f).exists():raise RuntimeError((r/f).read_text())
  f=r/'postprocessing_sequence.json'
  if f.exists() and json.loads(f.read_text())['success']:break
  time.sleep(5)
 for script in ['report_phase12.py','package_phase12.py']:
  cmd=[sys.executable,str(ROOT/'scripts'/script),'--run-dir',str(r)];start=time.perf_counter()
  with (r/script.replace('.py','.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  done.append(dict(script=script,command=cmd,seconds=time.perf_counter()-start));print('DONE',script,flush=True)
 dump_json(r/'delivery_sequence.json',dict(success=True,completed=done));print('PHASE12_DELIVERY_READY',flush=True)
except BaseException:
 dump_json(r/'delivery_failure.json',dict(error=traceback.format_exc(),completed=done));raise

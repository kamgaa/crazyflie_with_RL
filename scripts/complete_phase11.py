"""Wait for the single authorized training run, then evaluation and reporting. Never restarts training."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
from ppo_common import ROOT
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();pid=json.loads((r/'training_started.json').read_text())['pid']
while True:
 if (r/'failure.json').exists():raise RuntimeError((r/'failure.json').read_text())
 try:os.kill(pid,0)
 except ProcessLookupError:break
 time.sleep(5)
assert json.loads((r/'training_result.json').read_text())['success']
for script in ['finalize_phase11.py','audit_phase11.py','analyze_phase11.py','plot_phase11_comparison.py','report_phase11.py','package_phase11.py']:
 command=[sys.executable,str(ROOT/'scripts'/script),'--run-dir',str(r)]
 with (r/(script.replace('.py','.log'))).open('w') as out:subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True)
 print('DONE',script,flush=True)
print('PHASE11_COMPLETE',flush=True)

"""Offline completion only; no learning or simulator execution."""
import argparse,os,json,time,subprocess,sys
from pathlib import Path
from ppo_common import ROOT
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--analysis-pid',type=int,required=True);a=p.parse_args();r=Path(a.run_dir)
while True:
 try:os.kill(a.analysis_pid,0)
 except ProcessLookupError:break
 time.sleep(3)
assert json.loads((r/'verification.json').read_text())['success']
for script in ['plot_phase11_comparison.py','report_phase11.py','package_phase11.py']:
 with (r/script.replace('.py','.log')).open('w') as f:subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--run-dir',str(r)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',script,flush=True)
print('PHASE11_COMPLETE',flush=True)

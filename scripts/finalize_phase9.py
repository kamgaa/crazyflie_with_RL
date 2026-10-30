import argparse,subprocess,sys,time,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--workflow-pid',type=int,required=True);a=p.parse_args();r=Path(a.run_dir)
while not (r/'execution_complete.json').exists():
 if not Path(f'/proc/{a.workflow_pid}').exists():raise RuntimeError('Workflow ended before completion; inspect workflow.log')
 time.sleep(10)
for script in ['audit_phase9_trajectory.py','analyze_phase9.py','verify_phase9_delivery.py','report_phase9.py','package_phase9.py']:
 with (r/('finalize_'+script+'.log')).open('w') as f:
  ret=subprocess.run([sys.executable,'scripts/'+script,'--run-dir',str(r)],stdout=f,stderr=subprocess.STDOUT)
 if ret.returncode:raise RuntimeError(f'{script} failed, inspect finalize log')
print('FINALIZATION_COMPLETE',flush=True)

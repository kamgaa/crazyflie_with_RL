"""Wait for isolated evaluations, then produce the comparison and report."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args();r=Path(a.experiment).resolve()
while not (r/'evaluation_finished.json').exists():time.sleep(5)
results=[]
for script in ['analyze_phase5.py','report_phase5.py']:
    cmd=[sys.executable,str(ROOT/'scripts'/script),'--experiment',str(r)];started=time.time()
    with (r/(Path(script).stem+'.log')).open('w') as f:ret=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    results.append(dict(script=script,returncode=ret.returncode,wall_s=time.time()-started));dump_json(r/'postprocessing.json',results)
    if ret.returncode:raise SystemExit(ret.returncode)
dump_json(r/'postprocessing_finished.json',dict(time=time.time(),steps=results))

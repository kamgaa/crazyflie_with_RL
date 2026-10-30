"""Wait for evaluations then perform CPU analysis/report; no further learning."""
import argparse,json,sys,subprocess,time,traceback
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
while not (r/'evaluation_finished.json').exists():
    for name in ['preflight_failure.json','orchestration_failure.json','evaluation_failure.json']:
        if (r/name).exists():raise RuntimeError('Upstream failure: '+name)
    time.sleep(5)
try:
    jobs=[]
    for script,label in [('analyze_phase7.py','analysis'),('report_phase7.py','report')]:
        cmd=[sys.executable,str(ROOT/'scripts'/script),'--run-dir',str(r)];start=time.time()
        with (r/(label+'.log')).open('w') as f:res=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        jobs.append(dict(script=script,returncode=res.returncode,wall_s=time.time()-start))
        dump_json(r/'analysis_jobs.json',jobs);assert res.returncode==0,(script,res.returncode)
    dump_json(r/'analysis_finished.json',dict(time=time.time(),jobs=jobs));print('ANALYSIS_AND_REPORT_FINISHED',flush=True)
except BaseException as e:
    dump_json(r/'analysis_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise


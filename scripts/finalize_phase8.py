import argparse,subprocess,sys,time,json
from pathlib import Path
from ppo_common import ROOT,dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir)
while not (r/'evaluation_finished.json').exists():
    for name in ['evaluation_failure.json','orchestration_failure.json','preflight_failure.json']:
        if (r/name).exists():raise RuntimeError(str(r/name))
    time.sleep(3)
records=[]
for name in ['analyze_phase8.py','plot_phase8.py','report_phase8.py']:
    tick=time.time();cmd=[sys.executable,str(ROOT/'scripts'/name),'--run-dir',str(r)]
    res=subprocess.run(cmd,cwd=ROOT);records.append(dict(script=name,returncode=res.returncode,wall_s=time.time()-tick))
    dump_json(r/'finalization_jobs.json',records)
    if res.returncode:raise RuntimeError(name)
dump_json(r/'finalization_finished.json',dict(success=True,time=time.time(),jobs=records))
print('FINALIZATION_FINISHED',flush=True)

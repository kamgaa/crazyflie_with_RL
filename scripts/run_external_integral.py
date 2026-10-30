"""Run exactly the six fixed evaluations sequentially, preserving completed output."""
import json,subprocess,sys,time
from pathlib import Path
from ppo_common import ROOT,dump_json
r=Path(sys.argv[1]);executions=[]
assert json.loads((r/'functional_verification.json').read_text())['success']
for mode in ['off','on']:
 for task in ['hover','T10','T5']:
  dest=r/'evaluation'/mode/task
  cmd=[sys.executable,str(ROOT/'scripts/evaluate_external_integral.py'),'--run-dir',str(r),'--integral',mode,'--task',task,'--device','cuda:0','--headless','--physics','isaacsim_physx']
  if dest.exists():
   existing=json.loads((dest/'result.json').read_text());assert existing['success'];executions.append(dict(mode=mode,task=task,reused_existing=True,command=cmd));continue
  t=time.perf_counter()
  with (r/f'eval_{mode}_{task}.log').open('w') as f:x=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  j=json.loads((dest/'result.json').read_text()) if (dest/'result.json').exists() else {}
  executions.append(dict(mode=mode,task=task,command=cmd,returncode=x.returncode,process_wall_s=time.perf_counter()-t,success=j.get('success',False)))
  dump_json(r/'execution_commands.json',executions)
  assert x.returncode==0 and j.get('success'),executions[-1]
  print(mode,task,j['metrics']['completed'],j['metrics']['full']['position_rmse_m'],flush=True)
print('ALL_SIX_FINISHED',flush=True)

"""Sequential phase8 training; exactly three independent 256-iteration branches."""
import argparse,json,subprocess,sys,time,traceback,shutil,fcntl
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
lock=open(ROOT/'runs/.phase8_experiment.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (r/'comparison_ready.json').exists():
    if (r/'preflight_failure.json').exists():raise RuntimeError('Premise/preflight failed; no training allowed')
    time.sleep(2)
assert not (r/'training_orchestration_started.json').exists()
plan=json.loads((r/'experiment_plan.json').read_text());jobs=[]
parent=Path(plan['parent'])
assert json.loads((r/'parent_reproduction.json').read_text())['success']
# Final source inventory before any updates; original source_snapshot remains reviewable.
files=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'src').rglob('*.py'))+list((ROOT/'src').rglob('*.usda'))
for src in files:
    dest=r/'source_snapshot'/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
dump_json(r/'source_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in files})
dump_json(r/'training_orchestration_started.json',dict(time=time.time(),parent_sha256=sha256(parent/'final.pt'),order=plan['order']))
try:
    for name in plan['order']:
        out=r/name
        cmd=[sys.executable,str(ROOT/'scripts/train_phase8.py'),'--run-dir',str(out),'--additional-iterations','256','--device','cuda:0','--headless','--physics','isaacsim_physx']
        start=time.time();print('TRAIN_START',name,flush=True)
        with (out/'training.log').open('w') as f:res=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        job=dict(branch=name,command=cmd,returncode=res.returncode,start_time=start,wall_s=time.time()-start);jobs.append(job)
        dump_json(r/'training_jobs.json',jobs);print('TRAIN_END',name,res.returncode,flush=True)
        if res.returncode:
            failure=json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else dict(error=(out/'training.log').read_text()[-3000:])
            expected_numerical=any(k in str(failure).lower() for k in ['out of memory','non-finite','nan','simulator error'])
            if not expected_numerical:raise RuntimeError('Unexpected premise/implementation error: '+str(failure))
        else:
            result=json.loads((out/'training_result.json').read_text())
            assert result['additional_iterations']==256 and result['additional_transitions']==8388608
            assert result['additional_optimizer_updates']==plan['branches'][name]['updates']
    dump_json(r/'training_finished.json',dict(time=time.time(),jobs=jobs,
        all_success=all(x['returncode']==0 for x in jobs),
        actual_additional_transitions=sum(json.loads((r/x['branch']/'training_result.json').read_text())['additional_transitions'] for x in jobs if x['returncode']==0),
        actual_additional_updates=sum(json.loads((r/x['branch']/'training_result.json').read_text())['additional_optimizer_updates'] for x in jobs if x['returncode']==0)))
    print('TRAINING_BUDGETS_FINISHED',flush=True)
except BaseException as e:
    dump_json(r/'orchestration_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),jobs=jobs));raise


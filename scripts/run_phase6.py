"""Sequential phase-6 initial evaluation and exact continuation; never retrains on failure."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback,shutil
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args()
run=Path(a.run_dir).resolve();plan=json.loads((run/'experiment_plan.json').read_text())
lock=open(ROOT/'runs/.phase6_pipeline.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (run/'orchestration_started.json').exists()
assert json.loads((run/'preflight.json').read_text())['success']
dump_json(run/'orchestration_started.json',dict(pid=os.getpid(),time=time.time()))
jobs=[]
def job(label,cmd,log):
    print('START',label,flush=True);start=time.time()
    with log.open('w') as out:p=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    rec=dict(label=label,command=cmd,returncode=p.returncode,start_time=start,wall_s=time.time()-start)
    jobs.append(rec);dump_json(run/'jobs.json',jobs);print('END',label,p.returncode,flush=True)
    if p.returncode:raise RuntimeError(f'{label} failed; no replacement training permitted')
try:
    source_files=list(ROOT.glob('scripts/*.py'))+list((ROOT/'src').rglob('*.py'))+list((ROOT/'src').rglob('*.usda'))
    manifest={}
    for f in source_files:
        rel=f.relative_to(ROOT);dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,dest);manifest[str(rel)]=sha256(f)
    dump_json(run/'source_sha256.json',manifest)
    job('resume_start_evaluation',[sys.executable,str(ROOT/'scripts/evaluate_phase6.py'),'--checkpoints',plan['checkpoint'],
        '--run-dir',str(run),'--output',str(run/'evaluation/resume_start'),'--device','cuda:0','--headless','--physics','isaacsim_physx'],
        run/'evaluation/resume_start.log')
    initial=json.loads((run/'evaluation/resume_start/results.json').read_text());assert len(initial)==5
    reloads=json.loads((run/'evaluation/resume_start/reload_checks.json').read_text())
    assert all(x['max_action_error']==0 and x['optimizer_updates_in_process']==0 for x in reloads)
    job('additional_training',[sys.executable,str(ROOT/'scripts/train_phase6.py'),'--run-dir',str(run),
        '--additional-iterations','768','--device','cuda:0','--headless','--physics','isaacsim_physx'],run/'training.log')
    tr=json.loads((run/'training_result.json').read_text())
    assert tr['success'] and tr['additional_iterations']==768 and tr['additional_optimizer_updates']==15360
    assert tr['iterations']==1024 and tr['transitions']==33554432 and tr['adam_steps']==[20480]
    dump_json(run/'training_finished.json',dict(time=time.time(),success=True,jobs=jobs))
    print('EXACT_TRAINING_FINISHED',flush=True)
except BaseException as e:
    dump_json(run/'orchestration_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),jobs=jobs))
    raise


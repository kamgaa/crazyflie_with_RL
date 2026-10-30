"""Evaluate the 12 saved phase-6 policies, select best with precision tie-breaks, reload best/final."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback,shutil
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args()
run=Path(args.run_dir).resolve();jobs=[]
lock=open(ROOT/'runs/.phase5_evaluation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (run/'training_finished.json').exists():
    if (run/'orchestration_failure.json').exists():raise RuntimeError('Training pipeline failed; no dependent evaluation')
    time.sleep(5)
assert not (run/'evaluation_started.json').exists()
dump_json(run/'evaluation_started.json',dict(pid=os.getpid(),time=time.time()))
def evaluate(label,cps,dest):
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_phase6.py'),'--checkpoints',*[str(p) for p in cps],
        '--run-dir',str(run),'--output',str(dest),'--device','cuda:0','--headless','--physics','isaacsim_physx']
    start=time.time();print('EVAL_START',label,len(cps),flush=True)
    with dest.with_suffix('.log').open('w') as out:result=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    jobs.append(dict(label=label,command=cmd,returncode=result.returncode,start_time=start,wall_s=time.time()-start))
    dump_json(run/'evaluation_jobs.json',jobs);print('EVAL_END',label,result.returncode,flush=True)
    assert result.returncode==0,(label,result.returncode)
    checks=json.loads((dest/'reload_checks.json').read_text())
    assert all(x['max_action_error']==0 and x['optimizer_updates_in_process']==0 for x in checks)
def summary(rs):
    assert len(rs)==5
    return dict(checkpoint=rs[0]['checkpoint'],iteration=rs[0]['iteration'],transitions=rs[0]['transitions'],
        completed_count=sum(v['completed_8s'] for v in rs),mean_duration_s=sum(v['duration_s'] for v in rs)/5,
        mean_position_rmse_m=sum(v['position_rmse_m'] for v in rs)/5,
        worst_condition_rmse_m=max(v['position_rmse_m'] for v in rs))
def rank(v):
    # Precision tie-breaks apply only when all five episodes completed 8 seconds.
    full=v['completed_count']==5
    return (v['completed_count'],v['mean_duration_s'],
        -v['mean_position_rmse_m'] if full else 0.,
        -v['worst_condition_rmse_m'] if full else 0.,-v['iteration'])
try:
    cps=sorted([p for p in run.glob('checkpoint_*.pt') if not p.stem.endswith('_reference')],
        key=lambda p:int(p.stem.split('_')[1]))+[run/'final.pt']
    assert len(cps)==12
    evaluate('periodic',cps,run/'evaluation/periodic')
    initial=json.loads((run/'evaluation/resume_start/results.json').read_text())
    rs=json.loads((run/'evaluation/periodic/results.json').read_text());assert len(rs)==60
    assert sorted({v['iteration'] for v in rs})==list(range(320,1025,64))
    candidates=[summary(initial)]+[summary([v for v in rs if v['iteration']==i]) for i in range(320,1025,64)]
    best=max(candidates,key=rank);source=Path(best['checkpoint'])
    shutil.copy2(source,run/'best.pt')
    shutil.copy2(source.with_name(source.stem+'_reference.pt'),run/'best_reference.pt')
    dump_json(run/'selection.json',dict(best=best,candidates=candidates,
        rule='completed_count desc, mean_duration desc; if all5 complete mean_rmse asc then worst_rmse asc; earliest iteration',
        source_sha256=sha256(source),best_sha256=sha256(run/'best.pt')))
    evaluate('best_new_process_reload',[run/'best.pt'],run/'evaluation/best_reload')
    evaluate('final_new_process_reload',[run/'final.pt'],run/'evaluation/final_reload')
    dump_json(run/'evaluation_finished.json',dict(pid=os.getpid(),time=time.time(),jobs=jobs))
    print('ALL_EVALUATIONS_FINISHED',flush=True)
except BaseException as e:
    dump_json(run/'evaluation_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),jobs=jobs))
    raise


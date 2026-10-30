"""Evaluate completed phase7 branches after training; each best/final reload is a new process."""
import argparse,fcntl,json,subprocess,sys,time,traceback,shutil
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
while not (r/'training_finished.json').exists():
    if (r/'orchestration_failure.json').exists():raise RuntimeError('Training orchestration failed; inspect before evaluation')
    time.sleep(5)
lock=open(ROOT/'runs/.phase5_evaluation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (r/'evaluation_started.json').exists()
dump_json(r/'evaluation_started.json',dict(time=time.time()))
plan=json.loads((r/'experiment_plan.json').read_text());jobs=[]
def read(p):return json.loads(Path(p).read_text())
def summary(rs):
    assert len(rs)==5
    return dict(checkpoint=rs[0]['checkpoint'],iteration=rs[0]['iteration'],transitions=rs[0]['transitions'],
        additional_iterations=rs[0]['iteration']-1024,completed_count=sum(x['completed_8s'] for x in rs),
        mean_duration_s=sum(x['duration_s'] for x in rs)/5,mean_position_rmse_m=sum(x['position_rmse_m'] for x in rs)/5,
        worst_condition_rmse_m=max(x['position_rmse_m'] for x in rs))
def rank(s):
    full=s['completed_count']==5
    return (s['completed_count'],s['mean_duration_s'],-s['mean_position_rmse_m'] if full else 0,
        -s['worst_condition_rmse_m'] if full else 0,-s['iteration'])
def evaluate(branch,label,cps,dest):
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_phase7.py'),'--checkpoints',*[str(x) for x in cps],
        '--run-dir',str(r/branch),'--output',str(dest),'--device','cuda:0','--headless','--physics','isaacsim_physx']
    start=time.time();print('EVAL_START',branch,label,flush=True)
    with dest.with_suffix('.log').open('w') as f:res=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    jobs.append(dict(branch=branch,label=label,command=cmd,returncode=res.returncode,start_time=start,wall_s=time.time()-start))
    dump_json(r/'evaluation_jobs.json',jobs);assert res.returncode==0,(branch,label,res.returncode)
    assert all(x['max_action_error']==0 and x['optimizer_updates_in_process']==0 for x in read(dest/'reload_checks.json'))
    print('EVAL_END',branch,label,flush=True)
try:
    start=read(r/'common/evaluation_initial/results.json')
    for name in plan['order']:
        out=r/name
        if not (out/'training_result.json').exists():continue
        cps=sorted([x for x in out.glob('checkpoint_*.pt') if not x.stem.endswith('_reference')],key=lambda x:int(x.stem.split('_')[1]))+[out/'final.pt']
        assert len(cps)==4
        evaluate(name,'periodic',cps,out/'evaluation/periodic')
        rs=read(out/'evaluation/periodic/results.json')
        assert sorted({x['iteration'] for x in rs})==[1088,1152,1216,1280]
        candidates=[summary(start)]+[summary([x for x in rs if x['iteration']==i]) for i in [1088,1152,1216,1280]]
        best=max(candidates,key=rank);source=Path(best['checkpoint'])
        shutil.copy2(source,out/'best.pt');shutil.copy2(source.with_name(source.stem+'_reference.pt'),out/'best_reference.pt')
        dump_json(out/'selection.json',dict(best=best,best_is_parent=best['iteration']==1024,candidates=candidates,
            rule='completed count desc; mean duration desc; all5 complete mean RMSE asc then worst RMSE asc; earliest',
            source_sha256=sha256(source),best_sha256=sha256(out/'best.pt')))
        evaluate(name,'best_new_process_reload',[out/'best.pt'],out/'evaluation/best_reload')
        evaluate(name,'final_new_process_reload',[out/'final.pt'],out/'evaluation/final_reload')
    dump_json(r/'evaluation_finished.json',dict(time=time.time(),jobs=jobs))
    print('EVALUATIONS_FINISHED',flush=True)
except BaseException as e:
    dump_json(r/'evaluation_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),jobs=jobs));raise


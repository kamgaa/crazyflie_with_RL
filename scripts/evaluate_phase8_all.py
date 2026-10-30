"""Separate-process 8s selection and held-out 30s evaluations, after GPU training."""
import argparse,json,subprocess,sys,time,traceback,shutil,fcntl
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
while not (r/'training_finished.json').exists():
    if (r/'orchestration_failure.json').exists() or (r/'preflight_failure.json').exists():raise RuntimeError('Training/premise failure')
    time.sleep(3)
assert json.loads((r/'training_finished.json').read_text())['all_success']
assert not (r/'evaluation_started.json').exists()
lock=open(ROOT/'runs/.phase5_evaluation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((r/'experiment_plan.json').read_text());jobs=[];selections={};reuses=[]
dump_json(r/'evaluation_started.json',dict(time=time.time(),training_finished_before_evaluation=True))
def read(p):return json.loads(Path(p).read_text())
def evaluate(branch,checkpoints,out,horizon,label):
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_phase8.py'),'--checkpoints',*map(str,checkpoints),
        '--run-dir',str(r/branch),'--output',str(out),'--horizon',str(horizon),'--device','cuda:0','--headless','--physics','isaacsim_physx']
    start=time.time()
    with (r/branch/(label+'.log')).open('w') as f:res=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    job=dict(branch=branch,label=label,horizon_s=horizon,command=cmd,returncode=res.returncode,start_time=start,wall_s=time.time()-start,output=str(out));jobs.append(job)
    dump_json(r/'evaluation_jobs.json',jobs);assert res.returncode==0,job
    checks=read(out/'reload_checks.json');assert all(x['max_action_error']<=1e-6 and x['optimizer_updates_in_process']==0 for x in checks)
    print('EVAL_FINISHED',branch,label,horizon,flush=True)
    return read(out/'results.json')
def aggregate(rows):
    completed=sum(x['completed'] for x in rows)
    return dict(completed_count=completed,mean_duration_s=sum(x['duration_s'] for x in rows)/5,
        mean_position_rmse_m=sum(x['position_rmse_m'] for x in rows)/5,
        mean_tail_body_omega_rms_rad_s=sum(x['tail_body_omega_rms_rad_s'] for x in rows)/5 if completed==5 else None)
def rank(x):
    return (-x['completed_count'],-x['mean_duration_s'],x['mean_position_rmse_m'] if x['completed_count']==5 else 0,
        x['mean_tail_body_omega_rms_rad_s'] if x['completed_count']==5 else 0,x['additional_iterations'])
try:
    for branch in plan['order']:
        d=r/branch;checkpoints=sorted(p for p in d.glob('checkpoint_*.pt') if not p.stem.endswith('_reference'))+[d/'final.pt']
        rows=evaluate(branch,checkpoints,d/'evaluation/periodic_8s',8,'periodic_8s')
        candidates=[dict(checkpoint=str(d/'initial.pt'),additional_iterations=0,**aggregate(read(r/'common/evaluation_8s/results.json')))]
        for cp in checkpoints:
            rr=[x for x in rows if x['checkpoint']==str(cp)]
            assert len(rr)==5
            candidates.append(dict(checkpoint=str(cp),additional_iterations=rr[0]['iteration']-1280,**aggregate(rr)))
        selected=min(candidates,key=rank);source=Path(selected['checkpoint'])
        shutil.copy2(source,d/'best.pt');shutil.copy2(source.with_name(source.stem+'_reference.pt'),d/'best_reference.pt')
        selections[branch]=dict(best=selected,candidates=candidates,best_is_parent=selected['additional_iterations']==0,
            criterion='8s completion count, mean duration, if5/5 mean3DRMSE, tail body omega RMS, earliest',
            selection_horizon_s=8,selection_time=time.time(),best_sha256=sha256(d/'best.pt'),final_sha256=sha256(d/'final.pt'))
        dump_json(d/'selection.json',selections[branch]);dump_json(r/'checkpoint_selection.json',selections)
        evaluate(branch,[d/'best.pt',d/'final.pt'],d/'evaluation/reload_8s',8,'reload_8s')
    # No 30s result is available to selection above.
    evaluate('common',[r/'common/initial.pt'],r/'common/evaluation_30s',30,'evaluation_30s')
    for branch in plan['order']:
        d=r/branch
        same=sha256(d/'best.pt')==sha256(d/'final.pt')
        cps=[d/'final.pt'] if same else [d/'best.pt',d/'final.pt']
        evaluate(branch,cps,d/'evaluation/reload_30s',30,'reload_30s')
        if same:reuses.append(dict(branch=branch,horizon_s=30,best_result_shared_with='final',reason='best.pt and final.pt byte-identical SHA256',sha256=sha256(d/'final.pt'),both_explicitly_reloaded_at_8s=True))
    dump_json(r/'evaluation_reuse.json',reuses)
    dump_json(r/'evaluation_finished.json',dict(success=True,time=time.time(),jobs=jobs,selections=selections,reuses=reuses))
    print('ALL_EVALUATIONS_FINISHED',flush=True)
except BaseException as e:
    dump_json(r/'evaluation_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),jobs=jobs));raise


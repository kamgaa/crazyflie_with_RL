"""After all sequential training, evaluate saved policies in isolated processes."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args();root=Path(args.experiment).resolve()
lock=open(ROOT/'runs/.phase5_evaluation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (root/'orchestration_finished.json').exists():time.sleep(5)
assert not (root/'evaluation_started.json').exists()
dump_json(root/'evaluation_started.json',dict(pid=os.getpid(),time=time.time()))
baseline=json.loads((root/'common/evaluation_initial/results.json').read_text());jobs=[]
def run_eval(n,cps,dest):
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_phase5.py'),'--checkpoints',*[str(c) for c in cps],'--run-dir',str(root/f'N{n}'),'--output',str(dest),
        '--device','cuda:0','--headless','--physics','isaacsim_physx']
    start=time.time();print('EVAL_START',n,dest.name,len(cps),flush=True)
    with dest.with_suffix('.log').open('w') as f:result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    rec=dict(num_envs=n,command=cmd,returncode=result.returncode,start=start,wall_s=time.time()-start)
    jobs.append(rec);dump_json(root/'evaluation_jobs.json',jobs);print('EVAL_END',n,dest.name,result.returncode,flush=True)
    return result.returncode==0

def group_summary(rs):
    return dict(transitions=rs[0]['transitions'],iteration=rs[0]['iteration'],checkpoint=rs[0]['checkpoint'],
        checkpoint_name=rs[0]['checkpoint_name'],completed_count=sum(v['completed_8s'] for v in rs),
        mean_duration_s=sum(v['duration_s'] for v in rs)/len(rs),
        mean_rmse_all_m=sum(v['position_rmse_m'] for v in rs)/len(rs),
        mean_rmse_completed_m=sum(v['position_rmse_m'] for v in rs if v['completed_8s'])/sum(v['completed_8s'] for v in rs) if any(v['completed_8s'] for v in rs) else None)
for n in [64,256,1024]:
    r=root/f'N{n}';cps=sorted((p for p in r.glob('checkpoint_*.pt') if not p.stem.endswith('_reference')),key=lambda p:int(p.stem.split('_')[1]))
    if (r/'final.pt').exists():cps.append(r/'final.pt')
    if not cps:continue
    out=r/'evaluation/periodic'
    if not run_eval(n,cps,out):continue
    results=json.loads((out/'results.json').read_text());groups=[group_summary(baseline)]
    for trans in sorted({x['transitions'] for x in results}):groups.append(group_summary([x for x in results if x['transitions']==trans]))
    best=max(groups,key=lambda x:(x['completed_count'],x['mean_duration_s']))
    source=Path(best['checkpoint']);shutil.copy2(source,r/'best.pt');shutil.copy2(source.with_name(source.stem+'_reference.pt'),r/'best_reference.pt')
    dump_json(r/'selection.json',dict(best=best,checkpoints=groups,criteria=['8-second completed condition count descending','mean duration descending'],
        tie_rule='earliest evaluated transition count if both specified criteria tie; RMSE is not used',source_sha256=sha256(source),best_sha256=sha256(r/'best.pt')))
    verify=[r/'best.pt']+([r/'final.pt'] if (r/'final.pt').exists() else [])
    run_eval(n,verify,r/'evaluation/best_final_reload')
dump_json(root/'evaluation_finished.json',dict(pid=os.getpid(),time=time.time(),jobs=jobs));print('ALL_EVALUATIONS_FINISHED',flush=True)

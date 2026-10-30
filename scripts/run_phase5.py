"""Sequential GPU jobs only. Independent failures do not cancel other variants."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args();root=Path(args.experiment).resolve()
lock=open(ROOT/'runs/.phase5_orchestration.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (root/'common/evaluation_initial/results.json').exists(),'Complete common initial evaluation first'
assert not (root/'orchestration_started.json').exists(),'Orchestration already started'
dump_json(root/'orchestration_started.json',dict(pid=os.getpid(),order=[64,256,1024],time=time.time(),total_budget=25165824))
jobs=[]
def job(n,preflight):
    mode='preflight' if preflight else 'train';cmd=[sys.executable,str(ROOT/'scripts/train_phase5.py'),'--experiment',str(root),'--num-envs',str(n),'--device','cuda:0','--headless','--physics','isaacsim_physx']
    if preflight:cmd.append('--preflight')
    started=time.time();print('START',mode,n,flush=True)
    with (root/f'N{n}'/(mode+'.log')).open('w') as out:result=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    record=dict(num_envs=n,mode=mode,command=cmd,returncode=result.returncode,start_wall_time=started,end_wall_time=time.time())
    jobs.append(record);dump_json(root/'jobs.json',jobs);print('END',mode,n,result.returncode,flush=True)
    return result.returncode==0
for n in [256,1024]:job(n,True)
for n in [64,256,1024]:
    status=root/f'N{n}/preflight.json'
    if status.exists() and json.loads(status.read_text()).get('success'):job(n,False)
    else:print('SKIP_FAILED_PREFLIGHT',n,flush=True)
dump_json(root/'orchestration_finished.json',dict(pid=os.getpid(),time=time.time(),jobs=jobs));print('BOUNDED_TRAINING_FINISHED',flush=True)

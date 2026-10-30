"""All premise checks must pass before any branch optimizer update."""
import argparse,json,subprocess,sys,time,traceback,shutil
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
from phase6_common import differences
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
plan=json.loads((r/'experiment_plan.json').read_text());jobs=[]
try:
    assert json.loads((r/'diagnosis_ready.json').read_text())['success']
    for name in plan['order']:
        out=r/name;cmd=[sys.executable,str(ROOT/'scripts/train_phase7.py'),'--run-dir',str(out),'--additional-iterations','256','--preflight','--device','cuda:0','--headless','--physics','isaacsim_physx']
        tick=time.time()
        with (out/'preflight.log').open('w') as f:res=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        jobs.append(dict(branch=name,command=cmd,returncode=res.returncode,wall_s=time.time()-tick))
        dump_json(r/'preflight_jobs.json',jobs)
        assert res.returncode==0,(name,res.returncode)
    proofs=[json.loads((r/n/'preflight_resume_verification.json').read_text()) for n in plan['order']]
    for key in ['actor_sha256','critic_sha256','mean_sha256','std_sha256','critic_output_sha256']:
        assert len({v[key] for v in proofs})==1,key
    c0=json.loads((r/'C0_control/agent_config.json').read_text())
    diff={}
    for name in plan['order']:
        cfg=json.loads((r/name/'agent_config.json').read_text());dd=differences(c0,cfg)
        allowed={'run_name'}|({'algorithm.num_mini_batches'} if name=='M64_minibatch' else set())
        assert {x['path'] for x in dd}<=allowed,dd
        reward=json.loads((r/name/'reward_config.json').read_text())
        rd=differences(json.loads((r/'C0_control/reward_config.json').read_text()),reward)
        assert {x['path'] for x in rd}<=({'position_scale','weights.xy','weights.z'} if name=='R2_position' else set())
        diff[name]=dict(agent=dd,reward=rd)
    dump_json(r/'condition_diff.json',diff)
    files=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'src').rglob('*.py'))+list((ROOT/'src').rglob('*.usda'))
    for src in files:
        dst=r/'source_snapshot'/src.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    dump_json(r/'source_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in files})
    dump_json(r/'comparison_ready.json',dict(success=True,optimizer_updates=0,time=time.time(),branches=plan['order'],
        initial_outputs_identical=True,diagnosis_sha256=sha256(r/'diagnosis_before_training_ko.txt')))
    print('COMPARISON_READY',flush=True)
except BaseException as e:
    dump_json(r/'preflight_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),jobs=jobs));raise


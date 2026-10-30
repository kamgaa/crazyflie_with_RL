"""Prepare three bounded phase7 branches; no optimizer update."""
import argparse,json,shutil,socket,subprocess,time
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
assert socket.gethostname()=='MRL-server' and ROOT/'runs' in r.parents
r.mkdir(parents=True,exist_ok=False)
for n in ['evidence','common','plots','diagnosis','source_snapshot']:(r/n).mkdir()
parent=ROOT/'runs/phase6_n1024_resume_20261011_112937';ref=ROOT/'runs/phase5_parallel_20261010_145354/N64/final.pt'
pt=json.loads((parent/'training_result.json').read_text());assert sha256(parent/'final.pt')==pt['final_sha256']
assert (pt['iterations'],pt['transitions'],pt['optimizer_updates'])==(1024,33554432,20480)
project=[p for p in ROOT.rglob('*') if p.is_file() and not any(k in p.relative_to(ROOT).parts for k in ['runs','.git','__pycache__']) and 'phase7' not in p.name]
dump_json(r/'evidence/project_before_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in project})
prior=[]
for d in (ROOT/'runs').iterdir():
    if d.is_dir() and d.name.startswith(('phase3_','phase4_','phase5_','phase6_')):
        prior += [p for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
dump_json(r/'evidence/prior_runs_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in prior})
for name in ['REPORT_phase5_ko.txt','REPORT_phase6_ko.txt']:shutil.copy2(ROOT/name,r/'evidence'/name)
for name in ['agent_config.json','env_config.json','resolved_config.json','training_result.json','final_reference.pt','verification.json']:
    shutil.copy2(parent/name,r/'evidence'/('parent_'+name))
ps=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
(r/'evidence/processes_before.txt').write_text(ps)
(r/'evidence/git_before.txt').write_text(subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True))
dump_json(r/'evidence/server_before.json',dict(hostname=socket.gethostname(),cwd=str(ROOT),time=time.time(),
    parent_sha256=sha256(parent/'final.pt'),reference_sha256=sha256(ref),prior_files=len(prior),project_files=len(project)))
branches={'C0_control':dict(minibatches=4,position_scale=1,updates=5120),
    'M64_minibatch':dict(minibatches=64,position_scale=1,updates=81920),
    'R2_position':dict(minibatches=4,position_scale=2,updates=5120)}
plan=dict(parent=str(parent),checkpoint=str(parent/'final.pt'),reference=str(ref),parent_sha256=sha256(parent/'final.pt'),
    seed=42,num_envs=1024,rollout=32,epochs=5,base_iterations=1024,base_transitions=33554432,base_optimizer_updates=20480,
    additional_iterations=256,additional_transitions_per_branch=8388608,total_additional_transitions=25165824,
    total_additional_updates=92160,evaluation_additional_iterations=[0,64,128,192,256],order=list(branches),branches=branches)
dump_json(r/'experiment_plan.json',plan)
for name,change in branches.items():
    out=r/name;out.mkdir()
    for n in ['evidence','tensorboard','evaluation','plots']:(out/n).mkdir()
    cfg=json.loads((parent/'agent_config.json').read_text())
    cfg.update(max_iterations=1280,load_run=str(parent),load_checkpoint='final.pt',resume=True,
        save_interval=64,experiment_name='crazyflie_phase7',run_name=name)
    cfg['algorithm']['num_mini_batches']=change['minibatches']
    dump_json(out/'agent_config.json',cfg)
    dump_json(out/'reward_config.json',dict(variant='A',position_scale=change['position_scale'],
        weights=dict(xy=10*change['position_scale'],z=6*change['position_scale'],velocity=.005,tilt=6,angular_velocity=.001,yaw=1,termination=10),dt_multiplier=False))
    dump_json(out/'experiment_plan.json',dict(plan,branch=name,experiment=str(r),**change))
print(r,flush=True)


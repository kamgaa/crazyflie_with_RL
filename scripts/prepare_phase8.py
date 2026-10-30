"""Prepare phase8 without changing prior task/default behavior."""
import argparse,json,shutil,socket,subprocess,time,copy
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
assert socket.gethostname()=='MRL-server' and ROOT/'runs' in r.parents
r.mkdir(parents=True,exist_ok=False)
for n in ['evidence','common','plots','source_snapshot']:(r/n).mkdir()
parent=ROOT/'runs/phase7_diagnosis_ablation_20261012_135358/M64_minibatch'
pt=json.loads((parent/'training_result.json').read_text())
assert sha256(parent/'final.pt')==pt['final_sha256']
assert (pt['iterations'],pt['transitions'],pt['optimizer_updates'])==(1280,41943040,102400)
project=[p for p in ROOT.rglob('*') if p.is_file() and not any(k in p.relative_to(ROOT).parts for k in ['runs','.git','__pycache__']) and 'phase8' not in p.name]
dump_json(r/'evidence/project_before_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in project})
prior=[]
for d in (ROOT/'runs').iterdir():
    if d.is_dir() and d.name.startswith(('phase3_','phase4_','phase5_','phase6_','phase7_')):
        prior += [p for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
dump_json(r/'evidence/prior_runs_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in prior})
for name in ['REPORT_phase5_ko.txt','REPORT_phase6_ko.txt','REPORT_phase7_ko.txt','README.rst']:shutil.copy2(ROOT/name,r/'evidence'/name)
for name in ['agent_config.json','env_config.json','resolved_config.json','training_result.json','final_reference.pt','resume_verification.json']:
    shutil.copy2(parent/name,r/'evidence'/('parent_'+name))
(r/'evidence/processes_before.txt').write_text(subprocess.check_output(['ps','-eo','pid,etime,args'],text=True))
(r/'evidence/git_before.txt').write_text(subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True))
branches={n:dict(minibatches=64,updates=81920,position_mode=mode,omega_coefficient=w) for n,mode,w in [('Q_base','quadratic',.001),('E_position','exponential',.001),('E_omega','exponential',.010)]}
plan=dict(experiment=str(r),parent=str(parent),checkpoint=str(parent/'final.pt'),parent_sha256=sha256(parent/'final.pt'),
    seed=42,num_envs=1024,rollout=32,epochs=5,base_iterations=1280,base_transitions=41943040,base_optimizer_updates=102400,
    additional_iterations=256,additional_transitions_per_branch=8388608,total_additional_transitions=25165824,
    total_additional_updates=245760,final_iterations=1536,final_transitions=50331648,final_updates=184320,
    evaluation_additional_iterations=[0,64,128,192,256],evaluation_horizons_s=[8,30],order=list(branches),branches=branches)
dump_json(r/'experiment_plan.json',plan)
for name,change in branches.items():
    out=r/name;out.mkdir()
    for n in ['evidence','tensorboard','evaluation','plots']:(out/n).mkdir()
    cfg=json.loads((parent/'agent_config.json').read_text())
    cfg.update(max_iterations=1536,load_run=str(parent),load_checkpoint='final.pt',resume=True,save_interval=64,experiment_name='crazyflie_phase8',run_name=name)
    assert cfg['algorithm']['num_mini_batches']==64
    dump_json(out/'agent_config.json',cfg)
    reward=dict(variant='A',position_mode=change['position_mode'],sigma_xy_m=.10,sigma_z_m=.05,
        exponential_weight_xy=.100,exponential_weight_z=.015,
        weights=dict(xy=10.,z=6.,velocity=.005,tilt=6.,angular_velocity=change['omega_coefficient'],yaw=1.,termination=10.),
        dt_multiplier=False)
    dump_json(out/'reward_config.json',reward)
    dump_json(out/'experiment_plan.json',dict(plan,branch=name,**change))
shutil.copy2(r/'Q_base/agent_config.json',r/'common/agent_config.json')
shutil.copy2(r/'Q_base/reward_config.json',r/'common/reward_config.json')
shutil.copy2(parent/'final.pt',r/'common/initial.pt');shutil.copy2(parent/'final_reference.pt',r/'common/initial_reference.pt')
dump_json(r/'evidence/server_before.json',dict(hostname=socket.gethostname(),cwd=str(ROOT),time=time.time(),parent_sha256=sha256(parent/'final.pt'),project_files=len(project),prior_files=len(prior)))
print(r,flush=True)


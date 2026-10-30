"""Prepare a fresh phase-6 run without starting training."""
import argparse,json,shutil,socket,subprocess
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
p.add_argument('--parent',default=str(ROOT/'runs/phase5_parallel_20261010_145354'))
a=p.parse_args();r=Path(a.run_dir).resolve();parent=Path(a.parent).resolve()
assert socket.gethostname()=='MRL-server';assert ROOT/'runs' in r.parents
cp=parent/'N1024/final.pt';result=json.loads((parent/'N1024/training_result.json').read_text())
assert sha256(cp)==result['final_sha256']
assert (result['iterations'],result['transitions'],result['optimizer_updates'])==(256,8388608,5120)
r.mkdir(parents=True,exist_ok=False)
for n in ['evidence','tensorboard','plots','evaluation','source_snapshot']:(r/n).mkdir()
files=[x for x in ROOT.rglob('*') if x.is_file() and not any(k in x.relative_to(ROOT).parts for k in ['runs','.git','__pycache__'])]
dump_json(r/'evidence/project_before_sha256.json',{str(x.relative_to(ROOT)):sha256(x) for x in files})
dump_json(r/'evidence/parent_before_sha256.json',{str(x.relative_to(parent)):sha256(x) for x in parent.rglob('*') if x.is_file()})
dump_json(r/'evidence/server_before.json',dict(hostname=socket.gethostname(),project=str(ROOT),parent_sha256=sha256(cp)))
(r/'evidence/git_before.txt').write_text(subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True))
for src in [ROOT/'REPORT_phase5_ko.txt']+list((parent/'N1024').glob('*.json')):
    shutil.copy2(src,r/'evidence'/src.name)
cfg=json.loads((parent/'N1024/agent_config.json').read_text())
cfg.update(max_iterations=1024,resume=True,load_run=str(parent/'N1024'),load_checkpoint='final.pt',
    experiment_name='crazyflie_phase6',run_name='N1024_resume',save_interval=64)
dump_json(r/'agent_config.json',cfg)
dump_json(r/'experiment_plan.json',dict(parent=str(parent),checkpoint=str(cp),hover_reference=str(parent/'N64/final.pt'),
    num_envs=1024,rollout=32,base_iterations=256,additional_iterations=768,final_iterations=1024,
    base_transitions=8388608,additional_transitions=25165824,final_transitions=33554432,
    base_optimizer_updates=5120,additional_optimizer_updates=15360,final_optimizer_updates=20480,
    checkpoint_interval_transitions=2097152,checkpoint_interval_additional_iterations=64,
    evaluation_iterations=list(range(320,1025,64)),evaluation_points=12,seed=42,
    environment_state_restored=False,environment_reset_seed=42,strict_adam_restore_required=True))
print(r)


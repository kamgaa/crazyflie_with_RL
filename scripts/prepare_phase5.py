"""Prepare a NEW bounded phase-5 experiment; does not start training."""
import argparse,json,hashlib,shutil,socket,subprocess
from pathlib import Path
from ppo_common import ROOT,dump_json
from phase5_common import COUNTS,INTERVAL,BUDGET
p=argparse.ArgumentParser();p.add_argument('--experiment',required=True)
p.add_argument('--phase4-run',default=str(ROOT/'runs/phase4_20261009_191202_seed42'));a=p.parse_args()
r=Path(a.experiment).resolve();baseline=Path(a.phase4_run).resolve();r.mkdir(parents=True,exist_ok=False)
for name in ['evidence','common','plots','source_snapshot']:(r/name).mkdir()
for name in ['agent_config.json','env_config.json','optimizer_config.json','checkpoint_selection.json']:
    shutil.copy2(baseline/name,r/'evidence'/('phase4_'+name))
shutil.copy2(ROOT/'REPORT_phase4_ko.txt',r/'evidence/REPORT_phase4_ko.txt')
files=[p for p in ROOT.rglob('*') if p.is_file() and not any(k in p.relative_to(ROOT).parts for k in ['runs','.git','__pycache__'])]
dump_json(r/'evidence/project_before_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
dump_json(r/'evidence/server_before.json',dict(hostname=socket.gethostname(),project=str(ROOT),git_status=subprocess.check_output(['git','-C',str(ROOT),'status','--short','--branch'],text=True)))
for n,it in COUNTS.items():
    out=r/f'N{n}';out.mkdir()
    for folder in ['evidence','tensorboard','evaluation']:(out/folder).mkdir()
    cfg=json.loads((baseline/'agent_config.json').read_text());cfg.update(max_iterations=it,save_interval=INTERVAL//(n*32),experiment_name='crazyflie_phase5',run_name=f'N{n}')
    dump_json(out/'agent_config.json',cfg)
for p in list((ROOT/'scripts').glob('*phase5*.py'))+[ROOT/'scripts/phase4_metrics.py',ROOT/'scripts/ppo_common.py']+list((ROOT/'src').rglob('*.py')):
    dest=r/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
dump_json(r/'experiment_plan.json',dict(order=['N64','N256','N1024'],seed=42,rollout=32,transitions_per_run=BUDGET,total_budget=BUDGET*3,evaluation_interval=INTERVAL))
# Copy the documented metric contract for identical reporting on reproduced runs.
contract=ROOT/'runs/phase5_parallel_20261010_145354/METRIC_CONTRACT_ko.txt'
if contract.exists():shutil.copy2(contract,r/'METRIC_CONTRACT_ko.txt')
print(r)

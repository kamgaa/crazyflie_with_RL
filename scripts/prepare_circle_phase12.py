import argparse,json,shutil
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256,protected_hashes
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();source=Path(a.run_dir).resolve();r=source/'circle';r.mkdir()
for d in ['models/phase12_best','evidence','cases','plots','tensorboard']:(r/d).mkdir(parents=True)
sel=json.loads((source/'best_selection.json').read_text());srcplan=json.loads((source/'experiment_plan.json').read_text());base=ROOT/'runs/circle_phase11_best_20261014_214656';plan=json.loads((base/'experiment_plan.json').read_text())
for name in ['best.pt','best_reference.pt','agent_config.json','reward_config.json']:shutil.copy2(source/name,r/'models/phase12_best'/name)
plan.update(experiment='phase12 final selected best circle evaluation; no training',run_dir=str(r),branch=srcplan['branch'],selected_iteration=sel['selected']['additional_iteration'],models={'phase12_best':dict(source=str(source/'best.pt'),copy=str(r/'models/phase12_best/best.pt'),sha256=sha256(source/'best.pt'),reference_sha256=sha256(source/'best_reference.pt'))},controllers=['phase12_best'],model_selection=sel['selected']['checkpoint'],historical_phase11=str(base),historical_reused=['PID','phase11_best'],new_trials=2,reward_monitoring='same phase11 signed13 terms with explicitly configured roll/pitch weight; monitor only')
assert protected_hashes()==json.loads((base/'evidence/protected_physics_sha256.json').read_text())
dump_json(r/'experiment_plan.json',plan);dump_json(r/'evidence/protected_physics_sha256.json',protected_hashes());dump_json(r/'model_info.json',sel);print(r)

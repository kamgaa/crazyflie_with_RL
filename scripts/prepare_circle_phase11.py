import json,datetime,socket,shutil,subprocess
from pathlib import Path
from ppo_common import ROOT,sha256,dump_json,protected_hashes
import torch
assert socket.gethostname()=='MRL-server'
source=ROOT/'runs/phase11_axis_reward_fresh_20261014_210351'
old=ROOT/'runs/circle_pid_qe_20261013_160146';prior=ROOT/'runs/circle_phase9_track_return_best_20261014_142835'
r=ROOT/'runs'/('circle_phase11_best_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));r.mkdir()
for name in ['models/phase11_best','evidence','cases','plots','tensorboard','source_snapshot/scripts']:(r/name).mkdir(parents=True)
selection=json.loads((source/'best_selection.json').read_text());assert selection['selected']['additional_iteration']==512
cp=torch.load(source/'best.pt',map_location='cpu',weights_only=False);assert cp['infos']['completed_iterations']==512 and cp['infos']['optimizer_updates']==163840
for name in ['best.pt','best_reference.pt','agent_config.json','reward_config.json']:
 shutil.copy2(source/name,r/'models/phase11_best'/name)
plan=json.loads((prior/'experiment_plan.json').read_text());plan.update(experiment='phase11 best circle evaluation; zero training',created=datetime.datetime.now().isoformat(),run_dir=str(r),models={'phase11_best':dict(source=str(source/'best.pt'),sha256=sha256(source/'best.pt'),copy=str(r/'models/phase11_best/best.pt'),reference_sha256=sha256(source/'best_reference.pt'))},controllers=['phase11_best'],model_selection=str(source/'checkpoint_512.pt'),historical_phase9=str(prior),historical_reused=['PID','Q_base','E_position','phase9_best'],reward_monitoring='phase11 signed13 axis reward; evaluation only, no optimizer; moving target position only',gui_executed=False)
assert protected_hashes()==json.loads((prior/'evidence/protected_physics_sha256.json').read_text())
assert protected_hashes()==json.loads((old/'evidence/protected_physics_sha256.json').read_text())
dump_json(r/'experiment_plan.json',plan);dump_json(r/'evidence/protected_physics_sha256.json',protected_hashes())
existing=[q for d in ['scripts','src'] for q in (ROOT/d).rglob('*') if q.is_file() and '__pycache__' not in q.parts and 'circle_phase11' not in q.name]
dump_json(r/'evidence/existing_sources_sha256.json',{str(q.relative_to(ROOT)):sha256(q) for q in existing})
history=[q for base in [old,prior] for q in base.rglob('*') if q.is_file() and (q.suffix in ['.json','.csv'] or q.name.startswith('REPORT'))]
dump_json(r/'evidence/historical_files_sha256.json',{str(q.relative_to(ROOT)):sha256(q) for q in history})
dump_json(r/'model_info.json',dict(checkpoint=str(source/'best.pt'),sha256=sha256(source/'best.pt'),infos=cp['infos'],best_selection=selection['selected']))
for name,cmd in [('git_status.txt',['git','status','--short']),('processes_before.txt',['ps','-eo','pid,etime,args'])]:
 z=subprocess.run(cmd,capture_output=True,text=True);(r/'evidence'/name).write_text(z.stdout+z.stderr)
(ROOT/'runs/.circle_phase11_current_path').write_text(str(r));print(r)

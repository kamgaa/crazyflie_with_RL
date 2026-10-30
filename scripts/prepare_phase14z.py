import copy,datetime,json,shutil,socket,subprocess
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256,protected_hashes
from phase6_common import differences
assert socket.gethostname()=='MRL-server'
base=ROOT/'runs/phase13_vz_squared_corrected_20261017_221820'
r=ROOT/'runs'/('phase14_vz_weight3_fresh_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));r.mkdir()
for d in ['evidence','plots','source_snapshot','evaluation','tensorboard']:(r/d).mkdir()
a=json.loads((base/'agent_config.json').read_text());rc0=json.loads((base/'reward_config.json').read_text())
assert rc0['axis_weights']['attitude']==[.075,.075,1.] and rc0['axis_weights']['velocity']==[.005]*3
assert a['max_iterations']==512 and not a['resume'] and not a['load_run'] and not a['load_checkpoint']
rc=copy.deepcopy(rc0);rc['reward_id']='phase14z_axis_vz_squared';rc['vertical_velocity_mode']='absolute_squared';rc['axis_weights']['velocity'][2]=.015
plan=copy.deepcopy(json.loads((base/'experiment_plan.json').read_text()))
plan.update(research_name='Z축 절대 속도 제곱 벌점 가중치 3배 / 자세 벌점 0.075',
experiment=str(r),configuration_source=str(base),comparison_checkpoint=str(base/'best.pt'),
comparison_sha256=sha256(base/'best.pt'),branch='phase14z_vz_squared',baseline='phase13sq',
fresh=True,checkpoint_load_in_training=False,velocity_mode='xy_return_z_absolute_squared',
additional_iterations=512,seed=42,no_tuning=True,
velocity_z_formula='-0.015 * world_vz**2',
same_numeric_weight_different_units=False,velocity_z_weight_multiplier=3.,
circle=dict(radius_m=1,periods_s=[10,5],laps=2,position_only_reference=True,
source_protocol=str(base/'circle/experiment_plan.json')))
project={str(p.relative_to(ROOT)):sha256(p) for d in ['scripts','src'] for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'phase14z' not in p.name}
protected={str(p):sha256(p) for p in (ROOT/'runs').rglob('*') if p.is_file() and r not in p.parents and (p.suffix=='.pt' or p.name.startswith('REPORT_') or p.name in ['resolved_config.json','training_result.json'])}
dump_json(r/'evidence/project_before_sha256.json',project);dump_json(r/'evidence/prior_artifacts_sha256.json',protected);dump_json(r/'evidence/physics_before_sha256.json',protected_hashes())
for n,data in [('experiment_plan.json',plan),('agent_config.json',a),('reward_config.json',rc),
('condition_diff.json',dict(agent=[],reward=differences(rc0,rc),physical=[])),
('baseline_info.json',dict(path=str(base),checkpoint=str(base/'best.pt'),sha256=sha256(base/'best.pt'),used_only_for_comparison=True,training_checkpoint_loaded=False))]:
 dump_json(r/n,data)
shutil.copy2(base/'evaluation/initial_positions.json',r/'evaluation/initial_positions.json')
dump_json(r/'evaluation/positions_provenance.json',dict(source=str(base/'evaluation/initial_positions.json'),sha256=sha256(r/'evaluation/initial_positions.json'),previously_used=True,OOD=False))
for n,cmd in [('git_before.txt',['git','status','--short','--branch']),('git_diff_before.patch',['git','diff']),('processes_before.txt',['ps','-eo','user,pid,etime,args'])]:
 z=subprocess.run(cmd,text=True,capture_output=True);(r/'evidence'/n).write_text(z.stdout+z.stderr)
(ROOT/'runs/.phase14z_current_path').write_text(str(r));print(r,flush=True)

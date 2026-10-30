"""Two fresh phase11-equivalent branches; only roll/pitch reward multipliers vary."""
import copy,datetime,json,shutil,socket,subprocess
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256,protected_hashes
from phase6_common import differences
assert socket.gethostname()=='MRL-server'
base=ROOT/'runs/phase11_axis_reward_fresh_20261014_210351'
r=ROOT/'runs'/('phase12_attitude_weights_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));r.mkdir()
for d in ['evidence','plots','source_snapshot']:(r/d).mkdir()
a=json.loads((base/'agent_config.json').read_text());rc0=json.loads((base/'reward_config.json').read_text());bp=json.loads((base/'experiment_plan.json').read_text())
assert rc0['axis_weights']['attitude']==[.15,.15,1.]
assert a['algorithm']['num_mini_batches']==64 and a['max_iterations']==512 and not a['resume']
assert not a['load_run'] and not a['load_checkpoint']
branches={};diffs={}
project={str(p.relative_to(ROOT)):sha256(p) for d in ['scripts','src'] for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'phase12' not in p.name}
protected={str(p):sha256(p) for p in (ROOT/'runs').rglob('*') if p.is_file() and r not in p.parents and (p.suffix=='.pt' or p.name.startswith('REPORT_') or p.name in ['resolved_config.json','training_result.json'])}
dump_json(r/'evidence/previous_artifacts_sha256.json',protected);dump_json(r/'evidence/project_before_sha256.json',project);dump_json(r/'evidence/physics_sha256.json',protected_hashes())
for branch,mult in [('phase12-1',.5),('phase12-2',2.)]:
 d=r/branch;d.mkdir();branches[branch]=str(d)
 for folder in ['evidence','evaluation','plots','tensorboard','source_snapshot']:(d/folder).mkdir()
 rc=copy.deepcopy(rc0);rc['axis_weights']['attitude'][:2]=[.15*mult]*2
 diff=differences(rc0,rc);diffs[branch]=dict(agent=[],reward=diff)
 expected=copy.deepcopy(rc);expected['axis_weights']['attitude'][:2]=[.15,.15];assert expected==rc0
 plan=copy.deepcopy(bp);plan.update(research_name=f'기준 제어기 자세 벌점 비교 {branch}',experiment=str(d),configuration_source=str(base),comparison_checkpoint=str(base/'best.pt'),comparison_sha256=sha256(base/'best.pt'),branch=branch,attitude_multiplier=mult,attitude_weight_roll=.15*mult,attitude_weight_pitch=.15*mult,baseline='phase11',fresh=True,checkpoint_load_in_training=False)
 for name,data in [('experiment_plan.json',plan),('agent_config.json',a),('reward_config.json',rc),('configuration_plan_diff.json',diffs[branch])]:dump_json(d/name,data)
 dump_json(d/'evidence/project_before_sha256.json',project);dump_json(d/'evidence/prior_artifacts_sha256.json',protected)
 shutil.copy2(base/'evaluation/initial_positions.json',d/'evaluation/initial_positions.json')
 shutil.copy2(base/'REPORT_phase11_ko.txt',d/'evidence/REPORT_phase11_ko.txt')
 dump_json(d/'evaluation/positions_provenance.json',dict(source=str(base/'evaluation/initial_positions.json'),sha256=sha256(d/'evaluation/initial_positions.json'),previously_used=True,OOD=False))
plan=dict(experiment='phase12 attitude weight comparison',created=datetime.datetime.now().isoformat(),baseline=str(base),baseline_initial_sha256=sha256(base/'initial.pt'),baseline_best_sha256=sha256(base/'best.pt'),branches=branches,order=list(branches),num_fresh_runs=2,seed=42,fresh_no_checkpoint_resume=True,per_branch_iterations=512,per_branch_transitions=16777216,per_branch_optimizer_updates=163840,total_new_transitions=33554432,total_new_optimizer_updates=327680,best_selection=bp['best_rule'],latest_best_meaning='After all 512 iterations, use the last finalized best.pt selected from checkpoints 0/64/.../512 by unchanged fixed7 30s hover rule. It is not necessarily final.pt. Circle evaluation is not used for model selection.',circle=dict(radius_m=1,periods_s=[10,5],laps=2,position_only_reference=True,source_protocol=str(ROOT/'runs/circle_phase11_best_20261014_214656/experiment_plan.json')),no_tuning=True,no_additional_conditions=True)
dump_json(r/'experiment_plan.json',plan);dump_json(r/'condition_diff.json',diffs)
for n,cmd in [('git_before.txt',['git','status','--short','--branch']),('git_diff_before.patch',['git','diff']),('processes_before.txt',['ps','-eo','user,pid,etime,args'])]:
 z=subprocess.run(cmd,text=True,capture_output=True);(r/'evidence'/n).write_text(z.stdout+z.stderr)
(ROOT/'runs/.phase12_current_path').write_text(str(r));print(r,flush=True)

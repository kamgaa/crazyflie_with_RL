import json,copy,shutil,socket,datetime,subprocess
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
from phase6_common import differences
assert socket.gethostname()=='MRL-server'
r=ROOT/'runs'/('phase11_axis_reward_fresh_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));r.mkdir()
for n in ['evidence','evaluation','plots','tensorboard','source_snapshot']:(r/n).mkdir()
p=ROOT/'runs/phase10_track_return_continue_20261014_165918'
a0=json.loads((p/'agent_config.json').read_text());a=copy.deepcopy(a0)
a.update(seed=42,max_iterations=512,resume=False,load_run='',load_checkpoint='',experiment_name='crazyflie_phase11',run_name='axis_reward_fresh')
assert a['algorithm']['num_mini_batches']==64 and a['actor']['distribution_cfg']['std_type']=='log'
assert abs(a['actor']['distribution_cfg']['init_std']-0.22313016014842982)<1e-12
rc=dict(reward_id='phase11_axis',position_mode='axis_exponential',velocity_mode='return',dt_multiplier=False,return_gain_s_inv=4.,return_speed_limit_m_s=1.5,position={k:dict(weight=w,sigma_m=s) for k,w,s in [('x',.1,.1),('y',.1,.1),('z',.015,.05)]},axis_weights=dict(velocity=[.005]*3,attitude=[.15,.15,1.],angular_velocity=[.001]*3),failure_weight=10.,sigma_xy_m=.1,sigma_z_m=.05,exponential_weight_xy=.1,exponential_weight_z=.015,weights=dict(velocity=.005,tilt=6.,angular_velocity=.001,yaw=1.,termination=10.),legacy_fields_note='weights/sigma_xy/exponential_weight_xy only used for read-only common phase9 reward and legacy physical telemetry; actual 13 terms use position/axis_weights/failure_weight')
plan=dict(research_name='기준 제어기 실험 1 — 축별 보상과 완화된 자세 벌점',experiment=str(r),configuration_source=str(p),comparison_checkpoint=str(p/'final.pt'),comparison_sha256=sha256(p/'final.pt'),branch='axis_reward_fresh',seed=42,num_envs=1024,rollout=32,epochs=5,minibatches=64,updates=163840,position_mode='axis_exponential',velocity_mode='return',base_iterations=0,base_transitions=0,base_optimizer_updates=0,additional_iterations=512,additional_transitions=16777216,final_iterations=512,final_transitions=16777216,final_optimizer_updates=163840,evaluation_iterations=list(range(0,513,64)),evaluation_horizon_s=30,best_rule=['maximize fixed7 completed count','if below7 maximize mean duration','if7 minimize mean3D RMSE','earliest iteration'],fresh=True,checkpoint_load_in_training=False,random20_not_for_selection=True,reset_rng='fresh cuda Generator seed42 installed BEFORE first wrapper reset')
for name,data in [('experiment_plan.json',plan),('agent_config.json',a),('reward_config.json',rc),('configuration_plan_diff.json',dict(agent=differences(a0,a),reward=differences(json.loads((p/'reward_config.json').read_text()),rc)))]:dump_json(r/name,data)
for src,dst in [(p/'evaluation/initial_positions.json',r/'evaluation/initial_positions.json'),(p/'REPORT_phase10_ko.txt',r/'evidence/REPORT_phase10_ko.txt'),(p/'COMMANDS.txt',r/'evidence/COMMANDS_phase10.txt')]:shutil.copy2(src,dst)
pos=json.loads((r/'evaluation/initial_positions.json').read_text());assert len(pos['fixed'])==7 and len(pos['random20'])==20
assert [x['offset'] for x in pos['fixed']]==[[0,0,0],[.05,0,0],[-.05,0,0],[0,.05,0],[0,-.05,0],[0,0,.05],[0,0,-.05]]
dump_json(r/'evaluation/positions_provenance.json',dict(source=str(p/'evaluation/initial_positions.json'),sha256=sha256(r/'evaluation/initial_positions.json'),previously_used=True,OOD=False))
files=[q for d in ['scripts','src','assets'] if (ROOT/d).exists() for q in (ROOT/d).rglob('*') if q.is_file() and '__pycache__' not in q.parts and 'phase11' not in q.name]
dump_json(r/'evidence/project_before_sha256.json',{str(q.relative_to(ROOT)):sha256(q) for q in files})
# Preserve all old checkpoints and their reports/configs, without modifying them.
protected=[q for q in (ROOT/'runs').rglob('*') if q.is_file() and r not in q.parents and (q.suffix=='.pt' or q.name.startswith('REPORT_') or q.name in ['resolved_config.json','training_result.json'])]
dump_json(r/'evidence/prior_artifacts_sha256.json',{str(q):sha256(q) for q in protected})
for name,cmd in [('git_before.txt',['git','status','--short','--branch']),('git_diff_before.patch',['git','diff']),('processes_before.txt',['ps','-eo','pid,etime,args'])]:
 z=subprocess.run(cmd,capture_output=True,text=True);(r/'evidence'/name).write_text(z.stdout+z.stderr)
(ROOT/'runs/.phase11_current_path').write_text(str(r));print(r,flush=True)

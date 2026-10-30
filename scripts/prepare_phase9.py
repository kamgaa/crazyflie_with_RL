import argparse,json,shutil,socket,subprocess,copy,time
from pathlib import Path
import torch,numpy as np
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
assert socket.gethostname()=='MRL-server' and ROOT/'runs' in r.parents
r.mkdir(parents=True,exist_ok=False)
for n in ['evidence','common','plots','source_snapshot','evaluation']:(r/n).mkdir()
parent=ROOT/'runs/phase8_exponential_20261012_164652/E_position';cp=parent/'best.pt'
c=torch.load(cp,map_location='cpu',weights_only=False);info=c['infos'];steps=sorted(set(int(x['step']) for x in c['optimizer_state_dict']['state'].values()))
assert (info['completed_iterations'],info['collected_transitions'],info['optimizer_updates'],c['iter'],steps)==(1408,46137344,143360,1407,[143360])
pi=dict(checkpoint=str(cp),sha256=sha256(cp),reference=str(parent/'best_reference.pt'),reference_sha256=sha256(parent/'best_reference.pt'),infos=info,native_iter=c['iter'],adam_steps=steps,adam_parameters=len(c['optimizer_state_dict']['state']),rng_keys=list(c['rng_state']),source='checkpoint best.pt metadata; training_result.json describes final.pt, not this selected best')
dump_json(r/'parent_info.json',pi)
project=[p for p in ROOT.rglob('*') if p.is_file() and not any(k in p.relative_to(ROOT).parts for k in ['runs','.git','__pycache__']) and 'phase9' not in p.name]
dump_json(r/'evidence/project_before_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in project})
prior=[p for d in (ROOT/'runs').iterdir() if d.is_dir() and d!=r and d.name.startswith(('phase','circle_')) for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'tensorboard' not in p.parts]
dump_json(r/'evidence/prior_runs_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in prior})
for name in ['REPORT_phase8_ko.txt','REPORT_phase7_ko.txt']:
 if (ROOT/name).exists():shutil.copy2(ROOT/name,r/'evidence'/name)
for name in ['agent_config.json','env_config.json','resolved_config.json','training_result.json','best_reference.pt','resume_verification.json']:shutil.copy2(parent/name,r/'evidence'/('parent_'+name))
for name,cmd in [('processes_before.txt',['ps','-eo','pid,etime,args']),('git_before.txt',['git','status','--short','--branch'])]:
 v=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True);(r/'evidence'/name).write_text(v.stdout+v.stderr)
branches={n:dict(minibatches=64,updates=81920,position_mode='exponential',velocity_mode=mode,korean=ko) for n,mode,ko in [('control_stop','stop','정지 속도 보상'),('track_return','return','복귀 속도 추종 보상')]}
plan=dict(experiment=str(r),parent=str(parent),checkpoint=str(cp),parent_sha256=sha256(cp),seed=42,reset_seed=42,num_envs=1024,rollout=32,epochs=5,base_iterations=1408,base_transitions=46137344,base_optimizer_updates=143360,additional_iterations=256,additional_transitions_per_branch=8388608,total_additional_transitions=16777216,total_additional_updates=163840,final_iterations=1664,final_transitions=54525952,final_updates=225280,evaluation_additional_iterations=[0,64,128,192,256],order=list(branches),branches=branches,reset_box_half_width_m=.05)
dump_json(r/'experiment_plan.json',plan)
reward=json.loads((parent/'reward_config.json').read_text());assert reward['position_mode']=='exponential' and reward['weights']['angular_velocity']==.001
for name,change in branches.items():
 out=r/name;out.mkdir()
 for n in ['evidence','tensorboard','evaluation','plots']:(out/n).mkdir()
 cfg=json.loads((parent/'agent_config.json').read_text());cfg.update(max_iterations=1664,load_run=str(parent),load_checkpoint='best.pt',resume=True,save_interval=64,experiment_name='crazyflie_phase9',run_name=name)
 assert cfg['algorithm']['num_mini_batches']==64
 dump_json(out/'agent_config.json',cfg);dump_json(out/'reward_config.json',dict(reward,velocity_mode=change['velocity_mode'],return_gain_s_inv=4.,return_speed_limit_m_s=1.5))
 dump_json(out/'experiment_plan.json',dict(plan,branch=name,**change))
 shutil.copy2(cp,out/'initial.pt');shutil.copy2(parent/'best_reference.pt',out/'initial_reference.pt')
for name in ['agent_config.json','reward_config.json']:shutil.copy2(r/'control_stop'/name,r/'common'/name)
shutil.copy2(cp,r/'common/initial.pt');shutil.copy2(parent/'best_reference.pt',r/'common/initial_reference.pt')
cases=[dict(name='nominal',offset=[0.,0.,0.])]
for i,axis in enumerate('xyz'):
 for sign in [1,-1]:
  v=[0.,0.,0.];v[i]=sign*.05;cases.append(dict(name=axis+('_plus' if sign==1 else '_minus'),offset=v))
random=np.random.default_rng(20261013).uniform(-.05,.05,(20,3))
dump_json(r/'evaluation/initial_positions.json',dict(fixed=cases,random20=[dict(name=f'random_{i:02d}',offset=v.tolist()) for i,v in enumerate(random)],seed=20261013,generator='NumPy PCG64 default_rng',units='m; local target-relative XYZ box offsets'))
print(r,flush=True)

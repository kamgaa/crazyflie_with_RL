import csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2')
os.chdir(root);sys.path.insert(0,str(root))
comparison=Path(Path('/tmp/crazyflie-ab-training-comparison-dir').read_text())
runs={label:Path(json.loads((comparison/f'{label}_run.json').read_text())['run_dir']) for label in 'ab'}
while True:
 manifests={label:json.loads(next((run/'manifests').glob('*manifest_*.json')).read_text()) for label,run in runs.items()}
 if any(m['status']=='failed' for m in manifests.values()):raise RuntimeError('A/B training failed; inspect run logs')
 if all(m['status']=='completed' and (runs[label]/'training_stdout.log').read_text().rstrip().endswith('done.') for label,m in manifests.items()):break
 time.sleep(5)
models={}
training_records={}
initial_hashes=[]
for label,run in runs.items():
 m=manifests[label]
 source=json.loads((run/'training_source_record.json').read_text())
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source['source_sha256'].items())
 initial=json.loads(next((run/'metrics').glob('*initial-policy*.json')).read_text())
 initial_hashes.append(initial['initial_policy_sha256'])
 with next((run/'metrics').glob('*training-episodes*.csv')).open() as f:episodes=list(csv.DictReader(f))
 actual=m['result']['actual_total_timesteps']
 assert sum(int(r['length_steps']) for r in episodes)==actual
 assert int(episodes[-1]['end_timestep'])==actual
 for kind in ('final','best'):
  if m['models'][kind]:models[f'{label.upper()}_{kind}']=str(run/m['models'][kind]['path'])
 intermediate=[r for r in m['model_history'] if r['kind']=='intermediate']
 assert len(intermediate)==10
 assert all(r['normalization']=={'enabled':False} for r in intermediate)
 completed=[r for r in episodes if r['episode_ended']=='True']
 reasons={}
 for row in completed:reasons[row['end_reason']]=reasons.get(row['end_reason'],0)+1
 training_records[label]={'run_dir':str(run),'requested_timesteps':m['result']['requested_total_timesteps'],'actual_timesteps':actual,'completed_training_episodes':len(completed),'unfinished_episode_records':len(episodes)-len(completed),'training_end_reasons':reasons,'intermediate_timesteps':[r['timestep'] for r in intermediate],'initial_policy_sha256':initial['initial_policy_sha256'],'best':m['models']['best'],'final':m['models']['final']}
assert initial_hashes[0]==initial_hashes[1]
models['historical_nominal']=str(root/'artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip')
args=[sys.executable,'compare_dr_policies.py','--config','configs/eval_velocity_ab.yaml']
for name,path in models.items():args+=['--model',f'{name}={path}']
args+=['--cases','hover','step-005','--seed','42']
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/crazyflie-dr-mpl')
result=subprocess.run(args,cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(comparison/'evaluation_stdout.log').write_text(result.stdout)
result.check_returncode()
eval_dir=next(line.split('results: ',1)[1] for line in result.stdout.splitlines() if line.startswith('results: '))
before=json.loads((comparison/'preserved_hashes_before.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
report={'training':training_records,'models':models,'evaluation_dir':eval_dir,'evaluation_command':args,'same_initial_policy':True,'historical_artifacts_unchanged':True,'no_return_based_ranking':True}
with (comparison/'completion.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2),flush=True)

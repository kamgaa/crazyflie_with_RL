import contextlib, hashlib, json, os, sys, traceback
from pathlib import Path
root=Path('/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2')
os.chdir(root);sys.path.insert(0,str(root))
from crazyflie_rl.config import load_config
from crazyflie_rl.artifacts import ArtifactManager
from crazyflie_rl.training import PPOTrainer
label=sys.argv[1]
comparison=Path(Path('/tmp/crazyflie-ab-training-comparison-dir').read_text())
config_path=f'configs/e2e_train_velocity_ab_{label}.yaml'
config=load_config(config_path)
assert config.training.seed==42 and config.training.total_timesteps==1000000
artifacts=ArtifactManager.create(config,command=['train_ppo_02.py','--config',config_path])
record={'label':label,'run_dir':str(artifacts.run_dir.resolve()),'config':config_path,'fresh_initialization':True}
with (comparison/f'{label}_run.json').open('x') as f:json.dump(record,f,indent=2)
files=['crazyflie_rl/training.py','crazyflie_rl/training_observers.py','crazyflie_rl/environment.py','crazyflie_rl/config.py','crazyflie_rl/velocity_reference.py','crazyflie_rl/artifacts.py','crazyflie_rl/factories.py','configs/e2e_train_velocity_ab_a.yaml','configs/e2e_train_velocity_ab_b.yaml','configs/e2e_train_position_velocity_error.yaml']
source={'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},'initialization':'fresh PPO; no resume or checkpoint load','threads':{'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS')}}
(artifacts.run_dir/'training_source_record.json').write_text(json.dumps(source,indent=2)+'\n')
print(json.dumps(record),flush=True)
with (artifacts.run_dir/'training_stdout.log').open('x',buffering=1) as output:
 with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
  try:PPOTrainer(config,artifact_manager=artifacts).train()
  except BaseException:
   traceback.print_exc();raise
print('completed '+label,flush=True)

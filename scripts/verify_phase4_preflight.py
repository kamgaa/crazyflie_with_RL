"""Focused telemetry boundary checks; zero optimizer updates."""
import json,os,sys,types,csv,math
from pathlib import Path
import torch
from phase4_metrics import Telemetry,TERMS,reward_terms
from ppo_common import dump_json
r=Path(os.environ['RUN'])
a=types.SimpleNamespace(num_envs=2,device='cuda:0')
o=torch.zeros((2,15),device=a.device);o[:,6]=1;o[:,14]=1;o[:,0]=torch.tensor([.1,.2],device=a.device)
a.policy_observation=lambda:o
a.actions=torch.zeros((2,4),device=a.device);a.held_wrench=a.actions.clone();a.b_inv=torch.eye(4,device=a.device)
a.force_b=torch.zeros((2,1,3),device=a.device);a.torque_b=a.force_b.clone()
a.motors=types.SimpleNamespace(command=a.actions.clone(),force=a.actions.clone(),omega=a.actions.clone())
a.reset_terminated=torch.zeros(2,dtype=torch.bool,device=a.device);a.reset_time_outs=a.reset_terminated.clone()
a._get_rewards=lambda:reward_terms(o,a.reset_terminated).sum(-1)
t=Telemetry(a)
for step in range(32):a._get_rewards()
first=t.finish_rollout();assert first['Episode/completed_count']==0 and 'Episode/return' not in first
for step in range(32):
    a.reset_terminated[0]=step==7;a._get_rewards()
second=t.finish_rollout()
assert second['Episode/completed_count']==1 and second['Episode/length_steps']==40
assert abs(second['Episode/return']+14)<1e-5
assert abs(float(t.ep[0,8])-24)==0 and abs(float(t.ep[1,8])-64)==0
assert abs(float(t.ep[1,7])+25.6)<1e-5
old=list(csv.DictReader(open('runs/phase3_20261009_162844_seed42/evaluation/trajectory.csv')))
new=list(csv.DictReader(open(r/'evaluation/baseline_50/final/nominal.csv')))
keys=['x','y','z','qw','qx','qy','qz','wx','wy','wz','reward']
errors={k:max(abs(float(x[k])-float(y[k])) for x,y in zip(old,new)) for k in keys}
assert len(old)==len(new)==74 and max(errors.values())<1e-5
checks=dict(success=True,optimizer_updates=0,no_episode_mean_when_count_zero=True,
    episode_cross_rollout_length=second['Episode/length_steps'],episode_cross_rollout_return=second['Episode/return'],
    other_env_accumulator_preserved=True,baseline_phase3_length_match=True,baseline_vs_phase3_max_errors=errors,
    reward_sum_max_abs_error=max(json.loads((r/'evaluation/baseline_50/results.json').read_text())[i]['reward_sum_max_abs_error'] for i in range(5)))
dump_json(r/'evidence/preflight.json',checks)
rows=new
summary=dict(duration_s=.74,termination='max_tilt',
    max_action_abs=max(abs(float(x[f'raw_action_{i}'])) for x in rows for i in range(4)),
    maximum_body_angular_speed_rad_s=max(math.sqrt(sum(float(x[k])**2 for k in ['wx','wy','wz'])) for x in rows),
    max_request_actual_force_error_N=max(abs(float(x['request_3'])-float(x['actual_3'])) for x in rows),
    max_request_actual_torque_error_Nm=max(math.sqrt(sum((float(x[f'request_{i}'])-float(x[f'actual_{i}']))**2 for i in range(3))) for x in rows),
    native_motor_thrust_min_N=[min(float(x[f'thrust_{i}']) for x in rows) for i in range(4)],
    native_motor_thrust_max_N=[max(float(x[f'thrust_{i}']) for x in rows) for i in range(4)],
    final_row=rows[-1],existing_phase3_missing=['reward_terms','requested_applied_wrench','motor_thrust','KL','PPO_clip_fraction','exact_backward_total_loss'],
    new_baseline_note='Reward/wrench/motor telemetry newly measured at iteration 50 checkpoint; not backfilled into historic training')
dump_json(r/'evidence/phase3_diagnosis.json',summary)
print(json.dumps(checks));print('DIAGNOSTIC',json.dumps({k:v for k,v in summary.items() if k!='final_row'}))

"""Focused offline verification; does not execute a simulator or optimizer."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np,torch
from ppo_common import dump_json,sha256
from phase11_evalmetrics import extra_metrics
from phase13_reward import components
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir)
init=torch.load(r/'initial_reset.pt',weights_only=True,map_location='cpu');pre=torch.load(r/'preflight_initial_reset.pt',weights_only=True,map_location='cpu');sub=torch.load(r/'preflight_subsequent_reset.pt',weights_only=True,map_location='cpu')
o=init['observations'];e=o[:,:3];actual=init['world_position']-init['env_origins']-torch.tensor([0.,0.,1.])
def stats(x):return dict(min=x.min(0).values.tolist(),max=x.max(0).values.tolist(),mean=x.mean(0).tolist(),std=x.std(0,unbiased=False).tolist(),unique_rows=len(torch.unique(x,dim=0)),correlation=np.corrcoef(x.numpy().T).tolist())
s=dict(first_actual=stats(actual),first_generated=stats(init['delta']),subsequent_selected_actual=stats(sub['after'][:,:3]),source='saved float32 root_link_pos_w and actual policy observation before first action',first_actual_vs_generated_max_abs_m=float((actual-init['delta']).abs().max()),first_actual_vs_policy_error_max_m=float((actual-e).abs().max()),first_preflight_training_identical=torch.equal(pre['observations'],o),subsequent_positions_changed=not torch.equal(sub['before'][:,:3],sub['after'][:,:3]),raw_world_dtype=str(init['world_position'].dtype),origin_range=[init['env_origins'].min(0).values.tolist(),init['env_origins'].max(0).values.tolist()])
checks=dict(first_range=bool((actual.abs()<=.05001).all()),all_1024_unique=len(torch.unique(actual,dim=0))==1024,not_same_xyz_random=not torch.equal(actual[:,0],actual[:,1]),first_actual_policy_match=torch.equal(actual,e),first_preflight_training_same=s['first_preflight_training_identical'],first_velocity_zero=bool((o[:,3:6]==0).all()),first_omega_zero=bool((o[:,10:13]==0).all()),first_quaternion_horizontal=torch.equal(o[:,6:10],torch.tensor([1.,0,0,0]).expand(1024,4)),target_fixed=json.loads((r/'initial_reset.json').read_text())['target']==[0.,0.,1.],subsequent_new=s['subsequent_positions_changed'],subsequent_range=bool((sub['after'][:,:3].abs()<=.05001).all()),partial_reset=json.loads((r/'preflight.json').read_text())['partial_reset_preserves_other_envs'])
initial=torch.load(r/'initial.pt',map_location='cpu',weights_only=False)
comparison=torch.load(Path(json.loads((r/'experiment_plan.json').read_text())['comparison_checkpoint']),map_location='cpu',weights_only=False)
checks['fresh_actor_differs_trained_baseline']=any(not torch.equal(v,comparison['actor_state_dict'][k]) for k,v in initial['actor_state_dict'].items())
checks['fresh_critic_differs_trained_baseline']=any(not torch.equal(v,comparison['critic_state_dict'][k]) for k,v in initial['critic_state_dict'].items())
checks['initial_adam_empty']=len(initial['optimizer_state_dict']['state'])==0
checks['initial_counters_zero']=all(initial['infos'][k]==0 for k in ['completed_iterations','collected_transitions','optimizer_updates'])
checks['initial_std_log_minus1_5']=torch.equal(initial['actor_state_dict']['distribution.log_std_param'],torch.full((4,),-1.5))
# Recompute 13 reward terms on saved first stochastic rollout, without optimizer.
trace=torch.load(r/'first_rollout.pt',weights_only=True,map_location='cpu');cfg=json.loads((r/'reward_config.json').read_text());terms=components(trace['obs'].reshape(-1,15),trace['terminated'].reshape(-1),cfg).reshape_as(trace['terms'])
reward_diff=float((terms-trace['terms']).abs().max());checks['first_rollout_reward_cpu_matches_gpu']=reward_diff<1e-5
checks['first_rollout_actual_total']=float((trace['terms'].sum(-1)-trace['reward']).abs().max())<1e-5
checks['first_rollout_finite']=all(bool(torch.isfinite(v).all()) for v in trace.values())
# Numerical definition tests for the non-heuristic settling calculation.
def fake(errors):
 return [dict(time_s=(i+1)*.01,ex=x,ey=0.,ez=0.,vx=0.,vy=0.,vz=0.,roll_rad=0.,pitch_rad=0.,yaw_rad=0.,wx=0.,wy=0.,wz=0.,tilt_deg=0.) for i,x in enumerate(errors)]
a=extra_metrics(fake([.02]*49+[.005]*151),[.05,0,0],True)
b=extra_metrics(fake([.005]*199+[.02]),[.05,0,0],True)
c=extra_metrics(fake([.005]*200),[.05,0,0],False)
checks['settling_min1s_and_no_reexit']=a['settling_time_1cm_1s_s']==.5 and b['settling_time_1cm_1s_s'] is None
checks['failure_not_settled']=c['settling_time_1cm_1s_s'] is None
end_checks=[]
for path in (r/'evaluation').glob('*/*/*.csv'):
 with path.open() as f:rows=list(csv.DictReader(f))
 if not rows or 'terminated' not in rows[0]:continue
 terminated=[i for i,x in enumerate(rows) if x['terminated']=='True' or x['truncated']=='True']
 assert not terminated or terminated==[len(rows)-1],path
 end_checks.append(str(path))
checks['all_terminal_snapshots_last']=len(end_checks)>0
checks={k:bool(v) for k,v in checks.items()};dump_json(r/'initial_position_validation.json',s);dump_json(r/'focused_verification.json',dict(success=all(checks.values()),checks=checks,reward_cpu_gpu_max_error=reward_diff,terminal_files=len(end_checks),preflight_optimizer_updates=0));assert all(checks.values()),checks
print('FOCUSED_VERIFICATION_OK')

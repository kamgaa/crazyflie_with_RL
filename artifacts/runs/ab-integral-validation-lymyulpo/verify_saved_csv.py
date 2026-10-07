"""Independent saved-CSV audit; no simulation, inference, learning or tuning."""
from pathlib import Path
import csv,json,sys
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2];sys.path.insert(0,str(ROOT))
from crazyflie_rl.integral_eval import read_columns
m=json.loads((OUT/'manifest.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
assert m['status']=='completed' and len(summary)==52

def vec(f,key,n=3):return np.column_stack([f[f'{key}_{i}'] for i in range(n)])

def recover(f,event,start,end,duration):
 if event is None or duration<end-1e-9:return dict.fromkeys(('xy','z','3d'))
 mask=(f.time_post>start+1e-9)&(f.time_post<=end+1e-9)
 times=np.r_[start,f.time_post[mask]]
 errors=np.vstack((vec(event,'e_true_after')[0],vec(f,'position_error_world')[mask]))
 speeds=np.vstack((vec(event,'velocity')[0],vec(f,'velocity')[mask]))
 out={}
 for axis,sl in [('xy',slice(0,2)),('z',slice(2,3)),('3d',slice(0,3))]:
  good=(np.linalg.norm(errors[:,sl],axis=1)<=.005)&(np.linalg.norm(speeds[:,sl],axis=1)<=.02)
  suffix=np.logical_and.accumulate(good[::-1])[::-1]
  ids=np.flatnonzero(suffix&(end-times>=1-1e-9))
  out[axis]=float(times[ids[0]]-start) if len(ids) else None
 return out

n_control=n_physics=n_events=0;max_recurrence=max_identity=0.;prefix_checks={}
for r in summary:
 f=read_columns(OUT/(r['key']+'.csv'));p=read_columns(OUT/(r['key']+'-physics.csv'))
 n=len(f.time);n_control+=n;n_physics+=len(p.physics_time)
 assert len(p.physics_time)==5*n
 np.testing.assert_allclose(f.time,np.arange(n)*.01,atol=1e-14)
 np.testing.assert_allclose(f.time_post,f.time+.01,atol=1e-14)
 scenario=m['scenarios'][r['condition']]
 target=np.tile([0.,0.,1.],(n,1))
 if scenario['target_after30'] is not None:target[f.time>=30]=scenario['target_after30']
 np.testing.assert_array_equal(vec(f,'reference'),target)
 np.testing.assert_array_equal(vec(f,'reference_post'),target)
 np.testing.assert_array_equal(vec(f,'p_target'),target)
 np.testing.assert_array_equal(vec(f,'p_target_post'),target)
 before=vec(f,'position_before');position=vec(f,'position');xi=vec(f,'xi_t');nxt=vec(f,'xi_next')
 np.testing.assert_array_equal(before[1:],position[:-1]);np.testing.assert_array_equal(xi[0],0)
 np.testing.assert_array_equal(xi[1:],nxt[:-1]);np.testing.assert_array_equal(vec(f,'e_true_before'),before-target)
 np.testing.assert_array_equal(vec(f,'e_true_post'),position-target)
 np.testing.assert_array_equal(vec(f,'position_error_world'),position-target)
 np.testing.assert_array_equal(vec(f,'p_cmd'),target+xi)
 np.testing.assert_allclose(vec(f,'e_actor_before'),before-target-xi,atol=1e-15)
 native=vec(f,'observation',15);actor=vec(f,'policy_raw_observation',15)
 np.testing.assert_array_equal(native[:,3:],actor[:,3:]);np.testing.assert_allclose(actor[:,:3],before-target-xi,atol=1e-8,rtol=1e-7)
 np.testing.assert_allclose(actor[:,3:6],vec(f,'velocity_before'),atol=1e-7,rtol=1e-7)
 if r['gain']=='no_integral':np.testing.assert_array_equal(native,actor)
 raw=-4*(position-target);norm=np.linalg.norm(raw,axis=1);ids=norm>1.5;raw[ids]*=(1.5/norm[ids])[:,None]
 np.testing.assert_allclose(vec(f,'desired_velocity'),raw,atol=1e-14)
 np.testing.assert_allclose(vec(f,'internal_velocity_error'),vec(f,'velocity')-raw,atol=1e-14)
 np.testing.assert_array_equal(vec(f,'reference_velocity'),0)
 np.testing.assert_array_equal(vec(f,'reference_velocity_post'),0)
 clipping=vec(p,'allocator_clipped',4).reshape(n,5,4).astype(bool).any(axis=(1,2))
 esc=(vec(p,'esc_lower',4).astype(bool)|vec(p,'esc_upper',4).astype(bool)).reshape(n,5,4).any(axis=(1,2))
 action=(np.abs(vec(f,'action_applied',4))>=1-1e-9).any(axis=1)
 np.testing.assert_array_equal(f.interval_allocator_clipping,clipping)
 np.testing.assert_array_equal(f.interval_esc_boundary,esc);np.testing.assert_array_equal(f.interval_action_boundary,action)
 freeze=(clipping|esc|action)&f.integral_enabled;np.testing.assert_array_equal(f.integral_frozen,freeze)
 candidate=xi-.01*vec(f,'integral_gain_s_inv')*(before-target)
 np.testing.assert_array_equal(vec(f,'xi_candidate'),candidate)
 projected=candidate.copy();norm=np.linalg.norm(projected[:,:2],axis=1);ids=norm>.4
 projected[ids,:2]*=(.4/norm[ids])[:,None];projected[:,2]=np.clip(projected[:,2],-.15,.15)
 expected=np.where((f.integral_enabled&~freeze)[:,None],projected,xi)
 max_recurrence=max(max_recurrence,float(np.abs(expected-nxt).max()));np.testing.assert_allclose(expected,nxt,atol=1e-15)
 efficiency=np.ones((len(p.physics_time),4))
 for e in scenario['events']:
  if e['motor_number'] is not None:efficiency[p.physics_time>=e['time']-1e-9,e['motor_number']-1]=e['effectiveness']
 np.testing.assert_array_equal(vec(p,'motor_effectiveness',4),efficiency)
 for stage,control in [('thrust','applied_force_ctrl'),('reaction','applied_torque_ctrl')]:
  actual=vec(p,'motor_'+stage+'_actual',4)
  np.testing.assert_array_equal(actual,efficiency*vec(p,'motor_'+stage+'_nominal',4))
  np.testing.assert_array_equal(actual,vec(p,control,4))
 if scenario['events']:
  ev=read_columns(OUT/(r['key']+'-events.csv'));n_events+=len(ev.policy_input_time)
  expected_events=[e for e in scenario['events'] if e['time']<f.time_post[-1]-1e-9]
  assert len(ev.policy_input_time)==len(expected_events)
  for i,e in enumerate(expected_events):
   assert ev.policy_input_time[i]==e['time'] and ev.control_step[i]==round(e['time']/.01)
   np.testing.assert_array_equal(vec(ev,'xi_before')[i],vec(ev,'xi_after')[i])
   np.testing.assert_array_equal(vec(ev,'xi_before')[i],xi[int(ev.control_step[i])])
   np.testing.assert_array_equal(vec(ev,'p_cmd_after')[i],vec(ev,'target_after')[i]+vec(ev,'xi_after')[i])
  for e in r['event_results']:
   ids=np.flatnonzero(ev.policy_input_time==e['start_s'])
   event={k:v[ids] for k,v in ev.items()} if len(ids) else None
   assert e['recovery_s']==recover(f,event,e['start_s'],e['end_s'],f.time_post[-1])
 for name,metrics in r['windows'].items():
  if name in m['common_windows']:start,end=m['common_windows'][name]
  else:
   evdef=next(e for e in r['event_results'] if name==f"{e['kind']}_{e['start_s']:g}_{e['end_s']:g}");start,end=evdef['start_s'],evdef['end_s']
  mask=(f.time_post>start+1e-9)&(f.time_post<=end+1e-9)
  if f.time_post[-1]<end-1e-9:
   assert metrics is None
   continue
  assert metrics is not None and metrics['sample_count']==int(mask.sum())
  error=(position-target)[mask];mean=error.mean(0);variance=np.mean((error-mean)**2,axis=0);squares=np.mean(error**2,axis=0)
  max_identity=max(max_identity,float(np.max(np.abs(squares-mean**2-variance))))
  np.testing.assert_allclose([metrics['offset_xy_m'],metrics['mean_error_z_m'],metrics['sway_xy_rms_m'],metrics['sway_z_rms_m']],
    [np.linalg.norm(mean[:2]),mean[2],np.sqrt(variance[:2].sum()),np.sqrt(variance[2])],atol=1e-15)
 # Restore / target cases must reproduce their old matching prefix through
 # t=30 BEFORE the new event, without re-running old scenarios.
 if scenario['restore'] or scenario['target_after30'] is not None:
  oldname='motor_70' if scenario['mass']==0 else 'combined_70'
  old=read_columns(Path(m['previous_results'])/f"{oldname}-{r['label']}-{r['gain']}.csv")
  cols=[k for k in old if k in f and old[k].dtype.kind in 'fbiu' and f[k].dtype.kind in 'fbiu']
  maximum=0.
  for k in cols:
   np.testing.assert_allclose(old[k][:3000],f[k][:3000],rtol=1e-12,atol=1e-12,err_msg=k)
   maximum=max(maximum,float(np.max(np.abs(old[k][:3000].astype(float)-f[k][:3000].astype(float)))))
  prefix_checks[r['key']]=dict(control_samples=3000,numeric_columns=len(cols),max_abs_error=maximum)
 print('verified',r['key'],flush=True)
report=dict(rollouts=len(summary),control_samples=n_control,physics_samples=n_physics,events=n_events,
 max_recurrence_error_m=max_recurrence,max_bias_variance_identity_error_m2=max_identity,
 event_recovery_stops_at_next_event=True,reference_boundary_left_right_limits_verified=True,
 physical_state_and_xi_continuity_verified=True,force_and_reaction_efficiency_once=True,
 previous_30s_prefix_checks=prefix_checks)
(OUT/'independent_csv_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='previous_30s_prefix_checks'},indent=2))

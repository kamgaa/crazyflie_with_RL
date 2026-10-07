"""Independently recompute controller recurrence and metrics from saved signals.
Run after completion, from repository root. No simulation or policy execution.
"""
import sys
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from crazyflie_rl.integral_eval import read_columns

out=Path(__file__).resolve().parent
summary=json.loads((out/'summary.json').read_text())
manifest=json.loads((out/'manifest.json').read_text())
assert manifest['status']=='completed' and len(summary)==56

def vec(f,name,n=3):
 return np.column_stack([f[f'{name}_{i}'] for i in range(n)])

max_identity=0.; max_recurrence=0.; ncontrol=nphysics=0
for r in summary:
 f=read_columns(out/(r['key']+'.csv')); p=read_columns(out/(r['key']+'-physics.csv'))
 t=f.time; dt=f.control_dt; n=len(t); ncontrol+=n; nphysics+=len(p.physics_time)
 np.testing.assert_allclose(t,np.arange(n)*.01,atol=1e-14)
 np.testing.assert_allclose(f.time_post,t+dt,atol=1e-14)
 assert len(p.physics_time)==5*n
 position=vec(f,'position'); before=vec(f,'position_before'); target=vec(f,'p_target')
 xi=vec(f,'xi_t'); nextxi=vec(f,'xi_next'); error=before-target
 np.testing.assert_array_equal(target,np.tile([0,0,1.],(n,1)))
 np.testing.assert_array_equal(xi[0],np.zeros(3))
 np.testing.assert_array_equal(xi[1:],nextxi[:-1])
 np.testing.assert_array_equal(before[1:],position[:-1])
 np.testing.assert_array_equal(vec(f,'e_true_before'),error)
 np.testing.assert_array_equal(vec(f,'p_cmd'),target+xi)
 np.testing.assert_allclose(vec(f,'e_actor_before'),before-target-xi,atol=1e-15)
 native=vec(f,'observation',15); actor=vec(f,'policy_raw_observation',15)
 np.testing.assert_array_equal(native[:,3:],actor[:,3:])
 np.testing.assert_allclose(actor[:,:3],before-target-xi,atol=1e-8,rtol=1e-7)
 if r['gain']=='no_integral':np.testing.assert_array_equal(native,actor)
 np.testing.assert_allclose(actor[:,3:6],vec(f,'velocity_before'),atol=1e-7,rtol=1e-7)
 gains=vec(f,'integral_gain_s_inv'); candidate=xi-dt[:,None]*gains*error
 np.testing.assert_array_equal(candidate,vec(f,'xi_candidate'))
 cfg=r['integral_settings']; projected=candidate.copy(); norms=np.linalg.norm(projected[:,:2],axis=1)
 scale=np.ones(n);mask=norms>cfg['xy_limit_m'];scale[mask]=cfg['xy_limit_m']/norms[mask]
 projected[:,:2]*=scale[:,None];projected[:,2]=np.clip(projected[:,2],-cfg['z_limit_m'],cfg['z_limit_m'])
 clipping=np.any(vec(p,'allocator_clipped',4).reshape(n,5,4),axis=(1,2))
 esc=np.any((vec(p,'esc_lower',4)|vec(p,'esc_upper',4)).reshape(n,5,4),axis=(1,2)) if p.esc_lower_0.dtype.kind=='b' else np.any((vec(p,'esc_lower',4).astype(bool)|vec(p,'esc_upper',4).astype(bool)).reshape(n,5,4),axis=(1,2))
 action=np.any(np.abs(vec(f,'action_applied',4))>=1-1e-9,axis=1)
 np.testing.assert_array_equal(f.interval_allocator_clipping,clipping)
 np.testing.assert_array_equal(f.interval_esc_boundary,esc)
 np.testing.assert_array_equal(f.interval_action_boundary,action)
 freeze=(clipping|esc|action)&f.integral_enabled
 np.testing.assert_array_equal(f.integral_frozen,freeze)
 allowed=f.integral_enabled&~freeze
 np.testing.assert_array_equal(f.integral_update_allowed,allowed)
 expected=np.where(allowed[:,None],projected,xi)
 max_recurrence=max(max_recurrence,float(np.max(np.abs(expected-nextxi))))
 np.testing.assert_allclose(expected,nextxi,atol=1e-15,rtol=1e-14)
 np.testing.assert_allclose(f.integral_frozen_total_s,np.cumsum(freeze)*.01,atol=1e-12)
 assert np.max(np.linalg.norm(nextxi[:,:2],axis=1))<=cfg['xy_limit_m']+1e-12
 assert np.max(np.abs(nextxi[:,2]))<=cfg['z_limit_m']+1e-12
 actual_error=position-target
 np.testing.assert_array_equal(actual_error,vec(f,'position_error_world'))
 desired=-4*actual_error; norms=np.linalg.norm(desired,axis=1); mask=norms>1.5
 desired[mask]*=(1.5/norms[mask])[:,None]
 np.testing.assert_allclose(vec(f,'desired_velocity'),desired,atol=1e-14)
 np.testing.assert_allclose(vec(f,'internal_velocity_error'),vec(f,'velocity')-desired,atol=1e-14)
 for stage in ('thrust','reaction'):
  np.testing.assert_array_equal(vec(p,f'motor_{stage}_actual',4),vec(p,'motor_effectiveness',4)*vec(p,f'motor_{stage}_nominal',4))
 for name,(start,end) in manifest['windows'].items():
  m=r['windows'][name]; mask=(f.time_post>start+1e-9)&(f.time_post<=end+1e-9)
  if f.time_post[-1]<end-1e-9:
   assert m is None
   continue
  assert m and m['sample_count']==int(mask.sum())
  e=actual_error[mask];mu=e.mean(axis=0);var=np.mean((e-mu)**2,axis=0);square=np.mean(e**2,axis=0)
  np.testing.assert_allclose(m['mean_error_xy_m'],mu[:2],atol=1e-15)
  np.testing.assert_allclose([m['offset_xy_m'],m['mean_error_z_m'],m['sway_xy_rms_m'],m['sway_z_rms_m']],
    [np.linalg.norm(mu[:2]),mu[2],np.sqrt(var[:2].sum()),np.sqrt(var[2])],atol=1e-15)
  residual=square-mu**2-var;max_identity=max(max_identity,float(np.abs(residual).max()))
 if r['fault_applied']:
  events=read_columns(out/(r['key']+'-events.csv'))
  assert len(events.control_step)==1 and events.control_step[0]==500 and events.policy_input_time[0]==5
  np.testing.assert_array_equal(vec(events,'xi_before'),vec(events,'xi_after'))
  np.testing.assert_array_equal(vec(events,'xi_before')[0],xi[500])
  for dim,axes in [('xy',[0,1]),('z',[2]),('3d',[0,1,2])]:
   mask=f.time_post>=5-1e-9;times=f.time_post[mask];e=actual_error[mask][:,axes];v=vec(f,'velocity')[mask][:,axes]
   good=(np.linalg.norm(e,axis=1)<=.005)&(np.linalg.norm(v,axis=1)<=.02)
   suffix=np.logical_and.accumulate(good[::-1])[::-1]
   candidates=np.flatnonzero(suffix&(60-times>=1-1e-9))
   expected=float(times[candidates[0]]-5) if len(candidates) and r['completed'] else None
   assert r[f'recovery_{dim}_s']==expected
 print('verified',r['key'],flush=True)
report=dict(rollouts=len(summary),control_samples=ncontrol,physics_samples=nphysics,
            max_recurrence_error_m=max_recurrence,max_bias_variance_identity_error_m2=max_identity,
            true_target_metrics=True,raw_velocity_unchanged=True,interval_saturation_verified=True,
            force_and_reaction_efficiency_applied_once=True,fixed_windows_and_suffix_recovery_verified=True)
(out/'independent_csv_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

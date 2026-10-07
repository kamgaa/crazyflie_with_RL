"""Independent checks of completed saved CSVs; no policy inference or simulation."""
import csv,json,sys
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];sys.path.insert(0,str(ROOT))
from crazyflie_rl.integral_eval import read_columns
from crazyflie_rl.dr_policy import sha256

manifest=json.loads((OUT/'manifest.json').read_text())
results=json.loads((OUT/'summary.json').read_text())
assert manifest['status']=='completed' and len(results)==40

def vec(f,name,n=3):return np.column_stack([f[f'{name}_{i}'] for i in range(n)])

def recovery(f,event,index,start,end):
    if f.time_post[-1]<end-1e-9:return dict.fromkeys(('xy','z','3d'))
    take=(f.time_post>start+1e-9)&(f.time_post<=end+1e-9)
    ts=np.r_[start,f.time_post[take]]
    es=np.vstack((vec(event,'e_true_after')[index],vec(f,'position_error_world')[take]))
    vs=np.vstack((vec(event,'velocity')[index],vec(f,'velocity')[take]))
    out={}
    for axis,sl in [('xy',slice(0,2)),('z',slice(2,3)),('3d',slice(0,3))]:
        good=(np.linalg.norm(es[:,sl],axis=1)<=.005)&(np.linalg.norm(vs[:,sl],axis=1)<=.02)
        suffix=np.logical_and.accumulate(good[::-1])[::-1]
        ids=np.flatnonzero(suffix & (end-ts>=1-1e-9))
        out[axis]=float(ts[ids[0]]-start) if len(ids) else None
    return out

counts=dict(control=0,physics=0,events=0)
event_boundaries=[]
maxima=dict(allocation_command_error_n=0.,allocation_residual_error=0.,actuator_residual_error=0.,integral_recurrence_m=0.,bias_variance_m2=0.)
for r in results:
    f=read_columns(OUT/(r['key']+'.csv'));p=read_columns(OUT/(r['key']+'-physics.csv'))
    n=len(f.time);counts['control']+=n;counts['physics']+=len(p.physics_time)
    assert len(p.physics_time)==5*n
    np.testing.assert_allclose(f.time,np.arange(n)*.01,atol=1e-12)
    np.testing.assert_allclose(f.time_post,f.time+.01,atol=1e-12)
    np.testing.assert_allclose(f.motor_sample_time_post,f.time_post,atol=1e-8)
    np.testing.assert_array_equal(vec(f,'position_before')[1:],vec(f,'position')[:-1])
    np.testing.assert_array_equal(vec(f,'xi_t')[1:],vec(f,'xi_next')[:-1])
    np.testing.assert_array_equal(vec(f,'xi_t')[0],0)
    target=np.tile([0.,0.,1.],(n,1));np.testing.assert_array_equal(vec(f,'reference'),target)
    np.testing.assert_array_equal(vec(f,'p_target'),target)
    np.testing.assert_array_equal(vec(f,'p_cmd'),target+vec(f,'xi_t'))
    np.testing.assert_array_equal(vec(f,'e_true_before'),vec(f,'position_before')-target)
    np.testing.assert_array_equal(vec(f,'e_true_post'),vec(f,'position')-target)
    native=vec(f,'observation',15);actor=vec(f,'policy_raw_observation',15)
    np.testing.assert_array_equal(native[:,3:],actor[:,3:])
    np.testing.assert_allclose(actor[:,:3],vec(f,'position_before')-vec(f,'p_cmd'),atol=1e-8,rtol=1e-7)
    np.testing.assert_array_equal(actor[:,3:6],vec(f,'velocity_before').astype(np.float32))
    if r['gain']=='no_integral':np.testing.assert_array_equal(native,actor)
    scenario=manifest['scenarios'][r['condition']]
    eta=np.ones((len(p.physics_time),4))
    for event in scenario['events']:eta[p.physics_time>=event['time']-1e-9,event['motor_number']-1]=event['effectiveness']
    used=eta if r['allocator_mode']=='oracle' else np.ones_like(eta)
    np.testing.assert_array_equal(vec(p,'plant_efficiency',4),eta)
    np.testing.assert_array_equal(vec(p,'allocator_efficiency',4),used)
    physical=manifest['runs'][r['key']]['physical']
    B0=np.array(physical['B0_allocator']);Bxml=np.array(physical['B0_plant_xml'])
    desired=vec(p,'desired_wrench',4);raw=np.empty_like(desired)
    for eff in np.unique(used,axis=0):
        indices=np.all(used==eff,axis=1);raw[indices]=desired[indices]@np.linalg.pinv(B0*eff[None,:]).T
    maxima['allocation_command_error_n']=max(maxima['allocation_command_error_n'],float(np.max(np.abs(raw-vec(p,'motor_thrust_unclipped',4)))))
    np.testing.assert_allclose(raw,vec(p,'motor_thrust_unclipped',4),atol=1e-14,rtol=1e-13)
    cmd=np.clip(raw,0,.2);np.testing.assert_allclose(cmd,vec(p,'motor_thrust_command',4),atol=1e-14,rtol=1e-13)
    nominal=vec(p,'motor_thrust_nominal',4);actual=vec(p,'motor_thrust_actual',4)
    np.testing.assert_array_equal(actual,eta*nominal)
    np.testing.assert_array_equal(vec(p,'motor_reaction_actual',4),eta*vec(p,'motor_reaction_nominal',4))
    np.testing.assert_array_equal(actual,vec(p,'applied_force_ctrl',4))
    np.testing.assert_array_equal(vec(p,'motor_reaction_actual',4),vec(p,'applied_torque_ctrl',4))
    predicted=(eta*vec(p,'motor_thrust_command',4))@Bxml.T
    actual_wrench=actual@Bxml.T;actual_wrench[:,2]=vec(p,'motor_reaction_actual',4).sum(1)
    for key,values,maximum in [('allocation_residual_xml',desired-predicted,'allocation_residual_error'),('actuator_response_residual_xml',predicted-actual_wrench,'actuator_residual_error')]:
        error=float(np.max(np.abs(vec(p,key,4)-values)));maxima[maximum]=max(maxima[maximum],error)
        np.testing.assert_allclose(vec(p,key,4),values,atol=1e-14,rtol=1e-13)
    clipping=vec(p,'allocator_clipped',4).reshape(n,5,4).astype(bool).any((1,2))
    esc=(vec(p,'esc_lower',4).astype(bool)|vec(p,'esc_upper',4).astype(bool)).reshape(n,5,4).any((1,2))
    action=(np.abs(vec(f,'action_applied',4))>=1-1e-9).any(1)
    np.testing.assert_array_equal(f.interval_allocator_clipping,clipping)
    np.testing.assert_array_equal(f.interval_esc_boundary,esc)
    np.testing.assert_array_equal(f.interval_action_boundary,action)
    freeze=(clipping|esc|action)&f.integral_enabled;np.testing.assert_array_equal(f.integral_frozen,freeze)
    candidate=vec(f,'xi_t')-.01*vec(f,'integral_gain_s_inv')*vec(f,'e_true_before')
    np.testing.assert_array_equal(candidate,vec(f,'xi_candidate'))
    projected=candidate.copy();norm=np.linalg.norm(projected[:,:2],axis=1);mask=norm>.4
    projected[mask,:2]*=(.4/norm[mask])[:,None];projected[:,2]=np.clip(projected[:,2],-.15,.15)
    expected=np.where((f.integral_enabled&~freeze)[:,None],projected,vec(f,'xi_t'))
    maxima['integral_recurrence_m']=max(maxima['integral_recurrence_m'],float(np.max(np.abs(expected-vec(f,'xi_next')))))
    np.testing.assert_allclose(expected,vec(f,'xi_next'),atol=1e-15)
    if scenario['events']:
        ev=read_columns(OUT/(r['key']+'-events.csv'));counts['events']+=len(ev.policy_input_time)
        for i,e in enumerate(r['event_results']):
            if not e['applied']:assert e['recovery_s']==dict.fromkeys(('xy','z','3d'));continue
            assert ev.policy_input_time[i]==e['start_s']
            np.testing.assert_array_equal(vec(ev,'xi_before')[i],vec(ev,'xi_after')[i])
            np.testing.assert_array_equal(vec(ev,'xi_after')[i],vec(f,'xi_t')[int(ev.control_step[i])])
            assert e['recovery_s']==recovery(f,ev,i,e['start_s'],e['end_s'])
            j=int(round(e['start_s']*500))
            record=dict(key=r['key'],event=e['kind'],time_s=e['start_s'])
            for field in ('desired_wrench','allocator_efficiency','plant_efficiency','motor_thrust_unclipped',
                          'motor_thrust_command','motor_thrust_nominal','motor_thrust_actual',
                          'allocation_residual_xml','allocation_residual_b0','actuator_response_residual_xml','allocator_clipped'):
                record[field]=json.dumps(vec(p,field,4)[j].tolist())
            event_boundaries.append(record)
    for group in ('windows','partial_observed_windows'):
        for name,metric in r[group].items():
            if metric is None:continue
            mask=(f.time_post>=metric['first_sample_s']-1e-9)&(f.time_post<=metric['last_sample_s']+1e-9)
            e=vec(f,'position_error_world')[mask];mu=e.mean(0);var=((e-mu)**2).mean(0);sq=(e*e).mean(0)
            maxima['bias_variance_m2']=max(maxima['bias_variance_m2'],float(np.max(np.abs(sq-mu**2-var))))
            np.testing.assert_allclose([metric['offset_xy_m'],metric['mean_error_z_m'],metric['sway_xy_rms_m'],metric['sway_z_rms_m']],
                [np.linalg.norm(mu[:2]),mu[2],np.sqrt(var[:2].sum()),np.sqrt(var[2])],atol=1e-14)
    if not r['completed']:assert r['windows']['tail_58_60'] is None and r['windows']['full_0_60'] is None
    print('verified',r['key'],flush=True)

protected=json.loads((OUT/'initial_protected_hashes.json').read_text())
assert all(sha256(p)==h for p,h in protected.items())
report=dict(rollouts=40,counts=counts,maximum_numeric_errors=maxima,initial_protected_files_unchanged=len(protected),
    single_efficiency_application=True,current_efficiency_only=True,events_preserve_state=True,
    integral_uses_actual_allocator_saturation=True,event_recovery_capped_at_next_event=True,
    existing_and_oracle_equivalence_checks='regression_comparisons.json')
(OUT/'independent_csv_verification.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'event_boundary_signals.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,list(event_boundaries[0]));writer.writeheader();writer.writerows(event_boundaries)
print(json.dumps(report,indent=2),flush=True)

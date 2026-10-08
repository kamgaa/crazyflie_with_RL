"""Independent CSV arithmetic, causal routing, event and preservation checks."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
from pathlib import Path
import numpy as np
from crazyflie_rl.circle_fault_eval import CONFIG,CircleProtocol,trace_comparison
from crazyflie_rl.config import load_config
from crazyflie_rl.dr_transfer import ROOT
from crazyflie_rl.dr_policy import sha256
from crazyflie_rl.estimated_allocation import ConfirmedEfficiency
from crazyflie_rl.fault_estimation_eval import nominal_model,trace_rows,estimate_from_row,write_json
from crazyflie_rl.fault_estimator import EstimatorSettings,MotorEfficiencyEstimator
from crazyflie_rl.integral_eval import read_columns
from crazyflie_rl.motor_layout import native_from_user
from crazyflie_rl.oracle_allocation import efficiency_matrix
from crazyflie_rl.oracle_eval import flat_csv


def vector(c,key,n=3):return np.column_stack([c[f'{key}_{j}'] for j in range(n)])


def verify(directory):
    d=Path(directory);manifest=json.loads((d/'manifest.json').read_text());results=json.loads((d/'summary.json').read_text())
    assert manifest['status']=='complete' and len(results)==(3 if manifest['smoke'] else 20)
    settings=EstimatorSettings(**json.loads((d/'estimator_settings.json').read_text()))
    assert sha256(d/'estimator_settings.json')==manifest['estimator_settings_sha256']
    report=dict(runs={},prefault_comparisons={},causality={},preservation={},reference_checks={})
    compact=[]
    for r in results:
        key=r['key'];c=read_columns(d/(key+'.csv'));t=c.time_post;times=c.time;dt=t-times
        protocol=CircleProtocol(r['period_s']);f=protocol.fault_time;has_fault=r['variant']!='healthy'
        reference_key=f"T{r['period_s']:g}-{r['label']}-healthy"
        assert manifest['runs'][key]['initial']==manifest['runs'][reference_key]['initial']
        np.testing.assert_array_equal(c.estimator_update_count,np.arange(1,len(t)+1));np.testing.assert_array_equal(c.estimate_time,t)
        np.testing.assert_allclose(c.gyro_observation_time,t-.002,atol=1e-12);np.testing.assert_allclose(dt,.01,atol=1e-12)
        truth=vector(c,'truth_efficiency_user',4);expected=np.ones_like(truth);post=times>=f-1e-9
        if has_fault:expected[post,0]=.7
        np.testing.assert_array_equal(truth,expected)
        eta=vector(c,'estimated_efficiency_user',4);used=vector(c,'allocator_efficiency_user',4)
        holder=ConfirmedEfficiency();B0=np.array(manifest['runs'][key]['metadata']['B0_allocator']);inverse=np.linalg.pinv(B0)
        raw=vector(c,'motor_thrust_unclipped',4);wrench=vector(c,'wrench_command',4);max_alloc=0.
        for i in range(len(t)):
            expected_used=np.ones(4) if r['allocation_mode']=='blind' else truth[i] if r['allocation_mode']=='oracle' else holder.eta_user.copy()
            np.testing.assert_array_equal(used[i],expected_used)
            source=str(c.estimate_source_time_used[i])
            assert source=='' if i==0 else float(source)==t[i-1] and float(source)<=times[i]+1e-12
            native=native_from_user(expected_used);B=efficiency_matrix(B0,native)
            pinv=inverse if np.all(native==1) else np.linalg.pinv(B)
            max_alloc=max(max_alloc,float(np.max(abs(pinv@wrench[i]-raw[i]))))
            holder.update(state=c.estimator_state[i],motor=int(c.estimated_motor[i]),
                hypothesis_alpha_user=np.array([c[f'hypothesis_alpha_user_{j}'][i] for j in range(4)]),estimate_time=t[i])
            np.testing.assert_array_equal(holder.eta_user,[c[f'held_efficiency_next_user_{j}'][i] for j in range(4)])
        assert max_alloc<1e-14
        # Independent analytic expression, not the generator implementation.
        omega=2*np.pi/r['period_s'];s=np.clip((t-5)/2,0,1)
        theta=np.where(t<7,2*omega*(s**3-s**4/2),omega*(t-6))
        rate=np.where(t<7,omega*(3*s*s-2*s**3),omega)
        ref=np.column_stack([np.cos(theta),np.sin(theta),np.ones(len(t))])
        vref=np.column_stack([-rate*np.sin(theta),rate*np.cos(theta),np.zeros(len(t))])
        np.testing.assert_allclose(ref,vector(c,'reference_post'),atol=1e-14)
        np.testing.assert_allclose(vref,vector(c,'reference_velocity_post'),atol=1e-14)
        e=vector(c,'position')-ref;ve=vector(c,'velocity')-vref
        np.testing.assert_allclose(e,vector(c,'position_error_world'),atol=1e-14)
        np.testing.assert_allclose(ve,vector(c,'velocity_error'),atol=1e-14)
        np.testing.assert_allclose(c.radial_error_m,e[:,0]*np.cos(theta)+e[:,1]*np.sin(theta),atol=1e-14)
        np.testing.assert_allclose(c.tangential_error_m,-e[:,0]*np.sin(theta)+e[:,1]*np.cos(theta),atol=1e-14)
        # Velocity channel is original raw actual velocity, even while moving.
        np.testing.assert_array_equal(vector(c,'policy_raw_observation',15)[:,3:],vector(c,'observation',15)[:,3:])
        np.testing.assert_allclose(vector(c,'policy_raw_observation',15)[:,3:6],vector(c,'velocity_before'),rtol=1e-7,atol=1e-7)
        np.testing.assert_array_equal(vector(c,'p_target'),vector(c,'reference'))
        np.testing.assert_array_equal(vector(c,'p_cmd'),vector(c,'p_target')+vector(c,'xi_t'))
        np.testing.assert_allclose(vector(c,'e_true_before'),vector(c,'position_before')-vector(c,'reference'),atol=1e-15)
        if r['label']=='PID':
            np.testing.assert_array_equal(vector(c,'xi_next'),np.zeros((len(t),3)))
            np.testing.assert_array_equal(vector(c,'action',4),np.zeros((len(t),4)))
        for name,(a,b) in protocol.windows().items():
            mask=(t>a+1e-9)&(t<=b+1e-9);stat=r['windows'][name]
            if t[-1]<b-1e-9:
                assert stat is None
                stat=r['partial_observed_windows'][name]
            if not mask.any():assert stat is None;continue
            values=[np.sqrt(np.mean(np.sum(e[mask,:2]**2,axis=1))),np.sqrt(np.mean(e[mask,2]**2)),
                np.linalg.norm(e[mask,:2],axis=1).max(),np.abs(e[mask,2]).max(),
                np.sqrt(np.mean(np.sum(ve[mask,:2]**2,axis=1))),np.sqrt(np.mean(ve[mask,2]**2))]
            saved=[stat[k] for k in ('position_rmse_xy','position_rmse_z','max_xy_error_m','max_abs_z_error_m',
                                    'velocity_reference_error_rms_xy','velocity_reference_error_rms_z')]
            np.testing.assert_allclose(values,saved,atol=1e-12,rtol=1e-12)
            flags=vector(c,'physics_allocator_clipped',20)[mask].astype(bool).reshape(-1,4);h=np.repeat(dt[mask]/5,5)
            np.testing.assert_allclose(np.sum(flags*h[:,None],axis=0),stat['clipping']['duration_s_per_channel'],atol=1e-10)
            assert abs(np.sum(h[np.any(flags,axis=1)])-stat['clipping']['duration_any_s'])<1e-10
        command=vector(c,'motor_thrust_command',4)
        assert np.min(command)>=0 and np.max(command)<=.289
        native=truth[:,::-1]
        np.testing.assert_array_equal(vector(c,'motor_thrust_actual',4),native*vector(c,'motor_thrust_nominal',4))
        np.testing.assert_array_equal(vector(c,'motor_reaction_actual',4),native*vector(c,'motor_reaction_nominal',4))
        geometry=np.array(manifest['runs'][key]['metadata']['B0_physical_xml'])
        static=(geometry@(native*command).T).T
        np.testing.assert_allclose(static,vector(c,'static_rotor_wrench_xml',4),atol=1e-15)
        np.testing.assert_allclose(wrench-static,vector(c,'allocation_residual_xml',4),atol=1e-15)
        np.testing.assert_allclose(static-vector(c,'actual_rotor_wrench_xml',4),vector(c,'actuator_response_residual_xml',4),atol=1e-15)
        events=manifest['runs'][key]['events']
        assert len(events)==int(has_fault and t[-1]>f+1e-9)
        for event in events:
            assert event['policy_input_time']==f and event['native_motor_index']==3 and event['user_motor_id']==1
            np.testing.assert_allclose(event['theta'],4*np.pi,atol=1e-14)
            np.testing.assert_allclose(event['target_after'],[1,0,1],atol=1e-14)
            assert event['physical_and_actuator_state_unchanged'] and event['integral_state_preserved'] and event['pid_state_preserved']
            assert event['estimator_updates_preserved']==round(f/.01) and not event['estimator_reset']
        est=r['estimation'];healthy=~post if has_fault else np.ones(len(t),bool)
        fp=healthy&(c.estimator_state=='fault')
        assert abs(np.sum(dt[fp])-est['false_positive_seconds'])<1e-10
        if has_fault:
            hit=np.flatnonzero(post&(c.estimator_state=='fault')&(c.estimated_motor==1))
            detection=float(t[hit[0]]) if len(hit) else None;assert detection==est['confirmed_detection_time']
            for name,a,b in [('post_first_05s',f,f+.5),('post_first_1s',f,f+1),('post_all',f,protocol.horizon),('last_lap',protocol.horizon-protocol.period,protocol.horizon)]:
                mask=(t>a+1e-9)&(t<=b+1e-9);value=est['efficiency_windows'][name]['full' if t[-1]>=b-1e-9 else 'partial']
                if mask.any():
                    err=eta[mask,0]-truth[mask,0]
                    np.testing.assert_allclose([np.mean(abs(err)),np.sqrt(np.mean(err**2))],[value['mae'],value['rmse']],atol=1e-14)
                else:assert value is None
        change=np.flatnonzero(np.any(used!=1,axis=1));apply=float(times[change[0]]) if len(change) else None
        assert apply==est['first_compensation_time']
        if key!=reference_key:
            comparison=trace_comparison(d/(key+'.csv'),d/(reference_key+'.csv'),min(f,t[-1]))
            report['prefault_comparisons'][key]=comparison
            if not comparison['equal']:
                # A confirmed false alarm is a result, not a reason to tune it away.
                assert r['allocation_mode']=='estimated' and est['false_positive_seconds']>0
                comparison['cause']='confirmed prefault false positive changed estimated allocation'
        decisions=[]
        for i in range(len(t)):
            if i==0 or c.estimator_state[i]!=c.estimator_state[i-1] or c.estimated_motor[i]!=c.estimated_motor[i-1]:
                decisions.append(dict(kind='decision_transition',available_time=float(t[i]),
                    state=str(c.estimator_state[i]),confirmed_user_motor=int(c.estimated_motor[i]),
                    estimated_efficiency_user=eta[i],first_possible_command_time=float(t[i])))
        if apply is not None:
            i=int(change[0]);decisions.append(dict(kind='first_allocator_compensation',available_time=apply,
                efficiency_used_user=used[i],estimate_source_time=str(c.estimate_source_time_used[i]),
                allocation_mode=r['allocation_mode']))
        flat_csv(d/(key+'-decision-events.csv'),decisions)
        report['runs'][key]=dict(samples=len(t),max_allocation_recomputation_error_n=max_alloc,
            csv_metrics_recomputed=True,events_preserve_state=True,raw_velocity_unchanged=True)
        compact.append(dict(key=key,period_s=r['period_s'],label=r['label'],variant=r['variant'],completed=r['completed'],
            duration_s=r['duration_s'],termination_reasons=r['termination_reasons'],
            prefault=r['windows']['prefault_lap'],post=r['windows']['post_all'],last_lap=r['windows']['last_lap'],
            observed_partial_post=r['partial_observed_windows']['post_all'],estimation=est))
    # Mutation tests on allowed-only replay. No truth, event, or whole env object
    # reaches estimator. Future altered samples may affect later outputs only.
    selected=next((r for r in results if r['label']=='A_best' and r['variant']=='estimated_fault'),
                  next(r for r in results if r['label']=='A_best'))
    model,_=nominal_model(load_config(CONFIG));saved=trace_rows(d/(selected['key']+'.csv'))
    estimators=[MotorEfficiencyEstimator(model,settings) for _ in range(3)];holders=[ConfirmedEfficiency() for _ in range(3)]
    max_truth=max_future=0.;switch=min(len(saved)//2,round((6+2*selected['period_s']+.5)/.01))
    for i,row in enumerate(saved):
        fake=dict(row,truth_efficiency_user=np.zeros(4),fault_time=-999,scenario='motor4_other_truth',qacc=np.ones(10)*99,motor_omega=np.ones(4)*999)
        future=dict(row)
        if i>=switch:
            future['velocity']=row['velocity']+[1,2,3];future['delivered_esc_native']=row['delivered_esc_native']*.5
        outputs=[estimate_from_row(e,x,.002) for e,x in zip(estimators,[row,fake,future])]
        for h,out in zip(holders,outputs):h.consume(out)
        arrays=[]
        for h,out in zip(holders,outputs):
            B=efficiency_matrix(B0,native_from_user(h.eta_user));cmd=np.linalg.pinv(B)@np.array([.001,-.001,.0001,.43])
            arrays.append(np.r_[out['hypothesis_scores'],out['estimated_efficiency_user'],h.eta_user,cmd])
        max_truth=max(max_truth,float(np.max(abs(arrays[0]-arrays[1]))))
        if i<switch:max_future=max(max_future,float(np.max(abs(arrays[0]-arrays[2]))))
    assert max_truth==max_future==0
    report['causality']=dict(truth_metadata_mutation_max_error=max_truth,future_mutation_prefix_max_error=max_future,prefix_samples=switch,
        includes_estimates_and_allocated_commands=True,source_trace=selected['key'])
    original=json.loads((d/'protected_hashes_before.json').read_text())
    def check(item):
        name,old=item;return name,sha256(ROOT/name)==old
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:checks=dict(pool.map(check,original.items()))
    changed=[k for k,v in checks.items() if not v]
    allowed=['crazyflie_rl/integral_validation.py']
    assert set(changed)<=set(allowed),changed
    # Check modified old file has only the explicit continuous-reference guard.
    if changed:
        before=(d/'integral_validation_before.py.txt').read_text()
        replacement="        # Fixed/event setpoints use the interval's held reference. Continuous\n        # missions may evaluate the post-state against reference(t + dt).\n        if getattr(self, 'fixed_reference', True):\n            np.testing.assert_array_equal(row['reference_post'], env.pos_des)"
        assert (ROOT/allowed[0]).read_text()==before.replace("        np.testing.assert_array_equal(row['reference_post'], env.pos_des)",replacement)
    for policy in manifest['models']:assert sha256(Path(policy['path']))==policy['sha256']
    for name,value in manifest['source_sha256'].items():assert sha256(ROOT/name)==value,name
    report['preservation']=dict(checked_files=len(checks),intentional_common_source_change=changed,
        all_other_preexisting_files_unchanged=True,checkpoints_and_settings_unchanged=True)
    report['scope']='Focused tests plus all-run CSV checks; not a full-repository test run. Ideal simulator states; no compute-delay model.'
    write_json(d/'logging_contract.json',dict(
        state='position, quaternion, velocity, omega and same-time tracking reference at time_post; cached gyro actually at time_post-0.002',
        policy='raw observation, true/control target, xi_t, action and eta_used at time; xi_next and estimate available at time_post',
        held_reference='reference is command held over [time,time_post]; reference_post is analytic evaluation target at time_post',
        pid='zero residual action; existing CascadePID executes at 500Hz. Per-substep commands are physics_wrench_command. Single wrench_command and motor_* signals are LAST physics substep, timestamp motor_sample_time, output timestamp motor_sample_time_post.',
        ppo='deterministic policy at 100Hz; same wrench held for five physics substeps; external integral advances once per control interval',
        motor_arrays='native index order in existing fields; user efficiency arrays explicit. User ids 1,2,3,4 correspond to native indices 3,2,1,0.',
        wrench='all diagnostics native FLU body origin [tau_x,tau_y,tau_z,T]; T positive thrust magnitude, no gravity or external torque included',
        physics_arrays='five chronological substeps, flattened row-major (substep then native motor or wrench component); dt=0.002',
        residuals='allocation: requested minus true-efficiency/clipped-command static XML wrench; response: that static wrench minus lagged actual rotor wrench; geometry difference kept separate from B0',
        internal_desired_velocity='environment reward diagnostic against interval-held position; analytic reference_velocity_post is a separate tracking metric only',
        segments='post-state samples (start,end]; endpoint at fault belongs to pre-fault physics interval; changed eta first applied at fault before next step',
        partial='fixed windows unavailable after early termination are null, observed partial windows named separately',
        no_hover_recovery_metric=True))
    flat_csv(d/'comparison_table.csv',compact)
    analysis=[]
    for item in results:
        c=read_columns(d/(item['key']+'.csv'))
        post=c.time>=6+2*item['period_s']-1e-9
        dt=c.time_post-c.time
        xi=vector(c,'xi_next')
        physics_esc=vector(c,'delivered_esc_native',20).reshape(-1,5,4)
        analysis.append(dict(key=item['key'],observed_post_samples=int(post.sum()),
            post_integral_xy_limit_seconds=float(dt[post&(np.linalg.norm(xi[:,:2],axis=1)>=.4-1e-12)].sum()),
            post_integral_z_limit_seconds=float(dt[post&(np.abs(xi[:,2])>=.15-1e-12)].sum()),
            post_esc_max=float(physics_esc[post].max()) if post.any() else None,
            post_motor_nominal_max_n=float(vector(c,'motor_thrust_nominal',4)[post].max()) if post.any() else None,
            post_action_max_abs=np.max(np.abs(vector(c,'action',4)[post]),axis=0) if post.any() else None))
    flat_csv(d/'control_diagnostics.csv',analysis)
    if not manifest['smoke']:
        from crazyflie_rl.plotting import _pyplot
        plt=_pyplot()
        for T in (10.,5.):
            for label in ('A_best','B_best'):
                fig,axes=plt.subplots(4,3,figsize=(16,11),sharex=True,sharey='row',layout='constrained')
                for col,mode in enumerate(('blind','oracle','estimated')):
                    c=read_columns(d/f'T{T:g}-{label}-{mode}_fault.csv');x=c.motor_sample_time-(6+2*T)
                    for i in range(4):
                        ax=axes[i,col]
                        for field,color,style,title in [('wrench_command','k','-','Requested'),
                            ('static_rotor_wrench_xml','C1','--','Static, true eta + clipped command'),
                            ('actual_rotor_wrench_xml','C0','-','Actual, after motor lag')]:
                            ax.plot(x,c[f'{field}_{i}'],color=color,ls=style,label=title)
                        ax.axvline(0,color='grey',ls=':');ax.set_xlim(-.15,2.0);ax.grid(alpha=.25)
                        ax.set_ylabel(('Roll torque (Nm)','Pitch torque (Nm)','Yaw torque (Nm)','Positive total thrust (N)')[i])
                        if i==0:ax.set_title(mode+' | '+label+' PPO + I');ax.legend(fontsize=7)
                        if i==3:ax.set_xlabel('Time from fault (s), last physics substep')
                fig.suptitle(f'T={T:g}s | Native FLU body-origin rotor wrench; XML arm 0.03536m vs B0 arm 0.035355m')
                fig.savefig(d/f'T{T:g}-{label}-wrench-components.png',dpi=135);plt.close(fig)
    write_json(d/'independent_verification.json',report)
    write_json(d/'completion.json',dict(status='complete',runs=len(results),completed=sum(r['completed'] for r in results),
        terminated=sum(r['terminated'] for r in results),no_training_or_tuning=True,verification_passed=True,
        preserved_existing_files=len(checks)-len(changed)))
    print(json.dumps(dict(verified=len(results),causality=report['causality'],preservation=report['preservation']),indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    verify(p.parse_args().run_dir)

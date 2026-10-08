"""Independent CSV checks for the fixed 66-run closed-loop experiment."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import numpy as np
from crazyflie_rl.config import load_config
from crazyflie_rl.dr_transfer import ROOT
from crazyflie_rl.dr_policy import sha256
from crazyflie_rl.estimated_allocation import ConfirmedEfficiency
from crazyflie_rl.fault_estimation_eval import nominal_model, trace_rows, estimate_from_row, write_json
from crazyflie_rl.fault_estimator import MotorEfficiencyEstimator, EstimatorSettings
from crazyflie_rl.integral_eval import read_columns
from crazyflie_rl.motor_layout import native_from_user
from crazyflie_rl.oracle_allocation import efficiency_matrix
from crazyflie_rl.oracle_eval import flat_csv


def vector(c,key,n=3):
    return np.column_stack([c[f'{key}_{j}'] for j in range(n)])


def suffix(times,good,end,hold,start):
    if not len(times) or not good[-1]:return None
    bad=np.flatnonzero(~good);i=int(bad[-1]+1) if len(bad) else 0
    return float(times[i]-start) if end-times[i]>=hold-1e-9 else None


def verify(directory):
    d=Path(directory);manifest=json.loads((d/'manifest.json').read_text())
    results=json.loads((d/'summary.json').read_text())
    assert manifest['status']=='complete' and len(results)==66
    old=json.loads((Path(manifest['source_shadow_results'])/'manifest.json').read_text())
    settings=EstimatorSettings(**json.loads((d/'estimator_settings.json').read_text()))
    assert sha256(d/'estimator_settings.json')==manifest['estimator_settings_sha256']==old['estimator_settings_sha256']
    report={'runs':{},'application_rule':{},'causality':{},'preservation':{},'pooled_runtime_ms':{}}
    runtime={mode:[] for mode in ('blind','oracle','estimated')}
    for r in results:
        key=r['key'];c=read_columns(d/(key+'.csv'));t=c.time_post;times=c.time;dt=t-times
        blind_key=f"{r['label']}-{r['scenario']}-blind"
        assert manifest['runs'][key]['initial']==manifest['runs'][blind_key]['initial']
        assert manifest['runs'][key]['initial']==old['runs'][f"{r['label']}-{r['scenario']}"]['initial']
        np.testing.assert_allclose(dt,.01,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(c.estimator_update_count,np.arange(1,len(t)+1))
        np.testing.assert_array_equal(c.estimate_time,t)
        np.testing.assert_allclose(c.gyro_observation_time,t-.002,atol=1e-12)
        events=manifest['runs'][key]['events'];fault=r['fault_user_motor']
        truth=vector(c,'truth_efficiency_user',4);expected=np.ones_like(truth);post=times>=5-1e-9
        if fault:expected[post,fault-1]=r['fault_efficiency']
        np.testing.assert_array_equal(truth,expected)
        est=vector(c,'estimated_efficiency_user',4);used=vector(c,'allocator_efficiency_user',4)
        latch=ConfirmedEfficiency();B0=np.array(manifest['runs'][key]['metadata']['allocator']);B0_inverse=np.linalg.pinv(B0)
        wrench_array=vector(c,'wrench_command',4);raw_array=vector(c,'motor_thrust_unclipped',4)
        max_allocation=0.
        for i in range(len(t)):
            source=str(c.estimate_source_time_used[i])
            confirmation=str(c.held_confirmation_time_used[i])
            if i==0:assert source==''
            else:assert float(source)==t[i-1] and float(source)<=times[i]+1e-12
            if latch.last_confirmation_time is None:assert confirmation==''
            else:assert float(confirmation)==latch.last_confirmation_time
            expected_used=(np.ones(4) if r['allocation_mode']=='blind' else truth[i] if r['allocation_mode']=='oracle' else latch.eta_user.copy())
            np.testing.assert_array_equal(used[i],expected_used)
            native=native_from_user(expected_used);B=efficiency_matrix(B0,native)
            inverse=B0_inverse if np.all(native==1) else np.linalg.pinv(B)
            wrench=wrench_array[i]
            max_allocation=max(max_allocation,float(np.max(np.abs(inverse@wrench-raw_array[i]))))
            latch.update(state=c.estimator_state[i],motor=int(c.estimated_motor[i]),
                hypothesis_alpha_user=np.array([c[f'hypothesis_alpha_user_{j}'][i] for j in range(4)]),estimate_time=t[i])
            np.testing.assert_array_equal(latch.eta_user,[c[f'held_efficiency_next_user_{j}'][i] for j in range(4)])
        assert max_allocation<1e-14
        changed=np.flatnonzero(np.any(used!=1,axis=1))
        applied=float(times[changed[0]]) if len(changed) else None
        assert applied==r['first_allocator_compensation_time']
        if r['allocation_mode']=='estimated' and applied is not None:
            assert applied>=r['confirmed_detection_time']-1e-12
        p=vector(c,'position');ref=vector(c,'reference_post');e=p-ref;v=vector(c,'velocity');q=vector(c,'quaternion',4);w=vector(c,'omega')
        np.testing.assert_allclose(e,vector(c,'position_error_world'),atol=0,rtol=0)
        for name,a,b in [('full_0_20',0,20),('pre_3_5',3,5),('post_5_20',5,20),('tail_18_20',18,20)]:
            mask=(t>a+1e-9)&(t<=b+1e-9);record=r['windows'][name]
            if t[-1]<b-1e-9:
                assert record is None;continue
            selected=e[mask];mean=selected.mean(0);variance=np.mean((selected-mean)**2,axis=0)
            values=[np.linalg.norm(mean[:2]),mean[2],np.sqrt(np.mean(np.sum(selected[:,:2]**2,axis=1))),
                np.sqrt(np.mean(selected[:,2]**2)),np.sqrt(variance[:2].sum()),np.sqrt(variance[2])]
            saved=[record[k] for k in ('offset_xy_m','mean_error_z_m','rmse_xy_m','rmse_z_m','sway_xy_rms_m','sway_z_rms_m')]
            np.testing.assert_allclose(values,saved,atol=1e-12,rtol=1e-12)
        for bits,key in [('physics_allocator_clipped','clipping'),('physics_esc_boundary','esc_boundary')]:
            flags=vector(c,bits,20).astype(bool).reshape(-1,4);h=np.repeat(dt/5,5)
            np.testing.assert_allclose((flags*h[:,None]).sum(0),r[key]['duration_s_per_channel'],atol=1e-10)
            assert abs(np.sum(h[np.any(flags,axis=1)])-r[key]['duration_any_s'])<1e-10
        if events and r['completed']:
            event=events[0]
            tt=np.r_[5.,t[t>5+1e-9]];ee=np.vstack([event['e_true_after'],e[t>5+1e-9]])
            vv=np.vstack([event['velocity'],v[t>5+1e-9]])
            for name,sl in [('xy',slice(0,2)),('z',slice(2,3)),('3d',slice(0,3))]:
                good=(np.linalg.norm(ee[:,sl],axis=1)<=.005)&(np.linalg.norm(vv[:,sl],axis=1)<=.02)
                val=suffix(tt,good,20.,1.,5.);assert val==r[f'recovery_{name}_s']
            from crazyflie_rl.plotting import quaternion_to_euler_deg
            qq=np.vstack([event['quaternion'],q[t>5+1e-9]]);ww=np.vstack([event['omega'],w[t>5+1e-9]])
            yaw=np.r_[np.radians(quaternion_to_euler_deg(event['quaternion'])[2]),c.yaw_error_rad[t>5+1e-9]]
            tilt=np.degrees(np.arccos(np.clip(1-2*(qq[:,1]**2+qq[:,2]**2),-1,1)))
            joint=(np.linalg.norm(ee,axis=1)<=.005)&(np.linalg.norm(vv,axis=1)<=.02)&(tilt<=5)&(np.abs(yaw)<=np.radians(5))&(np.linalg.norm(ww,axis=1)<=.10)
            assert suffix(tt,joint,20.,1.,5.)==r['recovery_position_attitude_s']
        elif not r['completed']:
            assert all(r[f'recovery_{name}_s'] is None for name in ('xy','z','3d','position_attitude'))
        healthy=~post if fault else np.ones(len(t),bool);fp=healthy&(c.estimator_state=='fault')
        assert int(np.count_nonzero(fp&~np.r_[False,fp[:-1]]))==r['false_positive_episodes']
        assert abs(np.sum(dt[fp])-r['false_positive_seconds'])<1e-10
        if fault:
            hit=np.flatnonzero(post&(c.estimator_state=='fault')&(c.estimated_motor==fault))
            detection=float(t[hit[0]]) if len(hit) else None;assert detection==r['confirmed_detection_time']
            error=est[:,fault-1]-truth[:,fault-1]
            np.testing.assert_allclose([np.abs(error[post]).mean(),np.sqrt(np.mean(error[post]**2))],
                                      [r['efficiency_mae'],r['efficiency_rmse']],atol=1e-12)
            good=np.abs(error[post])<=.03
            delay=suffix(t[post],good,t[-1],.5,5.)
            assert delay==r['efficiency_settling_delay']
            for name,a,b in [('fault_0_05',5,5.5),('fault_0_1',5,6),('tail_18_20',18,20)]:
                mask=(t>a+1e-9)&(t<=b+1e-9);record=r['estimation_windows'][name]
                if t[-1]<b-1e-9:assert record is None
                else:np.testing.assert_allclose([np.abs(error[mask]).mean(),np.sqrt(np.mean(error[mask]**2))],[record['mae'],record['rmse']],atol=1e-12)
        for event in events:
            assert event['physical_and_actuator_state_unchanged'] and event['integral_state_preserved']
            assert event['estimator_updates_preserved']==500 and event['estimator_reset'] is False
        runtime[r['allocation_mode']].extend(c.estimator_runtime_ms)
        report['runs'][key]=dict(samples=len(t),csv_metrics_recomputed=True,events_state_preserved=True)
        report['application_rule'][key]=dict(max_allocation_error=max_allocation,first_applied=applied,causal_hold_rule_passed=True)
    # Estimator + latch + resulting allocation all tested against truth leakage
    # and future observation/command mutation on a real closed-loop trace.
    config=load_config('configs/eval_velocity_ab_user_frd.yaml');model,_=nominal_model(config)
    saved=trace_rows(d/'A_best-motor1_70-estimated.csv')
    estimators=[MotorEfficiencyEstimator(model,settings) for _ in range(3)]
    holders=[ConfirmedEfficiency() for _ in range(3)];max_truth=max_future=0.
    for i,row in enumerate(saved):
        changed=dict(row,truth_efficiency_user=np.zeros(4),scenario='fake_other_motor',fault_time=-999,
            qacc=np.ones(10)*999,motor_omega=np.ones(4)*999,actual_thrust=np.ones(4)*999)
        future=dict(row)
        if i>=600:
            future['velocity']=row['velocity']+[3,2,1]
            future['delivered_esc_native']=.5*row['delivered_esc_native']
        outputs=[estimate_from_row(est,r,.002) for est,r in zip(estimators,[row,changed,future])]
        for holder,out in zip(holders,outputs):holder.consume(out)
        arrays=[]
        for out,h in zip(outputs,holders):
            matrix=efficiency_matrix(B0,native_from_user(h.eta_user))
            command=np.linalg.pinv(matrix)@np.array([.001,-.001,.0001,.43])
            arrays.append(np.r_[out['hypothesis_scores'],out['estimated_efficiency_user'],h.eta_user,command])
        max_truth=max(max_truth,float(np.max(np.abs(arrays[0]-arrays[1]))))
        if i<600:max_future=max(max_future,float(np.max(np.abs(arrays[0]-arrays[2]))))
    assert max_truth==max_future==0
    report['causality']=dict(truth_metadata_mutation_max_error=max_truth,future_mutation_prefix_max_error=max_future,
        prefix_samples=600,includes_estimates_held_efficiencies_and_allocated_commands=True)
    report['common_initial_snapshots_match_all_modes_and_previous']=True
    report['runtime_interpretation']='Wall execution cost measured; no compute-delay model injected into simulation, no real-time guarantee.'
    for mode,values in runtime.items():
        a=np.array(values);report['pooled_runtime_ms'][mode]=dict(samples=len(a),median=float(np.median(a)),
            p95=float(np.percentile(a,95)),max=float(np.max(a)),over_10ms_count=int((a>10).sum()),over_10ms_fraction=float((a>10).mean()))
    original=json.loads((d/'protected_hashes_before.json').read_text())
    def check(item):
        name,expected=item;return name,sha256(Path(name))==expected
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: checks=dict(pool.map(check,original.items()))
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    report['preservation']=dict(files=len(checks),all_sha256_unchanged=True)
    compact=[]
    for r in results:
        post=r['windows']['post_5_20'];tail=r['windows']['tail_18_20']
        row={k:r[k] for k in ('key','label','scenario','allocation_mode','completed','actual_duration_sec','end_reason',
            'recovery_xy_s','recovery_z_s','recovery_3d_s','recovery_position_attitude_s','detection_delay','first_allocator_compensation_time',
            'efficiency_mae','efficiency_rmse','efficiency_settling_delay','false_positive_episodes','false_positive_seconds',
            'misidentification_seconds','uncertain_seconds','insufficient_data_seconds','integral_freeze_seconds')}
        if post:row.update({'post_'+k:post[k] for k in ('max_xy_error_m','max_abs_z_error_m','rmse_xy_m','rmse_z_m','max_tilt_deg','yaw_max_abs_deg')})
        if tail:row.update({'tail_'+k:tail[k] for k in ('offset_xy_m','mean_error_z_m','sway_xy_rms_m','sway_z_rms_m')})
        compact.append(row)
    flat_csv(d/'comparison.csv',compact)
    differences=[]
    for label in ('A_best','B_best'):
        for scenario in dict.fromkeys(r['scenario'] for r in results):
            group={r['allocation_mode']:r for r in results if r['label']==label and r['scenario']==scenario}
            out=dict(label=label,scenario=scenario)
            for mode in ('oracle','estimated'):
                for metric in ('max_xy_error_m','max_abs_z_error_m','rmse_xy_m','rmse_z_m','max_tilt_deg'):
                    ref=group['blind']['windows']['post_5_20'];test=group[mode]['windows']['post_5_20']
                    out[mode+'_minus_blind_'+metric]=test[metric]-ref[metric] if test and ref else None
            differences.append(out)
    flat_csv(d/'paired_differences.csv',differences)
    write_json(d/'independent_verification.json',report)
    source_paths=[ROOT/'configs/eval_velocity_ab_user_frd.yaml',ROOT/'configs/eval_velocity_ab.yaml',
        Path(manifest['completion']),Path(manifest['estimator_settings_source']),
        ROOT/'artifacts/runs/motor-layout-180a0id_/manifest.json',
        ROOT/'crazyflie_rl/estimated_allocation.py',ROOT/'crazyflie_rl/estimated_allocation_eval.py',
        ROOT/'crazyflie_rl/motor_limit_audit.py',ROOT/'compare_estimated_allocation.py',
        ROOT/'verify_estimated_allocation.py',ROOT/'tests/test_estimated_allocation.py']
    write_json(d/'input_and_new_source_hashes.json',{str(p):sha256(p) for p in source_paths})
    write_json(d/'completion.json',dict(status='complete',rollouts=66,fresh_rollouts=66,reused_rollouts=0,
        completed=sum(r['completed'] for r in results),terminated=sum(r['terminated'] for r in results),
        independent_verification_passed=True,protected_files=len(checks),no_training_or_tuning=True))
    print(json.dumps(dict(verified=len(results),preservation=report['preservation'],causality=report['causality'],runtime=report['pooled_runtime_ms']),indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    verify(p.parse_args().run_dir)

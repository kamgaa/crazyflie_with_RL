"""Independent CSV arithmetic, causal replay checks and saved-video alignment."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess
import concurrent.futures

import numpy as np

from crazyflie_rl.fault_estimator import ObservedState,DeliveredCommands,MotorEfficiencyEstimator,EstimatorSettings
from crazyflie_rl.fault_estimation_eval import nominal_model,trace_rows,estimate_from_row
from crazyflie_rl.config import load_config
from crazyflie_rl.integral_eval import read_columns
from crazyflie_rl.motor_layout import exposed_motor_index
from crazyflie_rl.dr_policy import sha256


def verify(directory):
    d=Path(directory);manifest=json.loads((d/'manifest.json').read_text());summaries=json.loads((d/'summary.json').read_text())
    assert manifest['status']=='complete' and len(summaries)==22
    settings=EstimatorSettings(**json.loads((d/'estimator_settings.json').read_text()))
    assert sha256(d/'estimator_settings.json')==manifest['estimator_settings_sha256']
    model,_=nominal_model(load_config(manifest['config']['source_path']))
    report=dict(runs={},truth_independence={},future_independence={},input_fields={
        'state':[f.name for f in fields(ObservedState)],'commands':[f.name for f in fields(DeliveredCommands)]})
    runtimes=[];prefixes={};report['common_prefault_prefix']={}
    for result in summaries:
        key=result['key'];c=read_columns(d/(key+'.csv'));t=c.time_post;dt=c.time_post-c.time
        runtimes.extend(c.estimator_runtime_ms.tolist())
        prefix_mask=t<=5+1e-9
        keys=[k for k in c if k.startswith(('position_','velocity_','omega_','quaternion_','action_',
              'wrench_command_','motor_thrust_command_','delivered_esc_native_','xi_t_','xi_next_'))]
        prefix={k:c[k][prefix_mask] for k in keys}
        if result['label'] not in prefixes:prefixes[result['label']]=prefix
        else:
            for k,v in prefix.items():np.testing.assert_array_equal(v,prefixes[result['label']][k])
        report['common_prefault_prefix'][key]=dict(samples=int(prefix_mask.sum()),max_abs_error=0.)
        np.testing.assert_allclose(dt,.01,atol=1e-12)
        np.testing.assert_allclose(c.estimate_time,t,atol=0)
        np.testing.assert_allclose(c.gyro_observation_time,np.maximum(0,t-.002),atol=1e-12)
        np.testing.assert_array_equal(c.estimator_update_count,np.arange(1,len(t)+1))
        assert (not result['terminated']) or all(v is None for k,v in result['estimation_windows'].items() if k=='tail_18_20')
        motor=result['fault_user_motor'];post=c.time>=5-1e-9
        est=np.column_stack([c[f'estimated_efficiency_user_{j}'] for j in range(4)])
        truth=np.column_stack([c[f'truth_efficiency_user_{j}'] for j in range(4)])
        expected=np.ones_like(truth)
        if motor:expected[post,motor-1]=result['fault_efficiency']
        np.testing.assert_array_equal(truth,expected)
        unhealthy=(c.estimator_state=='fault')
        healthy=~post if motor else np.ones(len(t),bool)
        fp=healthy&unhealthy
        episodes=int(np.count_nonzero(fp&~np.r_[False,fp[:-1]]))
        duration=float(np.sum(dt[fp]));assert episodes==result['false_positive_episodes']
        assert abs(duration-result['false_positive_seconds'])<1e-10
        independent=dict(false_positive_episodes=episodes,false_positive_seconds=duration,
            gyro_shift_s=.002,samples=len(t),event_state_preservation=True)
        normal=[j for j in range(4) if j+1!=motor]
        independent['normal_motor_ids']=[j+1 for j in normal]
        independent['normal_motor_bias']=np.mean(est[(post if motor else healthy)][:,normal]-1,axis=0).tolist()
        result['normal_motor_ids']=independent['normal_motor_ids']
        result['normal_motor_bias']=independent['normal_motor_bias']
        result['first_correct_candidate_time']=None
        if motor:
            candidate_indices=np.flatnonzero(post&(c.candidate_motor==motor))
            result['first_correct_candidate_time']=float(t[candidate_indices[0]]) if len(candidate_indices) else None
            independent['first_correct_candidate_time']=result['first_correct_candidate_time']
            correct=post&unhealthy&(c.estimated_motor==motor)
            indices=np.flatnonzero(correct);detection=float(t[indices[0]]) if len(indices) else None
            assert detection==result['confirmed_detection_time']
            err=est[:,motor-1]-truth[:,motor-1]
            mae=float(np.mean(np.abs(err[post])));rmse=float(np.sqrt(np.mean(err[post]**2)))
            np.testing.assert_allclose([mae,rmse],[result['efficiency_mae'],result['efficiency_rmse']],atol=1e-12)
            within=(np.abs(err[post])<=.03);tt=t[post];settled=None
            if len(tt) and within[-1]:
                bad=np.flatnonzero(~within);i=int(bad[-1]+1) if len(bad) else 0
                if tt[-1]-tt[i]>=.5-1e-9:settled=float(tt[i])
            assert settled==result['efficiency_settling_time']
            independent.update(confirmed_detection_time=detection,mae=mae,rmse=rmse,settled_time=settled)
            for name,a,b in [('fault_0_05',5,5.5),('fault_0_1',5,6),('tail_18_20',18,20)]:
                mask=(t>a+1e-9)&(t<=b+1e-9);record=result['estimation_windows'][name]
                if t[-1]<b-1e-9:assert record is None
                else:np.testing.assert_allclose([np.mean(np.abs(err[mask])),np.sqrt(np.mean(err[mask]**2))],
                                               [record['mae'],record['rmse']],atol=1e-12)
        for event in manifest['runs'][key]['events']:
            assert event['estimator_reset'] is False and event['estimator_updates_preserved']==500
            assert event['xi_before']==event['xi_after'] and event['physical_and_actuator_state_unchanged']
            if event['event']=='fault':assert event['native_motor_index']==exposed_motor_index(motor,'user_frd')
        report['runs'][key]=independent
    # Alter every forbidden diagnostic field, including future telemetry, while
    # checking earlier causal outputs on actual saved A/PPO observations.
    saved=trace_rows(d/'A_best-motor1_70.csv')
    estimators=[MotorEfficiencyEstimator(model,settings) for _ in range(3)]
    max_truth=0.;max_future=0.
    for i,row in enumerate(saved):
        changed=dict(row,truth_efficiency_user=np.zeros(4),motor_thrust_actual=np.ones(4)*100,
            motor_omega=np.ones(4)*1e6,scenario='wrong_motor4',fault_time=-30,qacc=np.ones(10)*100)
        future=dict(row)
        if i>=600:
            future['velocity']=row['velocity']+[3,2,1]
            future['delivered_esc_native']=.5*row['delivered_esc_native']
        outputs=[estimate_from_row(e,r,.002) for e,r in zip(estimators,[row,changed,future])]
        for k in ['hypothesis_scores','estimated_efficiency_user','hypothesis_alpha_user']:
            max_truth=max(max_truth,float(np.max(np.abs(outputs[0][k]-outputs[1][k]))))
            if i<600:max_future=max(max_future,float(np.max(np.abs(outputs[0][k]-outputs[2][k]))))
        assert outputs[0]['estimator_state']==outputs[1]['estimator_state']
        if i<600:assert outputs[0]['estimator_state']==outputs[2]['estimator_state']
    assert max_truth==max_future==0
    report['truth_independence']=dict(max_abs_error=max_truth,samples=len(saved))
    report['future_independence']=dict(max_abs_error=max_future,unchanged_prefix_samples=600)
    if (d/'video_metadata.json').exists():
        vm=json.loads((d/'video_metadata.json').read_text());vf=read_columns(d/'video_frames.csv')
        replay=np.load(d/'video_replay.npz');trace=read_columns(d/'A_best-motor1_70.csv')
        assert vm['source_trace_sha256']==sha256(d/'A_best-motor1_70.csv')
        assert vm['source_replay_sha256']==sha256(d/'video_replay.npz')
        for j in range(len(vf.frame)):
            i=int(vf.trace_row[j]);ri=int(vf.replay_row[j])
            assert vf.state_time_s[j]==trace.time_post[i]==vf.estimate_time_s[j]==replay['time'][ri]
            assert vf.estimated_eta1[j]==trace.estimated_efficiency_user_0[i]
            assert vf.estimator_state[j]==trace.estimator_state[i] and vf.estimated_motor[j]==trace.estimated_motor[i]
            assert vf.truth_eta1[j]==(.7 if vf.state_time_s[j]>=5-1e-9 else 1.)
            assert vf.qpos_sha256[j]==hashlib.sha256(replay['qpos'][ri].tobytes()).hexdigest()
        report['video_alignment']=dict(frames=len(vf.frame),passed=True,max_age_s=vm['max_state_age_s'],decode_passed=vm['full_decode_success'])
    original=json.loads((d/'protected_hashes_before.json').read_text())
    def check(item):
        path,expected=item;h=hashlib.sha256()
        with open(path,'rb') as file:
            while chunk:=file.read(4*1024**2):h.update(chunk)
        return path,h.hexdigest()==expected
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:checks=dict(pool.map(check,original.items()))
    assert all(checks.values()),[p for p,ok in checks.items() if not ok]
    report['protected_files']=dict(count=len(checks),all_sha256_unchanged=True)
    report['pooled_runtime_ms']=dict(samples=len(runtimes),median=float(np.median(runtimes)),p95=float(np.percentile(runtimes,95)),max=float(np.max(runtimes)))
    groups=[]
    for label in ('A_best','B_best'):
        for eta in (.8,.7):
            rows=[r for r in summaries if r['label']==label and r['fault_efficiency']==eta]
            def span(key):return [min(r[key] for r in rows),max(r[key] for r in rows)]
            groups.append(dict(policy=label,true_efficiency=eta,rollouts=len(rows),completed=sum(r['completed'] for r in rows),
                identified=sum(r['confirmed_detection_time'] is not None for r in rows),detection_delay_s=span('detection_delay'),
                mae=span('efficiency_mae'),rmse=span('efficiency_rmse'),within_003_suffix_delay_s=span('efficiency_settling_delay'),
                tail_mae_max=max(r['estimation_windows']['tail_18_20']['mae'] for r in rows)))
    headline=dict(groups=groups,main_rollouts=22,development_rollouts=1,disabled_comparison_rollouts=2,
        all_completed=sum(r['completed'] for r in summaries),false_positive_episodes=sum(r['false_positive_episodes'] for r in summaries),
        false_positive_seconds=sum(r['false_positive_seconds'] for r in summaries),misidentification_seconds=sum(r['misidentification_seconds'] for r in summaries),
        pooled_runtime_ms=report['pooled_runtime_ms'],initial_insufficient_s=[min(r['insufficient_data_seconds'] for r in summaries),max(r['insufficient_data_seconds'] for r in summaries)],
        no_real_sensor_noise_validation=True,shadow_only=True)
    (d/'headline_results.json').write_text(json.dumps(headline,indent=2))
    from crazyflie_rl.oracle_eval import flat_csv
    (d/'summary.json').write_text(json.dumps(summaries,indent=2))
    flat_csv(d/'summary.csv',summaries)
    flat_csv(d/'estimation_summary.csv',[{k:r[k] for k in ('key','completed','actual_duration_sec','end_reason','fault_user_motor',
        'fault_efficiency','first_correct_raw_candidate_time','first_correct_candidate_time','confirmed_detection_time','detection_delay','efficiency_mae','efficiency_rmse',
        'efficiency_settling_delay','false_positive_episodes','false_positive_seconds','misidentification_seconds','uncertain_seconds',
        'insufficient_data_seconds','allocator_clipping_seconds','esc_boundary_seconds','action_boundary_seconds','integral_freeze_seconds')} for r in summaries])
    (d/'independent_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'runs':len(report['runs']),'protected':len(checks),'truth_error':max_truth,'future_error':max_future,
                      'video':report.get('video_alignment')},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    verify(p.parse_args().run_dir)

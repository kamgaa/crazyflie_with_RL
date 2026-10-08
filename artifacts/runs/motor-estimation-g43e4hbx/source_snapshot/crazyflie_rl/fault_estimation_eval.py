"""No-learning shadow evaluation using the existing rollout and integral path."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, replace
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import time

import mujoco
import numpy as np

from .artifacts import _git_metadata
from .config import load_config
from .dr_transfer import ROOT, EvaluationAdapter, Case, Thresholds, run_case, summarize, write_json as _write_json, write_rollout
from .dr_policy import sha256
from .fault_estimator import ObservedState, DeliveredCommands, NominalModel, EstimatorSettings, MotorEfficiencyEstimator
from .integral_controller import IntegralController
from .integral_eval import parameter_digest, verify_integral_rows, read_columns
from .integral_validation import Scenario, Event, ScenarioObserver, GAINS, make_case
from .interactive_eval import InteractiveEnv, evaluation_config
from .motor_layout import layout_metadata, user_from_native, exposed_motor_index
from .payload_motor_eval import DEFAULT_RECORD, RecordedFaultEnv, select_models, segment_statistics
from .oracle_eval import flat_csv
from .oracle_recovery import clean


def write_json(path,value):
    _write_json(path,clean(value))


@dataclass(frozen=True)
class ShadowScenario(Scenario):
    efficiency: float = 1.
    move_axis: int | None = None
    development: bool = False

    def events(self):
        if self.development:
            return (Event(3.,'target_step',target=(.05,0.,1.)),Event(6.,'target_step',target=(.05,.05,1.)))
        if self.motor_number is not None:return (Event(5.,'fault',self.motor_number,self.efficiency),)
        if self.move_axis is not None:
            target=np.array([0.,0.,1.]);target[self.move_axis]=.05
            return (Event(5.,'target_step',target=tuple(target)),)
        return ()


SCENARIOS=(ShadowScenario('hover'),ShadowScenario('step_x',move_axis=0),ShadowScenario('step_y',move_axis=1))+tuple(
    ShadowScenario(f'motor{motor}_{percent:02}',motor_number=motor,efficiency=percent/100)
    for motor in range(1,5) for percent in (80,70))


def nominal_model(config):
    """Compile a separate nominal XML once; copy known constants, no flight data.

    Static geometry audit is reused. No force probe, qacc or actual flight motor
    state is used. Free hinge axial inertia is eliminated as in the audit.
    """
    from audit_coordinate_contracts import engine_geometry
    env=InteractiveEnv(config=config)
    try:
        adapter=EvaluationAdapter(env);adapter.reset_to_case_initial_state(Case('hover',1.,(0,0,1)),42)
        assert env._com_mw==0 and np.all(env.dist_torque_body==0)
        g=engine_geometry(env);I=g['locked_composite_inertia_native_kg_m2'].copy()
        for b in g['descendants']:
            if b['id']==env.drone_bid:continue
            jid=int(env.model.body_jntadr[b['id']]);axis=env.data.xaxis[jid]
            tensor=b['inertia_about_own_com_native_kg_m2']
            I-=np.outer(tensor@axis,axis)
        a=config.actuator;r=a.reaction_torque;v=config.vehicle
        if r.model!='legacy_ratio' or r.include_rotor_acceleration_torque:
            raise ValueError('this linear force hypothesis implementation requires the fixed legacy_ratio motor law')
        kwargs=dict(dt=1/v.physics_hz,motor_direction=v.motor_direction,thrust_min=v.thrust_min,
            thrust_max=v.thrust_max,time_constant_s=a.time_constant_s,steady_state_gain_rad_s=a.steady_state_gain_rad_s,
            thrust_polynomial_coefficients=a.thrust_polynomial_coefficients,
            omega_reference_rad_s=a.thrust_polynomial_omega_reference_rad_s,
            positive_branch_min_ratio=a.thrust_polynomial_positive_branch_min_ratio,max_ratio=a.thrust_polynomial_max_ratio,
            reaction_torque_model=r.model,legacy_ratio_m=r.legacy_ratio_m)
        model=NominalModel(g['total_mass_kg'],g['whole_com_native_m'],I,g['force'],g['torque_O'],
                           np.array([0.,0.,-v.gravity]),kwargs,v.mass)
        meta=dict(constants=asdict(model),layout=layout_metadata(env),geometry=g,
            xml=str(config.paths.mujoco_xml),xml_sha256=sha256(config.paths.mujoco_xml),
            approximations=['rigid composite effective inertia; free hinge axial inertia eliminated at nominal pose',
                'higher-order free-hinge gyroscopic coupling omitted (four 1e-9 kg m^2 inertias)',
                'completed-interval quaternion interpolation for force orientation',
                'midpoint gyro cross term; origin-to-COM velocity uses cached gyro with <=2 ms offset'],
            sensor_timing='qpos and origin qvel at t; unchanged cached gyro at max(0,t-physics_dt); Euler integrator',
            physical_rotor_arm_m=.03536,allocator_arm_m=v.arm_length,
            initial_motor_replica='own actuator reset from public nominal hover mass*g/4; never copy flight motor state')
        return model,meta
    finally:env.close()


def allowed_state(row, before, physics_dt):
    suffix='_before' if before else ''
    t=row['time'] if before else row['time_post']
    return ObservedState(t,row['position'+suffix],row['quaternion'+suffix],
        row['velocity'+suffix],row['omega'+suffix],max(0.,t-physics_dt))


def estimate_from_row(estimator,row,physics_dt):
    # Explicit allowlist. Evaluation labels, truth and all plant diagnostics are
    # deliberately absent from these three sealed input structures.
    return estimator.update(allowed_state(row,True,physics_dt),
        DeliveredCommands(row['time'],physics_dt,np.array(row['delivered_esc_native']).reshape((-1,4))),
        allowed_state(row,False,physics_dt))


class ShadowObserver(ScenarioObserver):
    def __init__(self,scenario,controller,model,settings,enabled=True,record_replay=False):
        super().__init__(scenario,controller)
        self.nominal=model;self.settings=settings;self.enabled=enabled;self.record_replay=record_replay
        self.estimator=None;self.replay=[]

    def on_reset(self,adapter):
        super().on_reset(adapter)
        self.estimator=MotorEfficiencyEstimator(self.nominal,self.settings) if self.enabled else None
        self.original_inverse=adapter.env.B_pinv.copy()
        self.metadata.update(layout=layout_metadata(adapter.env),estimator_mode='shadow' if self.enabled else 'disabled',
                             allocator='fault-unaware original B0 inverse, no oracle',ideal_simulator_observations=True)
        if self.record_replay:self.replay.append((0.,adapter.env.data.qpos.copy()))

    def before_step(self,adapter,step,t):
        count=self.estimator.update_count if self.enabled else None
        super().before_step(adapter,step,t)
        if self.enabled:assert self.estimator.update_count==count
        if self.events and self.events[-1]['control_step']==step:
            self.events[-1]['estimator_updates_preserved']=count
            self.events[-1]['estimator_reset']=False

    def after_step(self,env,row):
        super().after_step(env,row)
        np.testing.assert_array_equal(env.B_pinv,self.original_inverse)
        physics=self.physics_rows[self.physics_start:]
        row['delivered_esc_native']=np.array([r['motor_command'] for r in physics])
        row['allocator_clipping_seconds_native']=env.dt_phys*np.sum([r['allocator_clipped'] for r in physics],axis=0)
        row['allocator_clipping_union_seconds']=env.dt_phys*sum(bool(np.any(r['allocator_clipped'])) for r in physics)
        row['esc_boundary_union_seconds']=env.dt_phys*sum(bool(np.any(r['esc_lower']|r['esc_upper'])) for r in physics)
        row['truth_efficiency_user']=user_from_native(env.motor_effectiveness)
        if self.enabled:
            start=time.perf_counter_ns();result=estimate_from_row(self.estimator,row,env.dt_phys)
            row.update(result,estimator_runtime_ms=(time.perf_counter_ns()-start)/1e6)
        if self.record_replay:self.replay.append((row['time_post'],env.data.qpos.copy()))
        # Retain only one control interval's diagnostic physics rows in memory.
        # No dense physics CSVs and no use of true force in the estimator.
        self.physics_rows.clear()


TRACE_KEYS=('time','time_post','position_before','quaternion_before','velocity_before','omega_before',
    'position','quaternion','velocity','omega','reference','reference_post','position_error_world','yaw_error_rad',
    'action','wrench_command','motor_thrust_command','motor_thrust_actual','motor_reaction_actual',
    'delivered_esc_native','truth_efficiency_user','xi_t','xi_next','p_cmd','p_target','integral_frozen',
    'integral_stop_reason','integral_xy_at_limit','integral_z_at_limit','policy_action_at_bound',
    'allocator_clipping_seconds_native','allocator_clipping_union_seconds','esc_boundary_union_seconds',
    'terminated','truncated','estimate_time','gyro_observation_time','estimator_state','estimated_motor',
    'candidate_motor','raw_best_motor','estimated_efficiency_user','hypothesis_alpha_user','hypothesis_scores',
    'hypothesis_information','relative_improvement','score_margin','persistence_samples','window_samples',
    'nominal_delta_residual','observed_delta','normal_prediction_delta','nominal_force_impulse_native',
    'estimator_update_count','estimator_runtime_ms')


def execute(config,policy,scenario,model,settings,*,enabled=True,record_replay=False,horizon=20.):
    config=replace(config,environment=replace(config.environment,episode_sec=horizon))
    controller=IntegralController(policy,GAINS[1])
    observer=ShadowObserver(scenario,controller,model,settings,enabled,record_replay)
    case=make_case(scenario,horizon)
    rows,initial,error,reasons=run_case(config,case,controller,42,env_factory=RecordedFaultEnv,observer=observer,
                                      observation_transform=controller.prepare_observation)
    if error:raise RuntimeError(error)
    verify_integral_rows(rows,GAINS[1])
    return rows,initial,observer,case,reasons


def blocks(mask):
    mask=np.asarray(mask,bool)
    return int(np.sum(mask & ~np.r_[False,mask[:-1]])) if len(mask) else 0


def suffix_time(times,good,dwell=.5):
    if not len(times) or not good[-1]:return None
    bad=np.flatnonzero(~np.asarray(good));i=int(bad[-1]+1) if len(bad) else 0
    return float(times[i]) if times[-1]-times[i]>=dwell-1e-9 else None


def summarize_estimation(rows,scenario,case,reasons):
    result=summarize(rows,case,Thresholds(),None,reasons)
    dt=rows[0]['time_post']-rows[0]['time'];t=np.array([r['time_post'] for r in rows])
    eta=np.array([r['estimated_efficiency_user'] for r in rows]);truth=np.array([r['truth_efficiency_user'] for r in rows])
    motor=np.array([r['estimated_motor'] for r in rows]);raw=np.array([r['raw_best_motor'] for r in rows])
    states=np.array([r['estimator_state'] for r in rows]);runtime=np.array([r['estimator_runtime_ms'] for r in rows])
    post=np.array([r['time']>=5-1e-9 for r in rows]);fault=scenario.motor_number
    def first(mask):return float(t[np.flatnonzero(mask)[0]]) if np.any(mask) else None
    result.update(observed_control_metrics=segment_statistics(rows),scenario=scenario.name,fault_user_motor=fault,
        fault_efficiency=scenario.efficiency if fault else None,fault_applied=bool(fault and np.any(post)),
        estimator_runtime_ms=dict(median=float(np.median(runtime)),p95=float(np.percentile(runtime,95)),max=float(runtime.max())),
        uncertain_seconds=float(np.sum(states=='uncertain')*dt),insufficient_data_seconds=float(np.sum(states=='insufficient_data')*dt),
        final_estimator_state=states[-1],final_estimated_motor=int(motor[-1]),final_estimated_efficiency_user=eta[-1].tolist(),
        allocator_clipping_seconds=float(sum(r['allocator_clipping_union_seconds'] for r in rows)),
        allocator_clipping_seconds_native=np.sum([r['allocator_clipping_seconds_native'] for r in rows],axis=0).tolist(),
        esc_boundary_seconds=float(sum(r['esc_boundary_union_seconds'] for r in rows)),
        action_boundary_seconds=float(sum(np.any(r['policy_action_at_bound']) for r in rows)*dt),
        integral_freeze_seconds=float(sum(r['integral_frozen'] for r in rows)*dt),
        max_abs_yaw_error_deg=float(np.degrees(np.abs([r['yaw_error_rad'] for r in rows])).max()),
        estimation_windows={},efficiency_settling_time=None,efficiency_settling_delay=None,
        first_correct_raw_candidate_time=None,first_correct_candidate_time=None,confirmed_detection_time=None,detection_delay=None,
        misidentification_seconds=0.,efficiency_mae=None,efficiency_rmse=None,normal_motor_bias=None)
    healthy=~post if fault else np.ones(len(rows),bool)
    fp=healthy & (states=='fault')
    result['false_positive_episodes']=blocks(fp);result['false_positive_seconds']=float(fp.sum()*dt)
    result['healthy_observed_seconds']=float(healthy.sum()*dt)
    if fault:
        j=fault-1;error=eta[:,j]-truth[:,j]
        result['first_correct_raw_candidate_time']=first(post & (raw==fault))
        result['first_correct_candidate_time']=first(post & (np.array([r['candidate_motor'] for r in rows])==fault))
        detected=first(post & (states=='fault') & (motor==fault))
        result['confirmed_detection_time']=detected;result['detection_delay']=None if detected is None else detected-5
        result['misidentification_seconds']=float(np.sum(post & (states=='fault') & (motor!=fault))*dt)
        if np.any(post):
            result['efficiency_mae']=float(np.mean(np.abs(error[post])))
            result['efficiency_rmse']=float(np.sqrt(np.mean(error[post]**2)))
            result['normal_motor_bias']=np.mean((eta-truth)[post][:,np.arange(4)!=j],axis=0).tolist()
            settled=suffix_time(t[post],np.abs(error[post])<=.03)
            result['efficiency_settling_time']=settled
            result['efficiency_settling_delay']=None if settled is None else settled-5
        for name,a,b in [('fault_0_05',5,5.5),('fault_0_1',5,6),('tail_18_20',18,20)]:
            mask=(t>a+1e-9)&(t<=b+1e-9)
            result['estimation_windows'][name]=(dict(mae=float(np.mean(np.abs(error[mask]))),
                rmse=float(np.sqrt(np.mean(error[mask]**2))),mean=float(np.mean(eta[mask,j])),samples=int(mask.sum()))
                if t[-1]>=b-1e-9 and mask.any() else None)
    result['tail_control_metrics']=segment_statistics([r for r in rows if r['time_post']>18+1e-9]) if result['completed'] else None
    result['metric_scope']='full 20 seconds' if result['completed'] else 'partial observed interval only; unobserved fixed tails null'
    result['null_reasons']=dict(detection='no confirmed correct fault' if fault and result['detection_delay'] is None else 'not_applicable' if not fault else None,
        tail=None if result['completed'] else 'physical termination before 20 seconds',
        efficiency_settling='not_applicable' if not fault else 'no observed 0.5-second suffix within 0.03' if result['efficiency_settling_time'] is None else None)
    return result


def trace_rows(path):
    c=read_columns(path);rows=[]
    vectors={'position_before':3,'quaternion_before':4,'velocity_before':3,'omega_before':3,
        'position':3,'quaternion':4,'velocity':3,'omega':3,'delivered_esc_native':20,
        'estimated_efficiency_user':4,'hypothesis_alpha_user':4,'hypothesis_scores':5,
        'truth_efficiency_user':4,'nominal_delta_residual':6}
    for i in range(len(c['time'])):
        r={k:c[k][i] for k in ['time','time_post','estimator_state','estimated_motor','candidate_motor','raw_best_motor']}
        r.update({k:np.array([c[f'{k}_{j}'][i] for j in range(n)]) for k,n in vectors.items()})
        rows.append(r)
    return rows


def offline_verify(path,model,settings):
    rows=trace_rows(path);est=MotorEfficiencyEstimator(model,settings);maximum=0.
    for r in rows:
        out=estimate_from_row(est,r,model.actuator_kwargs['dt'])
        for key in ('estimated_efficiency_user','hypothesis_alpha_user','hypothesis_scores'):
            maximum=max(maximum,float(np.max(np.abs(out[key]-r[key]))))
            np.testing.assert_allclose(out[key],r[key],atol=1e-10,rtol=1e-10)
        assert out['estimator_state']==r['estimator_state'] and out['estimated_motor']==r['estimated_motor']
    return dict(samples=len(rows),max_abs_error=maximum)


def plots(directory,results):
    from .plotting import _pyplot
    plt=_pyplot()
    for result in results:
        key=result['key'];c=read_columns(directory/(key+'.csv'));t=c.time_post
        fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
        for i in range(4):
            axs[0].plot(t,c[f'estimated_efficiency_user_{i}'],label=f'Estimated M{i+1}',color=f'C{i}')
            axs[0].plot(t,c[f'truth_efficiency_user_{i}'],ls='--',color=f'C{i}',alpha=.6)
        for i in range(5):axs[1].plot(t,np.maximum(c[f'hypothesis_scores_{i}'],1e-12),label=f'H{i}')
        axs[1].set_yscale('log');axs[2].step(t,c.estimated_motor,where='post',label='Confirmed fault motor')
        axs[2].step(t,c.raw_best_motor,where='post',alpha=.35,label='Raw best fault hypothesis')
        for ax in axs:ax.grid(alpha=.25);ax.legend(ncol=3,fontsize=8)
        if result['scenario']!='hover':
            for ax in axs:ax.axvline(5,color='k',ls=':',label='event')
        if result['confirmed_detection_time'] is not None:
            for ax in axs:ax.axvline(result['confirmed_detection_time'],color='red',ls='--')
        axs[0].set_ylabel('Efficiency (user order)');axs[1].set_ylabel('Normalized residual score');axs[2].set_ylabel('Motor ID; 0 = no confirmed fault')
        axs[2].set_xlabel('Observation available time (s)');fig.suptitle(key+' | shadow only')
        fig.tight_layout();fig.savefig(directory/(key+'.png'),dpi=130);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/eval_velocity_ab_user_frd.yaml')
    p.add_argument('--completion',type=Path,default=DEFAULT_RECORD)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--stage',choices=['develop','evaluate'],required=True)
    p.add_argument('--estimator-settings',type=Path)
    args=p.parse_args(argv)
    config=evaluation_config(load_config(args.config))
    if config.vehicle.reaction_torque_layout!='user_frd' or config.environment.payload.mass!=0:
        raise ValueError('requires user_frd and zero payload')
    directory=args.output_dir or Path(tempfile.mkdtemp(prefix='motor-estimation-',dir=ROOT/'artifacts/runs'))
    directory.mkdir(parents=True,exist_ok=True)
    model,meta=nominal_model(config);write_json(directory/'nominal_model.json',meta) if not (directory/'nominal_model.json').exists() else None
    policies={x.provenance['label']:x for x in select_models(args.completion,config)}
    reference=json.loads((ROOT/'artifacts/runs/motor-layout-180a0id_/manifest.json').read_text())
    for label,policy in policies.items():
        old=next(x for x in reference['models']['user_frd'] if x['label']==label)
        assert old['path']==policy.provenance['path'] and old['sha256']==policy.provenance['sha256']
    from stable_baselines3 import PPO
    import torch
    def forbidden(*a,**k):raise RuntimeError('training and optimizer updates prohibited')
    PPO.learn=PPO.train=forbidden;torch.optim.Adam.step=forbidden
    digest={k:parameter_digest(v) for k,v in policies.items()}
    if args.stage=='develop':
        if (directory/'estimator_settings.json').exists():raise FileExistsError('development settings already frozen')
        setting=EstimatorSettings();scenario=ShadowScenario('development_no_fault',development=True)
        rows,initial,observer,case,reasons=execute(config,policies['A_best'],scenario,model,setting,horizon=10.)
        write_rollout(directory/'development_no_fault.csv',[{k:r[k] for k in TRACE_KEYS} for r in rows])
        residual=np.array([r['nominal_delta_residual'] for r in rows])
        scale=np.maximum(np.quantile(np.abs(residual),.99,axis=0),setting.residual_scales)
        setting=replace(setting,residual_scales=tuple(scale))
        est=MotorEfficiencyEstimator(model,setting)
        dev_out=[estimate_from_row(est,r,model.actuator_kwargs['dt']) for r in rows]
        threshold=max(9.,4*max(r['hypothesis_scores'][0] for r in dev_out))
        setting=replace(setting,h0_score_threshold=threshold)
        write_json(directory/'estimator_settings.json',asdict(setting))
        write_json(directory/'development.json',dict(condition=asdict(scenario),policy='A_best',horizon=10,
            completed=len(rows)==1000,reasons=reasons,calibration='one no-fault rollout only; scales=max(P99 abs raw residual, fixed floors); H0 threshold=max(9,4*maximum development H0 score)',
            residual_rms=np.sqrt(np.mean(residual**2,axis=0)),residual_max_abs=np.max(np.abs(residual),axis=0),
            settings=asdict(setting),frozen_before_main=True,created_unix=time.time(),initial=initial,
            model_manifest=meta,models=[x.provenance for x in policies.values()]))
        print('frozen settings:',directory/'estimator_settings.json',flush=True);return 0
    setting_path=args.estimator_settings or directory/'estimator_settings.json'
    settings=EstimatorSettings(**json.loads(setting_path.read_text()))
    if (directory/'manifest.json').exists():raise FileExistsError('evaluation output exists; choose a new directory')
    manifest=dict(status='running',models=[v.provenance for v in policies.values()],config=config.resolved_dict(),
        estimator=asdict(settings),estimator_settings_sha256=sha256(setting_path),window_duration_s=settings.window_samples/config.environment.policy_hz,
        scenarios=[asdict(s) for s in SCENARIOS],requested_rollouts=22,seed=42,integral=asdict(GAINS[1]),
        mode='shadow; original fault-unaware allocator; no oracle',nominal_model=meta,
        input_contract='only observed p/quaternion/origin-world-v/native-gyro with separate timestamps; delivered ESC command sequence; known nominal constants',
        observations='ideal simulator state; cached gyro timing preserved; no sensor noise or hardware validation',
        timing='row [time,time_post] command interval; estimate first available at time_post; event at t=5 before command; post t=5 belongs to previous interval',
        efficiency_error='reported efficiency vs ground truth; raw hypothesis estimates separately logged; scores are not probabilities',
        source_sha256={str(x.relative_to(ROOT)):sha256(x) for x in (ROOT/'crazyflie_rl').glob('*.py')},
        git=_git_metadata(ROOT),runs={})
    results=[];verification={'offline_replay':{},'shadow_equivalence':{},'input_isolation':{}}
    def save():
        write_json(directory/'manifest.json',manifest);write_json(directory/'summary.json',results)
        flat_csv(directory/'summary.csv',results);write_json(directory/'verification.json',verification)
    save()
    for label,policy in policies.items():
        for scenario in SCENARIOS:
            key=label+'-'+scenario.name;video=key=='A_best-motor1_70'
            print('RUN',key,flush=True)
            rows,initial,observer,case,reasons=execute(config,policy,scenario,model,settings,record_replay=video)
            result=summarize_estimation(rows,scenario,case,reasons);result.update(key=key,label=label)
            results.append(result);manifest['runs'][key]=dict(initial=initial,metadata=observer.metadata,events=observer.events)
            write_rollout(directory/(key+'.csv'),[{k:r[k] for k in TRACE_KEYS} for r in rows])
            write_json(directory/(key+'-events.json'),observer.events);flat_csv(directory/(key+'-events.csv'),observer.events)
            verification['offline_replay'][key]=offline_verify(directory/(key+'.csv'),model,settings)
            if video:
                np.savez_compressed(directory/'video_replay.npz',time=np.array([t for t,q in observer.replay]),qpos=np.array([q for t,q in observer.replay]))
                write_json(directory/'video_run.json',dict(key=key,summary=result,replay='video_replay.npz',trace=key+'.csv'))
            if key in ('A_best-motor1_70','B_best-step_y'):
                baseline,base_initial,_,_,base_reasons=execute(config,policy,scenario,model,settings,enabled=False)
                assert len(baseline)==len(rows) and base_initial==initial and base_reasons==reasons
                keys=('position','quaternion','velocity','omega','action','wrench_command','motor_thrust_actual',
                      'motor_thrust_command','motor_omega','delivered_esc_native','xi_t','xi_next')
                for r,b in zip(rows,baseline):
                    for k in keys:np.testing.assert_array_equal(r[k],b[k])
                verification['shadow_equivalence'][key]=dict(samples=len(rows),fields=keys,max_abs_error=0.,extra_disabled_rollout=True)
            assert parameter_digest(policy)==digest[label]
            save();print(key,result['actual_duration_sec'],result['final_estimator_state'],result['detection_delay'],flush=True)
    manifest['status']='complete';manifest['completed_rollouts']=len(results)
    verification['policy_parameters_unchanged']=True;verification['no_optimizer_updates']=True
    verification['allocator_unchanged_every_step']=True
    write_json(directory/'false_positives.json',[{k:r[k] for k in ('key','false_positive_episodes','false_positive_seconds','healthy_observed_seconds')} for r in results])
    command='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_motor_estimation.py'
    script='#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'
    script+='run_dir=$(mktemp -d artifacts/runs/motor-estimation-XXXXXX)\n'
    script+=command+' --stage develop --output-dir "$run_dir"\n'+command+' --stage evaluate --output-dir "$run_dir"\n'
    script+='MUJOCO_GL=egl python render_motor_estimation.py --run-dir "$run_dir"\n'
    script+='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_motor_estimation.py --run-dir "$run_dir"\n'
    (directory/'rerun.sh').write_text(script)
    plots(directory,results);save();print('RESULTS',directory,flush=True);return 0

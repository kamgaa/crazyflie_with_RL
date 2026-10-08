"""Fixed 66-run blind/oracle/estimated comparison, frozen policies and estimator."""
from __future__ import annotations
import argparse
import concurrent.futures
from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import numpy as np

from .artifacts import _git_metadata
from .config import load_config
from .dr_policy import sha256
from .dr_transfer import ROOT, Thresholds, run_case, write_rollout
from .estimated_allocation import ConfirmedEfficiency, EstimatedAllocationEnv
from .fault_estimator import EstimatorSettings, MotorEfficiencyEstimator
from .fault_estimation_eval import (SCENARIOS, TRACE_KEYS, nominal_model, estimate_from_row,
    summarize_estimation, offline_verify, write_json)
from .integral_controller import IntegralController
from .integral_eval import parameter_digest, verify_integral_rows, integral_statistics, read_columns
from .integral_validation import ScenarioObserver, GAINS, make_case, event_recovery
from .interactive_eval import evaluation_config
from .motor_layout import user_from_native, layout_metadata
from .motor_limit_audit import audit
from .oracle_eval import flat_csv
from .oracle_recovery import AttitudeCriteria, tilt_deg, suffix_latency, weighted_dwell
from .payload_motor_eval import DEFAULT_RECORD, select_models, segment_statistics

PREVIOUS = ROOT/'artifacts/runs/motor-estimation-g43e4hbx'
MODES = ('blind', 'oracle', 'estimated')
EXTRA_KEYS = ('attitude_deg','internal_velocity_error','desired_velocity','motor_thrust_unclipped',
    'motor_thrust_nominal','motor_reaction_nominal','allocator_lower_margin_n','allocator_upper_margin_n',
    'motor_omega','allocator_efficiency_native','allocator_efficiency_user','held_efficiency_next_user',
    'held_confirmation_time_used','estimate_source_time_used','estimate_accepted','confirmed_motor_held_next',
    'allocator_rank','allocator_condition','allocation_mode','physics_allocator_clipped','physics_esc_boundary',
    'allocation_residual_xml','actuator_response_residual_xml','actual_rotor_wrench_xml',
    'xi_candidate','control_dt','e_true_before','e_actor_before','integral_xy_projected','integral_z_projected',
    'interval_allocator_clipping','interval_esc_boundary','interval_action_boundary')


def capture_protected_files(directory):
    """A fresh preservation baseline on every rerun; no dependency on /tmp."""
    names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode().split('\0')
    paths=sorted({ROOT/name for name in names if name and (ROOT/name).is_file()
                  and not (ROOT/name).resolve().is_relative_to(directory.resolve())})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hashes=dict(pool.map(lambda path:(str(path.relative_to(ROOT)),sha256(path)),paths))
    write_json(directory/'protected_hashes_before.json',hashes)


class AllocationObserver(ScenarioObserver):
    def __init__(self, scenario, controller, model, settings, mode):
        super().__init__(scenario, controller)
        self.model = model; self.settings = settings; self.mode = mode

    def on_reset(self, adapter):
        super().on_reset(adapter)
        self.estimator = MotorEfficiencyEstimator(self.model, self.settings)
        self.held = ConfirmedEfficiency()
        self.metadata.update(layout=layout_metadata(adapter.env), allocation_source=self.mode,
            estimator_settings=asdict(self.settings), estimator_reset_only_at_rollout_start=True,
            normalization_frozen=True, ideal_simulator_observations=True)

    def before_step(self, adapter, step, t):
        count = self.estimator.update_count
        super().before_step(adapter, step, t)
        assert self.estimator.update_count == count
        if self.events and self.events[-1]['control_step'] == step:
            self.events[-1].update(estimator_updates_preserved=count, estimator_reset=False)
        env = adapter.env
        physical = (env.data.qpos.copy(), env.data.qvel.copy(), env.actuator_model.omega.copy(), self.controller.xi.copy())
        env.set_estimated_efficiency(self.held.eta_user)
        self.source_time = self.held.last_observation_time
        self.confirmation_time = self.held.last_confirmation_time
        if self.source_time is not None: assert self.source_time <= t+1e-12
        for a,b in zip(physical,(env.data.qpos,env.data.qvel,env.actuator_model.omega,self.controller.xi)):
            np.testing.assert_array_equal(a,b)

    def after_step(self, env, row):
        super().after_step(env, row)  # Integral freezes on THIS allocator's clipping.
        physics = self.physics_rows[self.physics_start:]
        assert len(physics) == env.substeps
        row['delivered_esc_native'] = np.array([r['motor_command'] for r in physics])
        clipping = np.array([r['allocator_clipped'] for r in physics])
        boundary = np.array([r['esc_lower'] | r['esc_upper'] for r in physics])
        row.update(allocator_clipping_seconds_native=env.dt_phys*clipping.sum(0),
            allocator_clipping_union_seconds=env.dt_phys*np.any(clipping,axis=1).sum(),
            esc_boundary_union_seconds=env.dt_phys*np.any(boundary,axis=1).sum(),
            physics_allocator_clipped=clipping, physics_esc_boundary=boundary,
            truth_efficiency_user=user_from_native(env.motor_effectiveness),
            allocator_efficiency_native=env.allocator_efficiency.copy(),
            allocator_efficiency_user=user_from_native(env.allocator_efficiency),
            estimate_source_time_used=self.source_time, held_confirmation_time_used=self.confirmation_time,
            allocator_rank=env.allocator_rank, allocator_condition=env.allocator_condition,
            allocation_mode=self.mode)
        for k in ('allocation_residual_xml','actuator_response_residual_xml','actual_rotor_wrench_xml'):
            row[k] = physics[-1][k].copy()
        # Diagnostic truth checks NEVER feed the observable-only estimator.
        for p in physics:
            np.testing.assert_allclose(p['motor_thrust_actual'],p['motor_effectiveness']*p['motor_thrust_nominal'],atol=0,rtol=0)
            np.testing.assert_allclose(p['motor_reaction_actual'],p['motor_effectiveness']*p['motor_reaction_nominal'],atol=0,rtol=0)
        begin = time.perf_counter_ns()
        result = estimate_from_row(self.estimator,row,env.dt_phys)
        row.update(result,estimator_runtime_ms=(time.perf_counter_ns()-begin)/1e6)
        next_eta,accepted = self.held.consume(result)
        row.update(held_efficiency_next_user=next_eta, estimate_accepted=accepted,
                   confirmed_motor_held_next=self.held.confirmed_motor)
        self.physics_rows.clear()  # Preserve only control-rate traces and 5x4 boundary bits.


def execute(config, policy, scenario, model, settings, mode, horizon=20.):
    controller = IntegralController(policy,GAINS[1])
    observer = AllocationObserver(scenario,controller,model,settings,mode)
    case = make_case(scenario,horizon)
    rows,initial,error,reasons = run_case(config,case,controller,42,
        env_factory=partial(EstimatedAllocationEnv,allocation_source=mode), observer=observer,
        observation_transform=controller.prepare_observation)
    if error: raise RuntimeError(error)
    verify_integral_rows(rows,GAINS[1])
    return rows,initial,observer,case,reasons


def metrics(rows, scenario, observer, case, reasons):
    r = summarize_estimation(rows,scenario,case,reasons)
    dt = np.array([x['time_post']-x['time'] for x in rows])
    runtime = np.array([x['estimator_runtime_ms'] for x in rows])
    r['estimator_runtime_ms'].update(over_10ms_count=int(np.sum(runtime>10)),over_10ms_fraction=float(np.mean(runtime>10)))
    windows = {}; partial_windows = {}
    for name,a,b in [('full_0_20',0,20),('pre_3_5',3,5),('post_5_20',5,20),('tail_18_20',18,20)]:
        selected = [x for x in rows if a+1e-9 < x['time_post'] <= b+1e-9]
        stat = segment_statistics(selected)
        if stat:
            yaw = np.array([x['yaw_error_rad'] for x in selected]);att = np.array([x['attitude_deg'] for x in selected])
            stat.update(roll_rms_deg=float(np.sqrt(np.mean(att[:,0]**2))),
                pitch_rms_deg=float(np.sqrt(np.mean(att[:,1]**2))),
                yaw_max_abs_deg=float(np.degrees(np.abs(yaw)).max()),yaw_mean_deg=float(np.degrees(yaw).mean()))
        available = r['actual_duration_sec'] >= b-1e-9
        windows[name] = stat if available else None
        partial_windows[name] = stat if not available else None
    r.update(windows=windows,partial_observed_windows=partial_windows)
    event = next(iter(scenario.events()),None);record = observer.events[0] if observer.events else None
    recovery = event_recovery(rows,record,event,case.horizon,r['actual_duration_sec'],Thresholds()) if event else dict.fromkeys(('xy','z','3d'))
    if not r['completed']: recovery = dict.fromkeys(('xy','z','3d'))
    r.update({f'recovery_{k}_s':v for k,v in recovery.items()})
    joint = None
    if event and record and r['completed']:
        initial = dict(time_post=event.time,position_error_world=record['e_true_after'],velocity=record['velocity'],
            quaternion=record['quaternion'],omega=record['omega'],yaw_error_rad=0.)
        from .plotting import quaternion_to_euler_deg
        initial['yaw_error_rad']=np.radians(quaternion_to_euler_deg(initial['quaternion'])[2])
        selected=[initial]+[x for x in rows if x['time_post']>event.time+1e-9]
        e,v,w=(np.array([x[k] for x in selected]) for k in ('position_error_world','velocity','omega'))
        good=(np.linalg.norm(e,axis=1)<=.005)&(np.linalg.norm(v,axis=1)<=.02)&(np.linalg.norm(w,axis=1)<=.10)
        good &= (tilt_deg([x['quaternion'] for x in selected])<=5)&(np.abs([x['yaw_error_rad'] for x in selected])<=np.radians(5))
        joint=suffix_latency([x['time_post'] for x in selected],good,True,start=event.time,end=case.horizon)
    r.update(recovery_position_attitude_s=joint,recovery_reason=('not_applicable' if event is None else 'physical_termination' if not r['completed'] else 'suffix_criterion'),
        integral=integral_statistics(rows,float(dt[0])),
        xi_max_abs_m=np.max(np.abs([x['xi_next'] for x in rows]),axis=0),
        allocation_rank_min=min(x['allocator_rank'] for x in rows),
        allocation_nonfinite_condition_count=sum(x['allocator_condition'] is None for x in rows))
    for key,col in [('clipping','physics_allocator_clipped'),('esc_boundary','physics_esc_boundary')]:
        bits=np.array([x[col] for x in rows]); subdt=np.repeat(dt/bits.shape[1],bits.shape[1])
        r[key]=weighted_dwell(bits.reshape(-1,4),subdt)
    for key,col in [('action_boundary','policy_action_at_bound'),('integral_stop','integral_frozen')]:
        r[key]=weighted_dwell([x[col] for x in rows],dt)
    changed=[x for x in rows if np.any(x['allocator_efficiency_user']!=1)]
    r['first_allocator_compensation_time']=changed[0]['time'] if changed else None
    r['first_allocator_compensation_reason']='applied' if changed else 'blind_by_design' if observer.mode=='blind' else 'no_confirmed_fault_or_no_fault'
    if not r['completed']:
        r['partial_position_rmse']={k:r[k] for k in ('position_rmse_xy','position_rmse_z','position_rmse_total')}
        for k in r['partial_position_rmse']:r[k]=None
    return r


def compare_trace(current, previous, *, prefix=False):
    a,b=read_columns(current),read_columns(previous)
    keys=[k for k in b if k in a and k.startswith(('position','quaternion','velocity','omega','action_',
        'wrench_command','motor_thrust','motor_reaction','delivered_esc','xi_t','xi_next'))]
    mask=a.time_post<=5+1e-9 if prefix else np.ones(len(a.time),bool)
    other=b.time_post<=5+1e-9 if prefix else np.ones(len(b.time),bool)
    maximum=0.
    for k in keys:
        np.testing.assert_array_equal(a[k][mask],b[k][other])
        maximum=max(maximum,float(np.max(np.abs(a[k][mask]-b[k][other]))))
    return dict(samples=int(mask.sum()),columns=len(keys),max_abs_error=maximum)


def plots(directory, results):
    from .plotting import _pyplot
    plt=_pyplot()
    for label in ('A_best','B_best'):
        fig,axes=plt.subplots(8,1,figsize=(14,22),sharex=True,layout='constrained')
        for mode,color in zip(MODES,('C0','C1','C2')):
            c=read_columns(directory/f'{label}-motor1_70-{mode}.csv');t=c.time_post
            e=np.array([c[f'position_error_world_{j}'] for j in range(3)]).T
            axes[0].plot(t,np.linalg.norm(e[:,:2],axis=1),color=color,label=mode)
            axes[1].plot(t,e[:,2],color=color,label=mode)
            for j,style in [(0,'-'),(1,'--'),(2,':')]:
                axes[2].plot(t,c[f'attitude_deg_{j}'],style,color=color,label=f'{mode} '+('roll','pitch','yaw')[j])
            axes[3].plot(t,c.estimated_efficiency_user_0,color=color,label=mode+' estimated')
            axes[3].step(c.time,c.allocator_efficiency_user_0,where='post',ls='--',color=color,label=mode+' used')
            axes[4].plot(t,c.motor_thrust_command_3,color=color,label=mode+' M1 command')
            axes[4].plot(t,c.motor_thrust_actual_3,ls='--',color=color,label=mode+' M1 actual')
            axes[5].step(c.time,c.allocator_clipping_union_seconds/.01,where='post',color=color,label=mode)
            for j,style in [(0,'-'),(1,'--'),(2,':')]:axes[6].plot(t,c[f'xi_next_{j}'],style,color=color,label=f'{mode} xi'+('x','y','z')[j])
            axes[7].step(c.time,c.integral_frozen.astype(float),where='post',color=color,label=mode)
        axes[3].step(c.time,c.truth_efficiency_user_0,where='post',color='black',lw=2,label='truth M1')
        for ax,ylabel in zip(axes,('XY error (m)','Z error (m)','Attitude (deg)','User M1 efficiency','User M1 thrust (N)','Clipping fraction','Integral xi (m)','Integral frozen')):
            ax.set_ylabel(ylabel,fontsize=10);ax.grid(alpha=.25);ax.axvline(5,color='k',ls=':')
            ax.legend(ncol=2 if ax in (axes[3],axes[4]) else 3,fontsize=8,
                      columnspacing=1.,handlelength=2.,labelspacing=.3)
            ax.tick_params(labelsize=9)
        axes[-1].set_xlabel('Simulation time (s); commands at interval start, state/estimate at end')
        fig.suptitle(label+' | user motor 1, efficiency 0.70 | fixed PPO + integral',fontsize=14)
        fig.savefig(directory/f'{label}-motor1_70-comparison.png',dpi=140)
        for ax in axes:ax.set_xlim(4.8,7.)
        fig.savefig(directory/f'{label}-motor1_70-transient.png',dpi=140);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/eval_velocity_ab_user_frd.yaml')
    p.add_argument('--completion',type=Path,default=DEFAULT_RECORD)
    p.add_argument('--shadow-results',type=Path,default=PREVIOUS)
    p.add_argument('--output-dir',type=Path,help='New empty directory; default unique artifacts/runs directory')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    config=evaluation_config(load_config(args.config))
    old=json.loads((args.shadow_results/'manifest.json').read_text())
    assert config.resolved_dict()==old['config'], 'common config differs from fixed shadow evaluation'
    assert config.vehicle.reaction_torque_layout=='user_frd' and config.environment.payload.mass==0
    setting_path=args.shadow_results/'estimator_settings.json'
    settings=EstimatorSettings(**json.loads(setting_path.read_text()))
    assert sha256(setting_path)==old['estimator_settings_sha256']
    model,meta=nominal_model(config)
    policies=select_models(args.completion,config)
    for policy in policies:
        previous=next(x for x in old['models'] if x['label']==policy.provenance['label'])
        assert all(previous[k]==policy.provenance[k] for k in ('path','sha256'))
    if args.dry_run:
        print(json.dumps(dict(combinations=66,modes=MODES,config=config.resolved_dict(),estimator=asdict(settings),
            models=[x.provenance for x in policies]),indent=2));return 0
    directory=args.output_dir or Path(tempfile.mkdtemp(prefix='estimated-allocation-',dir=ROOT/'artifacts/runs'))
    directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise FileExistsError('output directory must be empty')
    from stable_baselines3 import PPO
    import torch
    def forbidden(*a,**k):raise RuntimeError('learning and optimizer updates prohibited')
    PPO.learn=PPO.train=forbidden;torch.optim.Adam.step=forbidden
    digests={x.provenance['label']:parameter_digest(x) for x in policies}
    shutil.copyfile(setting_path,directory/'estimator_settings.json')
    capture_protected_files(directory)
    audit(directory,model,config)
    manifest=dict(status='running',requested_rollouts=66,all_runs_fresh=True,seed=42,deterministic=True,
        source_shadow_results=str(args.shadow_results.resolve()),shadow_manifest_sha256=sha256(args.shadow_results/'manifest.json'),
        common_config=config.resolved_dict(),actual_horizon_override_s=20.,nominal_model=meta,
        models=[x.provenance for x in policies],completion=str(args.completion.resolve()),completion_sha256=sha256(args.completion),
        estimator_settings_source=str(setting_path.resolve()),estimator_settings_sha256=sha256(setting_path),settings=asdict(settings),
        scenarios=[asdict(s) for s in SCENARIOS],integral=asdict(GAINS[1]),modes=MODES,
        thresholds=asdict(Thresholds()),attitude_criteria=asdict(AttitudeCriteria()),
        recovery='XY norm / |Z| / 3D norm <=0.005m and respective actual speed <=0.02m/s; suffix through20s >=1s. Joint adds tilt/yaw <=5deg and omega norm <=0.10rad/s. Event right-limit sample included.',
        timing='command/allocator eta at time; observed state and estimate available at time_post; next control uses previous row estimate only. Cached gyro time_post-0.002. Post boundary belongs to preceding interval.',
        application_rule='confirmed healthy -> ones; confirmed valid fault alpha -> single motor; uncertain/insufficient hold last confirmed vector. No smoothing/floor.',
        estimator_input='explicit observed p/q/v/omega and delivered ESC substeps only; truth restricted to oracle branch and evaluation diagnostics',
        source_sha256={str(x.relative_to(ROOT)):sha256(x) for x in (ROOT/'crazyflie_rl').glob('*.py')},
        git=_git_metadata(ROOT),runs={})
    results=[];verification={'offline_replay':{},'blind_reproduction':{},'prefault_equal':{},'nominal_equal':{}}
    def save():
        write_json(directory/'manifest.json',manifest);write_json(directory/'summary.json',results)
        flat_csv(directory/'summary.csv',results);write_json(directory/'verification.json',verification)
    save();print('RESULT_DIR',directory,flush=True)
    try:
        for policy in policies:
            label=policy.provenance['label']
            for scenario in SCENARIOS:
                for mode in MODES:
                    key=f'{label}-{scenario.name}-{mode}';print('RUN',key,flush=True)
                    rows,initial,observer,case,reasons=execute(config,policy,scenario,model,settings,mode)
                    result=metrics(rows,scenario,observer,case,reasons)
                    result.update(key=key,label=label,allocation_mode=mode,reused=False)
                    path=directory/(key+'.csv')
                    write_rollout(path,[{k:r[k] for k in TRACE_KEYS+EXTRA_KEYS} for r in rows])
                    write_json(directory/(key+'-events.json'),observer.events)
                    flat_csv(directory/(key+'-events.csv'),observer.events)
                    verification['offline_replay'][key]=offline_verify(path,model,settings)
                    blind=directory/f'{label}-{scenario.name}-blind.csv'
                    if mode=='blind':
                        verification['blind_reproduction'][key]=compare_trace(path,args.shadow_results/f'{label}-{scenario.name}.csv')
                    else:
                        verification['prefault_equal'][key]=compare_trace(path,blind,prefix=True)
                        if not scenario.has_fault: verification['nominal_equal'][key]=compare_trace(path,blind)
                    assert parameter_digest(policy)==digests[label]
                    results.append(result)
                    manifest['runs'][key]=dict(initial=initial,metadata=observer.metadata,events=observer.events,
                        csv_sha256=sha256(path),policy_unchanged=True)
                    save();print('DONE',len(results),key,result['actual_duration_sec'],result['recovery_3d_s'],flush=True)
        plots(directory,results)
        script='#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'
        script+='run_dir=$(mktemp -d artifacts/runs/estimated-allocation-XXXXXX)\n'
        script+='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_estimated_allocation.py --output-dir "$run_dir"\n'
        script+='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_estimated_allocation.py --run-dir "$run_dir"\n'
        (directory/'rerun.sh').write_text(script)
        manifest['status']='complete';manifest['completed_rollouts']=len(results)
        verification.update(no_learning_or_optimizer_updates=True,parameters_unchanged=True,single_efficiency_application_checked_every_substep=True)
        save();print('COMPLETE',directory,flush=True)
    except BaseException as exc:
        manifest['status']='interrupted_or_error';manifest['error']=repr(exc);save();raise
    return 0

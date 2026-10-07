"""Fixed A/B payload/fault experiment using the existing frozen rollout and plant."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shlex
import tempfile

import numpy as np

from .artifacts import _git_metadata
from .config import load_config
from .dr_policy import load_frozen_policy, sha256
from .dr_transfer import (ROOT, Case, Thresholds, reference_sequence, run_case,
                          summarize, validate_common_config, write_json, write_rollout)
from .interactive_eval import InteractiveEnv, evaluation_config, rotor_mapping
from .plotting import quaternion_to_euler_deg, save_transfer_comparison_plot
from .velocity_reference import observation_contract, velocity_semantics

DEFAULT_RECORD = ROOT/'artifacts/runs/velocity-ab-training-comparison-isygddsb/completion.json'
WINDOWS = {'full_0_20': (0.,20.), 'pre_3_5': (3.,5.), 'post_5_20': (5.,20.), 'tail_18_20': (18.,20.)}


@dataclass(frozen=True)
class Condition:
    name: str
    mass: float = 0.
    offset: tuple = (0., 0.)
    motor1: float = 1.

    @property
    def has_fault(self):
        return self.motor1 != 1.


CONDITIONS = (Condition('nominal'), Condition('centered_payload', .005),
              Condition('offset_payload', .005, (.03,0.)), Condition('motor_80', motor1=.8),
              Condition('motor_70', motor1=.7), Condition('combined_80', .005, (.03,0.), .8),
              Condition('combined_70', .005, (.03,0.), .7))


def condition_config(source, condition):
    fixed = evaluation_config(source)  # Turns off both reset samplers; preserves fixed payload.
    e = replace(fixed.environment, episode_sec=20., payload=replace(fixed.environment.payload,
                mass=condition.mass, offset=condition.offset, randomize=False))
    return replace(fixed, environment=e)


def motor_signals(env):
    raw = env.B_pinv @ env._last_wrench_cmd
    # Actual steady-state maximum at ESC=1 through the existing forward map,
    # also constrained by the unchanged allocator command maximum.
    model = env.actuator_model
    nominal_max = np.minimum(model.thrust_from_omega(model.steady_state_gain_rad_s), env.thrust_max)
    lam = env.motor_effectiveness.copy()
    return dict(motor_thrust_unclipped=raw, motor_thrust_command=env._last_f_cmd.copy(),
        motor_thrust_nominal=env.nominal_thrust.copy(), motor_thrust_actual=env._last_f.copy(),
        motor_reaction_nominal=env.nominal_reaction_torque.copy(), motor_reaction_actual=env._last_q_actual.copy(),
        motor_effectiveness=lam, motor_command=env._last_motor_cmd.copy(), motor_omega=env._last_omega.copy(),
        allocator_clipped=np.abs(raw-env._last_f_cmd)>1e-12,
        allocator_lower=env._last_f_cmd<=env.thrust_min+1e-9,
        allocator_upper=env._last_f_cmd>=env.thrust_max-1e-9,
        allocator_lower_margin_n=env._last_f_cmd-env.thrust_min,
        allocator_upper_margin_n=env.thrust_max-env._last_f_cmd,
        esc_lower=env._last_motor_cmd<=1e-9, esc_upper=env._last_motor_cmd>=1-1e-9,
        esc_lower_margin=env._last_motor_cmd.copy(), esc_upper_margin=1-env._last_motor_cmd,
        nominal_max_thrust_n=nominal_max, effective_max_thrust_n=lam*nominal_max,
        effective_max_margin_n=lam*nominal_max-env._last_f,
        actual_effective_upper=env._last_f>=lam*nominal_max-1e-9,
        thrust_lag_n=env._last_f_cmd-env.nominal_thrust,
        applied_force_ctrl=env.data.ctrl[env.act_force].copy(),
        applied_torque_ctrl=env.data.ctrl[env.act_torque].copy())


class RecordedFaultEnv(InteractiveEnv):
    """Observe every physics interval; parent applies the existing output loss once."""
    def __init__(self, *args, **kwargs):
        self.physics_rows = []
        super().__init__(*args, **kwargs)

    def _apply_control(self, wrench):
        super()._apply_control(wrench)
        self.physics_rows.append(dict(physics_time=float(self.data.time),
            physics_time_post=float(self.data.time+self.dt_phys), **motor_signals(self)))


class FaultObserver:
    def __init__(self, condition):
        self.condition = condition
        self.events = []
        self.metadata = None
        self.physics_rows = []

    def on_reset(self, adapter):
        env = adapter.env
        # Normal env reset has applied payload, material constants and all episode state.
        # Replace ONLY the motor initial state before any simulation with the same
        # nominal-mass reset in every condition (no payload trim or compensation).
        output = env.actuator_model.reset(airborne=True, episode_mass=env.mass,
                                         gravity_m_s2=env.gravity, randomize=False)
        env._record_actuator_output(output)
        env._write_applied_motor_controls()
        env.nominal_thrust = env._last_f.copy()
        env.nominal_reaction_torque = env._last_q_actual.copy()
        self.physics_rows = env.physics_rows
        bid = env.drone_bid
        self.metadata = dict(payload_mass_kg=env._com_mw, payload_offset_body_xyz_m=env._com_off3.tolist(),
            body_mass_kg=float(env.model.body_mass[bid]), body_ipos_m=env.model.body_ipos[bid].tolist(),
            body_inertia_kg_m2=env.model.body_inertia[bid].tolist(), body_iquat_wxyz=env.model.body_iquat[bid].tolist(),
            world_com_m=env.data.xipos[bid].tolist(), nominal_body_mass_kg=env._m0,
            nominal_body_ipos_m=env._ipos0.tolist(), policy_hover_mass_kg=env.mass,
            initial_motor_thrust_n=env._last_f.tolist(), initial_motor_omega=env._last_omega.tolist(),
            initial_motor_effectiveness=env.motor_effectiveness.tolist(), allocator=env.B.tolist(),
            action_scale=env.residual_scale.tolist(), disturbance_body_nm=env.dist_torque_body.tolist(),
            rotors=rotor_mapping(env), actuator=env.actuator_snapshot(),
            initial_motor_reset='existing actuator.reset(airborne=True, episode_mass=nominal policy mass); no payload trim')
        assert env._com_mw == self.condition.mass
        np.testing.assert_array_equal(env._com_off3[:2], self.condition.offset)
        np.testing.assert_allclose(env._last_f, np.full(4,env.mass*env.gravity/4), rtol=1e-12)

    def before_step(self, adapter, step, t):
        if self.condition.has_fault and step == round(5/adapter.control_dt):
            env = adapter.env
            before = adapter.snapshot()
            old = env.motor_effectiveness.copy()
            env.motor_effectiveness = np.array([self.condition.motor1,1.,1.,1.])
            # No stepping/reset of any physical/actuator/reference/observation state.
            assert adapter.snapshot() == before
            assert abs(float(env.data.time)-5.) < 1e-9 and t == 5.
            self.events.append(dict(event='motor_effectiveness', control_step=step, policy_input_time=t,
                simulation_time=float(env.data.time), motor_number=1, effectiveness_before=old,
                effectiveness_after=env.motor_effectiveness.copy(), physical_and_actuator_state_unchanged=True))

    def after_step(self, env, row):
        row.update(motor_signals(env))
        row['position_error_world'] = row['position']-row['reference_post']
        row['attitude_deg'] = quaternion_to_euler_deg(row['quaternion'])
        row['policy_action_at_bound'] = np.abs(row['action_applied'])>=1-1e-9
        row['policy_action_clipped'] = np.abs(row['action']-row['action_applied'])>1e-12
        # Correct effective bounds without changing the existing nominal-only logger default.
        row['motor_actual_at_thrust_upper_bound'] = row['motor_thrust_actual']>=row['effective_max_thrust_n']-1e-9
        row['motor_actual_at_thrust_lower_bound'] = row['motor_thrust_actual']<=env.motor_effectiveness*env.thrust_min+1e-9


def position_statistics(errors):
    e = np.asarray(errors, dtype=float).reshape((-1,3))
    if not len(e):
        return None
    mean = e.mean(0)
    variance = np.mean((e-mean)**2, axis=0)
    square = np.mean(e*e, axis=0)
    xy_bias2, xy_var = np.dot(mean[:2],mean[:2]), variance[:2].sum()
    xy_identity = float(square[:2].sum()-xy_bias2-xy_var)
    z_identity = float(square[2]-mean[2]**2-variance[2])
    np.testing.assert_allclose(square, mean**2+variance, rtol=1e-11, atol=1e-13)
    return dict(mean_error_xy_m=mean[:2].tolist(), offset_xy_m=float(np.sqrt(xy_bias2)),
        rmse_xy_m=float(np.sqrt(square[:2].sum())), sway_xy_rms_m=float(np.sqrt(xy_var)),
        max_xy_error_m=float(np.linalg.norm(e[:,:2],axis=1).max()),
        mean_error_z_m=float(mean[2]), offset_z_abs_m=float(abs(mean[2])), rmse_z_m=float(np.sqrt(square[2])),
        sway_z_rms_m=float(np.sqrt(variance[2])), max_abs_z_error_m=float(np.abs(e[:,2]).max()),
        rmse_3d_m=float(np.sqrt(square.sum())), xy_bias_variance_residual_m2=xy_identity,
        z_bias_variance_residual_m2=z_identity)


def segment_statistics(rows):
    if not rows:
        return None
    result = position_statistics([r['position_error_world'] for r in rows])
    result.update(sample_count=len(rows), first_sample_s=rows[0]['time_post'], last_sample_s=rows[-1]['time_post'])
    for key,prefix in (('velocity','actual_velocity'),('internal_velocity_error','velocity_error')):
        value = np.array([r[key] for r in rows])
        result[prefix+'_xy_rms_m_s'] = float(np.sqrt(np.mean(np.sum(value[:,:2]**2,axis=1))))
        result[prefix+'_z_rms_m_s'] = float(np.sqrt(np.mean(value[:,2]**2)))
    q=np.array([r['quaternion'] for r in rows]);w=np.array([r['omega'] for r in rows])
    result['max_tilt_deg']=float(np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1))).max())
    result['max_angular_speed_rad_s']=float(np.linalg.norm(w,axis=1).max())
    return result


def recovery_time(rows, thresholds, dimension, completed, *, horizon=20., start=5.):
    """Target-relative suffix recovery; defaults preserve the original 20s test."""
    if not completed:
        return None
    selected=[r for r in rows if start-1e-9<=r['time_post']<=horizon+1e-9]
    if not selected:
        return None
    times=np.array([r['time_post'] for r in selected])
    e=np.array([r['position_error_world'] for r in selected])
    v=np.array([r['velocity'] for r in selected])
    sl={'xy':slice(0,2),'z':slice(2,3),'3d':slice(0,3)}[dimension]
    good=(np.linalg.norm(e[:,sl],axis=1)<=thresholds.position_band_m)&(np.linalg.norm(v[:,sl],axis=1)<=thresholds.speed_band_m_s)
    suffix=np.logical_and.accumulate(good[::-1])[::-1]
    candidates=np.flatnonzero(suffix & (horizon-times>=thresholds.minimum_settle_sec-1e-9))
    return float(times[candidates[0]]-start) if len(candidates) else None


def dwell(values, dt):
    flags=np.asarray(values,dtype=bool)
    if flags.ndim==1:flags=flags[:,None]
    run=np.zeros(flags.shape[1],dtype=int);longest=run.copy()
    for row in flags:
        run=np.where(row,run+1,0);longest=np.maximum(longest,run)
    return dict(duration_s_per_channel=(flags.sum(0)*dt).tolist(),
                longest_continuous_s_per_channel=(longest*dt).tolist(),
                duration_any_s=float(np.any(flags,axis=1).sum()*dt))


def motor_statistics(rows, physics, dt, physics_dt):
    if not physics:
        return None
    result={'sampling':'every physics interval; dwell = interval count * physics_dt', 'physics_dt':physics_dt}
    for key in ('allocator_clipped','allocator_lower','allocator_upper','esc_lower','esc_upper','actual_effective_upper'):
        result[key]=dwell([r[key] for r in physics],physics_dt)
    for key in ('policy_action_at_bound','policy_action_clipped'):
        result[key]=dwell([r[key] for r in rows],dt)
    for key in ('allocator_lower_margin_n','allocator_upper_margin_n','esc_lower_margin','esc_upper_margin','effective_max_margin_n'):
        result[key+'_minimum_per_motor']=np.min([r[key] for r in physics],axis=0).tolist()
    for key in ('nominal_max_thrust_n','effective_max_thrust_n'):
        result[key+'_minimum_per_motor']=np.min([r[key] for r in physics],axis=0).tolist()
        result[key+'_maximum_per_motor']=np.max([r[key] for r in physics],axis=0).tolist()
    result['thrust_lag_rms_n_per_motor']=np.sqrt(np.mean(np.array([r['thrust_lag_n'] for r in physics])**2,axis=0)).tolist()
    return result


def analyze(rows, case, condition, observer, error, reasons, thresholds):
    result=summarize(rows,case,thresholds,error,reasons)
    # Existing summarize reports observed-only RMSE on failures. Keep it under
    # an explicitly partial key; full20 and fixed tail are unavailable then.
    if not result['completed']:
        result['partial_observed_3d_rmse_m']=result['position_rmse_total']
        for key in ('position_rmse_xy','position_rmse_z','position_rmse_total'):
            result[key]=None
    duration=result['actual_duration_sec'];full={};partial={}
    for name,(start,end) in WINDOWS.items():
        selected=[r for r in rows if start+1e-9<r['time_post']<=end+1e-9]
        available=duration>=end-1e-9 and not error
        if name in ('full_0_20','tail_18_20','post_5_20'):available=available and result['completed']
        full[name]=segment_statistics(selected) if available else None
        partial[name]=segment_statistics(selected) if selected and not available else None
    result.update(condition=condition.name, windows=full, partial_observed_windows=partial,
                  fault_scheduled=condition.has_fault, fault_applied=bool(observer.events),
                  recovery_xy_s=None,recovery_z_s=None,recovery_3d_s=None, fault_observed_metrics=None,
                  mean_shift_xy_m=None,mean_shift_z_m=None,mean_shift_scope=None)
    if condition.has_fault and observer.events:
        post=[r for r in rows if r['time_post']>5+1e-9]
        result['fault_observed_metrics']=segment_statistics(post)
        for dimension in ('xy','z','3d'):
            result[f'recovery_{dimension}_s']=recovery_time(rows,thresholds,dimension,result['completed'])
        pre=full['pre_3_5'];after=segment_statistics(post)
        if pre and after:
            result['mean_shift_xy_m']=(np.array(after['mean_error_xy_m'])-pre['mean_error_xy_m']).tolist()
            result['mean_shift_z_m']=after['mean_error_z_m']-pre['mean_error_z_m']
            result['mean_shift_scope']='(3,5] vs '+('(5,20]' if result['completed'] else '(5,termination], partial')
    dt=1/100
    result['motor_diagnostics']=motor_statistics(rows,observer.physics_rows,dt,dt/5)
    result['motor_diagnostics_post5']=motor_statistics(
        [r for r in rows if r['time']>=5-1e-9],
        [r for r in observer.physics_rows if r['physics_time']>=5-1e-9],dt,dt/5)
    return result


def select_models(record,config):
    data=json.loads(record.read_text());policies=[]
    for label,step in (('A',980000),('B',960000)):
        entry=data['training'][label.lower()];best=entry['best']
        path=(Path(entry['run_dir'])/best['path']).resolve(strict=True)
        if best['timestep']!=step or sha256(path)!=best['sha256']:
            raise ValueError(f'{label} best metadata/hash mismatch')
        if Path(data['models'][label+'_best']).resolve()!=path:
            raise ValueError('completion record model paths disagree')
        policy=load_frozen_policy(label+'_best',str(path),config)
        if policy.model.num_timesteps!=step:
            raise ValueError('checkpoint timestep mismatch')
        policies.append(policy)
    return policies


def save_summary(directory, results):
    write_json(directory/'summary.json',results)
    flat=[]
    for r in results:
        row={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        for window,metrics in r['windows'].items():
            if metrics:
                row.update({window+'__'+k:json.dumps(v) if isinstance(v,list) else v for k,v in metrics.items()})
        row['motor_diagnostics']=json.dumps(r['motor_diagnostics'])
        flat.append(row)
    import csv
    keys=list(dict.fromkeys(k for row in flat for k in row))
    with (directory/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(flat)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/eval_velocity_ab.yaml')
    parser.add_argument('--record',type=Path,default=DEFAULT_RECORD)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/runs')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    config=evaluation_config(load_config(args.config));validate_common_config(config)
    if velocity_semantics(config)['mode']!='absolute':raise ValueError('A/B must receive actual world velocity')
    policies=select_models(args.record,config)
    case=Case('hover',20.,(0.,0.,1.));thresholds=Thresholds()
    specs={c.name:condition_config(config,c) for c in CONDITIONS}
    manifest=dict(status='dry_run' if args.dry_run else 'running',seed=42,deterministic=True,
        models=[p.provenance for p in policies],selection_record=str(args.record.resolve()),
        selection_record_sha256=sha256(args.record),git=_git_metadata(ROOT),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'crazyflie_rl').glob('*.py')},
        common_resolved_config=config.resolved_dict(),conditions={c.name:asdict(c) for c in CONDITIONS},
        resolved_configs={k:v.resolved_dict() for k,v in specs.items()},
        observation_contract=observation_contract(config),thresholds=asdict(thresholds),
        normalization_order='native absolute raw observation -> frozen saved normalization/clipping if present -> deterministic predict',
        timing={'state':'post-state at t+dt; policy observation/action at t',
                'motor':'rollout row: last physics substep; physics.csv: every [physics_time,physics_time_post]',
                'fault':'one event at control step 500, t=5; first affected post-state t=5.01',
                'windows':'right-closed post-state samples (start,end]; t=5 belongs to pre, not post',
                'recovery':'target-relative suffix through t=20, >=1 second; latency from t=5; early termination => null',
                'axis_recovery':'XY norm(position/speed) or abs(Z position/speed), same thresholds .005m/.02m/s; 3D also retained'},
        case=case.description(.01),runs={},limitations=['One deterministic rollout per model/condition, not a success probability.',
            'Mean-centered RMS includes drift/transients; it is not by itself evidence of periodic oscillation.',
            'Thrust headroom is not a closed-loop stability or controllability guarantee.'])
    if args.dry_run:
        print(json.dumps(manifest,indent=2));return 0
    args.output_dir.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='ab-payload-motor-',dir=args.output_dir))
    write_json(directory/'manifest.json',manifest)
    command=['python','compare_ab_payload_motor.py','--config',str(args.config.resolve()),'--record',str(args.record.resolve()),'--output-dir',str(args.output_dir.resolve())]
    (directory/'rerun.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl '+shlex.join(command)+'\n')
    # Preserve hashes for all existing artifacts in the A/B training/comparison runs,
    # plus source physics and every config. No old output path is opened for writing.
    source_record=json.loads(args.record.read_text())
    protected_roots=[args.record.parent,Path(source_record['evaluation_dir'])]+[Path(r['run_dir']) for r in source_record['training'].values()]
    protected={str(p):sha256(p) for root in protected_roots for p in root.rglob('*') if p.is_file()}
    protected.update({str(p):sha256(p) for p in (ROOT/'configs').rglob('*.yaml')})
    for name in ('environment.py','actuators.py','motor_degradation.py','training.py'):
        p=ROOT/'crazyflie_rl'/name;protected[str(p)]=sha256(p)
    write_json(directory/'protected_hashes_before.json',protected)
    results=[]
    schedule=reference_sequence(case,.01);np.savez_compressed(directory/'reference.npz',**schedule)
    try:
        common_initial=None
        for condition in CONDITIONS:
            plotted={};first=None
            for policy in policies:
                label=policy.provenance['label'];key=condition.name+'-'+label
                observer=FaultObserver(condition)
                rows,snapshot,error,reasons=run_case(specs[condition.name],case,policy,42,
                    env_factory=RecordedFaultEnv,observer=observer)
                for row in rows:row.update(model_label=label,condition=condition.name)
                write_rollout(directory/(key+'.csv'),rows)
                write_rollout(directory/(key+'-physics.csv'),observer.physics_rows)
                write_events(directory/(key+'-events.csv'),observer.events)
                result=analyze(rows,case,condition,observer,error,reasons,thresholds)
                result['label']=label;results.append(result)
                manifest['runs'][key]=dict(initial_snapshot=snapshot,physical=observer.metadata,events_applied=len(observer.events),error=error)
                save_summary(directory,results);write_json(directory/'manifest.json',manifest)
                if error:raise RuntimeError(key+': '+error)
                if first is not None and snapshot!=first:raise AssertionError('A/B initial snapshots differ')
                first=snapshot
                comparable={k:snapshot[k] for k in ('position','quaternion','velocity','omega','qpos','qvel','reference','observation','previous_action','_last_f','_last_omega','_last_motor_cmd')}
                if common_initial is not None and comparable!=common_initial:raise AssertionError('conditions did not use same nominal initial state/motors')
                common_initial=comparable
                verify_signals(rows,observer.physics_rows,condition)
                plotted[label]=rows
                print(key, result['completed'], result['actual_duration_sec'],result['end_reason'],flush=True)
            save_transfer_comparison_plot(directory/(condition.name+'-comparison.png'),plotted,schedule,condition.name)
            save_fault_plot(directory/(condition.name+'-fault.png'),plotted,condition)
        assert all(sha256(p)==h for p,h in protected.items())
        manifest.update(status='completed',protected_files_unchanged=True,protected_file_count=len(protected),
                        initial_physical_and_motor_states_equal=True,ab_snapshots_equal=True,signal_verification_passed=True)
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_summary(directory,results);write_json(directory/'manifest.json',manifest)
        print('results:',directory,flush=True)
    return 0


def write_events(path,events):
    if events:write_rollout(path,events)
    else:path.write_text('event,control_step,policy_input_time,simulation_time,motor_number,effectiveness_before_0,effectiveness_after_0\n')


def verify_signals(rows,physics,condition):
    for r in physics:
        np.testing.assert_array_equal(r['motor_thrust_actual'],r['motor_effectiveness']*r['motor_thrust_nominal'])
        np.testing.assert_array_equal(r['motor_reaction_actual'],r['motor_effectiveness']*r['motor_reaction_nominal'])
        np.testing.assert_array_equal(r['applied_force_ctrl'],r['motor_thrust_actual'])
        np.testing.assert_array_equal(r['applied_torque_ctrl'],r['motor_reaction_actual'])
        expected=condition.motor1 if condition.has_fault and r['physics_time']>=5-1e-9 else 1.
        np.testing.assert_array_equal(r['motor_effectiveness'],[expected,1,1,1])
    for row in rows:
        np.testing.assert_allclose(row['policy_raw_velocity'],row['velocity_before'],atol=1e-7,rtol=1e-7)
        np.testing.assert_array_equal(row['reference'],[0,0,1])


def save_fault_plot(path,rollouts,condition):
    from .plotting import _new_path,_pyplot
    plt=_pyplot();target=_new_path(path)
    fig,axes=plt.subplots(4,2,figsize=(14,12),sharex=True)
    for j,(label,rows) in enumerate(rollouts.items()):
        if not rows:continue
        color=f'C{j}';t=np.array([r['time_post'] for r in rows]);ti=np.array([r['time'] for r in rows])
        e=np.array([r['position_error_world'] for r in rows])
        axes[0,0].plot(t,np.linalg.norm(e[:,:2],axis=1),color=color,label=label)
        axes[0,1].plot(t,e[:,2],color=color,label=label)
        for i in range(4):
            axes[1,0].step(ti,[r['motor_effectiveness'][i] for r in rows],where='post',color=color,ls=['-','--',':','-.'][i],label=f'{label} motor{i+1}')
            ax=axes[2+i//2,i%2]
            ax.plot(t,[r['motor_thrust_nominal'][i] for r in rows],color=color,ls='--',label=f'{label} pre-eff')
            ax.plot(t,[r['motor_thrust_actual'][i] for r in rows],color=color,label=f'{label} actual')
            ax.set_ylabel(f'Motor {i+1} thrust [N]')
        axes[1,1].plot(t,[np.degrees(np.arccos(np.clip(1-2*(r['quaternion'][1]**2+r['quaternion'][2]**2),-1,1))) for r in rows],color=color,label=label)
        if rows[-1]['terminated']:
            for ax in axes.flat:ax.axvline(t[-1],color=color,ls=':',alpha=.7,label=f'{label} terminated {t[-1]:.2f}s')
    axes[0,0].set_ylabel('Horizontal error norm [m]');axes[0,1].set_ylabel('Signed vertical error [m]')
    axes[1,0].set_ylabel('Rotor effectiveness');axes[1,1].set_ylabel('Tilt [deg]')
    for ax in axes.flat:
        if condition.has_fault:ax.axvline(5,color='k',ls='--',lw=1,label='fault t=5s')
        ax.set_xlim(0,20);ax.grid(alpha=.25);ax.legend(fontsize=7)
    for ax in axes[-1]:ax.set_xlabel('state: post time; effectiveness: control input time [s]')
    fig.suptitle(condition.name+(' — motor1 fault at t=5s' if condition.has_fault else ' — no fault event'))
    fig.tight_layout();fig.savefig(target,dpi=140);plt.close(fig)

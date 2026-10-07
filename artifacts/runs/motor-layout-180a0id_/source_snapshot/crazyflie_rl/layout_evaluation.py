"""Fixed 20 nominal + 10 fault rollouts; frozen native PPO, two physical layouts."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path
import shlex
import tempfile

import mujoco
import numpy as np

from .artifacts import _git_metadata
from .config import load_config
from .controllers import rotmat_from_quat_wxyz
from .dr_policy import sha256
from .dr_transfer import ROOT, Thresholds, run_case, summarize, write_json, write_rollout, validate_common_config
from .integral_controller import IntegralController
from .integral_eval import parameter_digest, verify_integral_rows, read_columns, integral_statistics
from .integral_validation import Scenario, GAINS, make_case, interval_statistics
from .interactive_eval import evaluation_config, rotor_mapping
from .motor_layout import S, P, layout_metadata, user_motor_signals, user_from_native, world_from_frd
from .oracle_allocation import OracleAllocationEnv
from .oracle_eval import OracleObserver, compare_csv, flat_csv, verify_oracle, residual_statistics
from .oracle_recovery import scenario_at, analyze_run, suffix_latency, tilt_deg, timed_diagnostics, clean
from .oracle_recovery_artifacts import vector, verify_csv
from .payload_motor_eval import DEFAULT_RECORD, condition_config, select_models, recovery_time, motor_statistics

PREVIOUS = ROOT/'artifacts/runs/oracle-recovery-m3dh3vij'
FAULTS = (('A_best','combined_motor2_70',70), ('A_best','combined_motor2_70',71),
          ('B_best','combined_motor3_70',72), ('B_best','combined_motor3_70',73),
          ('A_best','combined_motor3_70',66))
NOMINAL = (('hover',(0.,0.,1.),0.), ('step_x_005',(.05,0.,1.),0.),
           ('step_y_005',(0.,.05,1.),0.), ('yaw_cw_5',(0.,0.,1.),-5.),
           ('yaw_ccw_5',(0.,0.,1.),5.))


class LayoutObserver(OracleObserver):
    def __init__(self, scenario, controller, initial_yaw_deg=0.):
        super().__init__(scenario, controller)
        self.initial_yaw_deg = initial_yaw_deg

    def on_reset(self, adapter):
        if self.initial_yaw_deg:
            # Initialization only, before snapshot/first prediction. Same physical
            # native quaternion in both layouts; no FRD quaternion fed to PPO.
            angle = np.radians(self.initial_yaw_deg)/2
            adapter.env.data.qpos[3:7] = [np.cos(angle),0,0,np.sin(angle)]
            mujoco.mj_forward(adapter.env.model, adapter.env.data)
        super().on_reset(adapter)
        self.metadata.update(layout=layout_metadata(adapter.env),
            initial_yaw_native_deg=self.initial_yaw_deg,
            yaw_positive_top_view='CCW; world +Z, not FRD body +Z')

    def after_step(self, env, row):
        super().after_step(env, row)
        row.update(reaction_torque_layout=env.reaction_torque_layout, **user_motor_signals(row))
        row['R_world_from_FRD_post'] = world_from_frd(rotmat_from_quat_wxyz(row['quaternion']))
        # Metadata/extra columns only. No altered physical or controller signal.
        for r in self.physics_rows[self.physics_start:]:
            r.update(reaction_torque_layout=env.reaction_torque_layout, **user_motor_signals(r))


def configuration_pair(legacy_path, user_path):
    configs = {n:evaluation_config(load_config(p)) for n,p in (('legacy',legacy_path),('user_frd',user_path))}
    for n,c in configs.items():
        validate_common_config(c)
        if c.vehicle.reaction_torque_layout != n: raise ValueError('explicit layout profile mismatch')
    left,right = (c.resolved_dict() for c in configs.values())
    for d in (left,right):
        for key in ('source_path','experiment'): d.pop(key)
        for key in ('reaction_torque_layout','motor_direction'): d['vehicle'].pop(key)
    if left != right: raise ValueError('profiles differ beyond motor layout and metadata')
    return configs


def static_validation(configs, directory=None):
    from audit_coordinate_contracts import new_env, engine_geometry, increments, efficiency_loss_checks, establish_frames
    result = {}
    for layout,config in configs.items():
        env,adapter = new_env(config)
        try:
            g = engine_geometry(env); frames = establish_frames(env,g)
            H = np.eye(4); H[:3,:3] = S
            target = np.array([[1,1,-1,-1],[1,-1,-1,1],[-1,1,-1,1],[1,1,1,1]],dtype=float)
            target[:2] *= .03536; target[2] *= env.torque_coefficient
            converted = H@g['matrix_T']@P
            if layout == 'legacy': target[2] *= -1
            np.testing.assert_allclose(converted,target,atol=1e-14)
            np.testing.assert_array_equal(np.sign(env.B),np.sign(g['matrix_T']))
            np.testing.assert_allclose(np.abs(env.B[:2]-g['matrix_T'][:2]),.000005,atol=1e-14)
            yaw = []
            for sign in (-1,1):
                command = np.array([0.,0.,sign*.0001,env.mass*env.gravity])
                f = env.B_pinv@command
                actual = g['matrix_T']@f
                assert sign*actual[2] > 0 and np.all((f>env.thrust_min)&(f<env.thrust_max))
                np.testing.assert_allclose(actual[2],command[2],atol=1e-15)
                # Existing delayed actuator path, held physical pose. Record
                # transient separately from direct actual-force mapping probes.
                env.reset_actuator_state(airborne=True)
                samples=[]
                for k in range(150):
                    env._apply_control(command)
                    samples.append(dict(t=(k+1)*env.dt_phys, requested_yaw_nm=command[2],
                        nominal_thrust=env.nominal_thrust.copy(), actual_thrust=env._last_f.copy(),
                        actual_reaction=env._last_q_actual.copy(), yaw_nm=float(env._last_q_actual.sum())))
                assert sign*samples[-1]['yaw_nm']>0
                assert abs(samples[0]['yaw_nm'])<abs(samples[-1]['yaw_nm'])
                yaw.append(dict(sign=sign, static_wrench=actual, delayed_samples=samples))
            result[layout] = dict(allocator=env.B.copy(), physical_native=g['matrix_T'],
                physical_user_frd=converted, expected_user_frd=target, motor_mapping=rotor_mapping(env),
                layout=layout_metadata(env), yaw=yaw)
        finally: env.close()
        records,masses = increments(config,S)
        losses = efficiency_loss_checks(config,frames)
        motor1 = next(r for r in losses if r['test']=='user_FRD_motor_1')
        expected = np.array([-.03536,-.03536,(1 if layout=='user_frd' else -1)*.00594])*motor1['loss_magnitude_n']
        np.testing.assert_allclose(motor1['delta_torque_origin_frd_nm'],expected,atol=1e-14)
        assert max(max(r['errors'].values()) for r in records)<1e-8
        result[layout].update(rotor_increment_count=len(records), increments=records, masses=masses, efficiency_losses=losses)
    result = clean(result)
    if directory:
        write_json(directory/'static_validation.json',result)
        flat_csv(directory/'rotor_increments.csv',[dict(layout=n,**r) for n,x in result.items() for r in x['increments']])
        flat_csv(directory/'motor_mapping.csv',[dict(layout=n,**r) for n,x in result.items() for r in x['motor_mapping']])
        flat_csv(directory/'matrix_comparison.csv',[dict(layout=n,allocator=x['allocator'],physical_native=x['physical_native'],physical_user_frd=x['physical_user_frd']) for n,x in result.items()])
    return result


def nominal_analysis(rows,case,observer,error,reasons,config):
    r = summarize(rows,case,Thresholds(),error,reasons)
    r.update(windows={},partial_observed_windows={})
    for key,start,end in [('full',0.,8.),('tail',6.,8.)]:
        r['windows'][key],r['partial_observed_windows'][key] = interval_statistics(rows,observer.physics_rows,start,end,r['actual_duration_sec'],config)
    initial = dict(time_post=0.,position_error_world=rows[0]['position_before']-rows[0]['reference'],
                   velocity=rows[0]['velocity_before'],quaternion=rows[0]['quaternion_before'],
                   omega=rows[0]['omega_before'],yaw_error_rad=np.radians(observer.initial_yaw_deg))
    values=[initial]+rows
    for axis in ('xy','z','3d'):
        r['recovery_'+axis+'_s']=recovery_time(values,Thresholds(),axis,r['completed'],horizon=8.,start=0.)
    e=np.array([v['position_error_world'] for v in values]); v=np.array([v['velocity'] for v in values])
    yaw=np.abs(np.degrees([v['yaw_error_rad'] for v in values])); w=np.linalg.norm([v['omega'] for v in values],axis=1)
    good=(np.linalg.norm(e,axis=1)<=.005)&(np.linalg.norm(v,axis=1)<=.02)&(tilt_deg([v['quaternion'] for v in values])<=5)&(yaw<=5)&(w<=.10)
    times=[v['time_post'] for v in values]
    r['recovery_position_attitude_s']=suffix_latency(times,good,r['completed'],start=0.,end=8.)
    r['yaw_recovery_s']=suffix_latency(times,(yaw<=5)&(w<=.10),r['completed'],start=0.,end=8.)
    r.update(dwell_full=timed_diagnostics(rows,observer.physics_rows),
        integral_diagnostics=integral_statistics(rows,.01),motor_diagnostics=motor_statistics(rows,observer.physics_rows,.01,.002),
        wrench_diagnostics=residual_statistics(observer.physics_rows),tail=r['windows']['tail'])
    if not r['completed']:
        for key in ('position_rmse_xy','position_rmse_z','position_rmse_total'):
            r['partial_'+key]=r[key]; r[key]=None
    return r


def compact(r):
    keys=('key','group','label','layout','case','completed','terminated','actual_duration_sec','end_reason',
          'recovery_xy_s','recovery_z_s','recovery_3d_s','recovery_position_attitude_s','yaw_recovery_s')
    out={k:r.get(k) for k in keys}
    for prefix,metrics in [('tail',r['tail']),('observed',r['observed_metrics'])]:
        for k in ('rmse_xy_m','rmse_z_m','offset_xy_m','mean_error_xy_m','mean_error_z_m','sway_xy_rms_m','sway_z_rms_m',
                  'max_xy_error_m','max_abs_z_error_m','max_tilt_deg','max_angular_speed_rad_s','yaw_error_max_abs_deg'):
            out[prefix+'_'+k]=None if metrics is None else metrics.get(k)
    for k in ('allocator_clipped','esc_boundary_union','policy_action_at_bound','integral_frozen'):
        out[k+'_seconds']=r['dwell_full'][k]['duration_any_s']
    return out


def plots(directory, results):
    from .plotting import _pyplot
    plt=_pyplot()
    for group,case in sorted({(r['group'],r['case']) for r in results}):
        selected=[r for r in results if r['group']==group and r['case']==case]
        for zoom in ([False,True] if group=='fault' else [False]):
            fig,axes=plt.subplots(7,1,figsize=(12,19),sharex=True)
            for r in selected:
                c=read_columns(directory/(r['key']+'.csv'));t=c.time_post
                color='C0' if r['label']=='A_best' else 'C1';ls='-' if r['layout']=='legacy' else '--'
                label=r['label']+' / '+r['layout'];e=vector(c,'position_error_world');q=vector(c,'quaternion',4)
                for ax,y in zip(axes[:5],(np.linalg.norm(e[:,:2],axis=1),e[:,2],np.degrees(c.yaw_error_rad),tilt_deg(q),np.linalg.norm(vector(c,'omega'),axis=1))):
                    ax.plot(t,y,color=color,ls=ls,label=label)
                for i in range(4):
                    axes[5].plot(t,vector(c,'user_motor_thrust_actual',4)[:,i],color=f'C{i}',ls=ls,label=label+f' / user M{i+1}',alpha=.8)
                axes[6].plot(t,np.linalg.norm(vector(c,'allocation_residual_xml',4)[:,:3],axis=1),color=color,ls=ls,label=label+' allocation')
                axes[6].plot(t,np.linalg.norm(vector(c,'actuator_response_residual_xml',4)[:,:3],axis=1),color=color,ls=ls,alpha=.45,label=label+' lag')
                if r['terminated']:
                    for ax in axes:ax.axvline(r['actual_duration_sec'],color=color,ls=':',alpha=.5)
            for ax,unit in zip(axes,('XY error (m)','signed Z error (m)','yaw error (deg)','tilt (deg)','angular speed (rad/s)','actual thrust (N), user order','moment residual norm (Nm)')):
                ax.set_ylabel(unit);ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
                if group=='fault':ax.axvline(5,color='k',ls=':')
            if zoom:axes[-1].set_xlim(4.8,8.)
            axes[-1].set_xlabel('post-state time (s); motor sample is last interval substep')
            fig.suptitle(group+' / '+case+' — native PPO, physical reaction layout comparison')
            fig.tight_layout();fig.savefig(directory/(group+'-'+case+('-fault-zoom' if zoom else '')+'.png'),dpi=140);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-config',type=Path,default=ROOT/'configs/eval_velocity_ab.yaml')
    p.add_argument('--user-config',type=Path,default=ROOT/'configs/eval_velocity_ab_user_frd.yaml')
    p.add_argument('--previous-results',type=Path,default=PREVIOUS)
    p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/runs')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    configs=configuration_pair(args.legacy_config,args.user_config)
    policies={layout:{x.provenance['label']:x for x in select_models(DEFAULT_RECORD,c)} for layout,c in configs.items()}
    previous=json.loads((args.previous_results/'manifest.json').read_text())
    previous_results={r['key']:r for r in json.loads((args.previous_results/'summary.json').read_text())}
    for label,name,percent in FAULTS:
        s=scenario_at(name,percent);old=previous['scenarios'][name]
        assert s.mass==old['mass'] and list(s.offset)==list(old['offset']) and s.motor_number==old['motor_number']
    for layout,ps in policies.items():
        for label,policy in ps.items():
            old=next(v for v in previous['models'] if v['label']==label)
            assert policy.provenance['sha256']==old['sha256'] and policy.provenance['path']==old['path']
    manifest=dict(status='running',rollouts_requested=30,nominal_requested=20,fault_requested=10,
        seed=42,deterministic=True,configs={k:c.resolved_dict() for k,c in configs.items()},
        models={k:[v.provenance for v in ps.values()] for k,ps in policies.items()},
        nominal_tests=[dict(name=n,goal_world=g,initial_native_yaw_deg=y) for n,g,y in NOMINAL],fault_tests=FAULTS,
        gain_fault=asdict(GAINS[1]),gain_nominal=asdict(GAINS[0]),thresholds=asdict(Thresholds()),
        attitude_thresholds=dict(tilt_deg=5,yaw_error_deg=5,omega_norm_rad_s=.10),
        timing='policy input t; physics intervals [t,t+dt); post-state t+dt; tail (horizon-2,horizon]; event at t=5 before inference',
        recovery='suffix within position/speed band through horizon, minimum 1 second; fault latency from 5 seconds',
        yaw_recovery='diagnostic yaw <=5deg AND angular speed <=0.10rad/s, same suffix rule; a 5deg initial condition may already qualify',
        physical_change='evaluation-only native reaction signs + matching allocator yaw; no PPO coordinate change or trim',
        previous_results=str(args.previous_results.resolve()),audit_basis=str(ROOT/'artifacts/runs/coordinate-audit-j7zi124r'),
        git=_git_metadata(ROOT),runs={})
    if args.dry_run:
        manifest['status']='dry_run';print(json.dumps(manifest,indent=2));return 0
    # Defense in depth: this entrypoint cannot train or update optimizer state.
    from stable_baselines3 import PPO
    import torch
    def forbidden(*a,**kw):raise RuntimeError('learning/optimizer update forbidden in layout evaluation')
    PPO.learn=PPO.train=forbidden;torch.optim.Adam.step=forbidden
    args.output_dir.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='motor-layout-',dir=args.output_dir));print('results:',directory,flush=True)
    protected_paths=set(json.loads((args.previous_results/'protected_hashes_before.json').read_text()))
    protected_paths.update(str(x) for x in args.previous_results.rglob('*') if x.is_file())
    protected_paths.update(str(x) for x in (ROOT/'crazyflie_rl').glob('*.py'))
    protected={str(x):sha256(x) for x in protected_paths}
    write_json(directory/'protected_hashes_before.json',protected)
    manifest['source_sha256']={str(x.relative_to(ROOT)):sha256(x) for x in (ROOT/'crazyflie_rl').glob('*.py')}
    manifest['xml_sha256']=sha256(configs['legacy'].paths.mujoco_xml)
    manifest['config_sha256']={str(x):sha256(x) for x in (ROOT/'configs').glob('*.yaml')}
    cmd=['python','compare_motor_layouts.py','--legacy-config',str(args.legacy_config.resolve()),'--user-config',str(args.user_config.resolve()),'--previous-results',str(args.previous_results.resolve()),'--output-dir',str(args.output_dir.resolve())]
    (directory/'rerun.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl '+shlex.join(cmd)+'\n')
    parameters={layout:{label:parameter_digest(v) for label,v in ps.items()} for layout,ps in policies.items()}
    results=[];checks=dict(reproduction={},signals={},independent_csv={})
    def save():
        write_json(directory/'manifest.json',manifest);write_json(directory/'summary.json',clean(results))
        flat_csv(directory/'summary.csv',[compact(r) for r in results]);write_json(directory/'verification.json',checks)
    def execute(layout,label,scenario,case,yaw=0.,percent=None,native_scenario=None):
        group='fault' if percent is not None else 'nominal'
        case_id=scenario.name+(f'-eta{percent:03d}' if percent is not None else '')
        key=label+'-'+case_id+'-'+layout
        config=condition_config(configs[layout],scenario)
        config=replace(config,environment=replace(config.environment,episode_sec=case.horizon))
        gain=GAINS[1] if group=='fault' else GAINS[0]
        mode='oracle' if group=='fault' else 'existing'
        controller=IntegralController(policies[layout][label],gain);observer=LayoutObserver(scenario,controller,yaw)
        rows,snapshot,error,reasons=run_case(config,case,controller,42,
            env_factory=partial(OracleAllocationEnv,allocator_mode=mode),observer=observer,
            observation_transform=controller.prepare_observation)
        write_rollout(directory/(key+'.csv'),rows);write_rollout(directory/(key+'-physics.csv'),observer.physics_rows)
        write_rollout(directory/(key+'-events.csv'),observer.events)
        if error:raise RuntimeError(key+': '+error)
        r=(analyze_run(rows,case,observer,error,reasons,config,percent) if group=='fault'
           else nominal_analysis(rows,case,observer,error,reasons,config))
        r.update(key=key,label=label,layout=layout,case=case_id,group=group)
        if group=='fault':
            r['tail']=r['tail58_60'];r['observed_metrics']=r['post5_observed_metrics']
            event=observer.events[0] if observer.events else None
            vals=[x for x in rows if x['time_post']>=5-1e-9]
            r['yaw_recovery_s']=suffix_latency([x['time_post'] for x in vals],
                (np.abs(np.degrees([x['yaw_error_rad'] for x in vals]))<=5)&(np.linalg.norm([x['omega'] for x in vals],axis=1)<=.10),r['completed'])
        else:r['observed_metrics']=r['windows']['full'] or r['partial_observed_windows']['full']
        verify_oracle(rows,observer.physics_rows,native_scenario or scenario,mode,case=case);verify_integral_rows(rows,gain)
        checks['signals'][key]=dict(current_eta=True,single_effectiveness_application=True,event_state_preserved=True,integral_rule=True,
                                  control_samples=len(rows),physics_samples=len(observer.physics_rows))
        if group=='fault':
            checks['independent_csv'][key]=verify_csv(directory,r)
        if layout=='legacy' and group=='fault':
            oldkey=label+'-'+scenario.name+f'-eta{percent:03d}';old=previous_results[oldkey]
            for field in ('completed','actual_duration_sec','end_reason','recovery_3d_s','recovery_position_attitude_s'):
                assert r[field]==old[field],(key,field,r[field],old[field])
            checks['reproduction'][key]=dict(control=compare_csv(args.previous_results/(oldkey+'.csv'),directory/(key+'.csv')),
                physics=compare_csv(args.previous_results/(oldkey+'-physics.csv'),directory/(key+'-physics.csv'),physics=True))
        manifest['runs'][key]=dict(resolved_config=config.resolved_dict(),initial_snapshot=snapshot,
            physical=observer.metadata,events=clean(observer.events),model=policies[layout][label].provenance['sha256'],
            scenario_native=asdict(native_scenario or scenario),scenario_exposed=asdict(scenario),
            description=case.description(.01),initial_yaw_native_deg=yaw)
        results.append(r);save()
        print(f'{len(results)}/30 {key}: end={r["actual_duration_sec"]:.2f}s {r["end_reason"]}; position={r["recovery_3d_s"]}, joint={r["recovery_position_attitude_s"]}',flush=True)
    try:
        save();static_validation(configs,directory);checks['static_validation_passed']=True;save()
        for layout in configs:
            for label in ('A_best','B_best'):
                for name,goal,yaw in NOMINAL:
                    s=Scenario(name);case=replace(make_case(s,8.),goal=goal)
                    execute(layout,label,s,case,yaw)
        for label,name,percent in FAULTS:
            native=scenario_at(name,percent)
            for layout in configs:
                s=native if layout=='legacy' else replace(native,motor_number=5-native.motor_number)
                execute(layout,label,s,make_case(s),percent=percent,native_scenario=native)
        plots(directory,results)
        assert len(results)==30
        assert all(sha256(x)==h for x,h in protected.items())
        assert all(parameter_digest(p)==parameters[n][k] for n,ps in policies.items() for k,p in ps.items())
        manifest.update(status='completed',protected_files_unchanged=True,protected_file_count=len(protected),policy_parameters_unchanged=True)
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        manifest['counts']={group:dict(evaluated=sum(r['group']==group for r in results),
            completed=sum(r['group']==group and r['completed'] for r in results),
            physical_termination=sum(r['group']==group and r['terminated'] for r in results),
            position_recovered=sum(r['group']==group and r['recovery_3d_s'] is not None for r in results),
            joint_recovered=sum(r['group']==group and r['recovery_position_attitude_s'] is not None for r in results)) for group in ('nominal','fault')}
        save();write_json(directory/'completion.json',dict(status=manifest['status'],counts=manifest['counts']))
        print('results:',directory,flush=True)
    return 0

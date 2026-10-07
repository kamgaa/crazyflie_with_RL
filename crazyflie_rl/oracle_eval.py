"""Fixed 5 scenarios x 2 frozen PPOs x 4 allocator/integral configurations."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path
import shlex
import tempfile

import numpy as np

from .artifacts import _git_metadata
from .config import load_config
from .dr_policy import sha256
from .dr_transfer import ROOT, EvaluationAdapter, Thresholds, run_case, validate_common_config, write_json, write_rollout
from .integral_controller import IntegralController
from .integral_eval import parameter_digest, read_columns, verify_integral_rows
from .integral_validation import (SCENARIOS as VALIDATION_SCENARIOS, Scenario, ScenarioObserver, GAINS,
    make_case, analyze as base_analyze, verify_signals, write_event_tables)
from .interactive_eval import evaluation_config
from .payload_motor_eval import DEFAULT_RECORD, condition_config, select_models
from .oracle_allocation import OracleAllocationEnv, mass_geometry, plant_geometry, static_hover

PREVIOUS = ROOT/'artifacts/runs/ab-integral-validation-lymyulpo'
SCENARIOS = (Scenario('nominal'),)+tuple(next(s for s in VALIDATION_SCENARIOS if s.name == name) for name in (
    'combined_motor2_70', 'combined_motor3_70', 'motor1_70_restore', 'combined_motor1_70_restore'))
CONFIGURATIONS = tuple((mode, gain) for mode in ('existing', 'oracle') for gain in GAINS)


class OracleObserver(ScenarioObserver):
    def on_reset(self, adapter):
        super().on_reset(adapter)
        env = adapter.env
        env.geometry = plant_geometry(env)
        self.metadata.update(whole_aircraft=mass_geometry(env), B0_allocator=env.B.copy(),
            B0_plant_xml=env.geometry['matrix'], allocator_mode=env.allocator_mode,
            wrench_reference='drone body origin in body coordinates; [Mx,My,Mz,Fz] [Nm,Nm,Nm,N]')
        self.metadata = json.loads(json.dumps(self.metadata, default=lambda value:value.tolist()))

    def before_step(self, adapter, step, t):
        used_before = adapter.env.allocator_efficiency.copy()
        count = len(self.events)
        super().before_step(adapter, step, t)
        adapter.env.sync_allocator()
        if len(self.events) > count:
            self.events[-1].update(allocator_efficiency_before=used_before,
                allocator_efficiency_after=adapter.env.allocator_efficiency.copy(),
                allocator_mode=adapter.env.allocator_mode)

    def after_step(self, env, row):
        super().after_step(env, row)
        row.update(env.allocation_diagnostics())
        row['motor_sample_time'] = self.physics_rows[-1]['physics_time']
        row['motor_sample_time_post'] = self.physics_rows[-1]['physics_time_post']


def residual_statistics(physics):
    if not physics: return None
    result = dict(sample_count=len(physics), moment_reference='body origin, body frame; no gravity/external torque')
    for key in ('allocation_residual_xml', 'actuator_response_residual_xml', 'total_rotor_residual_xml',
                'allocation_residual_b0', 'geometry_wrench_difference'):
        values = np.array([r[key] for r in physics])
        result[key] = dict(moment_rms_xyz_nm=np.sqrt(np.mean(values[:, :3]**2, axis=0)).tolist(),
            moment_norm_rms_nm=float(np.sqrt(np.mean(np.sum(values[:, :3]**2, axis=1)))),
            moment_norm_max_nm=float(np.max(np.linalg.norm(values[:, :3], axis=1))),
            force_rms_n=float(np.sqrt(np.mean(values[:, 3]**2))), force_max_abs_n=float(np.max(np.abs(values[:, 3]))))
    return result


def analyze(rows, case, observer, error, reasons, config):
    result = base_analyze(rows, case, observer, error, reasons, config)
    dt = 1/config.environment.policy_hz
    for group in ('windows', 'partial_observed_windows'):
        for metrics in result[group].values():
            if metrics is None: continue
            start, end = metrics['first_sample_s'], metrics['last_sample_s']
            selected = [r for r in rows if start-1e-9 <= r['time_post'] <= end+1e-9]
            attitude = np.array([r['attitude_deg'] for r in selected])
            metrics.update(roll_pitch_rms_deg=np.sqrt(np.mean(attitude[:, :2]**2, axis=0)).tolist(),
                roll_pitch_max_abs_deg=np.max(np.abs(attitude[:, :2]), axis=0).tolist(),
                wrench_diagnostics=residual_statistics([r for r in observer.physics_rows
                    if start-dt-1e-9 <= r['physics_time'] < end-1e-9]))
    result['wrench_diagnostics'] = residual_statistics(observer.physics_rows)
    result['fault_to_termination_s'] = (result['actual_duration_sec']-5
        if result['terminated'] and result['fault_applied'] else None)
    for e in result['event_results']:
        name = f"{e['kind']}_{e['start_s']:g}_{e['end_s']:g}"
        e['observed_metrics'] = result['windows'][name] or result['partial_observed_windows'][name]
        e['observed_metrics_partial'] = result['windows'][name] is None
        if e['before_event']:
            from .plotting import quaternion_to_euler_deg
            e['before_event']['attitude_deg'] = quaternion_to_euler_deg(e['before_event']['quaternion']).tolist()
    return result


def static_audit(configs):
    records, lookup = [], {}
    for scenario in SCENARIOS:
        env = OracleAllocationEnv(config=configs[scenario.name])
        try:
            adapter = EvaluationAdapter(env); adapter.reset_to_case_initial_state(make_case(scenario), 42)
            OracleObserver(scenario, IntegralController(None, GAINS[0])).on_reset(adapter)
            stages = [('initial', 0., np.ones(4))]
            eta = np.ones(4)
            for e in scenario.events():
                eta = eta.copy(); eta[e.motor_number-1] = e.effectiveness
                stages.append((e.kind, e.time, eta))
            for stage, time, efficiency in stages:
                key = (scenario.mass, *scenario.offset, *efficiency)
                if key not in lookup:
                    item = static_hover(env, efficiency)
                    item.update(id=f'static_{len(records):02d}', payload_mass_kg=scenario.mass,
                        payload_offset_body_xy_m=list(scenario.offset), scenario_stages=[])
                    lookup[key] = item; records.append(item)
                lookup[key]['scenario_stages'].append(dict(scenario=scenario.name, stage=stage, time_s=time))
        finally: env.close()
    return json.loads(json.dumps(records, default=lambda value:value.tolist()))


def flat_csv(path, records):
    def flatten(value, prefix='', row=None):
        if row is None: row = {}
        for k, v in value.items():
            name = prefix+k
            if isinstance(v, dict): flatten(v, name+'__', row)
            elif isinstance(v, (np.ndarray, list, tuple)): row[name] = json.dumps(v.tolist() if isinstance(v, np.ndarray) else v, default=lambda x:x.tolist())
            else: row[name] = v
        return row
    rows = [flatten(r) for r in records]
    names = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, names); writer.writeheader(); writer.writerows(rows)


def write_reports(directory, results):
    write_event_tables(directory, results)
    # Preserve the common summary schema, but include allocator identity in
    # every derived table so existing/oracle rows cannot be conflated.
    windows, integral, motor, events = [], [], [], []
    for r in results:
        identity = {k:r[k] for k in ('key','label','condition','configuration','allocator_mode','gain','completed','actual_duration_sec','end_reason')}
        for group in ('windows','partial_observed_windows'):
            for name, metrics in r[group].items():
                windows.append(dict(identity,window=name,partial=group.startswith('partial'),available=metrics is not None,metrics=metrics))
        integral.append(dict(identity,diagnostics=r['integral_diagnostics']))
        motor.append(dict(identity,diagnostics=r['motor_diagnostics'],wrench=r['wrench_diagnostics']))
        events.extend(dict(identity,**e) for e in r['event_results'])
    for name, data in (('window_metrics',windows),('integral_summary',integral),('motor_summary',motor),('event_metrics',events)):
        flat_csv(directory/(name+'.csv'),data)


def compare_csv(first, second, *, end=None, physics=False):
    a, b = read_columns(first), read_columns(second)
    clock = 'physics_time' if physics else 'time_post'
    am = np.ones(len(a[clock]), dtype=bool) if end is None else a[clock] <= end+1e-9
    bm = np.ones(len(b[clock]), dtype=bool) if end is None else b[clock] <= end+1e-9
    assert am.sum() == bm.sum(), 'different observed durations'
    columns = [k for k in a if k in b and a[k].dtype.kind in 'fbiu' and b[k].dtype.kind in 'fbiu']
    maximum = 0.
    for key in columns:
        x, y = a[key][am].astype(float), b[key][bm].astype(float)
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12, err_msg=key)
        if len(x): maximum = max(maximum, float(np.max(np.abs(x-y))))
    return dict(samples=int(am.sum()), numeric_columns=len(columns), max_abs_error=maximum)


def verify_oracle(rows, physics, scenario, mode, *, case=None):
    verify_signals(rows, physics, scenario, case=case)
    for r in physics:
        expected = r['motor_effectiveness'] if mode == 'oracle' else np.ones(4)
        np.testing.assert_array_equal(r['allocator_efficiency'], expected)
        np.testing.assert_array_equal(r['plant_efficiency'], r['motor_effectiveness'])
        np.testing.assert_allclose(r['allocation_residual_xml']+r['actuator_response_residual_xml'],
                                   r['total_rotor_residual_xml'], atol=1e-15)


def save_comparison_plots(directory, results):
    """Use the existing matplotlib backend, CSV schema, units and four-trace style."""
    from .plotting import _pyplot, _new_path
    plt = _pyplot()
    def vector(f, name, n=3): return np.column_stack([f[f'{name}_{i}'] for i in range(n)])
    for scenario in SCENARIOS:
        for label in ('A_best', 'B_best'):
            chosen = [r for r in results if r['condition']==scenario.name and r['label']==label]
            traces = [(r, read_columns(directory/(r['key']+'.csv'))) for r in chosen]
            windows = [('full', 0, 60)]
            if scenario.has_fault: windows.append(('fault_zoom', 4.8, 8))
            if scenario.restore: windows.append(('restore_zoom', 29.8, 35))
            motor = (scenario.motor_number or 1)-1
            for window, start, end in windows:
                fig, axes = plt.subplots(7, 2, figsize=(16, 23), sharex=True)
                for index, (r, f) in enumerate(traces):
                    kw = dict(color='C0' if r['allocator_mode']=='existing' else 'C1',
                        ls=':' if r['gain']=='no_integral' else '-', label=r['configuration'])
                    t, ti, tm = f.time_post, f.time, f.motor_sample_time; error = vector(f,'position_error_world')
                    series = [np.linalg.norm(error[:,:2],axis=1), error[:,2], f.attitude_deg_0, f.attitude_deg_1,
                        np.degrees(np.arccos(np.clip(1-2*(f.quaternion_1**2+f.quaternion_2**2),-1,1))), f.attitude_deg_2,
                        np.linalg.norm(vector(f,'xi_t')[:,:2],axis=1), f.xi_t_2,
                        np.linalg.norm(vector(f,'allocation_residual_xml')[:,:3],axis=1), f.allocation_residual_xml_3,
                        np.linalg.norm(vector(f,'actuator_response_residual_xml')[:,:3],axis=1), f.actuator_response_residual_xml_3,
                        f[f'motor_thrust_actual_{motor}'], 2*index+f.integral_frozen.astype(float)]
                    for j, (ax, y) in enumerate(zip(axes.flat, series)):
                        ax.plot(ti if j in (6,7,13) else tm if j in (8,9,10,11,12) else t, y, **kw)
                    axes[6,0].plot(tm, f[f'motor_thrust_nominal_{motor}'], color=kw['color'], ls=kw['ls'],alpha=.35)
                    axes[6,1].step(ti,2*index+f.interval_allocator_clipping.astype(float),where='post',
                        color=kw['color'],ls='--',alpha=.4)
                    if r['terminated']:
                        for ax in axes.flat: ax.axvline(r['actual_duration_sec'],color=kw['color'],ls=kw['ls'],alpha=.4)
                names = ['True XY error [m]', 'Signed true Z error [m]', 'Roll [deg]', 'Pitch [deg]',
                    'Tilt [deg]','Yaw error [deg]','xi XY norm [m]','xi Z [m]',
                    'Allocation residual moment norm [Nm]','Allocation residual force [N]',
                    'Actuator residual moment norm [Nm]','Actuator residual force [N]',
                    f'Motor {motor+1} actual / faint nominal [N]','Freeze / dashed clip (stacked 0/1)']
                for ax, name in zip(axes.flat,names):
                    ax.set_ylabel(name); ax.grid(alpha=.25); ax.set_xlim(start,end)
                    for e in scenario.events(): ax.axvline(e.time,color='k',ls='--',lw=.8,label=f'{e.kind} {e.time:g}s')
                    ax.legend(fontsize=6,ncol=2)
                for ax in axes[-1]: ax.set_xlabel('state: t+dt; input/xi: t; wrench: last physics interval [s]')
                fig.suptitle(f'{scenario.name} / {label} / {window}; fixed integral 0.20; oracle knows ONLY effectiveness')
                fig.tight_layout(rect=(0,0,1,.98))
                fig.savefig(_new_path(directory/f'{scenario.name}-{label}-{window}.png'),dpi=130); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/eval_velocity_ab.yaml')
    parser.add_argument('--previous-results',type=Path,default=PREVIOUS)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/runs')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args(argv)
    previous = json.loads((args.previous_results/'manifest.json').read_text())
    config = evaluation_config(load_config(args.config)); validate_common_config(config)
    if previous['status']!='completed' or config.resolved_dict()!=previous['common_resolved_config']:
        raise ValueError('common evaluation config differs from completed independent validation')
    for s in SCENARIOS[1:]:
        if dict(**asdict(s),events=[asdict(e) for e in s.events()]) != previous['scenarios'][s.name]:
            # JSON converts tuples to lists; compare canonical JSON instead.
            if json.dumps(dict(**asdict(s),events=[asdict(e) for e in s.events()]),sort_keys=True)!=json.dumps(previous['scenarios'][s.name],sort_keys=True):
                raise ValueError('scenario or event differs from prior validation: '+s.name)
    policies = select_models(DEFAULT_RECORD, config)
    for policy in policies:
        old = next(p for p in previous['models'] if p['label']==policy.provenance['label'])
        for key in ('path','sha256','normalization','observation_contract'):
            if policy.provenance[key]!=old[key]: raise ValueError('model contract mismatch: '+key)
    configs = {s.name: replace(condition_config(config,s),environment=replace(condition_config(config,s).environment,episode_sec=60.)) for s in SCENARIOS}
    manifest = dict(status='dry_run' if args.dry_run else 'running',expected_rollouts=40,
        seed=42,deterministic=True,models=[p.provenance for p in policies],checkpoint_timesteps={p.provenance['label']:int(p.model.num_timesteps) for p in policies},
        previous_results=str(args.previous_results.resolve()),selection_record=str(DEFAULT_RECORD),
        git=_git_metadata(ROOT),source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'crazyflie_rl').glob('*.py')},
        common_resolved_config=config.resolved_dict(),resolved_configs={k:v.resolved_dict() for k,v in configs.items()},
        scenarios={s.name:dict(**asdict(s),events=[asdict(e) for e in s.events()]) for s in SCENARIOS},
        configurations=[dict(allocator=mode,integral=asdict(gain)) for mode,gain in CONFIGURATIONS],thresholds=asdict(Thresholds()),
        timing=previous['timing'],normalization_order=previous['preprocessing'],runs={},
        oracle_contract='only pinv(B0*eta) replaces B0 inverse; exact B0 inverse retained at all-ones; no division afterward',
        diagnostic_wrench_contract='body-origin body-frame XML rotor wrench, excluding gravity/external torque; preserve old B0-based fields separately',
        static_contract='full subtree mass/COM; XML matrix is distinguished from B0 allocator geometry; no computed trim in any rollout',
        limitations=['Oracle efficiency is diagnostic privileged information, not an estimator.',
                     'One deterministic rollout per fixed combination; no stability or success-probability claim.'])
    if args.dry_run: print(json.dumps(manifest,indent=2)); return 0
    args.output_dir.mkdir(parents=True,exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='ab-oracle-',dir=args.output_dir)); print('results:',directory,flush=True)
    write_json(directory/'manifest.json',manifest); write_json(directory/'scenarios.json',manifest['scenarios'])
    command=['python','compare_ab_oracle.py','--config',str(args.config.resolve()),'--previous-results',str(args.previous_results.resolve()),'--output-dir',str(args.output_dir.resolve())]
    (directory/'rerun.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl '+shlex.join(command)+'\n')
    protected_paths=set(json.loads((args.previous_results/'protected_hashes_before.json').read_text()))
    protected_paths.update(str(p) for p in args.previous_results.rglob('*') if p.is_file())
    protected_paths.update(str(p) for p in (ROOT/'configs').rglob('*.yaml'))
    protected_paths.update(str(ROOT/'crazyflie_rl'/name) for name in ('environment.py','actuators.py','motor_degradation.py','integral_controller.py','training.py'))
    protected={p:sha256(p) for p in protected_paths}
    write_json(directory/'protected_hashes_before.json',protected)
    parameters={p.provenance['label']:parameter_digest(p) for p in policies}
    manifest['parameter_sha256_before']=parameters
    results=[]; comparisons={}
    try:
        statics=static_audit(configs); write_json(directory/'static_hover.json',statics); flat_csv(directory/'static_hover.csv',statics)
        assert len(statics)==8
        for scenario in SCENARIOS:
            case=make_case(scenario); initial=None
            for policy in policies:
                for mode,gain in CONFIGURATIONS:
                    label=policy.provenance['label']; configuration=mode+'_'+gain.name
                    key=scenario.name+'-'+label+'-'+configuration
                    controller=IntegralController(policy,gain); observer=OracleObserver(scenario,controller)
                    rows,snapshot,error,reasons=run_case(configs[scenario.name],case,controller,42,
                        env_factory=partial(OracleAllocationEnv,allocator_mode=mode),observer=observer,
                        observation_transform=controller.prepare_observation)
                    for row in rows: row.update(model_label=label,condition=scenario.name,integral_mode=gain.name,configuration=configuration)
                    write_rollout(directory/(key+'.csv'),rows); write_rollout(directory/(key+'-physics.csv'),observer.physics_rows)
                    if observer.events:write_rollout(directory/(key+'-events.csv'),observer.events)
                    else:(directory/(key+'-events.csv')).write_text('event,control_step,policy_input_time,simulation_time\n')
                    result=analyze(rows,case,observer,error,reasons,configs[scenario.name])
                    result.update(label=label,key=key,allocator_mode=mode,configuration=configuration);results.append(result)
                    manifest['runs'][key]=dict(initial_snapshot=snapshot,physical=observer.metadata,events_applied=len(observer.events),error=error)
                    write_reports(directory,results);write_json(directory/'manifest.json',manifest)
                    if error:raise RuntimeError(key+': '+error)
                    if initial is not None and initial!=snapshot:raise AssertionError('scenario initial state differs')
                    initial=snapshot
                    verify_oracle(rows,observer.physics_rows,scenario,mode);verify_integral_rows(rows,gain)
                    if mode=='existing':
                        olddir=(args.previous_results if scenario.name!='nominal' else Path(previous['previous_results']))
                        oldkey=scenario.name+'-'+label+'-'+gain.name
                        comparisons[key]=compare_csv(olddir/(oldkey+'.csv'),directory/(key+'.csv'))
                        comparisons[key+'-physics']=compare_csv(olddir/(oldkey+'-physics.csv'),directory/(key+'-physics.csv'),physics=True)
                    else:
                        basekey=scenario.name+'-'+label+'-existing_'+gain.name
                        end=None if scenario.name=='nominal' else 5.
                        comparisons[key]=compare_csv(directory/(basekey+'.csv'),directory/(key+'.csv'),end=end)
                        comparisons[key+'-physics']=compare_csv(directory/(basekey+'-physics.csv'),directory/(key+'-physics.csv'),end=None if end is None else end-.002,physics=True)
                    write_json(directory/'regression_comparisons.json',comparisons)
                    print(f'{len(results)}/40 {key}: completed={result["completed"]} duration={result["actual_duration_sec"]:.2f} end={result["end_reason"]}',flush=True)
                    del rows,observer,controller
        save_comparison_plots(directory,results)
        assert all(sha256(p)==h for p,h in protected.items())
        assert all(parameter_digest(p)==parameters[p.provenance['label']] for p in policies)
        manifest.update(status='completed',completed_rollouts=len(results),flights_completed=sum(r['completed'] for r in results),
            protected_files_unchanged=True,protected_file_count=len(protected),policy_parameters_unchanged=True,
            verification_passed=True,regression_comparisons='regression_comparisons.json')
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        write_reports(directory,results);write_json(directory/'manifest.json',manifest)
        write_json(directory/'completion.json',{k:manifest.get(k) for k in ('status','expected_rollouts','completed_rollouts',
            'flights_completed','error','protected_files_unchanged','policy_parameters_unchanged')})
        print('results:',directory,flush=True)
    return 0

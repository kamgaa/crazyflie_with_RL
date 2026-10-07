"""Empirical recovery of the existing delayed plant + frozen PPO/oracle/integral.

Only candidate scheduling and offline metrics are new. All physics, control,
initialization, event handling and primary recovery definitions are reused.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
from .integral_validation import Scenario, Event, GAINS, make_case
from .interactive_eval import evaluation_config
from .oracle_allocation import OracleAllocationEnv, static_hover
from .oracle_eval import SCENARIOS, OracleObserver, analyze as oracle_analyze, compare_csv, flat_csv, verify_oracle
from .payload_motor_eval import DEFAULT_RECORD, condition_config, select_models, recovery_time

PREVIOUS = ROOT/'artifacts/runs/ab-oracle-0r_ekf0x'
COMBINATIONS = (('A_best', 'combined_motor2_70'), ('B_best', 'combined_motor3_70'), ('A_best', 'combined_motor3_70'))
BASE_PERCENT = (100, 95, 90, 85, 80, 75, 70)
GAIN = GAINS[1]


@dataclass(frozen=True)
class AttitudeCriteria:
    tilt_deg: float = 5.
    yaw_abs_deg: float = 5.
    angular_speed_rad_s: float = .10


@dataclass(frozen=True)
class EfficiencyScenario(Scenario):
    efficiency_percent: int = 70

    def __post_init__(self):
        if type(self.efficiency_percent) is not int or not 1 <= self.efficiency_percent <= 100:
            raise ValueError('efficiency must be an integer percentage in [1,100]')
        if self.restore or self.target_after30 is not None or self.motor_number not in (2, 3):
            raise ValueError('only the specified one-event motor2/motor3 scenarios are permitted')

    def events(self):
        # A 100% event is an explicitly logged no-op, never a fault recovery.
        return (Event(5., 'fault', self.motor_number, self.efficiency_percent/100),)


def scenario_at(name, percent):
    source = next(s for s in SCENARIOS if s.name == name)
    return EfficiencyScenario(**asdict(source), efficiency_percent=percent)


def clean(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist(), allow_nan=False))


def tilt_deg(quaternions):
    q = np.asarray(quaternions)
    return np.degrees(np.arccos(np.clip(1-2*(q[..., 1]**2+q[..., 2]**2), -1, 1)))


def suffix_latency(times, good, completed, *, start=5., end=60., minimum=1.):
    if not completed or not len(times):
        return None
    times, good = np.asarray(times), np.asarray(good, dtype=bool)
    mask = (times >= start-1e-9) & (times <= end+1e-9)
    times, good = times[mask], good[mask]
    suffix = np.logical_and.accumulate(good[::-1])[::-1]
    found = np.flatnonzero(suffix & (end-times >= minimum-1e-9))
    return float(times[found[0]]-start) if len(found) else None


def joint_recovery(rows, event, completed, criteria=AttitudeCriteria()):
    if event is None or not completed:
        return None
    initial = dict(time_post=5., position_error_world=event['e_true_after'], velocity=event['velocity'],
                   quaternion=event['quaternion'], omega=event['omega'])
    # All cases have fixed yaw target zero; use the same wrapped Euler-yaw convention.
    from .plotting import quaternion_to_euler_deg
    initial['yaw_error_rad'] = float(np.radians(quaternion_to_euler_deg(initial['quaternion'])[2]))
    selected = [initial]+[r for r in rows if r['time_post'] > 5+1e-9]
    e, v = (np.array([r[k] for r in selected]) for k in ('position_error_world', 'velocity'))
    yaw = np.array([r['yaw_error_rad'] for r in selected])
    good = ((np.linalg.norm(e, axis=1) <= .005) & (np.linalg.norm(v, axis=1) <= .02)
            & (tilt_deg([r['quaternion'] for r in selected]) <= criteria.tilt_deg)
            & (np.abs(np.degrees(yaw)) <= criteria.yaw_abs_deg)
            & (np.linalg.norm([r['omega'] for r in selected], axis=1) <= criteria.angular_speed_rad_s))
    return suffix_latency([r['time_post'] for r in selected], good, completed)


def weighted_dwell(flags, intervals):
    """Per-channel and union dwell from actual recorded interval lengths."""
    flags = np.asarray(flags, dtype=bool)
    if flags.ndim == 1: flags = flags[:, None]
    dt = np.asarray(intervals)
    if len(flags) != len(dt) or np.any(dt <= 0): raise ValueError('invalid sample intervals')
    run = np.zeros(flags.shape[1]); longest = run.copy(); union_run = union_longest = 0.
    for mask, duration in zip(flags, dt):
        run = np.where(mask, run+duration, 0.); longest = np.maximum(longest, run)
        union_run = union_run+duration if mask.any() else 0.
        union_longest = max(union_longest, union_run)
    return dict(duration_s_per_channel=np.sum(flags*dt[:, None], axis=0).tolist(),
                longest_continuous_s_per_channel=longest.tolist(),
                duration_any_s=float(np.sum(dt[np.any(flags, axis=1)])),
                longest_continuous_any_s=float(union_longest))


def timed_diagnostics(rows, physics):
    result = {}
    for prefix, clock, finish, records, keys in (
        ('physics', 'physics_time', 'physics_time_post', physics,
         ('allocator_clipped', 'allocator_lower', 'allocator_upper', 'esc_lower', 'esc_upper')),
        ('control', 'time', 'time_post', rows,
         ('policy_action_at_bound', 'integral_frozen', 'interval_allocator_clipping', 'interval_esc_boundary',
          'interval_action_boundary', 'integral_xy_at_limit', 'integral_z_at_limit'))):
        if not records: continue
        dt = np.array([r[finish]-r[clock] for r in records])
        result[prefix+'_interval_sum_s'] = float(dt.sum())
        for key in keys: result[key] = weighted_dwell([r[key] for r in records], dt)
        if prefix == 'physics':
            result['esc_boundary_union'] = weighted_dwell([r['esc_lower']|r['esc_upper'] for r in records], dt)
    return result


def recovery_signature(result):
    # 100% is not a fault, but use its measured suffix position criterion to
    # detect changes to a neighbouring fault candidate without treating null as failure.
    return bool(result['terminated']), bool(result['position_criterion_met'])


def refinement_candidates(results):
    ordered = sorted((r for r in results if r.get('evaluated')), key=lambda r:r['efficiency_percent'])
    existing = {r['efficiency_percent'] for r in results}
    extra = set()
    for low, high in zip(ordered, ordered[1:]):
        if recovery_signature(low) != recovery_signature(high):
            extra.update(range(low['efficiency_percent']+1, high['efficiency_percent']))
    return sorted(extra-existing, reverse=True)


def analyze_run(rows, case, observer, error, reasons, config, percent):
    r = oracle_analyze(rows, case, observer, error, reasons, config)
    event = observer.events[0] if observer.events else None
    existing = r['event_results'][0]['recovery_s'].copy()
    joint = joint_recovery(rows, event, r['completed'])
    r.update(efficiency_percent=percent, efficiency=percent/100, evaluated=True,
             fault_scheduled=percent != 100, fault_applied=percent != 100 and event is not None,
             event_is_noop=percent == 100, position_criterion_met=existing['3d'] is not None,
             joint_criterion_met=joint is not None)
    if percent == 100:
        recovery = dict.fromkeys(('xy', 'z', '3d'))
        r['event_results'][0].update(recovery_s=recovery, recovery_reason='not_applicable', kind='no_op_efficiency')
        r['recovery_reason'] = 'not_applicable'
        joint = None
    else:
        recovery = existing
        r['recovery_reason'] = ('physical_termination' if r['terminated'] else
                                'recovered' if recovery['3d'] is not None else 'position_not_recovered')
    r.update({f'recovery_{axis}_s':value for axis, value in recovery.items()})
    r.update(recovery_position_attitude_s=joint,
        recovered_within_5s=None if percent == 100 else recovery['3d'] is not None and recovery['3d'] <= 5+1e-9,
        recovered_within_10s=None if percent == 100 else recovery['3d'] is not None and recovery['3d'] <= 10+1e-9,
        recovery_position_attitude_reason=('not_applicable' if percent == 100 else 'physical_termination' if r['terminated']
            else 'recovered' if joint is not None else 'position_attitude_not_recovered'),
        category=('physical_termination' if r['terminated'] else 'completed_position_unrecovered' if not r['position_criterion_met']
            else 'position_only_recovered' if not r['joint_criterion_met'] else 'position_attitude_recovered'),
        classification_note='100% category is suffix criterion attainment, not fault recovery' if percent == 100 else None,
        dwell_full=timed_diagnostics(rows, observer.physics_rows),
        dwell_post5=timed_diagnostics([x for x in rows if x['time'] >= 5-1e-9],
                                     [x for x in observer.physics_rows if x['physics_time'] >= 5-1e-9]),
        pre_event_snapshot=clean(event) if event else None,
        post5_observed_metrics=r['event_results'][0]['observed_metrics'],
        post5_metrics_partial=not r['completed'], tail58_60=r['windows']['tail_58_60'])
    if not r['completed']:
        r['tail58_60'] = None; r['windows']['tail_58_60'] = None
        assert recovery['3d'] is None and joint is None
    return r


def compact(r):
    keys = ('key','combination','label','condition','efficiency_percent','stage','evaluated','static_feasible',
            'skip_reason','completed','terminated','actual_duration_sec','end_reason','category','recovery_reason',
            'recovery_xy_s','recovery_z_s','recovery_3d_s','recovery_position_attitude_s',
            'recovered_within_5s','recovered_within_10s','position_criterion_met','joint_criterion_met')
    result = {key:r.get(key) for key in keys}
    for prefix, metrics in (('post5', r.get('post5_observed_metrics')), ('tail58_60', r.get('tail58_60'))):
        for key in ('mean_error_xy_m','mean_error_z_m','offset_xy_m','rmse_xy_m','rmse_z_m','sway_xy_rms_m','sway_z_rms_m',
                    'max_xy_error_m','max_abs_z_error_m','max_tilt_deg','max_angular_speed_rad_s',
                    'yaw_error_mean_deg','yaw_error_max_abs_deg'):
            result[prefix+'__'+key] = metrics.get(key) if metrics else None
    for key in ('allocator_clipped','esc_boundary_union','policy_action_at_bound','integral_frozen',
                'integral_xy_at_limit','integral_z_at_limit'):
        for field in ('duration_any_s','longest_continuous_any_s'):
            result[key+'__'+field] = r.get('dwell_full', {}).get(key, {}).get(field)
    return result


def save_results(directory, results, statics, checks):
    write_json(directory/'summary.json', results); flat_csv(directory/'summary.csv', [compact(r) for r in results])
    write_json(directory/'static_hover.json', statics); flat_csv(directory/'static_hover.csv', statics)
    write_json(directory/'verification.json', checks)
    flat_csv(directory/'event_metrics.csv', [dict(key=r['key'], combination=r['combination'], efficiency_percent=r['efficiency_percent'], **e)
        for r in results for e in r.get('event_results', [])])
    flat_csv(directory/'window_metrics.csv', [dict(key=r['key'], window=name, partial=partial, metrics=metrics)
        for r in results for partial, group in ((False, 'windows'), (True, 'partial_observed_windows'))
        for name, metrics in r.get(group, {}).items()])


def boundary_report(results):
    report = {}
    for label, name in COMBINATIONS:
        key = label+'-'+name
        rs = sorted([r for r in results if r['combination']==key and r['evaluated']], key=lambda r:r['efficiency_percent'])
        adjacent = [dict(lower_percent=a['efficiency_percent'], upper_percent=b['efficiency_percent'],
                         lower_category=a['category'], upper_category=b['category'])
                    for a, b in zip(rs, rs[1:]) if recovery_signature(a) != recovery_signature(b)]
        recovered = [r['efficiency_percent'] for r in rs if r['efficiency_percent']<100 and r['position_criterion_met']]
        failures = [r['efficiency_percent'] for r in rs if r['terminated']]
        nonmonotonic = [dict(lower_success=a['efficiency_percent'], higher_failure=b['efficiency_percent'])
            for a in rs for b in rs if a['efficiency_percent'] < b['efficiency_percent']
            and a['position_criterion_met'] and not b['position_criterion_met']]
        report[key] = dict(recovered_percent=recovered, physical_termination_percent=failures,
            completed_unrecovered_percent=[r['efficiency_percent'] for r in rs if r['completed'] and not r['position_criterion_met']],
            position_only_percent=[r['efficiency_percent'] for r in rs if r['category']=='position_only_recovered'],
            transition_intervals=adjacent, nonmonotonic_pairs=nonmonotonic,
            lower_bound_note='recovered at evaluated lower bound 66%; actual boundary unknown' if 66 in recovered else None)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/eval_velocity_ab.yaml')
    parser.add_argument('--previous-results', type=Path, default=PREVIOUS)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/runs', help='Parent of a NEW unique result directory')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    previous = json.loads((args.previous_results/'manifest.json').read_text())
    prior_results = {r['key']:r for r in json.loads((args.previous_results/'summary.json').read_text())}
    config = evaluation_config(load_config(args.config)); validate_common_config(config)
    if previous['status'] != 'completed' or config.resolved_dict() != previous['common_resolved_config']:
        raise ValueError('common resolved configuration differs from previous oracle evaluation')
    for _, name in COMBINATIONS:
        source = next(s for s in SCENARIOS if s.name==name)
        expected = dict(**asdict(source), events=[asdict(e) for e in source.events()])
        if clean(expected) != previous['scenarios'][name]: raise ValueError('scenario changed: '+name)
    policies = {p.provenance['label']:p for p in select_models(DEFAULT_RECORD, config)}
    for label, policy in policies.items():
        old = next(p for p in previous['models'] if p['label']==label)
        for key in ('path','sha256','normalization','observation_contract'):
            if old[key] != policy.provenance[key]: raise ValueError('model contract changed: '+key)
    configs = {}
    for _, name in COMBINATIONS:
        conditioned = condition_config(config, scenario_at(name,70))
        configs[name] = replace(conditioned, environment=replace(conditioned.environment, episode_sec=60.))
    criteria = AttitudeCriteria()
    manifest = dict(status='dry_run' if args.dry_run else 'running', seed=42, deterministic=True,
        basic_requested_rollouts=21, combinations=[dict(label=l, scenario=n) for l,n in COMBINATIONS],
        base_efficiency_percent=list(BASE_PERCENT), extension_percent=[68,66], refinement_step_percent=1,
        extension_rule='only combinations recovered at 70%; sequential 68,66; static-infeasible candidates skipped',
        refinement_rule='enumerate every integer inside adjacent physical-termination or position-recovery transitions; no bisection',
        position_thresholds=asdict(Thresholds()), attitude_criteria=asdict(criteria),
        recovery_definition='suffix through 60s, >=1s; XY norm and |Z| each use 0.005m and respective velocity norm <=0.02m/s; 3D uses norms',
        no_fault_definition='100% event is a no-op; recovery times/within flags null, reason not_applicable; suffix attainment separately classified',
        timing=previous['timing'], metric_samples='post-state (5,60], event right-limit snapshot at 5 for recovery; tail (58,60]; physics intervals [t,t+dt)',
        saturation_dwell='sum recorded interval lengths; union across motors, not sum of rotor dwell',
        controller=dict(name='frozen PPO + oracle allocator + external 3-axis integral', integral=asdict(GAIN), allocator='B0 @ diag(current eta)'),
        common_resolved_config=config.resolved_dict(), resolved_configs={n:c.resolved_dict() for n,c in configs.items()},
        scenarios={n:previous['scenarios'][n] for _,n in COMBINATIONS},
        models=[p.provenance for p in policies.values()], checkpoint_timesteps={k:int(p.model.num_timesteps) for k,p in policies.items()},
        selection_record=str(DEFAULT_RECORD), previous_results=str(args.previous_results.resolve()),
        normalization_order=previous['normalization_order'], git=_git_metadata(ROOT),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'crazyflie_rl').glob('*.py')}, runs={})
    if args.dry_run:
        print(json.dumps(manifest, indent=2)); return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='oracle-recovery-', dir=args.output_dir))
    print('results:', directory, flush=True)
    write_json(directory/'manifest.json', manifest)
    cmd = ['python','compare_oracle_recovery.py','--config',str(args.config.resolve()),'--previous-results',str(args.previous_results.resolve()),'--output-dir',str(args.output_dir.resolve())]
    (directory/'rerun.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl '+shlex.join(cmd)+'\n')
    protected_paths = set(json.loads((args.previous_results/'protected_hashes_before.json').read_text()))
    protected_paths.update(str(p) for p in args.previous_results.rglob('*') if p.is_file())
    protected_paths.update(str(p) for p in (ROOT/'configs').rglob('*.yaml'))
    protected_paths.update(str(p) for p in (ROOT/'crazyflie_rl').glob('*.py'))
    protected_paths.update([str(DEFAULT_RECORD)]+[p.provenance['path'] for p in policies.values()])
    protected = {p:sha256(p) for p in protected_paths}
    write_json(directory/'protected_hashes_before.json', protected)
    parameters = {label:parameter_digest(p) for label,p in policies.items()}
    results, statics, static_cache, checks = [], [], {}, dict(reproduction70={}, prefixes={}, direct_signals={}, independent_csv={})

    def get_static(name, percent):
        scenario = scenario_at(name, percent); key=(scenario.mass,*scenario.offset,scenario.motor_number,percent)
        if key not in static_cache:
            env = OracleAllocationEnv(config=configs[name], allocator_mode='oracle')
            try:
                adapter = EvaluationAdapter(env); adapter.reset_to_case_initial_state(make_case(scenario),42)
                observer = OracleObserver(scenario,IntegralController(None,GAIN)); observer.on_reset(adapter)
                before = adapter.snapshot()
                eta = np.ones(4); eta[scenario.motor_number-1] = percent/100
                item = clean(static_hover(env,eta)); assert before==adapter.snapshot()
                item.update(id=f'static_{len(statics):03d}', condition=name, efficiency_percent=percent,
                            payload_mass_kg=scenario.mass, payload_offset_body_xy_m=list(scenario.offset))
                statics.append(item); static_cache[key]=item
            finally: env.close()
        return static_cache[key]

    def execute(label, name, percent, stage):
        combination = label+'-'+name; key=combination+f'-eta{percent:03d}'
        if any(r['key']==key for r in results): return next(r for r in results if r['key']==key)
        initial_static = get_static(name,100); stat = get_static(name,percent)
        identity=dict(key=key, combination=combination, label=label, condition=name, efficiency_percent=percent,
                      stage=stage, static_feasible=stat['feasible_static_equilibrium'], static_id=stat['id'])
        if not identity['static_feasible']:
            result=dict(identity,evaluated=False,skip_reason='static_equilibrium_exceeds_output_limits')
            results.append(result); save_results(directory,results,statics,checks); return result
        s=scenario_at(name,percent); case=make_case(s)
        controller=IntegralController(policies[label],GAIN); observer=OracleObserver(s,controller)
        rows,snapshot,error,reasons=run_case(configs[name],case,controller,42,
            env_factory=partial(OracleAllocationEnv,allocator_mode='oracle'),observer=observer,
            observation_transform=controller.prepare_observation)
        write_rollout(directory/(key+'.csv'),rows);write_rollout(directory/(key+'-physics.csv'),observer.physics_rows)
        write_rollout(directory/(key+'-events.csv'),observer.events)
        if error: raise RuntimeError(key+': '+error)
        r=analyze_run(rows,case,observer,error,reasons,configs[name],percent);r.update(identity)
        r.update(allocator_mode='oracle',configuration='oracle_integral_020')
        results.append(r)
        manifest['runs'][key]=dict(initial_snapshot=snapshot,physical=observer.metadata,events=clean(observer.events),
                                   static_initial_id=initial_static['id'],static_post_id=stat['id'])
        verify_oracle(rows,observer.physics_rows,s,'oracle');verify_integral_rows(rows,GAIN)
        np.testing.assert_allclose(np.diff([0.]+[x['time_post'] for x in rows]),1/config.environment.policy_hz,atol=1e-12)
        checks['direct_signals'][key]=dict(control_samples=len(rows),physics_samples=len(observer.physics_rows),
            current_eta_and_single_application=True,event_state_continuity=True,integral_recurrence=True)
        if percent==70:
            oldkey=name+'-'+label+'-oracle_integral_020'; old=prior_results[oldkey]
            for field in ('completed','actual_duration_sec','end_reason','termination_reasons'):
                if r[field]!=old[field]:raise AssertionError('70% reproduction '+field)
            assert r['event_results'][0]['recovery_s']==old['event_results'][0]['recovery_s']
            assert snapshot==previous['runs'][oldkey]['initial_snapshot']
            checks['reproduction70'][key]=dict(control=compare_csv(args.previous_results/(oldkey+'.csv'),directory/(key+'.csv')),
                physics=compare_csv(args.previous_results/(oldkey+'-physics.csv'),directory/(key+'-physics.csv'),physics=True),
                termination_and_recovery_match=True)
        else:
            prefixkey=combination+'-eta070'
            assert snapshot==manifest['runs'][prefixkey]['initial_snapshot']
            checks['prefixes'][key]=dict(control=compare_csv(directory/(prefixkey+'.csv'),directory/(key+'.csv'),end=5.),
                physics=compare_csv(directory/(prefixkey+'-physics.csv'),directory/(key+'-physics.csv'),end=5.-1/config.vehicle.physics_hz,physics=True))
        save_results(directory,results,statics,checks);write_json(directory/'manifest.json',manifest)
        print(f'{len(results)} {key}: {r["category"]}, end={r["actual_duration_sec"]:.2f}s, position_recovery={r["recovery_3d_s"]}, joint={r["recovery_position_attitude_s"]}',flush=True)
        return r

    try:
        # Reproduce first. These three count within the 21 base candidates.
        for label,name in COMBINATIONS: execute(label,name,70,'base')
        for label,name in COMBINATIONS:
            for percent in BASE_PERCENT:
                if percent!=70: execute(label,name,percent,'base')
        for label,name in COMBINATIONS:
            r=next(r for r in results if r['combination']==label+'-'+name and r['efficiency_percent']==70)
            if r.get('completed') and r.get('position_criterion_met'):
                for percent in (68,66):execute(label,name,percent,'extension')
        for label,name in COMBINATIONS:
            chosen=[r for r in results if r['combination']==label+'-'+name]
            for percent in refinement_candidates(chosen):execute(label,name,percent,'refinement')
        boundaries=boundary_report(results);write_json(directory/'boundaries.json',boundaries)
        from .oracle_recovery_artifacts import save_plots, verify_csv
        for r in results:
            if r['evaluated']:
                checks['independent_csv'][r['key']]=verify_csv(directory,r)
                print('CSV verified:',r['key'],flush=True)
        save_plots(directory,results,boundaries)
        assert all(sha256(p)==h for p,h in protected.items())
        assert all(parameter_digest(p)==parameters[label] for label,p in policies.items())
        manifest.update(status='completed',protected_files_unchanged=True,protected_file_count=len(protected),
                        policy_parameters_unchanged=True,verification_passed=True)
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        manifest.update(evaluated_rollouts=sum(r['evaluated'] for r in results),
            base_rollouts=sum(r['evaluated'] and r['stage']=='base' for r in results),
            additional_rollouts=sum(r['evaluated'] and r['stage']!='base' for r in results),
            static_skipped=sum(not r['evaluated'] for r in results),
            flights_completed=sum(r.get('completed',False) for r in results),
            physical_terminations=sum(r.get('terminated',False) for r in results),
            fault_position_recoveries=sum(r.get('recovery_3d_s') is not None for r in results),
            fault_joint_recoveries=sum(r.get('recovery_position_attitude_s') is not None for r in results))
        save_results(directory,results,statics,checks);write_json(directory/'manifest.json',manifest)
        write_json(directory/'completion.json',{k:v for k,v in manifest.items() if not isinstance(v,(dict,list))})
        print('results:',directory,flush=True)
    return 0

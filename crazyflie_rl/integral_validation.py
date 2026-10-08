"""Predeclared validation of frozen PPO + the unchanged external integral.

Exactly 13 scenarios x A/B x no_integral/integral_020. No search or training.
Boundary post-states belong to the preceding control interval (left reference
limit); event snapshots separately represent the instantaneous reference change.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shlex
import tempfile

import numpy as np

from .artifacts import _git_metadata
from .config import load_config
from .dr_policy import sha256
from .dr_transfer import (ROOT, Case, Thresholds, reference_sequence, run_case,
                          summarize, validate_common_config, write_json, write_rollout)
from .integral_controller import IntegralController, IntegralSettings
from .integral_eval import (IntegralObserver, integral_statistics, parameter_digest,
                           save_plots, verify_integral_rows, write_tables)
from .interactive_eval import evaluation_config
from .payload_motor_eval import (condition_config, motor_statistics, recovery_time,
                                 RecordedFaultEnv, segment_statistics, select_models)
from .velocity_reference import observation_contract, velocity_semantics
from .motor_layout import exposed_motor_index, user_from_native

PREVIOUS = ROOT/'artifacts/runs/ab-integral-m4i5pqhy'
COMMON_WINDOWS = {'full_0_60': (0., 60.), 'pre_3_5': (3., 5.), 'tail_18_20': (18., 20.),
                  'pre_second_28_30': (28., 30.), 'tail_58_60': (58., 60.)}
GAINS = (IntegralSettings('no_integral', 0., 0.), IntegralSettings('integral_020', .2, .2))


@dataclass(frozen=True)
class Event:
    time: float
    kind: str
    motor_number: int | None = None
    effectiveness: float | None = None
    target: tuple | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    mass: float = 0.
    offset: tuple = (0., 0.)
    motor_number: int | None = None
    restore: bool = False
    target_after30: tuple | None = None

    @property
    def has_fault(self):
        return self.motor_number is not None

    def events(self):
        events = [Event(5., 'fault', self.motor_number, .7)] if self.has_fault else []
        if self.restore: events.append(Event(30., 'restore', self.motor_number, 1.))
        if self.target_after30 is not None: events.append(Event(30., 'target_step', target=self.target_after30))
        return tuple(events)

    def event_intervals(self):
        events = self.events()
        return [(event, events[i+1].time if i+1 < len(events) else 60.) for i, event in enumerate(events)]


SCENARIOS = (
    Scenario('motor2_70', motor_number=2), Scenario('motor3_70', motor_number=3),
    Scenario('motor4_70', motor_number=4), Scenario('payload_pos_y', .005, (0., .03)),
    Scenario('payload_neg_x', .005, (-.03, 0.)), Scenario('payload_neg_y', .005, (0., -.03)),
    Scenario('combined_motor2_70', .005, (0., -.03), 2),
    Scenario('combined_motor3_70', .005, (-.03, 0.), 3),
    Scenario('combined_motor4_70', .005, (0., .03), 4),
    Scenario('motor1_70_restore', motor_number=1, restore=True),
    Scenario('combined_motor1_70_restore', .005, (.03, 0.), 1, restore=True),
    Scenario('combined70_step_x', .005, (.03, 0.), 1, target_after30=(.05, 0., 1.)),
    Scenario('combined70_step_y', .005, (.03, 0.), 1, target_after30=(0., .05, 1.)),
)


@dataclass(frozen=True)
class ScenarioCase(Case):
    scenario: Scenario | None = None

    def reference(self, t):
        goal, phase = self.goal, 'INITIAL'
        for event in self.scenario.events():
            if t >= event.time:
                phase = event.kind.upper()
                if event.target is not None: goal = event.target
        return np.asarray(goal), phase

    def post_reference(self, t):
        # t=30 post-state precedes the t=30 event and belongs to (5,30].
        # The next policy inference at exactly t=30 sees reference(30), the
        # right limit. No jump differentiation; all reference velocities are 0.
        return self.reference(np.nextafter(float(t), -np.inf))


def make_case(scenario, horizon=60.):
    return ScenarioCase(scenario.name, horizon, (0., 0., 1.), scenario=scenario)


class ScenarioObserver(IntegralObserver):
    """Only schedules existing effectiveness/reference APIs; plant is unchanged."""
    def on_reset(self, adapter):
        super().on_reset(adapter)
        env = adapter.env
        expected = (env._m0*env._ipos0 + self.condition.mass*np.r_[self.condition.offset, 0.])/(env._m0+self.condition.mass)
        np.testing.assert_allclose(env.model.body_ipos[env.drone_bid], expected, atol=1e-15)
        self.metadata.update(payload_frame='body frame, added point mass position relative to body origin',
                             composite_com_expected_body_m=expected.tolist())

    def before_step(self, adapter, step, t):
        env = adapter.env
        for event in self.condition.events():
            if step != round(event.time/adapter.control_dt): continue
            if abs(t-event.time) > 1e-12 or abs(env.data.time-event.time) > 1e-8:
                raise AssertionError('event not at the specified simulation control boundary')
            before = adapter.snapshot(); xi = self.controller.xi.copy()
            old_eff = env.motor_effectiveness.copy(); old_target = env.pos_des.copy()
            if event.kind in ('fault', 'restore'):
                env.motor_effectiveness = old_eff.copy()
                index = exposed_motor_index(event.motor_number, env.reaction_torque_layout)
                env.motor_effectiveness[index] = event.effectiveness
            else:
                adapter.set_reference(event.target)
            after = adapter.snapshot()
            # A target event changes reference + pure raw observation only.
            unchanged = [key for key in before if key not in ('reference', 'observation')]
            if any(before[key] != after[key] for key in unchanged):
                raise AssertionError('event reset/changed physical, actuator or episode state')
            np.testing.assert_array_equal(self.controller.xi, xi)
            np.testing.assert_array_equal(np.array(before['observation'])[3:], np.array(after['observation'])[3:])
            target = env.pos_des.copy(); position = np.array(before['position']); velocity = np.array(before['velocity'])
            self.events.append(dict(event=event.kind, control_step=step, policy_input_time=t,
                simulation_time=float(env.data.time), motor_number=event.motor_number,
                native_motor_index=(exposed_motor_index(event.motor_number, env.reaction_torque_layout)
                                    if event.motor_number is not None else None),
                user_motor_id=(4-exposed_motor_index(event.motor_number, env.reaction_torque_layout)
                               if event.motor_number is not None else None),
                reaction_torque_layout=env.reaction_torque_layout,
                user_effectiveness_before=user_from_native(old_eff),
                user_effectiveness_after=user_from_native(env.motor_effectiveness),
                effectiveness_before=old_eff, effectiveness_after=env.motor_effectiveness.copy(),
                target_before=old_target, target_after=target,
                p_cmd_before=old_target+xi, p_cmd_after=target+xi, xi_before=xi, xi_after=self.controller.xi.copy(),
                position=position, velocity=velocity, quaternion=np.array(before['quaternion']), omega=np.array(before['omega']),
                e_true_before=position-old_target, e_true_after=position-target,
                physical_and_actuator_state_unchanged=True, integral_state_preserved=True))
        self.physics_start = len(self.physics_rows)

    def after_step(self, env, row):
        super().after_step(env, row)
        # Fixed/event setpoints use the interval's held reference. Continuous
        # missions may evaluate the post-state against reference(t + dt).
        if getattr(self, 'fixed_reference', True):
            np.testing.assert_array_equal(row['reference_post'], env.pos_des)
        yaw = np.radians(row['attitude_deg'][2]) - env.yaw_des
        row['yaw_error_rad'] = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
        row['p_target_post'] = row['reference_post'].copy()
        row['xi_xy_norm_t'] = float(np.linalg.norm(row['xi_t'][:2]))
        row['xi_xy_norm_next'] = float(np.linalg.norm(row['xi_next'][:2]))


def yaw_statistics(rows):
    if not rows: return {}
    yaw = np.array([r['yaw_error_rad'] for r in rows])
    return dict(yaw_error_mean_rad=float(np.mean(yaw)), yaw_error_max_abs_rad=float(np.max(np.abs(yaw))),
                yaw_error_mean_deg=float(np.degrees(np.mean(yaw))),
                yaw_error_max_abs_deg=float(np.degrees(np.max(np.abs(yaw)))))


def interval_statistics(rows, physics, start, end, duration, config):
    selected = [r for r in rows if start+1e-9 < r['time_post'] <= end+1e-9]
    if not selected: return None, None
    metrics = segment_statistics(selected); metrics.update(yaw_statistics(selected))
    metrics['integral'] = integral_statistics(selected, 1/config.environment.policy_hz)
    selected_physics = [r for r in physics if start-1e-9 <= r['physics_time'] < end-1e-9]
    metrics['motors'] = motor_statistics(selected, selected_physics, 1/config.environment.policy_hz, 1/config.vehicle.physics_hz)
    complete = duration >= end-1e-9
    return (metrics, None) if complete else (None, metrics)


def event_recovery(rows, event_record, event, end, duration, thresholds, error=None):
    # Include the event's RIGHT-limit error at latency zero. Crucially, the
    # preceding rollout row at t=30 carries the OLD target and is excluded.
    # At the interval end include its LEFT-limit post-state before next event.
    if event_record is None or duration < end-1e-9 or (error and duration <= end):
        return dict.fromkeys(('xy', 'z', '3d'))
    initial = dict(time_post=event.time, position_error_world=event_record['e_true_after'], velocity=event_record['velocity'])
    selected = [initial]+[r for r in rows if event.time+1e-9 < r['time_post'] <= end+1e-9]
    return {axis: recovery_time(selected, thresholds, axis, True, horizon=end, start=event.time)
            for axis in ('xy', 'z', '3d')}


def analyze(rows, case, observer, error, reasons, config):
    scenario = case.scenario; thresholds = Thresholds()
    result = summarize(rows, case, thresholds, error, reasons)
    # The inherited overshoot is for a single fixed x target. Moving-target
    # overshoot is explicitly reported only for the commanded axis below.
    result.pop('overshoot_m', None)
    if not result['completed']:
        result['partial_observed_3d_rmse_m'] = result['position_rmse_total']
        for key in ('position_rmse_xy', 'position_rmse_z', 'position_rmse_total'): result[key] = None
    duration = result['actual_duration_sec']; windows = dict(COMMON_WINDOWS)
    for event, end in scenario.event_intervals(): windows[event.kind+f'_{event.time:g}_{end:g}'] = (event.time, end)
    full, partial = {}, {}
    for name, (start, end) in windows.items():
        full[name], partial[name] = interval_statistics(rows, observer.physics_rows, start, end, duration, config)
    result.update(condition=scenario.name, gain=observer.controller.settings.name,
                  integral_settings=asdict(observer.controller.settings), windows=full, partial_observed_windows=partial,
                  event_results=[], fault_scheduled=scenario.has_fault, fault_applied=any(e['event']=='fault' for e in observer.events))
    for event, end in scenario.event_intervals():
        record = next((e for e in observer.events if e['event'] == event.kind and e['policy_input_time'] == event.time), None)
        name = event.kind+f'_{event.time:g}_{end:g}'
        recovery = event_recovery(rows, record, event, end, duration, thresholds, error)
        entry = dict(kind=event.kind, start_s=event.time, end_s=end, applied=record is not None,
                     fully_observed=full[name] is not None, recovery_s=recovery,
                     before_event=None, motion_command_metrics=None)
        if record:
            entry['before_event'] = {k: np.asarray(record[k]).tolist() for k in (
                'e_true_before', 'velocity', 'xi_before', 'position', 'quaternion', 'omega',
                'target_before', 'target_after', 'p_cmd_before', 'p_cmd_after', 'effectiveness_before', 'effectiveness_after')}
            entry['before_event']['position_band_3d'] = bool(np.linalg.norm(record['e_true_before']) <= thresholds.position_band_m)
            entry['before_event']['speed_band_3d'] = bool(np.linalg.norm(record['velocity']) <= thresholds.speed_band_m_s)
            if event.kind == 'target_step':
                selected = [r for r in rows if event.time+1e-9 < r['time_post'] <= end+1e-9]
                if selected:
                    delta = record['target_after']-record['target_before']; axis = int(np.flatnonzero(delta)[0]); other = 1-axis
                    positions = np.vstack((record['position'], [r['position'] for r in selected]))
                    e = positions-record['target_after']; direction = np.sign(delta[axis])
                    entry['motion_command_metrics'] = dict(axis='xy'[axis], partial=full[name] is None,
                        overshoot_m=float(max(0., np.max(direction*e[:, axis]))),
                        uncommanded_axis='xy'[other], uncommanded_max_target_error_m=float(np.max(np.abs(e[:, other]))),
                        uncommanded_max_displacement_from_event_m=float(np.max(np.abs(positions[:, other]-record['position'][other]))),
                        max_abs_altitude_error_m=float(np.max(np.abs(e[:, 2]))))
        result['event_results'].append(entry)
    dt, physdt = 1/config.environment.policy_hz, 1/config.vehicle.physics_hz
    result['integral_diagnostics'] = integral_statistics(rows, dt)
    result['motor_diagnostics'] = motor_statistics(rows, observer.physics_rows, dt, physdt)
    return result


def verify_signals(rows, physics, scenario, *, case=None):
    # Vectorized direct signal checks on every physics interval (not inference
    # from trajectory). This uses the existing signals without changing them.
    if physics:
        time = np.array([r['physics_time'] for r in physics]); expected = np.ones((len(physics), 4))
        for event in scenario.events():
            if event.motor_number is not None: expected[time >= event.time-1e-9, event.motor_number-1] = event.effectiveness
        efficiency = np.array([r['motor_effectiveness'] for r in physics])
        np.testing.assert_array_equal(efficiency, expected)
        for stage, ctrl in (('thrust', 'applied_force_ctrl'), ('reaction', 'applied_torque_ctrl')):
            actual = np.array([r[f'motor_{stage}_actual'] for r in physics])
            nominal = np.array([r[f'motor_{stage}_nominal'] for r in physics])
            np.testing.assert_array_equal(actual, efficiency*nominal)
            np.testing.assert_array_equal(actual, [r[ctrl] for r in physics])
    for row in rows:
        np.testing.assert_allclose(row['policy_raw_velocity'], row['velocity_before'], atol=1e-7, rtol=1e-7)
        expected, _ = (case or make_case(scenario)).reference(row['time'])
        np.testing.assert_array_equal(row['reference'], expected)
        np.testing.assert_array_equal(row['reference_post'], expected)  # fixed over this control interval
        true_error = row['position']-expected
        desired = -4*true_error; speed = np.linalg.norm(desired)
        if speed > 1.5: desired *= 1.5/speed
        np.testing.assert_allclose(row['desired_velocity'], desired, atol=1e-14)
        np.testing.assert_allclose(row['internal_velocity_error'], row['velocity']-desired, atol=1e-14)


def write_event_tables(directory, results):
    write_tables(directory, results)
    flat = []
    for r in results:
        for e in r['event_results']:
            row = dict(label=r['label'], condition=r['condition'], gain=r['gain'],
                       **{k: v for k, v in e.items() if not isinstance(v, dict)})
            row.update({f'recovery_{k}_s': v for k, v in e['recovery_s'].items()})
            for group in ('before_event', 'motion_command_metrics'):
                for key, value in (e[group] or {}).items():
                    row[group+'__'+key] = json.dumps(value) if isinstance(value, list) else value
            flat.append(row)
    with (directory/'event_metrics.csv').open('w', newline='') as file:
        fields = list(dict.fromkeys(k for r in flat for k in r)); writer = csv.DictWriter(file, fields)
        writer.writeheader(); writer.writerows(flat)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/eval_velocity_ab.yaml')
    parser.add_argument('--previous-results', type=Path, default=PREVIOUS)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/runs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    previous = json.loads((args.previous_results/'manifest.json').read_text())
    if previous['status'] != 'completed': raise ValueError('expected completed prior integral experiment')
    config = evaluation_config(load_config(args.config)); validate_common_config(config)
    if velocity_semantics(config)['mode'] != 'absolute': raise ValueError('A/B require actual velocity observations')
    if config.resolved_dict() != previous['common_resolved_config']:
        raise ValueError('common config changed since the fixed-gain selection experiment')
    record = Path(previous['selection_record']); policies = select_models(record, config)
    for policy in policies:
        old = next(p for p in previous['models'] if p['label'] == policy.provenance['label'])
        for key in ('path', 'sha256', 'observation_contract', 'normalization'):
            if old[key] != policy.provenance[key]: raise ValueError('previous model contract/path/hash/normalization mismatch: '+key)
    specs = {s.name: condition_config(config, s) for s in SCENARIOS}
    specs = {name: replace(c, environment=replace(c.environment, episode_sec=60.)) for name, c in specs.items()}
    parameter_hashes = {p.provenance['label']: parameter_digest(p) for p in policies}
    manifest = dict(status='dry_run' if args.dry_run else 'running', expected_rollouts=52, seed=42, deterministic=True,
        controller='frozen PPO + unchanged external three-axis integral compensator; fixed validation, no gain search',
        previous_results=str(args.previous_results.resolve()), previous_manifest_sha256=sha256(args.previous_results/'manifest.json'),
        selection_record=str(record), models=[p.provenance for p in policies], git=_git_metadata(ROOT),
        checkpoint_timesteps={p.provenance['label']: int(p.model.num_timesteps) for p in policies},
        mujoco_xml=dict(path=str(config.paths.mujoco_xml), sha256=sha256(config.paths.mujoco_xml)),
        policy_parameter_sha256_before=parameter_hashes,
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (ROOT/'crazyflie_rl').glob('*.py')},
        common_resolved_config=config.resolved_dict(), resolved_configs={k: v.resolved_dict() for k, v in specs.items()},
        scenarios={s.name: dict(**asdict(s), events=[asdict(e) for e in s.events()]) for s in SCENARIOS},
        integral_settings=[asdict(g) for g in GAINS], thresholds=asdict(Thresholds()),
        observation_contract=observation_contract(config),
        preprocessing='native raw observation -> copy position error only -> existing frozen normalization/clipping -> deterministic policy',
        timing={'input': 'state/target/xi at t AFTER scheduled event; update integral once AFTER physics using that pre-state error',
                'post_state': 't+dt BEFORE events at that boundary; reference is the LEFT limit there',
                'windows': '(start,end] post-state samples; t=5 and t=30 belong to preceding interval',
                'event_snapshot': 'same physical time with before/after target, command, xi and effectiveness',
                'recovery': 'event RIGHT-limit sample + (event,next_event] LEFT-limit post-states; suffix through interval end, hold>=1s',
                'reference_velocity': 'zero throughout; target jump is regulation, no velocity impulse'},
        common_windows=COMMON_WINDOWS, runs={},
        limitations=['One deterministic execution per predeclared combination, not a success probability or stability proof.',
                     'No gain selection or added controller structure. Mean-centered RMS includes convergence trends.'])
    if args.dry_run: print(json.dumps(manifest, indent=2)); return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='ab-integral-validation-', dir=args.output_dir))
    print('results:', directory, flush=True)
    write_json(directory/'manifest.json', manifest)
    write_json(directory/'scenarios.json', manifest['scenarios'])
    command = ['python', 'validate_ab_integral.py', '--config', str(args.config.resolve()),
               '--previous-results', str(args.previous_results.resolve()), '--output-dir', str(args.output_dir.resolve())]
    (directory/'rerun.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+
        '\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl '+shlex.join(command)+'\n')
    protected = {str(p): sha256(p) for p in args.previous_results.rglob('*') if p.is_file()}
    old_protected = json.loads((args.previous_results/'protected_hashes_before.json').read_text())
    protected.update({p: sha256(p) for p in old_protected if Path(p).is_file()})
    for name in ('integral_controller.py', 'environment.py', 'actuators.py', 'motor_degradation.py', 'training.py'):
        path = ROOT/'crazyflie_rl'/name; protected[str(path)] = sha256(path)
    protected[str(config.paths.mujoco_xml)] = sha256(config.paths.mujoco_xml)
    write_json(directory/'protected_hashes_before.json', protected)
    results = []; common_initial = None
    try:
        for scenario in SCENARIOS:
            case = make_case(scenario); first_snapshot = None
            np.savez_compressed(directory/(scenario.name+'-reference.npz'), **reference_sequence(case, .01))
            for policy in policies:
                for gain in GAINS:
                    label = policy.provenance['label']; key = scenario.name+'-'+label+'-'+gain.name
                    controller = IntegralController(policy, gain); observer = ScenarioObserver(scenario, controller)
                    rows, snapshot, error, reasons = run_case(specs[scenario.name], case, controller, 42,
                        env_factory=RecordedFaultEnv, observer=observer, observation_transform=controller.prepare_observation)
                    for row in rows: row.update(model_label=label, condition=scenario.name, integral_mode=gain.name)
                    write_rollout(directory/(key+'.csv'), rows); write_rollout(directory/(key+'-physics.csv'), observer.physics_rows)
                    if observer.events: write_rollout(directory/(key+'-events.csv'), observer.events)
                    else: (directory/(key+'-events.csv')).write_text('event,control_step,policy_input_time,simulation_time,motor_number,target_before,target_after,p_cmd_before,p_cmd_after,xi_before,xi_after,effectiveness_before,effectiveness_after\n')
                    result = analyze(rows, case, observer, error, reasons, specs[scenario.name]); result.update(label=label, key=key)
                    results.append(result)
                    manifest['runs'][key] = dict(initial_snapshot=snapshot, physical=observer.metadata,
                        initial_xi=rows[0]['xi_t'].tolist() if rows else None, events_applied=len(observer.events), error=error)
                    write_event_tables(directory, results); write_json(directory/'manifest.json', manifest)
                    if error: raise RuntimeError(key+': '+error)
                    if first_snapshot is not None and snapshot != first_snapshot: raise AssertionError('scenario controller initial states differ')
                    first_snapshot = snapshot
                    comparable = {k: snapshot[k] for k in ('position', 'quaternion', 'velocity', 'omega', 'qpos', 'qvel',
                        'reference', 'observation', 'previous_action', '_last_f', '_last_omega', '_last_motor_cmd')}
                    if common_initial is not None and common_initial != comparable: raise AssertionError('nominal initialization changed across scenarios')
                    common_initial = comparable
                    verify_signals(rows, observer.physics_rows, scenario); verify_integral_rows(rows, gain)
                    print(f'{len(results)}/52 {key}: completed={result["completed"]} duration={result["actual_duration_sec"]:.2f} end={result["end_reason"]}', flush=True)
                    del rows, controller, observer
        save_plots(directory, results, 'integral_020', conditions=SCENARIOS,
                   event_markers={s.name: [(e.time, e.kind) for e in s.events()] for s in SCENARIOS},
                   motor_numbers={s.name: s.motor_number or 1 for s in SCENARIOS})
        assert all(sha256(p) == h for p, h in protected.items())
        assert all(parameter_digest(p) == parameter_hashes[p.provenance['label']] for p in policies)
        manifest.update(status='completed', completed_rollouts=len(results), flights_completed=sum(r['completed'] for r in results),
                        protected_files_unchanged=True, protected_file_count=len(protected), policy_parameters_unchanged=True,
                        initialization_event_and_signal_verification_passed=True)
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        write_event_tables(directory, results); write_json(directory/'manifest.json', manifest)
        write_json(directory/'completion.json', {k: manifest.get(k) for k in ('status', 'expected_rollouts', 'completed_rollouts',
            'flights_completed', 'error', 'protected_files_unchanged', 'policy_parameters_unchanged')})
        print('results:', directory, flush=True)
    return 0

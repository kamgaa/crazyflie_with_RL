"""Fixed 56-rollout development experiment; reuse payload/motor plant and logger."""
from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import asdict, replace
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
from .interactive_eval import evaluation_config
from .payload_motor_eval import (CONDITIONS, DEFAULT_RECORD, FaultObserver, RecordedFaultEnv,
    condition_config, dwell, motor_statistics, recovery_time, save_summary,
    segment_statistics, select_models, verify_signals, write_events)
from .velocity_reference import observation_contract, velocity_semantics

PREVIOUS = ROOT / 'artifacts/runs/ab-payload-motor-791j3j2g'
WINDOWS = {'full_0_20': (0., 20.), 'pre_3_5': (3., 5.), 'post_5_20': (5., 20.),
           'tail_18_20': (18., 20.), 'full_0_60': (0., 60.),
           'post_5_60': (5., 60.), 'tail_58_60': (58., 60.)}


class IntegralObserver(FaultObserver):
    def __init__(self, condition, controller):
        super().__init__(condition)
        self.controller = controller
        self.physics_start = 0

    def before_step(self, adapter, step, t):
        xi = self.controller.xi.copy()
        n = len(self.events)
        super().before_step(adapter, step, t)
        np.testing.assert_array_equal(self.controller.xi, xi)
        if len(self.events) > n:
            self.events[-1].update(xi_before=xi, xi_after=self.controller.xi.copy(), integral_state_preserved=True)
        self.physics_start = len(self.physics_rows)

    def after_step(self, env, row):
        super().after_step(env, row)
        physics = self.physics_rows[self.physics_start:]
        if len(physics) != env.substeps:
            raise AssertionError('missing or duplicated physics substeps')
        # These are the existing, separately logged command predicates and
        # tolerances. Actual/effective force and efficiency are never inputs.
        clipped = any(np.any(r['allocator_clipped']) for r in physics)
        esc = any(np.any(r['esc_lower'] | r['esc_upper']) for r in physics)
        action = bool(np.any(row['policy_action_at_bound']))
        row.update(self.controller.finish_step(allocator_clipping=clipped,
                                               esc_boundary=esc, action_boundary=action))
        row['e_true_post'] = row['position_error_world'].copy()


def window_statistics(rows, duration, error=None):
    full, partial = {}, {}
    for name, (start, end) in WINDOWS.items():
        selected = [r for r in rows if start + 1e-9 < r['time_post'] <= end + 1e-9]
        available = duration >= end - 1e-9
        full[name] = segment_statistics(selected) if available else None
        partial[name] = segment_statistics(selected) if selected and not available else None
    return full, partial


def integral_statistics(rows, dt):
    result = {}
    for key in ('integral_frozen', 'interval_allocator_clipping', 'interval_esc_boundary',
                'interval_action_boundary', 'integral_xy_at_limit', 'integral_z_at_limit',
                'integral_xy_projected', 'integral_z_projected'):
        result[key] = dwell([r[key] for r in rows], dt)
    result['xi_final_m'] = rows[-1]['xi_next'].tolist() if rows else None
    result['xi_max_xy_norm_m'] = float(max(np.linalg.norm(r['xi_next'][:2]) for r in rows)) if rows else None
    result['xi_max_abs_z_m'] = float(max(abs(r['xi_next'][2]) for r in rows)) if rows else None
    return result


def analyze(rows, case, condition, observer, error, reasons, thresholds, config):
    result = summarize(rows, case, thresholds, error, reasons)
    if not result['completed']:
        result['partial_observed_3d_rmse_m'] = result['position_rmse_total']
        for key in ('position_rmse_xy', 'position_rmse_z', 'position_rmse_total'):
            result[key] = None
    full, partial = window_statistics(rows, result['actual_duration_sec'], error)
    result.update(condition=condition.name, gain=observer.controller.settings.name,
                  integral_settings=asdict(observer.controller.settings), windows=full,
                  partial_observed_windows=partial, fault_scheduled=condition.has_fault,
                  fault_applied=bool(observer.events), recovery_xy_s=None, recovery_z_s=None,
                  recovery_3d_s=None, recovery_through20={}, mean_shift_xy_m=None, mean_shift_z_m=None)
    if condition.has_fault and observer.events:
        for dimension in ('xy', 'z', '3d'):
            result[f'recovery_{dimension}_s'] = recovery_time(rows, thresholds, dimension,
                result['completed'], horizon=case.horizon)
            result['recovery_through20'][dimension] = recovery_time(rows, thresholds, dimension,
                full['full_0_20'] is not None, horizon=20.)
        post = [r for r in rows if r['time_post'] > 5 + 1e-9]
        result['fault_observed_metrics'] = segment_statistics(post)
        pre, after = full['pre_3_5'], segment_statistics(post)
        if pre and after:
            result['mean_shift_xy_m'] = (np.array(after['mean_error_xy_m']) - pre['mean_error_xy_m']).tolist()
            result['mean_shift_z_m'] = after['mean_error_z_m'] - pre['mean_error_z_m']
        result['mean_shift_scope'] = '(3,5] vs (5,actual end]; target not redefined'
    dt, physics_dt = 1/config.environment.policy_hz, 1/config.vehicle.physics_hz
    result['motor_diagnostics'] = motor_statistics(rows, observer.physics_rows, dt, physics_dt)
    result['motor_diagnostics_post5'] = motor_statistics(
        [r for r in rows if r['time'] >= 5 - 1e-9],
        [r for r in observer.physics_rows if r['physics_time'] >= 5 - 1e-9], dt, physics_dt)
    result['integral_diagnostics'] = integral_statistics(rows, dt)
    return result


def verify_integral_rows(rows, settings):
    last = np.zeros(3)
    gain = np.array([settings.k_xy, settings.k_xy, settings.k_z])
    for r in rows:
        np.testing.assert_array_equal(r['xi_t'], last)
        np.testing.assert_allclose(r['e_true_before'], r['position_before']-r['reference'], atol=1e-15)
        np.testing.assert_array_equal(r['p_target'], r['reference'])
        np.testing.assert_array_equal(r['p_cmd'], r['p_target']+r['xi_t'])
        np.testing.assert_allclose(r['e_actor_before'], r['position_before']-r['p_cmd'], atol=1e-15)
        np.testing.assert_array_equal(r['policy_raw_observation'][3:], r['observation'][3:])
        np.testing.assert_allclose(r['policy_raw_observation'][:3], r['e_actor_before'], rtol=1e-7, atol=1e-8)
        np.testing.assert_array_equal(r['xi_candidate'], r['xi_t']-r['control_dt']*gain*r['e_true_before'])
        if r['integral_frozen'] or not settings.enabled:
            np.testing.assert_array_equal(r['xi_next'], r['xi_t'])
        assert np.linalg.norm(r['xi_next'][:2]) <= settings.xy_limit_m+1e-12
        assert abs(r['xi_next'][2]) <= settings.z_limit_m+1e-12
        last = r['xi_next']


class RolloutColumns(dict):
    """Small numeric CSV view; no optional dataframe dependency for evaluation."""
    def __getattr__(self, name):
        return self[name]


def read_columns(path):
    with Path(path).open(newline='') as file:
        reader = csv.reader(file)
        names = next(reader)
        columns = {name: [] for name in names}
        for row in reader:
            for name, value in zip(names, row): columns[name].append(value)
    result = RolloutColumns()
    for name, values in columns.items():
        if values and all(v in ('True', 'False') for v in values):
            result[name] = np.array([v == 'True' for v in values])
        else:
            try: result[name] = np.asarray(values, dtype=float)
            except ValueError: result[name] = np.asarray(values)
    return result


def compare_prefix(previous, current, *, physics=False):
    """All common numeric columns, except the intentionally changed horizon flag."""
    old, current_columns = read_columns(previous), read_columns(current)
    count = len(next(iter(old.values())))
    new = {k: v[:count] for k, v in current_columns.items()}
    if len(next(iter(new.values()))) != count:
        raise AssertionError('no_integral terminated before the old 20s prefix')
    columns = [c for c in old if c in new and c != 'truncated' and
               old[c].dtype.kind in 'fbiu' and new[c].dtype.kind in 'fbiu']
    for col in columns:
        np.testing.assert_allclose(new[col], old[col], rtol=1e-12, atol=1e-12, err_msg=col)
    for col in ('terminated',):
        if col in old:
            np.testing.assert_array_equal(new[col], old[col])
    if not physics:
        assert not new['truncated'].any() and old['truncated'][-1]
    maximum = max((float(np.max(np.abs(new[c].astype(float)-old[c].astype(float)))) for c in columns), default=0.)
    return dict(samples=count, numeric_columns=len(columns), max_abs_error=maximum,
                excluded='20s horizon truncated flag differs intentionally' if not physics else None)


def write_tables(directory, results):
    save_summary(directory, results)
    windows, integrals, motors = [], [], []
    for r in results:
        identity = {k: r[k] for k in ('label', 'condition', 'gain', 'completed', 'actual_duration_sec', 'end_reason')}
        for kind in ('windows', 'partial_observed_windows'):
            for name, metrics in r[kind].items():
                windows.append(dict(identity, window=name, partial=kind.startswith('partial'),
                                    available=metrics is not None, **(metrics or {})))
        integrals.append(dict(identity, **r['integral_diagnostics']))
        diagnostic = r['motor_diagnostics']
        if diagnostic:
            for motor in range(4):
                row = dict(identity, motor=motor+1)
                for key, value in diagnostic.items():
                    if isinstance(value, list) and len(value) == 4:
                        row[key] = value[motor]
                    elif isinstance(value, dict):
                        for field, items in value.items():
                            if isinstance(items, list) and len(items) == 4:
                                row[key+'__'+field] = items[motor]
                motors.append(row)
    for name, values in (('window_metrics', windows), ('integral_summary', integrals), ('motor_summary', motors)):
        flattened = [{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in values]
        with (directory/(name+'.csv')).open('w', newline='') as f:
            fields = list(dict.fromkeys(k for r in flattened for k in r))
            writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(flattened)


def gain_comparison(results):
    """Report fixed candidates; select one common gain only if all 14 flights finish.

    Development ranking: completion, true-target recovery/settling count, worst
    60s residual norm, transient error, freeze dwell. No new gain is generated.
    """
    comparisons = []
    for name in dict.fromkeys(r['gain'] for r in results):
        group = [r for r in results if r['gain'] == name]
        full = [r for r in group if r['completed']]
        row = dict(gain=name, trials=len(group), completed=len(full),
                   recovered_fault_3d=sum(r['recovery_3d_s'] is not None for r in group),
                   settled_3d=sum(r['settled'] for r in group),
                   freeze_total_s=sum(r['integral_diagnostics']['integral_frozen']['duration_any_s'] for r in group),
                   allocator_clip_total_s=sum(r['motor_diagnostics']['allocator_clipped']['duration_any_s'] for r in group),
                   longest_freeze_s=max(r['integral_diagnostics']['integral_frozen']['longest_continuous_s_per_channel'][0] for r in group),
                   worst_recovery_3d_s=max((r['recovery_3d_s'] for r in group if r['recovery_3d_s'] is not None), default=None))
        for window in ('tail_18_20', 'tail_58_60', 'full_0_60'):
            metrics = [r['windows'][window] for r in group if r['windows'][window]]
            row[window] = {key: max(m[key] for m in metrics) if metrics else None for key in
                ('offset_xy_m', 'offset_z_abs_m', 'sway_xy_rms_m', 'sway_z_rms_m',
                 'max_xy_error_m', 'max_abs_z_error_m', 'max_tilt_deg')}
        nominal = [r for r in group if r['condition'] == 'nominal']
        row['nominal'] = {key: max(r['windows']['full_0_60'][key] for r in nominal)
                          if all(r['completed'] for r in nominal) else None for key in
                          ('max_xy_error_m', 'max_abs_z_error_m', 'max_tilt_deg')}
        comparisons.append(row)
    eligible = [r for r in comparisons if r['gain'] != 'no_integral' and r['completed'] == 14]
    chosen = min(eligible, key=lambda r: (-r['settled_3d'], -r['recovered_fault_3d'],
        np.hypot(r['tail_58_60']['offset_xy_m'], r['tail_58_60']['offset_z_abs_m']),
        r['full_0_60']['max_xy_error_m'], r['freeze_total_s']))['gain'] if eligible else None
    return dict(common_development_candidate=chosen, gains=comparisons,
                selection_scope='same development conditions used for comparison; not independent validation',
                selection_rule='all 14 completed; then 3D settling/recovery count, worst tail residual, transient XY, freeze time')


def parameter_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.model.policy.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_plots(directory, results, chosen, *, conditions=CONDITIONS, event_markers=None, motor_numbers=None):
    """Reuse the plotting backend/schema; all candidates plus four-trace detail."""
    from .plotting import _pyplot, _new_path
    plt = _pyplot()
    styles = {'no_integral': ':', 'integral_005': '--', 'integral_010': '-.', 'integral_020': '-'}
    def vector(frame, name, n=3):
        return np.column_stack([frame[f'{name}_{i}'] for i in range(n)])
    for condition in conditions:
        markers = (event_markers[condition.name] if event_markers is not None
                   else [(5., 'fault')] if condition.has_fault else [])
        motor = (motor_numbers or {}).get(condition.name, 1) - 1
        event_title = '; '.join(f'{label} @ {time:g}s' for time, label in markers)
        def mark_events(ax):
            for time, label in markers:
                ax.axvline(time, color='k', ls='--' if time == 5 else '-.', lw=.8,
                           label=f'{label} t={time:g}s')
        traces = {}
        for r in results:
            if r['condition'] == condition.name:
                traces[(r['label'], r['gain'])] = read_columns(directory/(r['key']+'.csv'))
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        for (model, gain), f in traces.items():
            color = 'C0' if model == 'A_best' else 'C1'
            kw = dict(color=color, ls=styles[gain], label=model+' '+gain)
            t = f.time_post; e = vector(f, 'position_error_world')
            axes[0, 0].plot(t, np.linalg.norm(e[:, :2], axis=1), **kw)
            axes[0, 1].plot(t, e[:, 2], **kw)
            axes[1, 0].plot(t, np.linalg.norm(vector(f, 'xi_next')[:, :2], axis=1), **kw)
            axes[1, 1].plot(t, f.xi_next_2, **kw)
            if bool(f.terminated[-1]):
                for ax in axes.flat: ax.axvline(t[-1], color=color, ls=styles[gain], alpha=.5)
        labels = ['True XY error norm [m]', 'True signed Z error [m]', 'Integral XY norm [m]', 'Integral Z [m]']
        for ax, label in zip(axes.flat, labels):
            ax.set_ylabel(label); ax.grid(alpha=.25); ax.legend(fontsize=7)
            mark_events(ax)
        for ax in axes[-1]: ax.set_xlabel('post-state time [s]')
        fig.suptitle(condition.name+' — '+event_title); fig.tight_layout(rect=(0,0,1,.97))
        fig.savefig(_new_path(directory/(condition.name+'-all-gains.png')), dpi=140); plt.close(fig)
        if chosen is None: continue
        fig, axes = plt.subplots(9, 2, figsize=(17, 30), sharex=True)
        selected = {k: f for k, f in traces.items() if k[1] in ('no_integral', chosen)}
        for j, ((model, gain), f) in enumerate(selected.items()):
            color = 'C0' if model == 'A_best' else 'C1'; ls = ':' if gain == 'no_integral' else '-'
            label = model+' '+gain; kw = dict(color=color, ls=ls, label=label)
            t, ti = f.time_post, f.time
            e = vector(f, 'position_error_world'); xi = vector(f, 'xi_t')
            axes[0, 0].plot(t, np.linalg.norm(e[:, :2], axis=1), **kw)
            axes[0, 1].plot(t, e[:, 2], **kw)
            for i in range(3):
                axes[1+i, 0].plot(t, f[f'position_{i}'], **kw)
                axes[1+i, 0].plot(ti, f[f'p_cmd_{i}'], color=color, ls=ls, alpha=.35, label=label+' cmd')
                axes[1+i, 1].plot(ti, xi[:, i], **kw)
                axes[4+i, 0].plot(t, f[f'attitude_deg_{i}'], **kw)
                axes[4+i, 1].plot(t, f[f'velocity_{i}'], **kw)
            axes[7, 0].plot(t, np.linalg.norm(vector(f, 'omega'), axis=1), **kw)
            # Stack binary traces to make overlapping freeze/saturation visible.
            axes[7, 1].step(ti, 2*j+f.integral_frozen.astype(int), where='post', **kw)
            axes[7, 1].step(ti, 2*j+f.interval_allocator_clipping.astype(int), where='post',
                            color=color, ls='--', alpha=.4, label=label+' clip')
            for boundary, style in (('interval_esc_boundary', '-.'), ('interval_action_boundary', ':')):
                if f[boundary].any():
                    axes[7, 1].step(ti, 2*j+f[boundary].astype(int), where='post', color=color,
                                    ls=style, alpha=.6, label=label+' '+boundary)
            axes[8, 0].plot(ti, f[f'motor_effectiveness_{motor}'], **kw)
            axes[8, 1].plot(t, f[f'motor_thrust_nominal_{motor}'], color=color, ls=ls, alpha=.35, label=label+' nominal')
            axes[8, 1].plot(t, f[f'motor_thrust_actual_{motor}'], **kw)
            if bool(f.terminated[-1]):
                for ax in axes.flat: ax.axvline(t[-1], color=color, ls=ls, alpha=.5)
        labels = [('True XY error [m]', 'True Z error [m]')]+[(f'{a}: position / cmd [m]', f'xi {a} [m]') for a in 'xyz']+[
            ('Roll [deg]', 'Actual vx [m/s]'), ('Pitch [deg]', 'Actual vy [m/s]'), ('Yaw [deg]', 'Actual vz [m/s]'),
            ('Angular speed [rad/s]', 'Freeze / allocator clip (stacked 0/1)'), (f'Motor {motor+1} effectiveness', f'Motor {motor+1} nominal / actual [N]')]
        for pair, names in zip(axes, labels):
            for ax, name in zip(pair, names):
                ax.set_ylabel(name); ax.grid(alpha=.25); ax.legend(fontsize=6)
                mark_events(ax)
        if selected:
            reference_trace = next(iter(selected.values()))
            for i in range(3):
                axes[1+i, 0].step(reference_trace.time, reference_trace[f'p_target_{i}'],
                                  where='post', color='k', lw=.8, label='true target')
                axes[1+i, 0].legend(fontsize=6)
        for ax in axes[-1]: ax.set_xlabel('input/xi: t; physical state: t+dt [s]')
        fig.suptitle(f'{condition.name} — frozen PPO + external integral; {chosen}; {event_title}')
        fig.tight_layout(rect=(0,0,1,.98)); fig.savefig(_new_path(directory/(condition.name+'-representative.png')), dpi=130); plt.close(fig)
        # All four rotor stages remain distinct, no duplicated PID trace.
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        for (model, gain), f in selected.items():
            color = 'C0' if model == 'A_best' else 'C1'; ls = ':' if gain == 'no_integral' else '-'
            for i, ax in enumerate(axes):
                for stage, alpha in (('command', .2), ('nominal', .45), ('actual', 1.)):
                    ax.plot(f.time_post, f[f'motor_thrust_{stage}_{i}'], color=color, ls=ls, alpha=alpha,
                            label=f'{model} {gain} {stage}')
        for i, ax in enumerate(axes):
            ax.set_ylabel(f'Motor {i+1} [N]'); ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=3)
            mark_events(ax)
        axes[-1].set_xlabel('post-state / last substep time [s]')
        fig.suptitle(condition.name+' — '+event_title); fig.tight_layout(rect=(0,0,1,.97))
        fig.savefig(_new_path(directory/(condition.name+'-motors.png')), dpi=140); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/eval_velocity_ab.yaml')
    parser.add_argument('--record', type=Path, default=DEFAULT_RECORD)
    parser.add_argument('--previous-results', type=Path, default=PREVIOUS)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/runs')
    parser.add_argument('--xi-xy-limit', type=float, default=.40, help='XY norm projection radius [m]')
    parser.add_argument('--xi-z-limit', type=float, default=.15, help='independent Z absolute limit [m]')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    config = evaluation_config(load_config(args.config)); validate_common_config(config)
    if velocity_semantics(config)['mode'] != 'absolute': raise ValueError('A/B require actual velocity inputs')
    policies = select_models(args.record, config)
    parameter_hashes = {p.provenance['label']: parameter_digest(p) for p in policies}
    gains = [IntegralSettings(name, gain, gain, args.xi_xy_limit, args.xi_z_limit) for name, gain in
             (('no_integral', 0.), ('integral_005', .05), ('integral_010', .10), ('integral_020', .20))]
    case = Case('hover', 60., (0., 0., 1.)); thresholds = Thresholds()
    specs = {c.name: condition_config(config, c) for c in CONDITIONS}
    specs = {name: replace(c, environment=replace(c.environment, episode_sec=60.)) for name, c in specs.items()}
    previous = json.loads((args.previous_results/'manifest.json').read_text())
    for p in policies:
        old = next(m for m in previous['models'] if m['label'] == p.provenance['label'])
        if old['sha256'] != p.provenance['sha256']: raise ValueError('prior comparison used a different checkpoint')
    manifest = dict(status='dry_run' if args.dry_run else 'running',
        controller='frozen PPO policy + external three-axis integral compensator', seed=42, deterministic=True,
        models=[p.provenance for p in policies], selection_record=str(args.record.resolve()),
        policy_parameter_sha256_before=parameter_hashes,
        selection_record_sha256=sha256(args.record), previous_results=str(args.previous_results.resolve()),
        git=_git_metadata(ROOT), source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (ROOT/'crazyflie_rl').glob('*.py')},
        common_resolved_config=config.resolved_dict(), resolved_configs={k: v.resolved_dict() for k, v in specs.items()},
        conditions=[asdict(c) for c in CONDITIONS], integral_settings=[asdict(g) for g in gains],
        observation_contract=observation_contract(config), thresholds=asdict(thresholds), windows=WINDOWS,
        normalization_order='native raw obs -> COPY position channel p-(true target+xi); velocity unchanged -> frozen saved normalization/clipping -> deterministic policy',
        integral_rule='xi_candidate = xi_t - control_dt * diag(kxy,kxy,kz) * PRE-state true error; freeze ALL axes for any command saturation; otherwise project XY norm and Z independently',
        saturation={'allocator_clipping_tolerance_n': 1e-12, 'esc_boundary_tolerance': 1e-9,
                    'action_boundary_tolerance': 1e-9, 'scope': 'any physics substep within control interval; action at control step',
                    'excluded_inputs': 'payload, COM, effectiveness, actual thrust; no gain/bias/allocator adaptation'},
        timing={'observation': 'state/reference/xi_t at time=t; native env reward and returned obs use true target',
                'integral': 'after physics, once per control interval using e_true(t) and that interval saturation; xi_next used next interval',
                'physical': 'post-state at time_post=t+dt; true error/desired velocity metrics same post time',
                'windows': '(start,end] post-state samples; unobserved fixed windows null, partial stored separately',
                'recovery': 'true-target suffix through t=60, position5mm/speed.02m/s, >=1s; fault latency from5; optional through20 separate'},
        case=case.description(.01), runs={}, expected_rollouts=56,
        limitations=['Conservative conditional integration and state projection, not an anti-windup or stability guarantee.',
                     'One deterministic execution per combination. Gain selection uses this development set.',
                     'Mean-centered RMS can include drift; force margin does not prove controllability.'])
    if args.dry_run:
        print(json.dumps(manifest, indent=2)); return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='ab-integral-', dir=args.output_dir))
    print('results:', directory, flush=True)
    write_json(directory/'manifest.json', manifest)
    command = ['python', 'compare_ab_integral.py', '--config', str(args.config.resolve()), '--record', str(args.record.resolve()),
               '--previous-results', str(args.previous_results.resolve()), '--output-dir', str(args.output_dir.resolve()),
               '--xi-xy-limit', str(args.xi_xy_limit), '--xi-z-limit', str(args.xi_z_limit)]
    (directory/'rerun.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+
        '\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl '+shlex.join(command)+'\n')
    record = json.loads(args.record.read_text())
    roots = [args.previous_results, args.record.parent, Path(record['evaluation_dir'])]+[Path(r['run_dir']) for r in record['training'].values()]
    protected = {str(p): sha256(p) for root in roots for p in root.rglob('*') if p.is_file()}
    # Include earlier protected historical nominal/D artifacts without rehashing source files intentionally edited here.
    old_protection = args.record.parent/'preserved_hashes_before.json'
    if old_protection.exists():
        older = json.loads(old_protection.read_text())
        for path in older:
            p = Path(path)
            if p.is_file() and 'artifacts' in p.parts: protected[str(p)] = sha256(p)
    protected.update({str(p): sha256(p) for p in (ROOT/'configs').rglob('*.yaml')})
    for name in ('environment.py', 'actuators.py', 'motor_degradation.py', 'training.py'):
        p = ROOT/'crazyflie_rl'/name; protected[str(p)] = sha256(p)
    write_json(directory/'protected_hashes_before.json', protected)
    np.savez_compressed(directory/'reference.npz', **reference_sequence(case, .01))
    results, prefixes = [], {}; common_initial = None
    try:
        # Gate all compensated rollouts on the complete set of 14 old-prefix checks.
        for gain in gains:
            if gain.enabled and len(prefixes) != 14: raise AssertionError('no_integral regression gate incomplete')
            for condition in CONDITIONS:
                for policy in policies:
                    label = policy.provenance['label']; key = condition.name+'-'+label+'-'+gain.name
                    controller = IntegralController(policy, gain); observer = IntegralObserver(condition, controller)
                    rows, snapshot, error, reasons = run_case(specs[condition.name], case, controller, 42,
                        env_factory=RecordedFaultEnv, observer=observer, observation_transform=controller.prepare_observation)
                    for row in rows: row.update(model_label=label, condition=condition.name, integral_mode=gain.name)
                    write_rollout(directory/(key+'.csv'), rows)
                    write_rollout(directory/(key+'-physics.csv'), observer.physics_rows)
                    write_events(directory/(key+'-events.csv'), observer.events)
                    result = analyze(rows, case, condition, observer, error, reasons, thresholds, specs[condition.name])
                    result.update(label=label, key=key); results.append(result)
                    manifest['runs'][key] = dict(initial_snapshot=snapshot, physical=observer.metadata,
                        initial_xi_m=rows[0]['xi_t'].tolist() if rows else None, events_applied=len(observer.events), error=error)
                    write_tables(directory, results); write_json(directory/'manifest.json', manifest)
                    if error: raise RuntimeError(key+': '+error)
                    if snapshot != previous['runs'][condition.name+'-'+label]['initial_snapshot']:
                        raise AssertionError('initial snapshot changed from previous payload/motor evaluation')
                    comparable = {k: snapshot[k] for k in ('position', 'quaternion', 'velocity', 'omega', 'qpos', 'qvel',
                        'reference', 'observation', 'previous_action', '_last_f', '_last_omega', '_last_motor_cmd')}
                    if common_initial is not None and comparable != common_initial:
                        raise AssertionError('initial physical/motor states differ across combinations')
                    common_initial = comparable
                    verify_signals(rows, observer.physics_rows, condition); verify_integral_rows(rows, gain)
                    if not gain.enabled:
                        old_key = condition.name+'-'+label
                        prefixes[old_key] = {kind: compare_prefix(args.previous_results/(old_key+suffix),
                            directory/(key+suffix), physics=kind=='physics') for kind, suffix in
                            (('control', '.csv'), ('physics', '-physics.csv'))}
                        write_json(directory/'no_integral_prefix_verification.json', prefixes)
                    print(f'{len(results)}/56 {key}: completed={result["completed"]} duration={result["actual_duration_sec"]:.2f} end={result["end_reason"]}', flush=True)
                    del rows, observer, controller
        comparison = gain_comparison(results); write_json(directory/'gain_comparison.json', comparison)
        save_plots(directory, results, comparison['common_development_candidate'])
        assert all(sha256(p) == h for p, h in protected.items())
        assert all(parameter_digest(p) == parameter_hashes[p.provenance['label']] for p in policies)
        manifest.update(status='completed', completed_rollouts=len(results),
            flights_completed=sum(r['completed'] for r in results), protected_files_unchanged=True,
            protected_file_count=len(protected), initial_states_equal=True,
            integral_and_force_signal_verification_passed=True, no_integral_prefixes_verified=len(prefixes),
            policy_parameters_unchanged=True,
            common_development_candidate=comparison['common_development_candidate'])
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_tables(directory, results); write_json(directory/'manifest.json', manifest)
        write_json(directory/'completion.json', {k: manifest.get(k) for k in
            ('status', 'expected_rollouts', 'completed_rollouts', 'flights_completed', 'error', 'protected_files_unchanged', 'common_development_candidate')})
        print('results:', directory, flush=True)
    return 0

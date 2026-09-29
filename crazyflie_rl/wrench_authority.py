"""Post-rollout 4D wrench/action diagnostics, without controller changes.

PID effort is requested torque and episode-hover-centered thrust. E2E action
contribution is scale * action: its thrust bias is configured vehicle mass*g,
which need not equal episode mass*g. Missing episode metadata is never inferred
from vehicle.mass or from the commanded thrust.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .yaw_authority import _correlation, _is_e2e_policy

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .eval_cli import RolloutTrace

AXES = ('tau_x', 'tau_y', 'tau_z', 'delta_fz')
UNITS = ('Nm', 'Nm', 'Nm', 'N')
SCALE_KEYS = ('tau_x_nm', 'tau_y_nm', 'tau_z_nm', 'delta_fz_n')
EPSILON = 1e-12
BOUNDARIES = ((.25, '0p25'), (.5, '0p5'), (.75, '0p75'),
              (.9, '0p9'), (.95, '0p95'), (.99, '0p99'))


def _finite(value) -> float | None:
    return float(value) if np.isfinite(value) else None


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or abs(denominator) <= EPSILON:
        return None
    return _finite(numerator / denominator)


def _stats(values: np.ndarray) -> dict[str, float | None]:
    keys = ('rms', 'abs_mean', 'abs_p50', 'abs_p90', 'abs_p95', 'abs_p99', 'abs_max')
    if not values.size or not np.all(np.isfinite(values)):
        return dict.fromkeys(keys)
    absolute = np.abs(values)
    percentiles = np.percentile(absolute, [50, 90, 95, 99])
    return dict(zip(keys, map(_finite, (
        np.sqrt(np.mean(values**2)), np.mean(absolute), *percentiles, np.max(absolute),
    ))))


def _hover_force(trace: RolloutTrace, config: ExperimentConfig) -> tuple[float | None, str]:
    mass = getattr(trace, 'episode_mass_kg', None)
    if mass is None or not np.isfinite(mass) or mass <= 0:
        return None, 'unavailable: episode_mass_kg was not recorded; no mass assumption'
    return float(mass * config.vehicle.gravity), 'trace.episode_mass_kg (_m0 + sampled _com_mw) * config.vehicle.gravity'


def _signals(trace: RolloutTrace, config: ExperimentConfig) -> dict[str, np.ndarray]:
    def array(name):
        value = getattr(trace, name)
        if value is None:
            return np.full((trace.sample_count, 4), np.nan)
        result = np.asarray(value, dtype=float)
        if result.shape != (trace.sample_count, 4):
            raise ValueError(f'{name} must have shape ({trace.sample_count}, 4)')
        return result

    command, actual = array('wrench_command'), array('wrench_actual')
    hover, _ = _hover_force(trace, config)
    center = np.array([0., 0., 0., hover if hover is not None else np.nan])
    action = array('control_input') if _is_e2e_policy(trace) else np.full_like(command, np.nan)
    physical = action * np.asarray(config.environment.residual_scale)
    # Match the implemented E2E bias, not the episode-hover force.
    bias = np.array([0., 0., 0., config.vehicle.mass * config.vehicle.gravity])
    return {
        'command': command, 'actual': actual,
        'effort': command - center, 'actual_effort': actual - center,
        'error': array('allocation_error') if trace.allocation_error is not None else command - actual,
        'action': action, 'physical': physical,
        'mapping_error': command - (bias + physical),
    }


def analyze_wrench_authority(trace: RolloutTrace, config: ExperimentConfig) -> dict[str, Any]:
    signals = _signals(trace, config)
    phases = np.asarray(trace.phases)
    if phases.shape != (trace.sample_count,):
        raise ValueError('phases must have one value per sample')
    hover, source = _hover_force(trace, config)

    def summarize(mask):
        result = {'sample_count': int(np.count_nonzero(mask)), 'axes': {}}
        for index, name in enumerate(AXES):
            selected = {key: value[mask, index] for key, value in signals.items()}
            action = selected['action']
            action_stats = {'u_' + key: value for key, value in _stats(action).items()}
            for boundary, suffix in BOUNDARIES:
                action_stats['fraction_abs_gt_' + suffix] = (
                    float(np.mean(np.abs(action) > boundary))
                    if action.size and np.all(np.isfinite(action)) else None
                )
            command, actual, error = (_stats(selected[key]) for key in ('command', 'actual', 'error'))
            tracking = {
                'command_rms': command['rms'], 'actual_rms': actual['rms'],
                'tracking_error_rms': error['rms'],
                'command_abs_p95': command['abs_p95'], 'command_abs_p99': command['abs_p99'],
                'actual_abs_p95': actual['abs_p95'], 'actual_abs_p99': actual['abs_p99'],
                'command_actual_correlation': _correlation(selected['command'], selected['actual']),
            }
            physical = _stats(selected['physical'])
            result['axes'][name] = {
                'unit': UNITS[index],
                'command_effort': _stats(selected['effort']),
                'actual_effort': _stats(selected['actual_effort']),
                'normalized_action': action_stats,
                'physical_action_contribution': {'physical_' + key: physical[key]
                                                for key in ('abs_mean', 'abs_p95', 'abs_p99', 'abs_max')},
                # Fourth channel here is ABSOLUTE Fz; effort above is centered.
                'absolute_wrench_tracking': tracking,
                'e2e_command_mapping_error_abs_max': _stats(selected['mapping_error'])['abs_max'],
            }
        return result

    return {
        'label': trace.label, 'control_mode': trace.control_mode,
        'episode_mass_kg': getattr(trace, 'episode_mass_kg', None),
        'hover_force_n': hover, 'hover_force_source': source,
        'e2e_command_bias_force_n': float(config.vehicle.mass * config.vehicle.gravity),
        'normalized_action_applicable': _is_e2e_policy(trace),
        'terminated_at': trace.terminated_at, 'truncated_at': trace.truncated_at,
        'overall': summarize(np.ones(trace.sample_count, dtype=bool)),
        'phases': {phase: summarize(phases == phase) for phase in dict.fromkeys(trace.phases)},
    }


def build_wrench_authority_report(
    traces: Sequence[RolloutTrace], config: ExperimentConfig, *, model: str | None = None,
) -> dict[str, Any]:
    scales = np.asarray(config.environment.residual_scale, dtype=float)
    sigma = _finite(np.exp(config.training.ppo.log_std_init))
    physical_sigma = {name: _finite(abs(scale) * sigma) if sigma is not None else None
                      for name, scale in zip(AXES, scales)}
    report = {
        'configured_action_scale': dict(zip(SCALE_KEYS, map(float, scales))),
        'initial_exploration_sigma_normalized': sigma,
        'initial_exploration_sigma_physical': physical_sigma,
        'exploration_source': 'config.training.ppo.log_std_init; initial pre-clipping Gaussian, not checkpoint learned std',
        'sampling': 'Last physics substep per policy step. Correlations have zero sample lag.',
        'effort_definition': 'Torque command; Fz command minus episode mass*g. Absolute thrust is separate.',
        'action_contribution_definition': 'scale * normalized action; E2E Fz bias uses configured vehicle.mass*g, not episode mass*g',
        'ratio_epsilon': EPSILON,
        'config': config.resolved_dict(), 'model': model,
        'floor': None, 'ppo': None,
        'scale_analysis': {'overall': {}, 'phases': {}},
    }
    for trace in traces:
        report['floor' if trace.policy == 'floor' else 'ppo'] = analyze_wrench_authority(trace, config)
    report['hover_force_n'] = {key: (report[key] or {}).get('hover_force_n') for key in ('floor', 'ppo')}
    report['hover_force_source'] = {key: (report[key] or {}).get('hover_force_source') for key in ('floor', 'ppo')}

    def compare(pid, ppo):
        result = {}
        for index, name in enumerate(AXES):
            effort = pid['axes'][name]['command_effort']
            p99 = effort['abs_p99']
            physical = ppo['axes'][name]['physical_action_contribution']['physical_abs_p99'] if ppo else None
            result[name] = {
                'pid_p95_over_action_scale': _ratio(effort['abs_p95'], abs(scales[index])),
                'pid_p99_over_action_scale': _ratio(p99, abs(scales[index])),
                'scale_if_pid_p99_maps_to_target_usage': {
                    f'target_usage_{suffix}': _finite(p99 / target) if p99 is not None else None
                    for target, suffix in ((.1, '0p1'), (.2, '0p2'), (.3, '0p3'))
                },
                'ppo_physical_p99_over_pid_p99': _ratio(physical, p99),
                'initial_sigma_over_pid_p99': _ratio(physical_sigma[name], p99),
            }
        return result

    pid, ppo = report['floor'], report['ppo']
    if pid is not None and pid['control_mode'] == 'residual':
        report['scale_analysis']['overall'] = compare(pid['overall'], ppo['overall'] if ppo else None)
        report['scale_analysis']['phases'] = {
            phase: compare(values, ppo['phases'].get(phase) if ppo else None)
            for phase, values in pid['phases'].items()
        }
    return report


def format_wrench_authority(report: dict[str, Any]) -> str:
    def number(x):
        return 'N/A' if x is None else f'{x:.5g}'
    lines = []
    phases = [p for p in ('GOTO', 'CIRCLE', 'LISSAJOUS') if p in report['scale_analysis']['phases']]
    for phase in phases or ['overall']:
        pid = ((report['floor'] or {}).get('overall') if phase == 'overall' else
               (report['floor'] or {}).get('phases', {}).get(phase))
        ppo = ((report['ppo'] or {}).get('overall') if phase == 'overall' else
               (report['ppo'] or {}).get('phases', {}).get(phase))
        ratios = report['scale_analysis']['overall'] if phase == 'overall' else report['scale_analysis']['phases'][phase]
        lines += [f'=== Wrench / Action Scale Diagnostic: {phase} ===',
                  'Axis/unit           PID P95       PID P99         Scale     P99/Scale']
        for index, axis in enumerate(AXES):
            scale = report['configured_action_scale'][SCALE_KEYS[index]]
            effort = pid['axes'][axis]['command_effort'] if pid else {}
            values = (effort.get('abs_p95'), effort.get('abs_p99'), scale,
                      ratios.get(axis, {}).get('pid_p99_over_action_scale'))
            lines.append(f'{axis + "/" + UNITS[index]:<15}' + ''.join(f'{number(v):>14}' for v in values))
        lines.append('PPO normalized action (evaluation config deterministic=' + str(report['config']['evaluation']['deterministic']) + ')')
        lines.append('Axis                     RMS           P95           P99           Max')
        for axis in AXES:
            m = ppo['axes'][axis]['normalized_action'] if ppo else {}
            lines.append(f'{axis:<15}' + ''.join(f'{number(m.get(k)):>14}' for k in ('u_rms','u_abs_p95','u_abs_p99','u_abs_max')))
        lines += ['Boundary usage', 'Axis                   >0.25          >0.5         >0.75          >0.9']
        for axis in AXES:
            m = ppo['axes'][axis]['normalized_action'] if ppo else {}
            lines.append(f'{axis:<15}' + ''.join(f'{number(m.get("fraction_abs_gt_" + k)):>14}' for k in ('0p25','0p5','0p75','0p9')))
        lines += ['Scale if PID P99 maps to normalized usage (diagnostic candidate only):',
                  'Axis                     10%           20%           30%']
        for axis in AXES:
            m = ratios.get(axis, {}).get('scale_if_pid_p99_maps_to_target_usage', {})
            lines.append(f'{axis:<15}' + ''.join(f'{number(m.get("target_usage_" + k)):>14}' for k in ('0p1','0p2','0p3')))
    return '\n'.join(lines)


def save_wrench_authority_plot(path: str | Path, traces: Sequence[RolloutTrace], config: ExperimentConfig) -> Path:
    from .plotting import _new_path, _pyplot
    target = _new_path(path)
    plt = _pyplot()
    fig, grid = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    axes = grid.ravel()
    try:
        colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')
        for trace in traces:
            s = _signals(trace, config)
            time = trace.time_sec
            if trace.policy == 'floor':
                for i in range(3):
                    axes[0].plot(time, s['effort'][:, i], color=colors[i], label=AXES[i])
                axes[1].plot(time, s['effort'][:, 3], label='PID command - episode hover')
            elif _is_e2e_policy(trace):
                for i in range(4):
                    axes[2].plot(time, s['action'][:, i], color=colors[i], label=AXES[i])
                for i in range(3):
                    axes[4].plot(time, s['command'][:, i], '--', color=colors[i], label=AXES[i]+' cmd')
                    axes[4].plot(time, s['actual'][:, i], color=colors[i], label=AXES[i]+' actual')
                for key, style in (('command', '--'), ('actual', '-')):
                    axes[3].plot(time, s[key][:, 3], style, label='PPO '+key)
                    axes[5].plot(time, s['effort' if key == 'command' else 'actual_effort'][:, 3], style, label='PPO '+key+' - hover')
            for i in range(1, trace.sample_count):
                if trace.phases[i] != trace.phases[i-1]:
                    for ax in axes:
                        ax.axvline(time[i], color='gray', alpha=.12, linewidth=.6)
        titles = ('PID requested torque', 'PID hover-centered thrust', 'PPO normalized action',
                  'PPO absolute thrust', 'PPO torque tracking', 'PPO hover-centered thrust')
        labels = ('Torque [Nm]', 'Delta Fz [N]', 'Normalized action', 'Fz [N]', 'Torque [Nm]', 'Delta Fz [N]')
        for ax, title, label in zip(axes, titles, labels):
            ax.set_title(title)
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=7, ncol=2)
            ax.set_xlabel('Time [s]')
        axes[2].set_ylim(-1, 1)
        for threshold in (-.9, .9):
            axes[2].axhline(threshold, color='gray', linestyle=':')
        for ax in (axes[0], axes[4]):
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        title = 'Wrench / action scale diagnostics'
        if any(_hover_force(t, config)[0] is None for t in traces):
            title += ' — missing episode mass: centered thrust unavailable'
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(target, dpi=140)
    finally:
        plt.close(fig)
    return target

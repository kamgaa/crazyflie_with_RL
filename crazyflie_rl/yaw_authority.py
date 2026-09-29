"""Observer-only yaw diagnostics from completed RolloutTrace arrays.

Wrenches are the final physics-substep samples already stored by the runner.
Actual yaw torque is the signed motor reaction-torque sum, not total external
body torque. Correlations are simultaneous sample correlations, without lag
compensation or causal interpretation.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .eval_cli import RolloutTrace

_STD_EPSILON = 1e-12
_SCALE_EPSILON = 1e-12


def _is_e2e_policy(trace: RolloutTrace) -> bool:
    return trace.policy != 'floor' and trace.control_mode == 'e2e'


def _column(trace: RolloutTrace, name: str, width: int) -> np.ndarray:
    value = getattr(trace, name)
    if value is None:
        return np.full(trace.sample_count, np.nan)
    array = np.asarray(value, dtype=float)
    if array.shape != (trace.sample_count, width):
        raise ValueError(f'{name} must have shape ({trace.sample_count}, {width})')
    return array[:, 2]


def _signals(trace: RolloutTrace, config: ExperimentConfig) -> dict[str, np.ndarray]:
    yaw = _column(trace, 'attitude_deg', 3)
    delta = np.deg2rad(yaw) - config.environment.yaw_target
    command = _column(trace, 'wrench_command', 4)
    actual = _column(trace, 'wrench_actual', 4)
    return {
        'yaw_deg': yaw,
        'yaw_error_deg': np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta))),
        'command': command,
        'actual': actual,
        'tracking_error': (
            _column(trace, 'allocation_error', 4)
            if trace.allocation_error is not None else command - actual
        ),
        'action': (
            _column(trace, 'control_input', 4) if _is_e2e_policy(trace)
            else np.full(trace.sample_count, np.nan)
        ),
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if (first.size < 2 or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))):
        return None
    if np.std(first) <= _STD_EPSILON or np.std(second) <= _STD_EPSILON:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return float(np.clip(value, -1, 1)) if np.isfinite(value) else None


def analyze_yaw_authority(
    trace: RolloutTrace, config: ExperimentConfig,
) -> dict[str, Any]:
    """Summarize present phases; missing/nonfinite signals yield null statistics.

    PID floor's normalized residual input is deliberately ineligible. A residual
    PPO policy is also ineligible for the E2E action-boundary interpretation.
    """
    phases = np.asarray(trace.phases)
    if phases.shape != (trace.sample_count,):
        raise ValueError('phases must have one entry per sample')
    signals = _signals(trace, config)

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        values = {key: array[mask] for key, array in signals.items()}
        result: dict[str, Any] = {'sample_count': int(np.count_nonzero(mask))}

        def reduce(array: np.ndarray, operation) -> float | None:
            if not array.size or not np.all(np.isfinite(array)):
                return None
            value = float(operation(array))
            return value if np.isfinite(value) else None

        rms = lambda x: np.sqrt(np.mean(x**2))
        yaw = np.abs(values['yaw_error_deg'])
        result['yaw_rms_deg'] = reduce(yaw, rms)
        result['yaw_mean_abs_deg'] = reduce(yaw, np.mean)
        result['yaw_peak_abs_deg'] = reduce(yaw, np.max)
        for signal, prefix in (('command', 'tau_z_cmd'), ('actual', 'tau_z_actual')):
            absolute = np.abs(values[signal])
            result[f'{prefix}_rms_nm'] = reduce(absolute, rms)
            result[f'{prefix}_abs_mean_nm'] = reduce(absolute, np.mean)
            result[f'{prefix}_abs_p95_nm'] = reduce(absolute, lambda x: np.percentile(x, 95))
            result[f'{prefix}_abs_p99_nm'] = reduce(absolute, lambda x: np.percentile(x, 99))
            result[f'{prefix}_abs_max_nm'] = reduce(absolute, np.max)
        error = np.abs(values['tracking_error'])
        result['tau_z_tracking_error_rms_nm'] = reduce(error, rms)
        result['tau_z_tracking_error_abs_p95_nm'] = reduce(error, lambda x: np.percentile(x, 95))
        result['tau_z_tracking_error_abs_max_nm'] = reduce(error, np.max)
        action = np.abs(values['action'])
        result['u_tau_z_rms'] = reduce(action, rms)
        result['u_tau_z_abs_mean'] = reduce(action, np.mean)
        result['u_tau_z_abs_p95'] = reduce(action, lambda x: np.percentile(x, 95))
        result['u_tau_z_abs_p99'] = reduce(action, lambda x: np.percentile(x, 99))
        result['u_tau_z_abs_max'] = reduce(action, np.max)
        for threshold, suffix in ((.8, '0p8'), (.9, '0p9'), (.95, '0p95'), (.99, '0p99')):
            result[f'u_tau_z_fraction_gt_{suffix}'] = reduce(
                action, lambda x: np.mean(x > threshold),
            )
        result['tau_z_actual_vs_cmd_correlation'] = _correlation(values['command'], values['actual'])
        result['u_tau_z_vs_tau_z_actual_correlation'] = _correlation(values['action'], values['actual'])
        return result

    return {
        'label': trace.label,
        'control_mode': trace.control_mode,
        'normalized_yaw_action_applicable': _is_e2e_policy(trace),
        'tracking_error_source': ('allocation_error[:,2]' if trace.allocation_error is not None
                                  else 'wrench_command[:,2] - wrench_actual[:,2]'),
        'terminated_at': trace.terminated_at,
        'truncated_at': trace.truncated_at,
        'overall': summarize(np.ones(trace.sample_count, dtype=bool)),
        'phases': {phase: summarize(phases == phase) for phase in dict.fromkeys(trace.phases)},
    }


def build_yaw_authority_report(
    traces: Sequence[RolloutTrace], config: ExperimentConfig, *, model: str | None = None,
) -> dict[str, Any]:
    scale = float(config.environment.residual_scale[2])
    report: dict[str, Any] = {
        'configured_tau_z_action_scale_nm': scale,
        'policy_tau_z_max_abs_nm': abs(scale),
        'yaw_target_rad': float(config.environment.yaw_target),
        'correlation_std_epsilon': _STD_EPSILON,
        'ratio_denominator_epsilon_nm': _SCALE_EPSILON,
        'sampling': 'Final physics substep per policy step; correlations have zero sample lag.',
        'actual_torque_definition': 'Signed sum of motor reaction torques; excludes external body torques.',
        'config': config.resolved_dict(),
        'model': model,
        'floor': None,
        'ppo': None,
        'comparisons': {'overall': {}, 'phases': {}},
    }
    for trace in traces:
        report['floor' if trace.policy == 'floor' else 'ppo'] = analyze_yaw_authority(trace, config)

    def ratios(metrics: dict[str, Any]) -> dict[str, float | None]:
        result = {}
        for percentile in (95, 99):
            value = metrics[f'tau_z_cmd_abs_p{percentile}_nm']
            result[f'pid_tau_z_p{percentile}_over_e2e_action_scale'] = (
                float(value / abs(scale)) if value is not None and abs(scale) > _SCALE_EPSILON else None
            )
        return result

    floor = report['floor']
    # Only a real PID floor is eligible for PID-command comparison ratios.
    if floor is not None and floor['control_mode'] == 'residual':
        report['comparisons']['overall'] = ratios(floor['overall'])
        report['comparisons']['phases'] = {phase: ratios(m) for phase, m in floor['phases'].items()}
    return report


def format_yaw_authority(report: dict[str, Any]) -> str:
    """Show one phase (prefer CIRCLE) with numeric observations only."""
    phase = next((name for name in ('CIRCLE', 'LISSAJOUS', 'GOTO')
                  if any(name in (report[key] or {}).get('phases', {}) for key in ('floor', 'ppo'))), None)
    values = [(report[key] or {}).get('phases', {}).get(phase) if phase else
              (report[key] or {}).get('overall') for key in ('floor', 'ppo')]
    rows = (
        ('Yaw RMS [deg]', 'yaw_rms_deg'), ('Yaw peak [deg]', 'yaw_peak_abs_deg'),
        ('Cmd RMS [Nm]', 'tau_z_cmd_rms_nm'), ('Cmd P95 abs [Nm]', 'tau_z_cmd_abs_p95_nm'),
        ('Cmd P99 abs [Nm]', 'tau_z_cmd_abs_p99_nm'), ('Actual RMS [Nm]', 'tau_z_actual_rms_nm'),
        ('Actual P95 abs [Nm]', 'tau_z_actual_abs_p95_nm'),
        ('Tracking error RMS [Nm]', 'tau_z_tracking_error_rms_nm'),
        ('PPO yaw action RMS', 'u_tau_z_rms'), ('PPO yaw action P95 abs', 'u_tau_z_abs_p95'),
        ('PPO yaw action P99 abs', 'u_tau_z_abs_p99'),
        ('P(abs(action) > 0.9)', 'u_tau_z_fraction_gt_0p9'),
        ('P(abs(action) > 0.95)', 'u_tau_z_fraction_gt_0p95'),
    )
    def number(value):
        return 'N/A' if value is None else f'{value:.5g}'
    lines = [f'=== Yaw Authority Diagnostic: {phase or "overall"} ===',
             f'{"Metric":<29}{"PID":>14}{"PPO":>14}']
    for label, key in rows:
        lines.append(f'{label:<29}' + ''.join(f'{number(v[key] if v else None):>14}' for v in values))
    comparisons = (report['comparisons']['phases'].get(phase, {}) if phase
                   else report['comparisons']['overall'])
    lines.append(f'Configured E2E tau-z scale: {report["configured_tau_z_action_scale_nm"]:g} Nm')
    lines.append('PID P99 / E2E scale: ' + number(comparisons.get('pid_tau_z_p99_over_e2e_action_scale')))
    return '\n'.join(lines)


def save_yaw_authority_plot(
    path: str | Path, traces: Sequence[RolloutTrace], config: ExperimentConfig,
) -> Path:
    """One three-panel PNG; no PID residual-action comparison curve."""
    from .plotting import _new_path, _pyplot

    target = _new_path(path)
    plt = _pyplot()
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    try:
        has_action = False
        boundaries: dict[float, set[str]] = {}
        for trace in traces:
            signals = _signals(trace, config)
            color = 'tab:blue' if trace.policy == 'floor' else 'tab:orange'
            label = 'PID' if trace.policy == 'floor' else 'PPO'
            axes[0].plot(trace.time_sec, signals['yaw_deg'], color=color, label=label)
            axes[1].plot(trace.time_sec, signals['command'], color=color, linestyle='--', label=f'{label} requested')
            axes[1].plot(trace.time_sec, signals['actual'], color=color, label=f'{label} actual')
            if _is_e2e_policy(trace):
                axes[2].plot(trace.time_sec, signals['action'], color=color, label='E2E PPO yaw action')
                has_action = True
            for index, phase in enumerate(trace.phases):
                if index == 0 or phase != trace.phases[index - 1]:
                    boundaries.setdefault(float(trace.time_sec[index]), set()).add(phase)
        yaw_target = config.environment.yaw_target
        reference = np.rad2deg(np.arctan2(np.sin(yaw_target), np.cos(yaw_target)))
        axes[0].axhline(reference, color='black', linestyle=':', label='Yaw reference')
        for threshold in (-.9, .9):
            axes[2].axhline(threshold, color='gray', linestyle=':', label=f'{threshold:+.1f}')
        axes[2].set_ylim(-1, 1)
        axes[2].text(.01, .04, 'PID residual yaw action: N/A', transform=axes[2].transAxes)
        if not has_action:
            axes[2].text(.5, .5, 'E2E PPO yaw action: N/A', ha='center', transform=axes[2].transAxes)
        for x, phases in sorted(boundaries.items()):
            for axis in axes:
                axis.axvline(x, color='gray', alpha=.2, linewidth=.6)
            axes[0].text(x, 1.01, '/'.join(sorted(phases)), fontsize=7,
                         transform=axes[0].get_xaxis_transform())
        for axis, ylabel in zip(axes, ('Yaw [deg]', 'Yaw torque [Nm]', 'Normalized yaw action')):
            axis.set_ylabel(ylabel)
            axis.grid(alpha=.2)
            axis.legend(loc='upper right', fontsize=8, ncol=2)
        axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axes[2].set_xlabel('Time [s]')
        fig.suptitle('Yaw authority diagnostics')
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(target, dpi=140)
    finally:
        plt.close(fig)
    return target

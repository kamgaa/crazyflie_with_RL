"""Post-process recorded motion with the current continuous state reward costs.

No simulator, policy, or RNG access. Costs exclude action, action-rate and crash
penalties and are not a reconstruction of total episode return. Trace precision
is limited by the float32 observations used by EvaluationRunner.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .eval_cli import RolloutTrace

_COMPONENTS = ('position', 'velocity', 'tilt', 'angular_velocity', 'yaw')
_EPSILON = 1e-12


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def reward_balance_metrics(
    trace: RolloutTrace, config: ExperimentConfig,
) -> dict[str, Any]:
    """Sample means overall/per phase, with progress only between adjacent samples.

    Repeated phase segments are pooled for state statistics, but progress never
    bridges intervening phases. Reduction rate is -sum(delta error)/sum(delta t),
    equivalent to -mean(delta error)/dt for the runner's fixed sample interval.
    Missing/nonfinite quantities produce null, including total/fractions when
    the complete continuous state cost cannot be reconstructed.
    """
    n = trace.sample_count
    times = np.asarray(trace.time_sec, dtype=float)
    phases = np.asarray(trace.phases)
    if times.shape != (n,) or phases.shape != (n,):
        raise ValueError('time and phases must have one entry per trace sample')

    def vector(name: str) -> np.ndarray:
        value = getattr(trace, name)
        if value is None:
            return np.full((n, 3), np.nan)
        array = np.asarray(value, dtype=float)
        if array.shape != (n, 3):
            raise ValueError(f'{name} must have shape ({n}, 3)')
        return array

    # This is the norm recorded against the actual sampled reference, not a
    # projection of velocity onto a fixed-reference position error.
    errors = np.asarray(trace.position_error, dtype=float)
    if errors.shape != (n,):
        raise ValueError('position_error must contain one norm per sample')
    velocity = vector('linear_velocity')
    from .velocity_reference import velocity_reward_semantics
    velocity_cost_signal = (vector('velocity_error')
        if velocity_reward_semantics(config)['mode'] != 'absolute' and getattr(trace,'control_mode',None) != 'residual'
        else velocity)
    omega = vector('angular_velocity')
    attitude = np.deg2rad(vector('attitude_deg'))
    cos_tilt = np.clip(np.cos(attitude[:, 0]) * np.cos(attitude[:, 1]), -1.0, 1.0)
    tilt = np.rad2deg(np.arccos(cos_tilt))
    yaw_delta = attitude[:, 2] - config.environment.yaw_target
    yaw_error = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    raw = {
        'position': errors**2,
        'velocity': np.sum(velocity**2, axis=1),
        'tilt': 1.0 - cos_tilt,
        'angular_velocity': np.sum(omega**2, axis=1),
        'yaw': yaw_error**2,
    }
    weights = {name: float(getattr(config.environment.reward, f'{name}_weight'))
               for name in _COMPONENTS}
    reward = config.environment.reward
    xy_weight = reward.effective_position_xy_weight
    z_weight = reward.effective_position_z_weight
    error_vectors = vector('position') - vector('reference_position')
    position_sq_xy = np.sum(error_vectors[:, :2] ** 2, axis=1)
    position_sq_z = error_vectors[:, 2] ** 2

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        count = int(np.count_nonzero(mask))

        def reduce(values: np.ndarray, operation=np.mean) -> float | None:
            selected = values[mask]
            if not selected.size or not np.all(np.isfinite(selected)):
                return None
            return _finite(operation(selected))

        means = {name: reduce(values) for name, values in raw.items()}
        result: dict[str, Any] = {'sample_count': count}
        for name, prefix, unit in (
            ('position', 'position_error', 'm'),
            ('velocity', 'velocity', 'mps'),
            ('angular_velocity', 'angular_velocity', 'radps'),
        ):
            mean_sq = means[name]
            result[f'{prefix}_rms_{unit}'] = _finite(np.sqrt(mean_sq)) if mean_sq is not None else None
            result[f'{prefix}_peak_{unit}'] = reduce(np.sqrt(raw[name]), np.max)
            result[f'{name}_sq_mean'] = mean_sq
        result['position_error_mean_m'] = reduce(errors)
        result['tilt_rms_deg'] = reduce(tilt, lambda x: np.sqrt(np.mean(x**2)))
        result['tilt_mean_deg'] = reduce(tilt)
        result['tilt_peak_deg'] = reduce(tilt, np.max)
        result['tilt_error_mean'] = means['tilt']
        result['yaw_error_sq_mean'] = means['yaw']
        costs = {name: _finite(weights[name] * mean) if mean is not None else None
                 for name, mean in means.items()}
        result['position_sq_xy_mean'] = reduce(position_sq_xy)
        result['position_sq_z_mean'] = reduce(position_sq_z)
        result['position_cost_xy_mean'] = reduce(xy_weight * position_sq_xy)
        result['position_cost_z_mean'] = reduce(z_weight * position_sq_z)
        for axis in ('xy', 'z'):
            mean_sq = result[f'position_sq_{axis}_mean']
            result[f'position_rmse_{axis}'] = _finite(np.sqrt(mean_sq)) if mean_sq is not None else None
        costs['position'] = (
            reduce(xy_weight * raw['position']) if xy_weight == z_weight
            else reduce(xy_weight * position_sq_xy + z_weight * position_sq_z)
        )
        error_signal = vector('velocity_error') if getattr(trace, 'velocity_error', None) is not None else velocity_cost_signal
        result['velocity_error_rms_mps'] = reduce(np.linalg.norm(error_signal,axis=1),lambda x: np.sqrt(np.mean(x*x)))
        result['velocity_reward_sq_mean'] = reduce(np.sum(velocity_cost_signal**2,axis=1))
        result['velocity_reward_signal'] = ('velocity_error' if velocity_reward_semantics(config)['mode'] != 'absolute'
            and getattr(trace,'control_mode',None) != 'residual' else 'absolute_velocity')
        costs['velocity'] = reduce(weights['velocity']*np.sum(velocity_cost_signal**2,axis=1))
        total = _finite(sum(costs.values())) if all(v is not None for v in costs.values()) else None
        result['mean_cost'] = {**costs, 'total': total}
        result['cost_fraction'] = {
            name: cost / (total + _EPSILON) if total is not None else None
            for name, cost in costs.items()
        }
        pairs = mask[:-1] & mask[1:]
        delta = np.diff(errors)[pairs]
        dt = np.diff(times)[pairs]
        result['progress_pair_count'] = int(delta.size)
        valid = bool(delta.size and np.all(np.isfinite(delta)))
        result['progress_fraction'] = float(np.mean(delta < 0)) if valid else None
        result['regress_fraction'] = float(np.mean(delta > 0)) if valid else None
        result['mean_error_delta_m_per_step'] = _finite(np.mean(delta)) if valid else None
        result['mean_error_reduction_rate_mps'] = (
            _finite(-np.sum(delta) / np.sum(dt))
            if valid and np.all(np.isfinite(dt)) and np.all(dt > 0) else None
        )
        return result

    return {
        'overall': summarize(np.ones(n, dtype=bool)),
        'phases': {phase: summarize(phases == phase) for phase in dict.fromkeys(trace.phases)},
    }


def build_reward_balance_report(
    traces: Sequence[RolloutTrace], config: ExperimentConfig, *, model: str | None = None,
) -> dict[str, Any]:
    """Score both controllers with one resolved config; ratios are diagnostics."""
    report: dict[str, Any] = {
        'reward_weights': {name: float(getattr(config.environment.reward, f'{name}_weight'))
                           for name in _COMPONENTS} | {
            'position_xy': config.environment.reward.effective_position_xy_weight,
            'position_z': config.environment.reward.effective_position_z_weight,
        },
        'scope': 'continuous_state_cost_only',
        'excluded_components': ['action', 'action_rate', 'crash'],
        'fraction_epsilon': _EPSILON,
        'ratio_denominator_epsilon': _EPSILON,
        'yaw_target_rad': float(config.environment.yaw_target),
        'precision': 'Reconstructed from float32-observation-derived trace; not exact environment reward.',
        'progress_rate_definition': '-sum(adjacent_error_delta)/sum(adjacent_time_delta); no phase-gap bridging',
        'config': config.resolved_dict(),
        'model': model,
        'floor': None,
        'ppo': None,
        'comparisons': {'overall': {}, 'phases': {}},
    }
    for trace in traces:
        key = 'floor' if trace.policy == 'floor' else 'ppo'
        report[key] = {
            'label': trace.label, 'control_mode': trace.control_mode,
            'terminated_at': trace.terminated_at, 'truncated_at': trace.truncated_at,
            'diverged_at': trace.diverged_at,
            **reward_balance_metrics(trace, config),
        }

    def ratios(pid: dict, ppo: dict) -> dict:
        result = {}
        for label, key in (
            ('position_error_rms', 'position_error_rms_m'),
            ('velocity_rms', 'velocity_rms_mps'),
            ('tilt_rms', 'tilt_rms_deg'),
            ('angular_velocity_rms', 'angular_velocity_rms_radps'),
        ):
            denominator, numerator = pid[key], ppo[key]
            result[label] = (_finite(numerator / denominator)
                             if numerator is not None and denominator is not None
                             and abs(denominator) > _EPSILON else None)
        for name in _COMPONENTS:
            denominator, numerator = pid['mean_cost'][name], ppo['mean_cost'][name]
            result[f'{name}_cost'] = (_finite(numerator / denominator)
                                     if numerator is not None and denominator is not None
                                     and abs(denominator) > _EPSILON else None)
        return {'ppo_over_pid': result}

    floor, ppo = report['floor'], report['ppo']
    if floor is not None and ppo is not None:
        report['comparisons']['overall'] = ratios(floor['overall'], ppo['overall'])
        report['comparisons']['phases'] = {
            phase: ratios(values, ppo['phases'][phase])
            for phase, values in floor['phases'].items() if phase in ppo['phases']
        }
    return report


def format_reward_balance(report: dict[str, Any]) -> str:
    """Compact numeric tables; deliberately no ranking or causal interpretation."""
    def number(value: Any) -> str:
        return 'n/a' if value is None else f'{value:.5g}'

    lines = ['Reward Balance: overall position RMS [m]']
    for key, label in (('floor', 'PID'), ('ppo', 'PPO')):
        values = report[key]
        lines.append(f"  {label}: {number(values['overall']['position_error_rms_m']) if values else 'n/a'}")
        if values:
            overall = values['overall']
            lines.append(f"    xy (horizontal distance): {number(overall['position_rmse_xy'])}, "
                         f"z: {number(overall['position_rmse_z'])}")
    for phase in ('GOTO', 'CIRCLE', 'LISSAJOUS'):
        values = [(report[k] or {}).get('phases', {}).get(phase) for k in ('floor', 'ppo')]
        if not any(values):
            continue
        lines.extend([f'\n=== Reward Balance: {phase} ===', f"{'Metric':<30}{'PID':>13}{'PPO':>13}"])
        rows = [
            ('Position RMS [m]', 'position_error_rms_m'),
            ('Position XY RMSE [m]', 'position_rmse_xy'),
            ('Position Z RMSE [m]', 'position_rmse_z'),
            ('Mean position XY cost', 'position_cost_xy_mean'),
            ('Mean position Z cost', 'position_cost_z_mean'),
            ('Velocity RMS [m/s]', 'velocity_rms_mps'),
            ('Tilt RMS [deg]', 'tilt_rms_deg'),
            ('Angular vel RMS [rad/s]', 'angular_velocity_rms_radps'),
        ]
        for label, key in rows:
            lines.append(f'{label:<30}' + ''.join(f'{number(v[key] if v else None):>13}' for v in values))
        for group, label in (('mean_cost', 'Mean cost'), ('cost_fraction', 'Cost fraction')):
            for name in _COMPONENTS:
                lines.append(f'{label + ": " + name:<30}' + ''.join(
                    f'{number(v[group][name] if v else None):>13}' for v in values))
    return '\n'.join(lines)

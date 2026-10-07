"""Deterministic, non-learning DR transfer comparison with explicit timing."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np

from .config import load_config
from .artifacts import _git_metadata
from .dr_policy import (labeled, load_frozen_policy, sha256, policy_raw_observation,
                        VELOCITY_SLICE, VELOCITY_INPUTS, OBSERVATION_CONTRACT)
from .environment import CrazyflieResidualEnv
from .eval_cli import EvaluationRunner
from .missions import mission_from_experiment
from .plotting import save_transfer_comparison_plot, quaternion_to_euler_deg

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ('step-005', 'step-050', 'circle')
CASE_NAMES = (*DEFAULT_CASES, 'circle-air', 'hover')
SCALE = (0.0075, 0.0075, 0.001, 0.5)


def display_name(label):
    return {'baseline': 'baseline (legacy position reset)',
            'posdr': 'POS-only DR (new reset)'}.get(label, label)


@dataclass(frozen=True)
class Thresholds:
    position_band_m: float = .005
    speed_band_m_s: float = .02
    minimum_settle_sec: float = 1.
    tail_sec: float = 2.

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('thresholds and windows must be positive and finite')
        if max(self.minimum_settle_sec, self.tail_sec) > 8:
            raise ValueError('step windows cannot exceed the 8-second horizon')


@dataclass(frozen=True)
class Case:
    name: str
    horizon: float
    goal: tuple | None = None
    mission: object = None
    floor_start: bool = False
    reference_time_offset_sec: float = 0.

    def reference(self, t):
        return (self.mission.reference(t + self.reference_time_offset_sec) if self.mission
                else (np.asarray(self.goal), 'STEP'))

    def reference_velocity(self, t):
        # A step target is a regulation command, not a differentiated jump.
        return (self.mission.reference_velocity(t + self.reference_time_offset_sec)
                if self.mission else np.zeros(3))

    def post_reference(self, t):
        """Same-time evaluation reference; event cases can declare a left limit."""
        return self.reference(t)

    def initial_position(self):
        if self.name == 'circle-air':
            return self.reference(0)[0].copy()
        return np.asarray([0., 0., .02 if self.floor_start else 1.])

    def description(self, dt):
        return {
            'name': self.name, 'horizon_sec': self.horizon,
            'policy_steps': round(self.horizon / dt),
            'initial_position': self.initial_position().tolist(),
            'initial_quaternion_wxyz': [1., 0., 0., 0.],
            'initial_velocity': [0., 0., 0.], 'initial_omega': [0., 0., 0.],
            'yaw_reference': 0., 'floor_start': self.floor_start,
            'actuator_reset': 'ground_zero' if self.floor_start else 'airborne_hover_equilibrium',
            'first_reference': self.reference(0)[0].tolist(),
            'first_reference_velocity_world_m_s': self.reference_velocity(0).tolist(),
            'mission_parameters': self.mission.effective_parameters() if self.mission else None,
            'source_mission_phase_boundaries': asdict(self.mission.boundaries) if self.mission else None,
            'reference_time_offset_sec': self.reference_time_offset_sec,
            'phase_boundaries': (
                {'circle_end': self.mission.boundaries.circle_end - self.reference_time_offset_sec,
                 'total': self.horizon} if self.name == 'circle-air'
                else asdict(self.mission.boundaries) if self.mission else None),
            'start_provenance': (
                'airborne at generator CIRCLE entry; original ramp and HOLD; TAKEOFF/GOTO/SETTLE excluded'
                if self.name == 'circle-air' else
                'current view_live_circle_eval mission convention; no historical run assumed'
                if self.mission else 'explicit step initial state, no pre-hover'),
        }


def validate_common_config(config):
    e, a = config.environment, config.actuator
    expected = [(e.control_mode == 'e2e', 'control_mode=e2e'),
                (e.policy_hz == 100 and config.vehicle.physics_hz == 500, '100/500 Hz'),
                (e.residual_scale == SCALE, 'action scale'),
                (e.reward.effective_position_xy_weight == 10 and e.reward.effective_position_z_weight == 6, 'xy/z weights 10/6'),
                (not e.payload.randomize and e.payload.mass == 0 and e.payload.offset == (0., 0.), 'nominal payload'),
                (a.enabled and a.model == 'cf21b_first_order' and not a.randomization.enabled, 'fixed first-order actuator'),
                (a.reset_rpm_mode == 'auto', 'actuator reset_rpm_mode=auto'),
                (e.position_perturbation == e.attitude_perturbation_deg == 0, 'legacy perturbations off'),
                (e.initial_pose_randomization is None or not e.initial_pose_randomization.enabled, 'new reset DR off'),
                (config.evaluation.deterministic, 'deterministic=True')]
    for valid, message in expected:
        if not valid:
            raise ValueError(f'common evaluation config requires {message}')


def make_cases(config, names):
    if len(set(names)) != len(names):
        raise ValueError('duplicate cases are not supported')
    result = []
    for name in names:
        if name not in CASE_NAMES:
            raise ValueError(f'unknown case: {name}')
        if name == 'hover':
            result.append(Case(name, 8., (0.,0.,1.)))
        elif name.startswith('step-'):
            result.append(Case(name, 8., (.05 if name == 'step-005' else .5, 0., 1.)))
        else:
            mission = mission_from_experiment(config, mission_type='circle')
            p = mission.effective_parameters()
            required = dict(center_xy=(.5, 0.), radius=.5, period=5., laps=2., altitude=1.,
                            direction='ccw', start_angle_deg=0., ramp_sec=2.)
            if any(p[k] != v for k, v in required.items()):
                raise ValueError('circle parameters must match the fixed DR comparison contract (period=5)')
            offset = mission.boundaries.settle2_end if name == 'circle-air' else 0.
            result.append(Case(name, mission.total_sec - offset, mission=mission,
                               floor_start=name == 'circle' and config.mission.force_floor_start,
                               reference_time_offset_sec=offset))
    return result


class EvaluationAdapter:
    """Isolate existing private state APIs without altering ordinary reset/RNG."""
    def __init__(self, env):
        self.env = env
        self.control_dt = env.dt_phys * env.substeps

    def read_state(self):
        p, q, v, omega = self.env._read_state()
        return dict(position=p, quaternion=q, velocity=v, omega=omega)

    def current_observation(self):
        # _obs is pure concatenation, has no history/normalization side effects.
        return self.env._obs(*self.env._read_state())

    def set_reference(self, reference):
        value = np.asarray(reference, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError('reference must be finite xyz')
        self.env.pos_des = value.copy()
        self.env.yaw_des = 0.

    def reset_to_case_initial_state(self, case, seed):
        import mujoco
        env = self.env
        # Reset at p0, NEVER at the displaced step target. This resets MuJoCo,
        # actuator parameters, PID, previous action, counters and control history.
        self.set_reference([0., 0., 1.])
        env.reset(seed=seed)
        env.dist_torque_body[:] = 0.
        if case.floor_start:
            ok, error = EvaluationRunner._force_floor_start(env)
            if not ok:
                raise RuntimeError(f'floor initialization failed: {error}')
        else:
            env.data.qpos[:3] = case.initial_position()
            env.data.qpos[3:7] = [1., 0., 0., 0.]
            env.data.qvel[:] = 0.
            mujoco.mj_forward(env.model, env.data)
            env.reset_actuator_state(airborne=True)
        env.pid.reset()
        env._prev_action[:] = 0.
        env._step = 0
        self.set_reference(case.reference(0)[0])
        return self.current_observation()

    def snapshot(self):
        env = self.env
        result = {k: v.tolist() for k, v in self.read_state().items()}
        result.update(time_sec=float(env.data.time), qpos=env.data.qpos.tolist(),
                      qvel=env.data.qvel.tolist(), ctrl=env.data.ctrl.tolist(),
                      reference=env.pos_des.tolist(), yaw_reference=env.yaw_des,
                      observation=self.current_observation().tolist(), step=env._step,
                      previous_action=env._prev_action.tolist(),
                      pid_i_velocity=env.pid._i_vel.tolist(), pid_i_rate=env.pid._i_rate.tolist(),
                      actuator=env.actuator_snapshot(),
                      mass=env.model.body_mass.tolist(), inertia=env.model.body_inertia.tolist())
        for name in ('_last_f', '_last_f_cmd', '_last_omega', '_last_motor_cmd',
                     '_last_q_actual', '_last_wrench_cmd', '_last_wrench_actual', '_last_allocation_error'):
            result[name] = np.asarray(getattr(env, name)).tolist()
        return result


def reference_sequence(case, dt):
    t = np.arange(round(case.horizon / dt) + 1) * dt
    pairs = [case.reference(float(v)) for v in t]
    return dict(time=t, reference=np.asarray([p[0] for p in pairs]), phase=np.asarray([p[1] for p in pairs]),
                reference_velocity=np.asarray([case.reference_velocity(float(v)) for v in t]))


def termination_reasons(env, state, control_reference):
    p, q = state['position'], state['quaternion']
    tilt = np.arccos(np.clip(1 - 2 * (q[1]**2 + q[2]**2), -1, 1))
    tests = {'min_altitude': p[2] < env.min_altitude, 'max_altitude': p[2] > env.max_altitude,
             'max_tilt': tilt > env.max_termination_tilt,
             'max_position_error': np.linalg.norm(p - control_reference) > env.max_position_error}
    return [k for k, hit in tests.items() if hit]


def run_case(config, case, policy, seed, velocity_input='absolute', *, env_factory=None, observer=None,
             observation_transform=None):
    from .velocity_reference import velocity_semantics, desired_velocity_semantics, velocity_reward_semantics
    if velocity_input != 'absolute' and velocity_semantics(config)['mode'] != 'absolute':
        raise ValueError('new position-generated velocity error cannot use an additional v_ref intervention')
    env = (env_factory or CrazyflieResidualEnv)(config=config, episode_sec=case.horizon)
    adapter = EvaluationAdapter(env)
    rows, initial, error, reasons = [], None, None, []
    try:
        policy.bind(env)
        adapter.reset_to_case_initial_state(case, seed)
        if observer is not None:
            observer.on_reset(adapter)
        initial = adapter.snapshot()
        for k in range(round(case.horizon / adapter.control_dt)):
            t = k * adapter.control_dt
            if observer is not None:
                observer.before_step(adapter, k, t)
            reference, phase = case.reference(t)
            adapter.set_reference(reference)
            before = adapter.read_state()
            obs = adapter.current_observation()
            reference_velocity = case.reference_velocity(t)
            policy_obs = policy_raw_observation(obs, reference_velocity, velocity_input, config=config)
            if observation_transform is not None:
                # Evaluation-only intervention, before frozen normalization. The
                # environment observation, reference, reward and state stay native.
                policy_obs = observation_transform(policy_obs, before['position'], reference, adapter.control_dt)
            action = np.asarray(policy.predict(policy_obs))
            if action.shape != (4,) or not np.all(np.isfinite(action)):
                raise ValueError('policy returned a non-finite or wrong-shape action')
            _, reward, terminated, truncated, info = env.step(action)
            after = adapter.read_state()
            post_time = (k + 1) * adapter.control_dt
            post_reference, post_phase = case.post_reference(post_time)
            post_reference_velocity = case.reference_velocity(post_time)
            row = dict(time=t, time_post=post_time, phase=phase, phase_post=post_phase,
                       reference=np.asarray(reference).copy(), reference_post=np.asarray(post_reference).copy(),
                       observation=obs.copy(), action=action.copy(), action_applied=np.clip(action, -1, 1),
                       reward=float(reward), terminated=terminated, truncated=truncated)
            row.update(policy_input_time=t, observation_velocity_mode=velocity_input,
                       reference_velocity=reference_velocity, reference_velocity_post=post_reference_velocity,
                       velocity_error_before=before['velocity'] - reference_velocity,
                       velocity_error=after['velocity'] - post_reference_velocity,
                       policy_raw_observation=policy_obs.copy(), policy_raw_velocity=policy_obs[VELOCITY_SLICE].copy())
            row.update({f'{key}_before': value for key, value in before.items()})
            row.update(desired_velocity_before=env.desired_velocity(before['position']),
                       internal_velocity_reference_mode=desired_velocity_semantics(config)['mode'],
                       environment_observation_velocity_mode=velocity_semantics(config)['mode'],
                       environment_reward_velocity_mode=velocity_reward_semantics(config)['mode'],
                       desired_velocity=env.desired_velocity(after['position']),
                       internal_velocity_error=after['velocity']-env.desired_velocity(after['position']))
            row.update(after)
            for key, attr in (('motor_thrust_command', '_last_f_cmd'), ('motor_thrust', '_last_f'),
                              ('wrench_command', '_last_wrench_cmd'), ('wrench_actual', '_last_wrench_actual'),
                              ('motor_omega', '_last_omega'), ('motor_command', '_last_motor_cmd')):
                row[key] = np.asarray(getattr(env, attr)).copy()
            row['motor_thrust_unclipped'] = env.B_pinv @ env._last_wrench_cmd
            row['motor_allocation_clipped'] = (np.abs(row['motor_thrust_unclipped']-row['motor_thrust_command']) > 1e-12)
            row['motor_command_at_lower_bound'] = row['motor_command'] <= 1e-9
            row['motor_command_at_upper_bound'] = row['motor_command'] >= 1-1e-9
            row['motor_actual_at_thrust_lower_bound'] = row['motor_thrust'] <= env.thrust_min+1e-9
            row['motor_actual_at_thrust_upper_bound'] = row['motor_thrust'] >= env.thrust_max-1e-9
            row.update({f'{group}_{key}': value for group, values in info.items()
                        if group.startswith('reward_') for key, value in values.items()})
            if observer is not None:
                observer.after_step(env, row)
            rows.append(row)
            if not all(np.all(np.isfinite(v)) for v in after.values()):
                raise ValueError('non-finite simulator state')
            if terminated:
                reasons = termination_reasons(env, after, reference)
            if terminated or truncated:
                break
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        env.close()
    return rows, initial, error, reasons


def rmse(errors):
    values = np.asarray(errors, dtype=float).reshape((-1, 3))
    if not len(values) or not np.all(np.isfinite(values)):
        return dict(position_rmse_xy=None, position_rmse_z=None, position_rmse_total=None)
    squares = np.mean(values**2, axis=0)
    return dict(position_rmse_xy=float(np.sqrt(sum(squares[:2]))),
                position_rmse_z=float(np.sqrt(squares[2])), position_rmse_total=float(np.sqrt(sum(squares))))


def summarize(rows, case, thresholds, error=None, reasons=()):
    n = len(rows)
    times = np.asarray([r['time_post'] for r in rows])
    errors = np.asarray([r['position'] - r['reference_post'] for r in rows]).reshape((-1, 3))
    positions = np.asarray([r['position'] for r in rows]).reshape((-1, 3))
    speeds = np.asarray([r['velocity'] for r in rows]).reshape((-1, 3))
    terminated = any(r['terminated'] for r in rows)
    truncated = any(r['truncated'] for r in rows)
    duration = float(times[-1]) if n else 0.
    completed = bool(not error and not terminated and n and abs(duration - case.horizon) < 1e-8)
    result = dict(case=case.name, sample_count=n, expected_duration_sec=case.horizon,
                  internal_velocity_reference_mode=rows[0].get('internal_velocity_reference_mode','absolute') if n else None,
                  actual_duration_sec=duration, completed=completed, terminated=terminated, truncated=truncated,
                  partial=not completed, metric_scope='full' if completed else 'partial_observed_only',
                  termination_reasons=list(reasons), error=error,
                  end_reason=error or (';'.join(reasons) if terminated else 'horizon' if completed else 'early_truncation' if truncated else 'no_samples'),
                  **rmse(errors))
    result.update(motion_metrics(rows, case))
    result.update(actuation_metrics(rows))
    result['completed_count'] = int(completed)
    result['trial_count'] = 1
    result['actual_speed_rms_m_s'] = float(np.sqrt(np.mean(np.sum(speeds**2, axis=1)))) if n else None
    result['internal_velocity_error_rmse_m_s'] = (float(np.sqrt(np.mean([
        np.dot(r['internal_velocity_error'], r['internal_velocity_error']) for r in rows])))
        if n and 'internal_velocity_error' in rows[0] else None)
    if case.name in ('circle', 'circle-air'):
        phases = np.asarray([r['phase_post'] for r in rows])
        # Evaluate at t+dt with phase(t+dt), including phase boundaries.
        mask = phases == 'CIRCLE'
        result.update(altitude_metrics(positions, errors))
        result['circle_phase'] = dict(sample_count=int(mask.sum()), partial=not completed,
                                      **rmse(errors[mask]), **altitude_metrics(positions[mask], errors[mask]),
                                      **motion_metrics([r for r, keep in zip(rows, mask) if keep], case))
        result['phases'] = {phase: dict(sample_count=int(np.sum(phases == phase)), **rmse(errors[phases == phase]))
                            for phase in dict.fromkeys(phases)}
        return result
    result.update(last_2s_position_rmse_xy=None, last_2s_position_rmse_z=None,
                  last_2s_position_rmse_total=None, last_2s_speed_rms=None,
                  last_2s_internal_velocity_error_rms_m_s=None,
                  last_2s_position_std_x=None, last_2s_position_std_y=None, last_2s_position_std_z=None,
                  first_position_entry_s=None, settling_time_s=None, settled=False, overshoot_m=None)
    if n and np.all(np.isfinite(errors)) and np.all(np.isfinite(speeds)):
        # Event times include the initial observation at t=0; RMSE still uses
        # post-transition samples only, as declared in the manifest.
        event_times = np.r_[rows[0]['time'], times]
        event_errors = np.vstack((rows[0]['position_before'] - rows[0]['reference'], errors))
        event_speeds = np.vstack((rows[0]['velocity_before'], speeds))
        within_position = np.linalg.norm(event_errors, axis=1) <= thresholds.position_band_m
        indices = np.flatnonzero(within_position)
        result['first_position_entry_s'] = float(event_times[indices[0]]) if indices.size else None
        result['overshoot_m'] = float(max(0., np.max(positions[:, 0] - case.goal[0])))
        if completed:
            # Exactly the fixed final window (6,8] by default; never the last
            # two seconds of a failed partial episode. No padded samples.
            tail = times > case.horizon - thresholds.tail_sec + 1e-9
            result.update({f'last_2s_{key}': value for key, value in rmse(errors[tail]).items()})
            result['last_2s_speed_rms'] = float(np.sqrt(np.mean(np.sum(speeds[tail]**2, axis=1))))
            if all('internal_velocity_error' in r for r in rows):
                velocity_errors=np.array([r['internal_velocity_error'] for r in rows])
                result['last_2s_internal_velocity_error_rms_m_s'] = float(np.sqrt(np.mean(np.sum(velocity_errors[tail]**2,axis=1))))
            for axis, std in zip('xyz', np.std(positions[tail], axis=0, ddof=0)):
                result[f'last_2s_position_std_{axis}'] = float(std)
            good = within_position & (np.linalg.norm(event_speeds, axis=1) <= thresholds.speed_band_m_s)
            suffix = np.logical_and.accumulate(good[::-1])[::-1]
            candidates = np.flatnonzero(suffix & (case.horizon - event_times >= thresholds.minimum_settle_sec - 1e-9))
            if candidates.size:
                result.update(settled=True, settling_time_s=float(event_times[candidates[0]]))
    return result


def altitude_metrics(positions, errors):
    """Signed post-state altitude deviation from the same-time reference, in metres."""
    keys = ('altitude_min_m', 'altitude_max_m', 'altitude_error_mean_m',
            'altitude_error_min_m', 'altitude_error_max_m', 'altitude_error_max_abs_m')
    if not len(positions):
        return dict.fromkeys(keys)
    z, ez = positions[:, 2], errors[:, 2]
    return dict(zip(keys, map(float, (z.min(), z.max(), ez.mean(), ez.min(), ez.max(), np.abs(ez).max()))))


def motion_metrics(rows, case):
    keys = ('velocity_rmse', 'horizontal_speed_rms_m_s', 'roll_rms_deg', 'pitch_rms_deg')
    if not rows:
        return dict.fromkeys(keys)
    velocity = np.array([r['velocity'] for r in rows])
    reference = np.array([case.reference_velocity(r['time_post']) for r in rows])
    rpy = np.array([quaternion_to_euler_deg(r['quaternion']) for r in rows])
    return dict(zip(keys, map(float, (np.sqrt(np.mean(np.sum((velocity-reference)**2, axis=1))),
                                    np.sqrt(np.mean(np.sum(velocity[:, :2]**2, axis=1))),
                                    np.sqrt(np.mean(rpy[:, 0]**2)), np.sqrt(np.mean(rpy[:, 1]**2))))))


def actuation_metrics(rows):
    """Last-physics-substep signals; lag/tracking error is NOT called saturation."""
    result = {'roll_max_abs_deg': None, 'pitch_max_abs_deg': None, 'tilt_max_deg': None,
              'motor_saturation_sample_convention': 'last physics substep of each control interval',
              'motor_allocation_clip_tolerance_n': 1e-12, 'motor_command_bound_tolerance': 1e-9}
    if rows:
        rpy = np.array([quaternion_to_euler_deg(r['quaternion']) for r in rows])
        q = np.array([r['quaternion'] for r in rows])
        result.update(roll_max_abs_deg=float(np.max(np.abs(rpy[:,0]))),
                      pitch_max_abs_deg=float(np.max(np.abs(rpy[:,1]))),
                      tilt_max_deg=float(np.max(np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1))))))
    for key in ('motor_allocation_clipped', 'motor_command_at_lower_bound', 'motor_command_at_upper_bound',
                'motor_actual_at_thrust_lower_bound', 'motor_actual_at_thrust_upper_bound'):
        available = bool(rows) and all(key in r for r in rows)
        values = np.array([r[key] for r in rows], dtype=bool) if available else None
        result[f'{key}_fraction_any_motor'] = float(np.mean(np.any(values,axis=1))) if available else None
        result[f'{key}_fraction_per_motor'] = np.mean(values,axis=0).tolist() if available else None
    if rows and all('motor_thrust' in r for r in rows):
        force=np.array([r['motor_thrust'] for r in rows])
        result.update(motor_thrust_min_n_per_motor=force.min(axis=0).tolist(),
                      motor_thrust_max_n_per_motor=force.max(axis=0).tolist())
    return result


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_rollout(path, rows):
    def flat(row):
        out = {}
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                out.update({f'{key}_{i}': float(v) for i, v in enumerate(value.flat)})
            else:
                out[key] = value
        return out
    with Path(path).open('x', newline='') as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(flat(rows[0])))
            writer.writeheader()
            writer.writerows(flat(r) for r in rows)
        else:
            file.write('time,time_post,phase,phase_post\n')


def write_summary(directory, summaries):
    write_json(directory / 'summary.json', summaries)
    flat = []
    for row in summaries:
        item = {}
        for key, value in row.items():
            if key == 'circle_phase':
                item.update({f'circle_phase_{k}': v for k, v in value.items()})
            elif isinstance(value, (dict, list)):
                item[key] = json.dumps(value)
            else:
                item[key] = value
        flat.append(item)
    with (directory / 'summary.csv').open('w', newline='') as file:
        if flat:
            fields = list(dict.fromkeys(k for row in flat for k in row))
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/eval_dr_transfer.yaml')
    p.add_argument('--model', action='append', required=True, metavar='LABEL=ZIP')
    p.add_argument('--manifest', action='append', default=[], metavar='LABEL=JSON', help='optional explicit training manifest')
    p.add_argument('--normalization', action='append', default=[], metavar='LABEL=PATH_OR_none', help='saved VecNormalize statistics or explicit declaration of raw observations')
    p.add_argument('--cases', nargs='+', choices=CASE_NAMES, default=list(DEFAULT_CASES))
    p.add_argument('--velocity-inputs', nargs='+', choices=VELOCITY_INPUTS, default=['absolute'],
                   help='absolute: native environment input unchanged; error: v_ref subtraction on legacy absolute inputs only')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path, help='parent for a unique run directory; never overwritten')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--position-band', type=float, default=.005)
    p.add_argument('--speed-band', type=float, default=.02)
    p.add_argument('--settle-hold-sec', type=float, default=1.)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.seed < 0:
            raise ValueError('seed must be nonnegative')
        if len(set(args.velocity_inputs)) != len(args.velocity_inputs):
            raise ValueError('duplicate velocity input modes')
        config = load_config(args.config)
        from .velocity_reference import observation_contract, velocity_semantics, velocity_reward_semantics
        if velocity_semantics(config)['mode'] != 'absolute' and args.velocity_inputs != ['absolute']:
            raise ValueError('new velocity-error policies require native inputs; --velocity-inputs error would double-subtract')
        validate_common_config(config)
        thresholds = Thresholds(args.position_band, args.speed_band, args.settle_hold_sec)
        paths, manifests, normalizations = labeled(args.model), labeled(args.manifest), labeled(args.normalization)
        if not paths:
            raise ValueError('at least one model label is required')
        if (manifests.keys() | normalizations.keys()) - paths.keys():
            raise ValueError('manifest/normalization label has no matching model')
        cases = make_cases(config, args.cases)
        policies = [load_frozen_policy(label, path, config, manifest=manifests.get(label),
                                      normalization=normalizations.get(label)) for label, path in paths.items()]
        for policy in policies:
            policy.provenance['display_name'] = display_name(policy.provenance['label'])
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    dt = 1 / config.environment.policy_hz
    manifest = dict(schema_version=2, created_at=datetime.now(timezone.utc).isoformat(),
                    status='dry_run' if args.dry_run else 'running', common_resolved_config=config.resolved_dict(),
                    seed=args.seed, deterministic=True, thresholds=asdict(thresholds),
                    git=_git_metadata(ROOT),
                    source_sha256={name: sha256(ROOT/name) for name in
                                   ('crazyflie_rl/dr_transfer.py', 'crazyflie_rl/dr_policy.py',
                                    'crazyflie_rl/missions.py', 'crazyflie_rl/environment.py', 'crazyflie_rl/plotting.py',
                                    'crazyflie_rl/velocity_reference.py','crazyflie_rl/config.py')},
                    velocity_inputs=args.velocity_inputs,
                    rollouts=[dict(model_label=label, observation_velocity_mode=mode, case=c.name)
                              for c in cases for label in paths for mode in args.velocity_inputs],
                    velocity_reward_semantics=velocity_reward_semantics(config),
                    policy_input_contract=dict(environment_raw=observation_contract(config), velocity_semantics=velocity_semantics(config), velocity_slice=[3, 6],
                        internal_reference='norm_clip(-Kp*(p-p_target),v_max); recompute at each state; no target differentiation',
                        velocity_frame='world', velocity_unit='m/s', absolute='native environment input unchanged', error='v_world-v_ref_world (legacy only)',
                        error_mode_interpretation='inference input intervention on an absolute-velocity-trained policy',
                        order=['environment raw observation', 'copy and replace velocity in error mode only',
                               'frozen saved observation normalization and clipping if present', 'deterministic policy predict'],
                        normalization_frozen=True, physical_state_and_reward_unchanged=True),
                    reference_velocity=dict(implementation='CircleMission.reference_velocity; derivative of ramped_phase',
                        circle='r*theta_dot*[-sin(theta),cos(theta),0]; theta_dot=omega*(1-cos(pi*s/ramp))/2 in ramp, omega afterwards',
                        step='zero for regulation targets; no impulse', hold='zero from HOLD boundary inclusive',
                        boundary='CIRCLE/HOLD velocity discontinuity; right-phase value, no spike',
                        timing='position and velocity use the same case time offset; input at t, metrics at t+dt'),
                    metric_units=dict(velocity_rmse='m/s', horizontal_speed_rms_m_s='m/s',
                                      roll_rms_deg='deg', pitch_rms_deg='deg'),
                    models=[p.provenance for p in policies],
                    cases={c.name: c.description(dt) for c in cases}, initial_snapshots={},
                    actual_overrides={c.name: {'episode_sec': c.horizon, 'initial_state': c.description(dt),
                                              'dist_torque_body': [0., 0., 0.]} for c in cases},
                    timing={'control': 'pre-state/ref/action at t; applied reference held on [t,t+dt]',
                            'metrics': 'post-state and mission.reference(t+dt); phase(t+dt)',
                            'thrust': 'last physics-substep requested/actual values of [t,t+dt]',
                            'reward': 'environment reward against held control reference, not post evaluation reference'},
                    termination_policy='stop on first terminated or truncated, including floor TAKEOFF guards',
                    limitations=['No automatic causal interpretation.',
                                 'Same-model/seed repetitions are determinism checks, not independent training runs.',
                                 'Ground start z=0.02 can immediately violate the unchanged min_altitude=0.2 guard.',
                                 'Historical training metadata without a checkpoint hash is path-associated, not cryptographic proof.'])
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    root = args.output_dir or config.paths.artifact_root / 'runs'
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='dr-transfer-', dir=root))
    write_json(directory / 'manifest.json', manifest)
    summaries = []
    try:
        for case in cases:
            schedule = reference_sequence(case, dt)
            schedule_path = directory / f'{case.name}-reference.npz'
            np.savez_compressed(schedule_path, **schedule)
            manifest['cases'][case.name]['reference_file'] = schedule_path.name
            manifest['cases'][case.name]['reference_sha256'] = sha256(schedule_path)
            manifest['initial_snapshots'][case.name] = {}
            first_snapshot = None
            plotted = {}
            for policy in policies:
                for mode in args.velocity_inputs:
                    label = policy.provenance['label']
                    rows, snapshot, error, reasons = run_case(config, case, policy, args.seed, mode)
                    key = label if args.velocity_inputs == ['absolute'] else f'{label}/{mode}'
                    filename = f'{case.name}-{label}' + ('' if args.velocity_inputs == ['absolute'] else f'--velocity-{mode}')
                    manifest['initial_snapshots'][case.name][key] = snapshot
                    if snapshot is not None:
                        if first_snapshot is not None and snapshot != first_snapshot:
                            raise RuntimeError(f'{case.name}: initial snapshots differ between policies')
                        first_snapshot = snapshot
                    for k, row in enumerate(rows):
                        np.testing.assert_array_equal(row['reference'], schedule['reference'][k])
                        np.testing.assert_array_equal(row['reference_post'], schedule['reference'][k + 1])
                        np.testing.assert_array_equal(row['reference_velocity'], schedule['reference_velocity'][k])
                        np.testing.assert_array_equal(row['reference_velocity_post'], schedule['reference_velocity'][k + 1])
                        row['model_label'] = label
                        row['case'] = case.name
                    write_rollout(directory / f'{filename}.csv', rows)
                    result = dict(label=label, display_name=display_name(label), observation_velocity_mode=mode,
                                  **summarize(rows, case, thresholds, error, reasons))
                    summaries.append(result)
                    input_label = ('position velocity error (native)'
                                   if velocity_semantics(config)['mode'] == 'position_error' else mode)
                    plotted[f'{display_name(label)} / {input_label}'] = rows
                    print(f"{case.name} {display_name(label)} / {input_label}: completed={result['completed']} duration={result['actual_duration_sec']:.2f}s reason={result['end_reason']}")
                    if sha256(policy.provenance['path']) != policy.provenance['sha256']:
                        raise RuntimeError('checkpoint changed during evaluation')
                    write_summary(directory, summaries)
                    write_json(directory / 'manifest.json', manifest)
            save_transfer_comparison_plot(directory / f'{case.name}-comparison.png', plotted, schedule, case.name)
        manifest['status'] = 'completed' if all(r['error'] is None for r in summaries) else 'failed'
        manifest['initial_snapshots_equal'] = all(all(s is not None for s in group.values()) for group in manifest['initial_snapshots'].values())
        manifest['reference_sequences_equal'] = True
        manifest['checkpoint_hashes_unchanged'] = True
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_summary(directory, summaries)
        write_json(directory / 'manifest.json', manifest)
        print(f'results: {directory}')
    return 0 if manifest['status'] == 'completed' else 1

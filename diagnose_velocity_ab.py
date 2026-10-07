"""Read-only velocity-channel diagnostics. Never calls PPO.learn or an optimizer.

Writes to a new directory and uses the existing frozen loader and environment.
The ordinary model compatibility checks stay enabled.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from crazyflie_rl.config import load_config
from crazyflie_rl.dr_policy import load_frozen_policy, sha256
from crazyflie_rl.velocity_reference import ABSOLUTE_CONTRACT, VELOCITY_SLICE

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT/'artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip'
D_RUN = ROOT/'artifacts/runs/ppo_e2e_hover_position-velocity-error-nominal_seed42_20261001-210046'


class RestoredAbsolutePolicy:
    """Explicit diagnostic adapter: restore RAW v before frozen preprocessing."""

    def __init__(self, policy):
        if policy.provenance['observation_contract'] != ABSOLUTE_CONTRACT:
            raise ValueError('restoration diagnostic requires an absolute-velocity policy')
        self.policy = policy

    def restore(self, error_observation, desired_velocity):
        obs, desired = np.asarray(error_observation), np.asarray(desired_velocity)
        if obs.shape != (15,) or desired.shape != (3,) or not np.isfinite(obs).all() or not np.isfinite(desired).all():
            raise ValueError('expected finite raw 15D observation and desired world velocity')
        restored = obs.copy()
        restored[VELOCITY_SLICE] = obs[VELOCITY_SLICE] + desired
        return restored

    def predict(self, error_observation, desired_velocity):
        return self.policy.predict(self.restore(error_observation, desired_velocity))


def observation_diagnostic(checkpoint, output, seed=42):
    from crazyflie_rl.dr_transfer import Case, EvaluationAdapter
    from crazyflie_rl.environment import CrazyflieResidualEnv
    absolute = load_config(ROOT/'configs/eval_dr_transfer.yaml')
    error = load_config(ROOT/'configs/eval_position_velocity_error.yaml')
    policy = load_frozen_policy('historical_baseline_diagnostic', str(checkpoint), absolute)
    adapter = RestoredAbsolutePolicy(policy)
    old_env, new_env = CrazyflieResidualEnv(config=absolute), CrazyflieResidualEnv(config=error)
    rng = np.random.default_rng(seed)
    originals, errors, restored, desireds, old_actions, restored_actions, clipped_flags = ([] for _ in range(7))
    try:
        case = Case('hover', 8., goal=(0., 0., 1.))
        for env in (old_env, new_env):
            EvaluationAdapter(env).reset_to_case_initial_state(case, seed)
        policy.bind(old_env)
        snapshots = [EvaluationAdapter(env).snapshot() for env in (old_env, new_env)]
        for clipped in (False, True):
            for index in range(256):
                direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
                norm = rng.uniform(.45, 1.5) if clipped else rng.uniform(0, .30)
                ep = norm * direction if index else (np.array([.5, 0., 0.]) if clipped else np.zeros(3))
                position = old_env.pos_des + ep
                velocity, omega = rng.normal(0, .5, 3), rng.normal(0, .3, 3)
                axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
                half_angle = rng.uniform(-.2, .2)
                quat = np.r_[np.cos(half_angle), np.sin(half_angle)*axis]
                original = old_env._obs(position, quat, velocity, omega)
                converted = new_env._obs(position, quat, velocity, omega)
                saved = converted.copy()
                desired = new_env.desired_velocity(position)
                reconstruction = adapter.restore(converted, desired)
                originals.append(original); errors.append(converted); restored.append(reconstruction)
                desireds.append(desired); clipped_flags.append(clipped)
                old_actions.append(policy.predict(original))
                restored_actions.append(adapter.predict(converted, desired))
                np.testing.assert_array_equal(converted, saved)
        for env, before in zip((old_env, new_env), snapshots):
            assert EvaluationAdapter(env).snapshot() == before
    finally:
        old_env.close(); new_env.close()
    original, restored = np.asarray(originals), np.asarray(restored)
    action, restored_action = np.asarray(old_actions), np.asarray(restored_actions)
    mask = np.asarray(clipped_flags)
    report = {'sample_kind': 'synthetic physical states, real frozen baseline PPO inference',
              'seed': seed, 'sample_count': len(mask), 'observation_atol': 5e-7, 'action_atol': 5e-6,
              'adapter_order': 'raw velocity error + desired velocity -> original raw observation -> frozen normalization/clipping -> deterministic predict',
              'compatibility_checks_enabled': True, 'input_arrays_and_simulator_state_unchanged': True,
              'policy': policy.provenance}
    for name, select in [('all', np.ones(len(mask), bool)), ('unclipped', ~mask), ('clipped', mask)]:
        report[name] = {'samples': int(select.sum()),
                        'max_observation_abs_error': float(np.max(np.abs(original[select]-restored[select]))),
                        'max_action_abs_error': float(np.max(np.abs(action[select]-restored_action[select])))}
    np.testing.assert_allclose(original, restored, rtol=0, atol=report['observation_atol'])
    np.testing.assert_allclose(action, restored_action, rtol=0, atol=report['action_atol'])
    keep = [0, 1, 2, *range(6, 15)]
    np.testing.assert_array_equal(original[:, keep], np.asarray(errors)[:, keep])
    report['passed'] = True
    np.savez_compressed(output/'observation_restoration.npz', original=original, converted=errors,
                        restored=restored, desired_velocity=desireds, original_action=action,
                        restored_action=restored_action, clipped=mask)
    return report


def training_audit(run, output):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    records = [json.loads(p.read_text()) for p in sorted((run/'metrics').glob('*evaluation-step*.json'))]
    if not records:
        raise ValueError(f'no saved policy evaluation records in {run}')
    manifest = json.loads(next((run/'manifests').glob('*manifest_*.json')).read_text())
    config = manifest['resolved_config']
    hz = config['environment']['policy_hz']
    with (output/'d_evaluation_history.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])+['mean_survival_sec'])
        writer.writeheader()
        for row in records: writer.writerow(dict(row, mean_survival_sec=row['policy_mean_episode_length']/hz))
    event = next(run.rglob('events.out.tfevents*'))
    accumulator = EventAccumulator(str(event), size_guidance={'scalars': 0})
    accumulator.Reload()
    tags = accumulator.Tags()['scalars']
    selected = [tag for tag in tags if tag.startswith('train/') or tag.startswith('rollout/')]
    scalars, scalar_summary = {}, {}
    for tag in selected:
        points = accumulator.Scalars(tag)
        scalars[tag] = [{'step': x.step, 'value': float(x.value)} for x in points]
        values = np.array([x.value for x in points])
        scalar_summary[tag] = dict(count=len(points), first=float(values[0]), last=float(values[-1]),
                                   minimum=float(values.min()), maximum=float(values.max()))
    write_json(output/'d_training_scalars.json', scalars)
    with (output/'d_training_scalars.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['tag', 'step', 'value'])
        for tag, points in scalars.items():
            for point in points: writer.writerow([tag, point['step'], point['value']])
    zip_paths = sorted(run.rglob('*.zip'))
    return {'run': str(run.resolve()), 'manifest_status': manifest['status'], 'result': manifest['result'],
            'evaluation_settings': config['evaluation'], 'termination_settings': config['environment']['termination'],
            'evaluation_count': len(records), 'first_evaluation': records[0], 'last_evaluation': records[-1],
            'best_mean_survival_record': max(records, key=lambda x: x['policy_mean_episode_length']),
            'min_disqualifications_record': min(records, key=lambda x: x['policy_disqualifications']),
            'zero_disqualification_evaluations': sum(x['policy_disqualifications'] == 0 for x in records),
            'full_mean_horizon_evaluations': sum(x['policy_mean_episode_length'] >= hz*config['environment']['episode_sec'] for x in records),
            'saved_as_best_count': sum(x['saved_as_best'] for x in records),
            'best_rule': 'strict score improvement AND zero tail-tilt disqualifications; beating floor or full horizon is not an extra gate',
            'checkpoints': [{'path': str(p.resolve()), 'sha256': sha256(p)} for p in zip_paths],
            'scalar_summary': scalar_summary,
            'training_episode_length_recorded': 'rollout/ep_len_mean' in tags,
            'termination_cause_history': 'not recorded in training/evaluation aggregates; final test max_tilt does not establish historical causes',
            'policy_std_definition': 'SB3 mean exp(log_std), not measured realized action std or motor saturation'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def plot_training_history(output):
    """Plot saved statistics only; no policy execution or training."""
    from crazyflie_rl.plotting import _pyplot
    target = output/'d_training_diagnostics.png'
    if target.exists():
        raise FileExistsError(target)
    with (output/'d_evaluation_history.csv').open() as f:
        history = list(csv.DictReader(f))
    scalars = json.loads((output/'d_training_scalars.json').read_text())
    plt = _pyplot()
    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    try:
        x = [int(r['timestep']) for r in history]
        axes[0].plot(x, [float(r['mean_survival_sec']) for r in history], label='mean deterministic evaluation survival')
        axes[0].axhline(8, color='gray', ls='--', label='evaluation horizon')
        axes[0].set_ylabel('seconds'); axes[0].legend()
        axes[1].plot(x, [int(r['policy_disqualifications']) for r in history])
        axes[1].set_ylabel('tail tilt disqualified / 30')
        for axis, tag in zip(axes[2:], ['train/approx_kl', 'train/clip_fraction', 'train/std']):
            if tag in scalars:
                axis.plot([p['step'] for p in scalars[tag]], [p['value'] for p in scalars[tag]])
            axis.set_ylabel(tag)
        for axis in axes: axis.grid(alpha=.25)
        axes[-1].set_xlabel('recorded training timestep')
        fig.suptitle('D: saved training diagnostics (no additional learning)')
        fig.tight_layout()
        fig.savefig(target, dpi=140)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--d-run', type=Path, default=D_RUN)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', type=Path, default=ROOT/'artifacts/runs')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='velocity-ab-nonlearning-', dir=args.output_root))
    before = sha256(args.baseline)
    try:
        observation = observation_diagnostic(args.baseline, output, args.seed)
        write_json(output/'observation_restoration.json', observation)
        training = training_audit(args.d_run, output)
        write_json(output/'d_training_audit.json', training)
        plot_training_history(output)
        assert sha256(args.baseline) == before
        write_json(output/'status.json', {'status': 'completed', 'learning_executed': False,
                                         'baseline_hash_unchanged': True})
        print(json.dumps(observation['all'], indent=2))
    except Exception as exc:
        write_json(output/'status.json', {'status': 'failed', 'error': str(exc), 'learning_executed': False})
        raise
    finally:
        print(f'results: {output}')


if __name__ == '__main__':
    main()

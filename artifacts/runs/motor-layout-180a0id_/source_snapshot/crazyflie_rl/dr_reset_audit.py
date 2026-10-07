"""Replay archived reset configurations and inspect existing transfer CSVs, without training."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import yaml

from .config import _build_config
from .dr_policy import sha256, training_manifest
from .dr_transfer import ROOT, display_name, write_json
from .environment import CrazyflieResidualEnv
from .plotting import quaternion_to_euler_deg


def archived_config(checkpoint):
    """Validate the archived resolved YAML against its manifest, never current train YAML."""
    path, manifest = training_manifest(checkpoint)
    if path is None:
        raise ValueError(f'archived training manifest missing: {checkpoint}')
    saved = checkpoint.parent.parent / manifest['resolved_config_path']
    raw = yaml.safe_load(saved.read_text())
    if raw != manifest['resolved_config']:
        raise ValueError(f'archived YAML differs from manifest: {saved}')
    data = copy.deepcopy(raw)
    source = Path(data.pop('source_path'))
    # resolved_dict serializes these dataclass fields flat; the validated input
    # schema nests them. All values must round-trip exactly, with no inheritance.
    actuator = data['actuator']
    actuator['thrust_polynomial'] = {
        key.removeprefix('thrust_polynomial_'): actuator.pop(key)
        for key in list(actuator) if key.startswith('thrust_polynomial_')}
    config = _build_config(data, source)
    if config.resolved_dict() != raw:
        raise ValueError('archived config reconstruction changed values')
    return config, manifest, dict(manifest_path=str(path), manifest_sha256=sha256(path),
                                 resolved_config_path=str(saved), resolved_config_sha256=sha256(saved))


def distribution_contract(config):
    e = config.environment
    new = e.initial_pose_randomization
    if new is None:
        a = e.position_perturbation
        return dict(sampler='legacy', precedence='new config absent: legacy branch',
                    distribution='independent Uniform(-a,a) per axis', bound_parameter_m=a,
                    axis_absolute_bounds_m=[a] * 3, theoretical_norm_max_m=float(np.sqrt(3)*a),
                    theoretical_norm_rms_m=a, attitude_randomized=e.attitude_perturbation_deg > 0,
                    attitude_max_angle_deg=e.attitude_perturbation_deg)
    radius = new.position.max_norm_m if new.enabled and new.position.enabled else 0.
    return dict(sampler='new', precedence='present new config overrides legacy, even when disabled',
                ignored_legacy_position_perturbation_m=e.position_perturbation,
                distribution='normalized Gaussian direction; radius R*Uniform(0,1); NOT volume-uniform',
                bound_parameter_m=radius, axis_absolute_bounds_m=[radius]*3,
                theoretical_norm_max_m=radius, theoretical_norm_rms_m=float(radius/np.sqrt(3)),
                attitude_randomized=bool(new.enabled and new.attitude.enabled and new.attitude.max_angle_deg > 0),
                attitude_max_angle_deg=new.attitude.max_angle_deg if new.enabled and new.attitude.enabled else 0.)


def statistics(values):
    x = np.asarray(values)
    return dict(min=float(x.min()), max=float(x.max()), mean=float(x.mean()),
                std=float(x.std()), rms=float(np.sqrt(np.mean(x*x))),
                p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)))


def sample_resets(config, count, seed):
    """Actual env.reset: seed once, then advance both environment RNG streams."""
    env = CrazyflieResidualEnv(config=config)
    positions, quaternions, velocities, motors = [], [], [], []
    initial = None
    try:
        for index in range(count):
            env.reset(seed=seed if index == 0 else None)
            positions.append(env.data.qpos[:3].copy() - env.pos_des)
            quaternions.append(env.data.qpos[3:7].copy())
            velocities.append(env.data.qvel.copy())
            motors.append(env._last_omega.copy())
            snapshot = env.actuator_snapshot()
            if initial is None:
                initial = snapshot
            elif snapshot != initial:
                raise ValueError('unexpected changing actuator state/parameters in fixed-reset audit')
        arrays = dict(position_offset=np.asarray(positions), quaternion_wxyz=np.asarray(quaternions),
                      generalized_velocity=np.asarray(velocities), motor_omega_rad_s=np.asarray(motors))
        report = dict(sample_count=count, seed=seed, seed_protocol='seed first reset only; subsequent reset(seed=None)',
                      unique_position_count=len(np.unique(arrays['position_offset'], axis=0)),
                      axes={axis: statistics(arrays['position_offset'][:, i]) for i, axis in enumerate('xyz')},
                      norm=statistics(np.linalg.norm(arrays['position_offset'], axis=1)),
                      all_level=bool(np.all(arrays['quaternion_wxyz'] == [1, 0, 0, 0])),
                      all_zero_velocity=bool(np.all(arrays['generalized_velocity'] == 0)),
                      motor_omega_constant=bool(np.all(arrays['motor_omega_rad_s'] == arrays['motor_omega_rad_s'][0])),
                      initial_motor_omega_rad_s=arrays['motor_omega_rad_s'][0].tolist(),
                      initial_motor_thrust_N=env._last_f.tolist(),
                      hover_total_thrust_N=float(env.mass * env.gravity),
                      hover_equilibrium_verified=bool(np.allclose(env._last_f.sum(), env.mass * env.gravity)
                                                      and np.all(arrays['motor_omega_rad_s'] > 0)),
                      actuator_snapshot=initial)
        return report, arrays
    finally:
        env.close()


def read_rows(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def vector(row, key, count=3):
    return np.array([float(row[f'{key}_{i}']) for i in range(count)])


def inspect_rollouts(directory, label):
    tail_path = directory / f'step-005-{label}.csv'
    step_path = directory / f'step-050-{label}.csv'
    summaries = json.loads((directory / 'summary.json').read_text())
    summary = next(s for s in summaries if s['label'] == label and s['case'] == 'step-005')
    tail = [r for r in read_rows(tail_path) if 6 + 1e-9 < float(r['time_post']) <= 8 + 1e-9]
    tail_report = None
    if summary['completed'] and tail:
        errors = np.array([vector(r, 'position')-vector(r, 'reference_post') for r in tail])
        mean, std = errors.mean(axis=0), errors.std(axis=0)
        bias_mse, variance = float(mean @ mean), float(std @ std)
        tail_report = dict(window='post-state (6,8] seconds', sample_count=len(tail),
                           mean_error_xyz_m=mean.tolist(), std_error_xyz_m=std.tolist(),
                           position_rmse_total_m=float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
                           mean_offset_mse_m2=bias_mse, fluctuation_mse_m2=variance,
                           mean_offset_fraction_of_mse=bias_mse/(bias_mse+variance) if bias_mse+variance else None)
    rows = read_rows(step_path)
    terminal = None
    if rows:
        row = rows[-1]
        def state(suffix, reference):
            return dict(position_m=vector(row, 'position'+suffix).tolist(),
                        position_error_m=(vector(row, 'position'+suffix)-vector(row, reference)).tolist(),
                        rpy_deg=quaternion_to_euler_deg(vector(row, 'quaternion'+suffix, 4)).tolist(),
                        omega_body_rad_s=vector(row, 'omega'+suffix).tolist())
        terminal = dict(control_time_s=float(row['time']), post_time_s=float(row['time_post']),
                        pre_control_state=state('_before', 'reference'), post_state=state('', 'reference_post'),
                        action=vector(row, 'action', 4).tolist(),
                        motor_thrust_command_N=vector(row, 'motor_thrust_command', 4).tolist(),
                        motor_thrust_actual_N=vector(row, 'motor_thrust', 4).tolist(),
                        wrench_command=vector(row, 'wrench_command', 4).tolist(),
                        wrench_actual=vector(row, 'wrench_actual', 4).tolist(),
                        terminated=row['terminated'].lower() == 'true', truncated=row['truncated'].lower() == 'true',
                        allocator_saturation='unmeasured: no pre-clipping allocation or saturation flag in CSV',
                        thrust_timing='last physics substep of final control interval, not interval mean')
    return dict(sources={str(p): sha256(p) for p in (tail_path, step_path, directory/'summary.json')},
                step_005_tail=tail_report, step_050_final_transition=terminal)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT/'artifacts/runs/dr-transfer-8rfoihws')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/runs')
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.seed < 0:
        parser.error('samples must be positive and seed nonnegative')
    source = args.source_run.resolve(strict=True)
    source_manifest = json.loads((source/'manifest.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='dr-reset-audit-', dir=args.output_dir))
    report = dict(source_evaluation=str(source), source_manifest_sha256=sha256(source/'manifest.json'),
                  scope='reset distribution replay using archived configs and current sampler, NOT visited rollout state distribution',
                  source_files={str(ROOT/p): sha256(ROOT/p) for p in
                                ('crazyflie_rl/environment.py', 'crazyflie_rl/initial_pose.py', 'crazyflie_rl/config.py')},
                  limitations=['Archived manifests record dirty working trees; exact historical executable source is unconfirmed.',
                               'Training manifests are path-associated; they do not contain hashes of these checkpoints.',
                               'Two checkpoints do not identify a causal effect of DR; both used reset position randomness.'],
                  models=[])
    rollouts = {}
    for model in source_manifest['models']:
        label, checkpoint = model['label'], Path(model['path']).resolve(strict=True)
        if sha256(checkpoint) != model['sha256']:
            raise ValueError(f'checkpoint differs from original comparison: {checkpoint}')
        config, manifest, evidence = archived_config(checkpoint)
        with zipfile.ZipFile(checkpoint) as archive:
            metadata = json.loads(archive.read('data'))
        matches = [record for record in manifest.get('models', {}).values()
                   if record.get('path') and (checkpoint.parent.parent/record['path']).resolve() == checkpoint]
        records = [r.get('timestep') for r in matches]
        samples, arrays = sample_resets(config, args.samples, args.seed)
        sample_path = output/f'{label}-reset-samples.npz'
        np.savez_compressed(sample_path, **arrays)
        item = dict(label=label, display_name=display_name(label), checkpoint_path=str(checkpoint),
                    checkpoint_sha256=model['sha256'], evidence=evidence, archived_git=manifest.get('git'),
                    historical_source_verification='unconfirmed: archived git dirty=True' if manifest.get('git', {}).get('dirty') else 'not independently reconstructed',
                    training_seed=config.training.seed, checkpoint_seed=metadata.get('seed'),
                    seed_status='unrecorded/unset' if config.training.seed is None else 'recorded',
                    checkpoint_num_timesteps=metadata.get('num_timesteps'), manifest_checkpoint_timesteps=records,
                    distribution=distribution_contract(config), resolved_training_config=config.resolved_dict(),
                    observation=dict(shape=model.get('observation_shape'), action_shape=model.get('action_shape'),
                                     contract=model.get('observation_contract'), normalization=model.get('normalization')),
                    action_scale=list(config.environment.residual_scale), reward=config.resolved_dict()['environment']['reward'],
                    actuator=config.resolved_dict()['actuator'], payload=config.resolved_dict()['environment']['payload'],
                    reset_samples=samples, sample_file=sample_path.name, sample_sha256=sha256(sample_path))
        report['models'].append(item)
        rollouts[label] = dict(display_name=display_name(label), **inspect_rollouts(source, label))
        if sha256(checkpoint) != model['sha256']:
            raise ValueError('checkpoint changed during audit')
        print(f'{display_name(label)}: {args.samples} resets; norm RMS={samples["norm"]["rms"]:.8f} m')
    report['checkpoint_hashes_unchanged'] = True
    write_json(output/'reset_audit.json', report)
    write_json(output/'existing_rollout_audit.json', rollouts)
    fields = ('display_name', 'sampler', 'bound_m', 'theoretical_norm_max_m', 'theoretical_norm_rms_m',
              'sample_norm_max_m', 'sample_norm_mean_m', 'sample_norm_rms_m', 'sample_norm_p50_m', 'sample_norm_p95_m',
              'attitude_randomized', 'training_seed', 'checkpoint_num_timesteps')
    with (output/'reset_comparison.csv').open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for m in report['models']:
            d, n = m['distribution'], m['reset_samples']['norm']
            writer.writerow(dict(display_name=m['display_name'], sampler=d['sampler'], bound_m=d['bound_parameter_m'],
                                 theoretical_norm_max_m=d['theoretical_norm_max_m'], theoretical_norm_rms_m=d['theoretical_norm_rms_m'],
                                 **{f'sample_norm_{key}_m': n[key] for key in ('max', 'mean', 'rms', 'p50', 'p95')},
                                 attitude_randomized=d['attitude_randomized'], training_seed=m['training_seed'],
                                 checkpoint_num_timesteps=m['checkpoint_num_timesteps']))
    print(f'audit: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

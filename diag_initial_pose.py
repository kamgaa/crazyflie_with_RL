"""Reset-only pose sampling diagnostic. No policy inference or PPO training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs/e2e_train_pose_dr_10cm_30deg_scale006.yaml'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--samples', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.seed < 0:
        parser.error('samples must be positive and seed must be nonnegative')
    import numpy as np
    from crazyflie_rl.config import load_config
    from crazyflie_rl.environment import CrazyflieResidualEnv

    config = load_config(args.config)
    settings = config.environment.initial_pose_randomization
    if settings is None:
        parser.error('this diagnostic requires explicit initial_pose_randomization settings')
    position_limit = settings.position.max_norm_m if settings.enabled and settings.position.enabled else 0.0
    angle_limit = settings.attitude.max_angle_deg if settings.enabled and settings.attitude.enabled else 0.0
    position_norms, angles, axes, norm_errors = [], [], [], []
    env = CrazyflieResidualEnv(config=config, seed=args.seed)
    try:
        for index in range(args.samples):
            # First seeded reset, then the normal continuous reset RNG stream.
            observation, info = env.reset(seed=args.seed if index == 0 else None)
            q = env.data.qpos[3:7].copy()
            position_norms.append(float(np.linalg.norm(env.data.qpos[:3] - env.pos_des)))
            angles.append(float(np.rad2deg(2*np.arccos(np.clip(abs(q[0])/np.linalg.norm(q),0,1)))))
            axes.append(info['initial_attitude_axis_xyz'])
            norm_errors.append(abs(float(np.linalg.norm(q)) - 1.0))
            assert observation.shape == (15,) and env.action_space.shape == (4,)
            assert np.array_equal(env.data.qvel, np.zeros_like(env.data.qvel))
    finally:
        env.close()

    def stats(values):
        return dict(zip(('max', 'mean', 'p50', 'p95', 'p99'), map(float, (
            np.max(values), np.mean(values), *np.percentile(values, [50, 95, 99]),
        ))))
    violations = {
        'position': int(np.count_nonzero(np.asarray(position_norms) > position_limit + 1e-12)),
        'attitude': int(np.count_nonzero(np.asarray(angles) > angle_limit + 1e-10)),
    }
    report = {
        'config': str(args.config.resolve()), 'seed': args.seed, 'samples': args.samples,
        'position_norm_m': stats(position_norms), 'attitude_angle_deg': stats(angles),
        'axis_mean_xyz': np.mean(axes, axis=0).tolist(),
        'fraction_abs_axis_z_gt_0p1': float(np.mean(np.abs(np.asarray(axes)[:,2]) > .1)),
        'max_quaternion_norm_error': max(norm_errors),
        'bound_violation_count': violations,
        'last_reset_info': info,
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text + '\n')
    print(text)
    return int(any(violations.values()))


if __name__ == '__main__':
    raise SystemExit(main())

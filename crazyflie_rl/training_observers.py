"""Observer-only training episode records and independent periodic snapshots."""
from __future__ import annotations

import csv
import hashlib
import time

import gymnasium as gym
import numpy as np


REWARD_COMPONENTS = ('position', 'velocity', 'tilt', 'angular_velocity', 'yaw',
                     'action', 'action_rate', 'crash')


def physical_termination_reasons(env, position, quaternion, reference):
    tilt = np.arccos(np.clip(1-2*(quaternion[1]**2+quaternion[2]**2), -1, 1))
    conditions = {'min_altitude': position[2] < env.min_altitude,
                  'max_altitude': position[2] > env.max_altitude,
                  'max_tilt': tilt > env.max_termination_tilt,
                  'max_position_error': np.linalg.norm(position-reference) > env.max_position_error}
    return [key for key, hit in conditions.items() if hit]


class EpisodeCSVRecorder(gym.Wrapper):
    """Record terminal data BEFORE DummyVecEnv auto-reset; never change rewards/RNG."""

    def __init__(self, env, path):
        super().__init__(env)
        self._file = open(path, 'x', newline='')
        fields = ['episode', 'start_timestep', 'end_timestep', 'length_steps', 'duration_sec',
                  'terminated', 'truncated', 'physical_termination', 'time_limit_reached',
                  'episode_ended', 'end_reason', 'return', 'final_position_error_m', 'final_tilt_deg']
        fields += [f'reward_sum_{key}' for key in REWARD_COMPONENTS]
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader(); self._file.flush()
        self.total_steps = self.episode = self.length = 0
        self._active = False
        self._started = time.perf_counter()

    def reset(self, **kwargs):
        if self._active and self.length:
            self._record(False, False, False, 'explicit_reset')
        result = self.env.reset(**kwargs)
        self.episode += 1; self.length = 0; self.return_sum = 0.
        self.component_sums = dict.fromkeys(REWARD_COMPONENTS, 0.)
        self._active = True
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1; self.length += 1; self.return_sum += float(reward)
        for key in REWARD_COMPONENTS:
            self.component_sums[key] += float(info.get('reward_terms', {}).get(key, 0.))
        if terminated or truncated:
            reason = self._record(terminated, truncated, True)
            info = dict(info)
            info['episode'] = {'r': self.return_sum, 'l': self.length,
                               't': time.perf_counter()-self._started}
            info['episode_end_reason'] = reason
        return obs, reward, terminated, truncated, info

    def _record(self, terminated, truncated, ended, explicit_reason=None):
        base = self.env.unwrapped
        p, q, _, _ = base._read_state()
        reasons = physical_termination_reasons(base, p, q, base.pos_des) if terminated else []
        if truncated: reasons.append('time_limit')
        reason = explicit_reason or ';'.join(reasons) or 'terminated_unknown'
        tilt = np.degrees(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2), -1, 1)))
        self._writer.writerow({
            'episode': self.episode, 'start_timestep': self.total_steps-self.length,
            'end_timestep': self.total_steps, 'length_steps': self.length,
            'duration_sec': self.length*base.dt_phys*base.substeps, 'terminated': bool(terminated),
            'truncated': bool(truncated), 'physical_termination': bool(terminated),
            'time_limit_reached': bool(truncated), 'episode_ended': bool(ended),
            'end_reason': reason, 'return': self.return_sum,
            'final_position_error_m': float(np.linalg.norm(p-base.pos_des)), 'final_tilt_deg': float(tilt),
            **{f'reward_sum_{key}': value for key, value in self.component_sums.items()}})
        self._file.flush(); self._active = False
        return reason

    def close(self):
        try:
            if not self._file.closed:
                if self._active and self.length:
                    self._record(False, False, False, 'collector_closed')
                self._file.close()
        finally:
            self.env.close()


def make_periodic_checkpoint_callback(artifacts, interval):
    from stable_baselines3.common.callbacks import BaseCallback

    class PeriodicSnapshots(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.next_step = interval

        def _on_training_start(self):
            digest = hashlib.sha256()
            for name, tensor in sorted(self.model.policy.state_dict().items()):
                digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
            artifacts.write_metrics('training-initial-policy', {
                'initial_policy_sha256': digest.hexdigest(), 'fresh_initialization': True,
                'seed': self.model.seed, 'checkpoint_interval': interval,
                'checkpoint_timing': 'rollout start after previous optimizer update; final after last update',
                'normalization_enabled': self.model.get_vec_normalize_env() is not None})

        def _maybe_save(self):
            if self.num_timesteps >= self.next_step:
                artifacts.save_model(self.model, 'intermediate', timestep=int(self.num_timesteps),
                    metadata={'requested_interval': interval, 'optimizer_update_completed': True})
                self.next_step = (self.num_timesteps//interval+1)*interval

        def _on_rollout_start(self):
            self._maybe_save()

        def _on_training_end(self):
            self._maybe_save()

        def _on_step(self):
            return True

    return PeriodicSnapshots()

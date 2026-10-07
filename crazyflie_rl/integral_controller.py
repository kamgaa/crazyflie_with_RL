"""Evaluation controller: frozen PPO + bounded external world-frame integral.

Only xi is persistent control state. No payload, effectiveness, thrust or plant
model enters the compensator; command saturation is supplied as boolean flags.
"""
from dataclasses import dataclass

import numpy as np

POSITION_SLICE = slice(0, 3)


@dataclass(frozen=True)
class IntegralSettings:
    name: str
    k_xy: float
    k_z: float
    xy_limit_m: float = .40
    z_limit_m: float = .15

    def __post_init__(self):
        values = (self.k_xy, self.k_z, self.xy_limit_m, self.z_limit_m)
        if not np.all(np.isfinite(values)) or min(values) < 0:
            raise ValueError('integral gains and limits must be finite and nonnegative')

    @property
    def enabled(self):
        return self.k_xy != 0 or self.k_z != 0


class IntegralController:
    """Wrap frozen inference without changing the environment's true reference.

    prepare_observation uses xi_t, then predict delegates to the original frozen
    normalization/inference. finish_step uses pre-step e_true and interval-wide
    command saturation to compute xi_(t+1), once per policy control interval.
    """
    def __init__(self, policy, settings):
        self.policy = policy
        self.settings = settings
        self.reset()

    def reset(self):
        self.xi = np.zeros(3)
        self.pending = None
        self.frozen_steps = self.frozen_run = self.longest_frozen_run = 0

    def bind(self, env):
        self.policy.bind(env)
        self.reset()

    def predict(self, observation):
        return self.policy.predict(observation)

    def prepare_observation(self, raw, position, target, dt):
        if self.pending is not None:
            raise RuntimeError('previous control interval has not been finished')
        if np.shape(raw) != (15,) or dt <= 0 or not np.isfinite(dt):
            raise ValueError('expected 15 raw channels and positive finite control dt')
        target = np.array(target, dtype=float, copy=True)
        error = np.asarray(position, dtype=float) - target
        if not np.all(np.isfinite(error)):
            raise ValueError('nonfinite position error')
        command = target + self.xi
        actor_error = np.asarray(position, dtype=float) - command
        self.pending = dict(p_target=target, p_cmd=command, e_true_before=error.copy(),
                            e_actor_before=actor_error.copy(), xi_t=self.xi.copy(), control_dt=float(dt))
        if not self.settings.enabled:
            return raw  # Preserve the exact original inference path at K_I=0.
        transformed = raw.copy()
        transformed[POSITION_SLICE] = actor_error
        return transformed

    def finish_step(self, *, allocator_clipping, esc_boundary, action_boundary):
        if self.pending is None:
            raise RuntimeError('missing pre-step observation')
        record = self.pending
        self.pending = None
        settings = self.settings
        reasons = [name for name, value in (
            ('allocator_clipping', allocator_clipping), ('esc_boundary', esc_boundary),
            ('policy_action_boundary', action_boundary)) if value]
        dt = record['control_dt']
        gain = np.array([settings.k_xy, settings.k_xy, settings.k_z])
        candidate = self.xi - dt * gain * record['e_true_before']
        projected = candidate.copy()
        xy_norm = np.linalg.norm(projected[:2])
        if xy_norm > settings.xy_limit_m:
            projected[:2] *= settings.xy_limit_m / xy_norm
        projected[2] = np.clip(projected[2], -settings.z_limit_m, settings.z_limit_m)
        frozen = settings.enabled and bool(reasons)
        allowed = settings.enabled and not frozen
        if allowed:
            self.xi = projected
        self.frozen_steps += int(frozen)
        self.frozen_run = self.frozen_run + 1 if frozen else 0
        self.longest_frozen_run = max(self.longest_frozen_run, self.frozen_run)
        record.update(xi_candidate=candidate, xi_next=self.xi.copy(), integral_gain_s_inv=gain,
            integral_enabled=settings.enabled, integral_update_allowed=allowed,
            integral_frozen=frozen, integral_stop_reason=';'.join(reasons) if settings.enabled else 'disabled',
            interval_allocator_clipping=bool(allocator_clipping), interval_esc_boundary=bool(esc_boundary),
            interval_action_boundary=bool(action_boundary),
            integral_xy_projected=allowed and xy_norm > settings.xy_limit_m,
            integral_z_projected=allowed and abs(candidate[2]) > settings.z_limit_m,
            integral_xy_at_limit=np.linalg.norm(self.xi[:2]) >= settings.xy_limit_m - 1e-12,
            integral_z_at_limit=abs(self.xi[2]) >= settings.z_limit_m - 1e-12,
            integral_frozen_total_s=self.frozen_steps * dt,
            integral_frozen_longest_s=self.longest_frozen_run * dt)
        return record

"""Evaluation-only confirmed-estimate routing; no estimator or plant changes."""
from __future__ import annotations

import numpy as np

from .motor_layout import native_from_user
from .oracle_allocation import OracleAllocationEnv, efficiency_matrix


class ConfirmedEfficiency:
    """Hold the last confirmed vector, not the diagnostic uncertain estimate.

    Consume only the decision/alpha fields of an already completed interval.
    No scenario, truth, environment or timing-of-failure is accepted.
    """
    def __init__(self):
        self.eta_user = np.ones(4)
        self.last_confirmation_time = None
        self.last_observation_time = None
        self.confirmed_motor = 0

    def update(self, *, state, motor, hypothesis_alpha_user, estimate_time):
        t = float(estimate_time)
        if not np.isfinite(t) or (self.last_observation_time is not None and t <= self.last_observation_time):
            raise ValueError('estimate times must strictly increase')
        self.last_observation_time = t
        alpha = np.asarray(hypothesis_alpha_user, float)
        if alpha.shape != (4,): raise ValueError('four hypothesis alphas required')
        accepted = False
        if state == 'healthy':
            self.eta_user = np.ones(4); self.confirmed_motor = 0; accepted = True
        elif state == 'fault':
            if motor not in (1, 2, 3, 4): raise ValueError('confirmed user motor must be 1..4')
            value = alpha[motor-1]
            if np.isfinite(value) and 0 <= value <= 1:
                self.eta_user = np.ones(4); self.eta_user[motor-1] = value
                self.confirmed_motor = int(motor); accepted = True
        elif state not in ('uncertain', 'insufficient_data'):
            raise ValueError('unknown estimator decision')
        if accepted: self.last_confirmation_time = t
        return self.eta_user.copy(), accepted

    def consume(self, result):
        return self.update(state=result['estimator_state'], motor=result['estimated_motor'],
            hypothesis_alpha_user=result['hypothesis_alpha_user'], estimate_time=result['estimate_time'])


class EstimatedAllocationEnv(OracleAllocationEnv):
    """Same B0/pseudoinverse/clipping/plant; explicit choice of efficiency source.

    Actual efficiency is read for CONTROL only in the oracle branch. Parent
    diagnostics read truth separately for all branches and never feed it back.
    """
    def __init__(self, *args, allocation_source='blind', **kwargs):
        if allocation_source not in ('blind', 'oracle', 'estimated'):
            raise ValueError(allocation_source)
        self.allocation_source = allocation_source
        self.estimated_efficiency_native = np.ones(4)
        super().__init__(*args, allocator_mode='oracle' if allocation_source == 'oracle' else 'existing', **kwargs)
        self.allocator_rank = 4
        self.allocator_condition = float(np.linalg.cond(self.B0))

    def set_estimated_efficiency(self, user_values):
        # Validation only; never touch qpos/qvel, motor state, target or integral.
        native = native_from_user(user_values)
        efficiency_matrix(self.B0, native)
        self.estimated_efficiency_native = native.copy()

    def sync_allocator(self):
        if self.allocation_source == 'oracle':
            eta = self.motor_effectiveness.copy()
        elif self.allocation_source == 'estimated':
            eta = self.estimated_efficiency_native.copy()
        else:
            eta = np.ones(4)
        if not np.array_equal(eta, self.allocator_efficiency):
            self.allocator_efficiency = eta.copy()
            self.allocator_matrix = efficiency_matrix(self.B0, eta)
            self.B_pinv = (self.B0_pinv.copy() if np.all(eta == 1)
                           else np.linalg.pinv(self.allocator_matrix))
            self.allocator_rank = int(np.linalg.matrix_rank(self.allocator_matrix))
            condition = float(np.linalg.cond(self.allocator_matrix))
            self.allocator_condition = condition if np.isfinite(condition) else None
            if not np.isfinite(self.B_pinv).all(): raise FloatingPointError('nonfinite allocation inverse')

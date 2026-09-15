"""Testable Lyapunov-candidate tracking reward components.

The quadratic form in this module is deliberately called a *Lyapunov
candidate*.  A positive-definite ``P`` makes the form positive definite in the
chosen tracking error, but does not establish decrease for the simulator,
controller, learned policy, or their closed-loop dynamics.  The decay term is
only a soft transition penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .controllers import (
    geometric_attitude_error as controller_attitude_error,
    quat_normalize_wxyz,
    rotmat_from_quat_wxyz,
)


REWARD_MODES = frozenset({"legacy", "lyapunov", "legacy_plus_lyapunov"})


def _finite_vector3(value: Sequence[float], name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite vector with shape (3,)") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return vector.copy()


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class TrackingError:
    """Four three-axis tracking-error blocks with explicit frame semantics.

    ``position`` and ``velocity`` are world-frame errors. ``attitude`` is the
    geometric SO(3) error vector expressed in the actual body frame, and
    ``angular_rate`` is the matched actual-body-frame rate error.
    """

    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray
    angular_rate: np.ndarray

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "attitude", "angular_rate"):
            value = _finite_vector3(getattr(self, name), name)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate(
            (self.position, self.velocity, self.attitude, self.angular_rate)
        )


@dataclass(frozen=True)
class LyapunovRewardTerms:
    """Scalar components for one transition's candidate-based reward."""

    reward_total: float
    v_before: float
    v_after: float
    delta_v: float
    potential_difference: float
    potential_shaping: float
    decay_target: float
    decay_violation: float
    decay_penalty: float
    state_cost: float
    terminal_potential_zeroed: bool
    v_decreased: bool
    decay_condition_satisfied: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "reward_total": self.reward_total,
            "v_before": self.v_before,
            "v_after": self.v_after,
            "delta_v": self.delta_v,
            "potential_difference": self.potential_difference,
            "potential_shaping": self.potential_shaping,
            "decay_target": self.decay_target,
            "decay_violation": self.decay_violation,
            "decay_penalty": self.decay_penalty,
            "state_cost": self.state_cost,
            "terminal_potential_zeroed": self.terminal_potential_zeroed,
            "v_decreased": self.v_decreased,
            "decay_condition_satisfied": self.decay_condition_satisfied,
        }


def desired_rotation_from_yaw(yaw: float) -> np.ndarray:
    """Return the level body-to-world rotation for one finite yaw target."""

    angle = _finite_scalar(yaw, "yaw")
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def geometric_attitude_error(
    actual_quaternion_wxyz: Sequence[float],
    desired_rotation: Sequence[Sequence[float]],
) -> np.ndarray:
    """Compute ``0.5 * (R_d.T R - R.T R_d)^vee``.

    Rotation matrices make the result invariant to the quaternion sign, so
    ``q`` and ``-q`` produce exactly the same error.  The SO(3) expression also
    avoids Euler-angle wrap discontinuities at +/-pi.  As with the standard
    geometric error, exactly 180-degree rotations are a critical point.
    """

    quaternion = np.asarray(actual_quaternion_wxyz, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("actual_quaternion_wxyz must be finite with shape (4,)")
    rotation = rotmat_from_quat_wxyz(quat_normalize_wxyz(quaternion))
    desired = np.asarray(desired_rotation, dtype=float)
    if desired.shape != (3, 3) or not np.all(np.isfinite(desired)):
        raise ValueError("desired_rotation must be finite with shape (3, 3)")
    return controller_attitude_error(rotation, desired)


def tracking_error_from_state(
    *,
    position: Sequence[float],
    quaternion_wxyz: Sequence[float],
    velocity_world: Sequence[float],
    angular_rate_body: Sequence[float],
    position_reference: Sequence[float],
    yaw_reference: float,
    velocity_reference_world: Sequence[float] = (0.0, 0.0, 0.0),
    angular_rate_reference_body: Sequence[float] = (0.0, 0.0, 0.0),
    desired_rotation: Sequence[Sequence[float]] | None = None,
) -> TrackingError:
    """Build the four tracking blocks using the environment's conventions.

    The current mission API supplies only position and yaw.  Therefore callers
    use its preserved zero velocity/rate and level-yaw references unless a
    future, independently validated reference pipeline supplies ``R_d`` and
    ``omega_d`` explicitly.
    """

    actual_position = _finite_vector3(position, "position")
    reference_position = _finite_vector3(position_reference, "position_reference")
    actual_velocity = _finite_vector3(velocity_world, "velocity_world")
    reference_velocity = _finite_vector3(
        velocity_reference_world, "velocity_reference_world"
    )
    actual_rate = _finite_vector3(angular_rate_body, "angular_rate_body")
    desired_rate = _finite_vector3(
        angular_rate_reference_body, "angular_rate_reference_body"
    )
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_wxyz must be finite with shape (4,)")
    actual_rotation = rotmat_from_quat_wxyz(quat_normalize_wxyz(quaternion))
    desired = (
        desired_rotation_from_yaw(yaw_reference)
        if desired_rotation is None
        else np.asarray(desired_rotation, dtype=float)
    )
    if desired.shape != (3, 3) or not np.all(np.isfinite(desired)):
        raise ValueError("desired_rotation must be finite with shape (3, 3)")

    return TrackingError(
        position=actual_position - reference_position,
        velocity=actual_velocity - reference_velocity,
        attitude=geometric_attitude_error(quaternion, desired),
        angular_rate=actual_rate - actual_rotation.T @ desired @ desired_rate,
    )


def normalized_tracking_vector(error: TrackingError, config: Any) -> np.ndarray:
    """Return ``z`` after positive per-axis normalization."""

    normalization = config.normalization
    blocks: list[np.ndarray] = []
    for name in ("position", "velocity", "attitude", "angular_rate"):
        scale = _finite_vector3(getattr(normalization, name), f"{name}_scale")
        if np.any(scale <= 0.0):
            raise ValueError(f"{name}_scale values must be positive")
        blocks.append(getattr(error, name) / scale)
    return np.concatenate(blocks)


def lyapunov_candidate(error: TrackingError, config: Any) -> float:
    """Evaluate the positive diagonal quadratic candidate ``z.T P z``."""

    normalized = normalized_tracking_vector(error, config)
    weights = config.matrix_weights
    block_weights = np.asarray(
        [
            getattr(weights, "position"),
            getattr(weights, "velocity"),
            getattr(weights, "attitude"),
            getattr(weights, "angular_rate"),
        ],
        dtype=float,
    )
    if block_weights.shape != (4,) or not np.all(np.isfinite(block_weights)):
        raise ValueError("Lyapunov candidate matrix weights must be finite")
    if np.any(block_weights <= 0.0):
        raise ValueError("Lyapunov candidate matrix weights must be positive")
    diagonal = np.repeat(block_weights, 3)
    return float(np.dot(diagonal * normalized, normalized))


def compute_lyapunov_reward(
    *,
    error_before: TrackingError,
    error_after: TrackingError,
    dt: float,
    gamma: float,
    config: Any,
    terminated: bool,
    truncated: bool,
) -> LyapunovRewardTerms:
    """Compute separated candidate cost, potential shaping, and decay penalty.

    ``config`` is the complete typed ``RewardConfig`` so its mode cannot drift
    from the parameters used by the environment.  ``gamma`` is supplied by the
    same ``training.ppo.gamma`` field passed to Stable-Baselines3.

    On an actual absorbing termination, the successor terminal potential is
    defined as zero.  A time-limit truncation retains ``-V_after`` because it
    is not an absorbing MDP terminal.  This is the episodic terminal convention
    under which the shaping terms telescope; policy invariance still requires
    all other standard potential-shaping MDP assumptions.
    """

    mode = str(config.mode)
    if mode not in REWARD_MODES:
        raise ValueError(f"unknown reward mode: {mode!r}")
    timestep = _finite_scalar(dt, "dt")
    discount = _finite_scalar(gamma, "gamma")
    if timestep <= 0.0:
        raise ValueError("dt must be positive")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("gamma must be within [0, 1]")
    if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
        truncated, (bool, np.bool_)
    ):
        raise ValueError("terminated and truncated must be booleans")

    settings = config.lyapunov
    decay_rate = _finite_scalar(settings.decay_rate, "decay_rate")
    decay_fraction = decay_rate * timestep
    if not 0.0 < decay_fraction < 1.0:
        raise ValueError("decay_rate must satisfy 0 < decay_rate * dt < 1")

    v_before = lyapunov_candidate(error_before, settings)
    v_after = lyapunov_candidate(error_after, settings)
    delta_v = v_after - v_before
    decay_target = (1.0 - decay_fraction) * v_before
    decay_violation = max(0.0, v_after - decay_target)

    use_candidate_terms = mode != "legacy"
    terminal_potential_zeroed = bool(terminated)
    potential_difference = (
        v_before
        if terminal_potential_zeroed
        else v_before - discount * v_after
    )
    potential_shaping = (
        _finite_scalar(settings.potential_weight, "potential_weight")
        * potential_difference
        if use_candidate_terms and settings.potential_shaping_enabled
        else 0.0
    )
    decay_penalty = (
        -_finite_scalar(settings.decay_weight, "decay_weight")
        * decay_violation**2
        if use_candidate_terms and settings.decay_penalty_enabled
        else 0.0
    )
    state_cost = (
        -_finite_scalar(settings.state_cost_weight, "state_cost_weight") * v_after
        if mode == "lyapunov"
        else 0.0
    )
    for name, weight in (
        ("potential_weight", settings.potential_weight),
        ("decay_weight", settings.decay_weight),
        ("state_cost_weight", settings.state_cost_weight),
    ):
        if _finite_scalar(weight, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")

    reward_total = state_cost + potential_shaping + decay_penalty
    values = (
        reward_total,
        v_before,
        v_after,
        delta_v,
        potential_difference,
        potential_shaping,
        decay_target,
        decay_violation,
        decay_penalty,
        state_cost,
    )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Lyapunov-candidate reward produced a non-finite value")

    return LyapunovRewardTerms(
        reward_total=float(reward_total),
        v_before=float(v_before),
        v_after=float(v_after),
        delta_v=float(delta_v),
        potential_difference=float(potential_difference),
        potential_shaping=float(potential_shaping),
        decay_target=float(decay_target),
        decay_violation=float(decay_violation),
        decay_penalty=float(decay_penalty),
        state_cost=float(state_cost),
        terminal_potential_zeroed=terminal_potential_zeroed,
        v_decreased=bool(v_after < v_before),
        decay_condition_satisfied=bool(v_after <= decay_target),
    )


__all__ = [
    "LyapunovRewardTerms",
    "REWARD_MODES",
    "TrackingError",
    "compute_lyapunov_reward",
    "desired_rotation_from_yaw",
    "geometric_attitude_error",
    "lyapunov_candidate",
    "normalized_tracking_vector",
    "tracking_error_from_state",
]

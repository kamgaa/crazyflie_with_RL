"""Bounded reset pose sampling around the environment's nominal identity attitude."""
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .config import InitialPoseRandomizationConfig


def sample_initial_pose(rng: np.random.Generator, settings: InitialPoseRandomizationConfig):
    """Return position offset, unit wxyz quaternion, axis and angle in radians.

    Radius and rotation angle are uniform in magnitude, NOT volume-uniform in
    a ball and NOT Haar-uniform on SO(3). Only the supplied reset RNG is used.
    Explicitly disabled settings mean nominal pose, not legacy perturbations.
    """
    offset = np.zeros(3)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.array([1.0, 0.0, 0.0])  # Deterministic axis for zero rotation.
    angle = 0.0
    if not settings.enabled:
        return offset, quaternion, axis, angle

    def direction(full_3d=True):
        vector = rng.normal(size=3)
        if not full_3d:
            vector[2] = 0.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.array([1.0, 0.0, 0.0])
        return vector / norm

    # Fixed draw sequence within the new mode, including disabled subcomponents
    # and zero bounds. Legacy mode never calls this sampler.
    position_direction = direction()
    radius_fraction = rng.uniform(0.0, 1.0)
    attitude_axis = direction(settings.attitude.full_3d)
    angle_fraction = rng.uniform(0.0, 1.0)
    if settings.position.enabled:
        # U gives uniform perturbation magnitude in [0, max_norm_m].
        offset = settings.position.max_norm_m * radius_fraction * position_direction
    if settings.attitude.enabled and settings.attitude.max_angle_deg > 0:
        axis = attitude_axis
        angle = np.deg2rad(settings.attitude.max_angle_deg) * angle_fraction
        half = 0.5 * angle
        quaternion = np.concatenate(([np.cos(half)], np.sin(half) * axis))
        quaternion /= np.linalg.norm(quaternion)
    return offset, quaternion, axis, float(angle)


def initial_pose_info(position, target, quaternion, axis):
    """JSON-safe reset diagnostics; quaternion follows the existing wxyz convention."""
    offset = np.asarray(position) - np.asarray(target)
    q = np.asarray(quaternion, dtype=float)
    angle = 2.0 * np.arccos(np.clip(abs(q[0]) / np.linalg.norm(q), 0.0, 1.0))
    return {
        'initial_position_offset_xyz_m': offset.tolist(),
        'initial_position_error_norm_m': float(np.linalg.norm(offset)),
        'initial_attitude_quaternion_wxyz': q.tolist(),
        'initial_attitude_axis_xyz': np.asarray(axis).tolist(),
        'initial_attitude_angle_deg': float(np.rad2deg(angle)),
    }

"""Shared named body-frame axes for deterministic attitude perturbations."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping, Sequence


_DIAGONAL_COMPONENT = math.sqrt(0.5)

ATTITUDE_AXIS_VECTORS: Mapping[str, tuple[float, float, float]] = MappingProxyType(
    {
        "roll_plus": (1.0, 0.0, 0.0),
        "roll_minus": (-1.0, 0.0, 0.0),
        "pitch_plus": (0.0, 1.0, 0.0),
        "pitch_minus": (0.0, -1.0, 0.0),
        "diagonal_pp": (_DIAGONAL_COMPONENT, _DIAGONAL_COMPONENT, 0.0),
        "diagonal_pm": (_DIAGONAL_COMPONENT, -_DIAGONAL_COMPONENT, 0.0),
        "diagonal_mp": (-_DIAGONAL_COMPONENT, _DIAGONAL_COMPONENT, 0.0),
        "diagonal_mm": (-_DIAGONAL_COMPONENT, -_DIAGONAL_COMPONENT, 0.0),
    }
)
ATTITUDE_AXIS_CHOICES = tuple(ATTITUDE_AXIS_VECTORS)
CARDINAL_ATTITUDE_AXIS_CHOICES = ATTITUDE_AXIS_CHOICES[:4]


def attitude_axis_vector(name: str) -> tuple[float, float, float]:
    """Return the unit body-frame axis associated with one canonical name."""

    try:
        return ATTITUDE_AXIS_VECTORS[str(name)]
    except KeyError as exc:
        choices = ", ".join(ATTITUDE_AXIS_CHOICES)
        raise ValueError(
            f"unknown attitude axis {name!r}; choose one of: {choices}"
        ) from exc


def axis_angle_quaternion_wxyz(
    axis_xyz: Sequence[float], angle_rad: float
) -> tuple[float, float, float, float]:
    """Build a normalized canonical body-to-world ``wxyz`` quaternion.

    A zero rotation is the identity regardless of the supplied axis.  For a
    nonzero rotation, the axis is normalized before applying the axis-angle
    formula.  The returned representative always follows the ``w >= 0`` sign
    convention.
    """

    try:
        axis = tuple(float(value) for value in axis_xyz)
    except (TypeError, ValueError) as exc:
        raise ValueError("attitude axis must be a finite three-vector") from exc
    if len(axis) != 3 or not all(math.isfinite(value) for value in axis):
        raise ValueError("attitude axis must be a finite three-vector")

    angle = float(angle_rad)
    if not math.isfinite(angle):
        raise ValueError("attitude angle must be finite")
    if angle == 0.0:
        return (1.0, 0.0, 0.0, 0.0)

    axis_norm = math.sqrt(math.fsum(value * value for value in axis))
    if axis_norm == 0.0:
        raise ValueError("nonzero attitude angle requires a nonzero axis")
    unit_axis = tuple(value / axis_norm for value in axis)
    half_angle = 0.5 * angle
    sine = math.sin(half_angle)
    quaternion = (
        math.cos(half_angle),
        unit_axis[0] * sine,
        unit_axis[1] * sine,
        unit_axis[2] * sine,
    )
    quaternion_norm = math.sqrt(math.fsum(value * value for value in quaternion))
    normalized = tuple(value / quaternion_norm for value in quaternion)
    if normalized[0] < 0.0:
        normalized = tuple(-value for value in normalized)
    return normalized[0], normalized[1], normalized[2], normalized[3]


def named_attitude_quaternion_wxyz(
    name: str, angle_deg: float
) -> tuple[float, float, float, float]:
    """Build the canonical quaternion for a named axis and degree angle."""

    degrees = float(angle_deg)
    if not math.isfinite(degrees):
        raise ValueError("attitude angle must be finite")
    return axis_angle_quaternion_wxyz(
        attitude_axis_vector(name), math.radians(degrees)
    )


__all__ = [
    "ATTITUDE_AXIS_CHOICES",
    "ATTITUDE_AXIS_VECTORS",
    "CARDINAL_ATTITUDE_AXIS_CHOICES",
    "attitude_axis_vector",
    "axis_angle_quaternion_wxyz",
    "named_attitude_quaternion_wxyz",
]

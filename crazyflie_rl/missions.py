"""Reference missions shared by interactive and headless evaluation.

The circle equations intentionally mirror the legacy ``circle_traj.py`` and
``view_live_hover.py`` implementation.  In particular, the angular-velocity
ramp contributes half of ``omega * ramp_sec`` to the phase and the final HOLD
reference returns to ``goto_xy``.  For the shipped presets that creates a
position-reference jump at the CIRCLE/HOLD boundary; preserving that observed
behaviour is part of this refactor's compatibility contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import CirclePresetConfig, ConfigError, ExperimentConfig, MissionConfig


PHASES = ("TAKEOFF", "SETTLE1", "GOTO", "SETTLE2", "CIRCLE", "HOLD")


def cosine_ease(value: float) -> float:
    """Legacy cosine ease-in/out clamped to the inclusive unit interval."""

    amount = min(1.0, max(0.0, float(value)))
    return 0.5 * (1.0 - math.cos(math.pi * amount))


@dataclass(frozen=True)
class MissionBoundaries:
    """Cumulative phase boundaries, in seconds from mission start."""

    takeoff_end: float
    settle1_end: float
    goto_end: float
    settle2_end: float
    circle_end: float
    total: float


@dataclass(frozen=True)
class CircleMission:
    """Stateful-in-time, immutable circle-reference definition."""

    config: MissionConfig
    preset: CirclePresetConfig

    def __post_init__(self) -> None:
        if self.config.type != "circle":
            raise ConfigError(
                f"CircleMission requires mission.type='circle', got {self.config.type!r}"
            )

    @classmethod
    def from_experiment(cls, config: ExperimentConfig, preset: str) -> "CircleMission":
        return cls(config.mission, config.mission.circle_preset(preset))

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.preset.period

    @property
    def center_xy(self) -> tuple[float, float]:
        goto_x, goto_y = self.config.goto_xy
        return goto_x - self.preset.radius, goto_y

    @property
    def boundaries(self) -> MissionBoundaries:
        takeoff_end = self.config.takeoff_sec
        settle1_end = takeoff_end + self.config.settle_sec
        goto_end = settle1_end + self.config.goto_sec
        settle2_end = goto_end + self.config.settle_sec
        circle_duration = (
            self.config.circle_ramp_sec
            + self.config.number_of_laps * self.preset.period
        )
        circle_end = settle2_end + circle_duration
        return MissionBoundaries(
            takeoff_end=takeoff_end,
            settle1_end=settle1_end,
            goto_end=goto_end,
            settle2_end=settle2_end,
            circle_end=circle_end,
            total=circle_end + self.config.post_hold_sec,
        )

    @property
    def total_sec(self) -> float:
        return self.boundaries.total

    def circle_phase(self, elapsed: float) -> float:
        """Return the exact legacy circle phase for time since CIRCLE entry."""

        tc = float(elapsed)
        ramp = self.config.circle_ramp_sec
        if tc < ramp:
            return self.omega * (
                0.5 * tc
                - (ramp / (2.0 * math.pi)) * math.sin(math.pi * tc / ramp)
            )
        return self.omega * (0.5 * ramp) + self.omega * (tc - ramp)

    def reference(self, time_sec: float) -> tuple[np.ndarray, str]:
        """Return ``(position_xyz, phase_name)`` at one mission time."""

        t = float(time_sec)
        bounds = self.boundaries
        altitude = self.config.hover_altitude
        goto_x, goto_y = self.config.goto_xy

        if t < bounds.takeoff_end:
            z = altitude * cosine_ease(t / self.config.takeoff_sec)
            return np.array([0.0, 0.0, z], dtype=float), "TAKEOFF"

        if t < bounds.settle1_end:
            return np.array([0.0, 0.0, altitude], dtype=float), "SETTLE1"

        if t < bounds.goto_end:
            amount = cosine_ease(
                (t - bounds.settle1_end) / self.config.goto_sec
            )
            return np.array(
                [goto_x * amount, goto_y * amount, altitude], dtype=float
            ), "GOTO"

        if t < bounds.settle2_end:
            return np.array([goto_x, goto_y, altitude], dtype=float), "SETTLE2"

        if t < bounds.circle_end:
            phi = self.circle_phase(t - bounds.settle2_end)
            center_x, center_y = self.center_xy
            return np.array(
                [
                    center_x + self.preset.radius * math.cos(phi),
                    center_y + self.preset.radius * math.sin(phi),
                    altitude,
                ],
                dtype=float,
            ), "CIRCLE"

        # This is deliberately the circle's nominal start point rather than
        # its actual end point.  See the module docstring compatibility note.
        return np.array([goto_x, goto_y, altitude], dtype=float), "HOLD"


__all__ = [
    "CircleMission",
    "MissionBoundaries",
    "PHASES",
    "cosine_ease",
]

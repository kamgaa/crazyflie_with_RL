"""Reference missions shared by interactive and headless evaluation.

The legacy circle constructor deliberately preserves the equations used by
``circle_traj.py`` and ``view_live_hover.py``, including their final HOLD jump.
Explicit circles and Lissajous paths instead enter the trajectory at their
mathematical first point and hold their actual final reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from .config import (
    CircleParameters,
    CirclePresetConfig,
    ConfigError,
    ExperimentConfig,
    HoverParameters,
    LissajousParameters,
    MissionConfig,
)


PHASES = (
    "HOVER",
    "TAKEOFF",
    "SETTLE1",
    "GOTO",
    "SETTLE2",
    "CIRCLE",
    "LISSAJOUS",
    "HOLD",
)


class ReferenceMission(Protocol):
    """Common immutable reference surface consumed by EvaluationRunner."""

    @property
    def name(self) -> str: ...

    @property
    def total_sec(self) -> float: ...

    def reference(self, time_sec: float) -> tuple[np.ndarray, str]: ...

    def effective_parameters(self) -> dict[str, object]: ...


def cosine_ease(value: float) -> float:
    """Legacy cosine ease-in/out clamped to the inclusive unit interval."""

    amount = min(1.0, max(0.0, float(value)))
    return 0.5 * (1.0 - math.cos(math.pi * amount))


def ramped_phase(elapsed: float, angular_speed: float, ramp_sec: float) -> float:
    """Integrate the legacy half-cosine angular-speed ramp.

    ``ramp_sec == 0`` represents an instantaneous transition to the requested
    angular speed.  For positive ramps this is algebraically identical to the
    pre-refactor circle formula.
    """

    time_sec = float(elapsed)
    ramp = float(ramp_sec)
    omega = float(angular_speed)
    if ramp < 0.0:
        raise ConfigError("ramp_sec must be non-negative")
    if ramp == 0.0:
        return omega * time_sec
    if time_sec < ramp:
        return omega * (
            0.5 * time_sec
            - (ramp / (2.0 * math.pi))
            * math.sin(math.pi * time_sec / ramp)
        )
    return omega * (0.5 * ramp) + omega * (time_sec - ramp)


@dataclass(frozen=True)
class MissionBoundaries:
    """Cumulative phase boundaries, in seconds from mission start."""

    takeoff_end: float
    settle1_end: float
    goto_end: float
    settle2_end: float
    circle_end: float
    total: float

    @property
    def trajectory_end(self) -> float:
        """General name for the legacy ``circle_end`` storage field."""

        return self.circle_end


def _trajectory_boundaries(config: MissionConfig, duration: float) -> MissionBoundaries:
    takeoff_end = config.takeoff_sec
    settle1_end = takeoff_end + config.settle_sec
    goto_end = settle1_end + config.goto_sec
    settle2_end = goto_end + config.settle_sec
    trajectory_end = settle2_end + duration
    return MissionBoundaries(
        takeoff_end=takeoff_end,
        settle1_end=settle1_end,
        goto_end=goto_end,
        settle2_end=settle2_end,
        circle_end=trajectory_end,
        total=trajectory_end + config.post_hold_sec,
    )


@dataclass(frozen=True)
class HoverMission:
    """Fixed XYZ/yaw reference held for a configured duration."""

    parameters: HoverParameters
    force_floor_start: bool = False

    def __post_init__(self) -> None:
        if self.parameters.duration <= 0.0:
            raise ConfigError("hover duration must be positive")
        if self.parameters.target[2] < 0.0:
            raise ConfigError("hover target altitude must be non-negative")

    @classmethod
    def from_experiment(cls, config: ExperimentConfig) -> "HoverMission":
        if config.mission.type != "hover":
            raise ConfigError(
                f"HoverMission requires mission.type='hover', got {config.mission.type!r}"
            )
        return cls(
            config.mission.hover,
            force_floor_start=config.mission.force_floor_start,
        )

    @property
    def name(self) -> str:
        return "hover"

    @property
    def total_sec(self) -> float:
        return self.parameters.duration

    def reference(self, time_sec: float) -> tuple[np.ndarray, str]:
        del time_sec
        return np.asarray(self.parameters.target, dtype=float).copy(), "HOVER"

    def effective_parameters(self) -> dict[str, object]:
        return {
            "target": self.parameters.target,
            "yaw_deg": self.parameters.yaw_deg,
            "duration": self.parameters.duration,
            "force_floor_start_requested": self.force_floor_start,
        }


@dataclass(frozen=True)
class CircleMission:
    """Immutable circle reference with a legacy-compatible constructor."""

    config: MissionConfig
    preset: CirclePresetConfig
    parameters: CircleParameters | None = None
    legacy_hold: bool = True

    def __post_init__(self) -> None:
        if self.config.type != "circle":
            raise ConfigError(
                f"CircleMission requires mission.type='circle', got {self.config.type!r}"
            )
        params = self.parameters
        if params is not None:
            if params.radius <= 0.0 or params.period <= 0.0 or params.laps <= 0.0:
                raise ConfigError("circle radius, period, and laps must be positive")
            if params.ramp_sec < 0.0:
                raise ConfigError("circle ramp_sec must be non-negative")
            if params.direction not in {"cw", "ccw"}:
                raise ConfigError("circle direction must be 'cw' or 'ccw'")

    @classmethod
    def from_experiment(cls, config: ExperimentConfig, preset: str) -> "CircleMission":
        """Build the exact legacy preset mission, including its HOLD jump."""

        return cls(config.mission, config.mission.circle_preset(preset))

    @classmethod
    def from_parameters(
        cls,
        config: MissionConfig,
        parameters: CircleParameters | None = None,
    ) -> "CircleMission":
        """Build a continuous explicit circle used by unified ``view_live``."""

        params = config.circle if parameters is None else parameters
        preset = CirclePresetConfig(
            key="runtime",
            description="explicit unified view_live circle",
            radius=params.radius,
            period=params.period,
        )
        return cls(config, preset, parameters=params, legacy_hold=False)

    @property
    def name(self) -> str:
        return "circle"

    @property
    def radius(self) -> float:
        return self.preset.radius if self.parameters is None else self.parameters.radius

    @property
    def period(self) -> float:
        return self.preset.period if self.parameters is None else self.parameters.period

    @property
    def laps(self) -> float:
        return (
            self.config.number_of_laps
            if self.parameters is None
            else self.parameters.laps
        )

    @property
    def ramp_sec(self) -> float:
        return (
            self.config.circle_ramp_sec
            if self.parameters is None
            else self.parameters.ramp_sec
        )

    @property
    def direction(self) -> str:
        return "ccw" if self.parameters is None else self.parameters.direction

    @property
    def omega(self) -> float:
        sign = -1.0 if self.direction == "cw" else 1.0
        return sign * 2.0 * math.pi / self.period

    @property
    def start_angle_rad(self) -> float:
        angle_deg = 0.0 if self.parameters is None else self.parameters.start_angle_deg
        return math.radians(angle_deg)

    @property
    def center_xy(self) -> tuple[float, float]:
        if self.parameters is not None:
            return self.parameters.center_xy
        goto_x, goto_y = self.config.goto_xy
        return goto_x - self.preset.radius, goto_y

    @property
    def goto_xy(self) -> tuple[float, float]:
        if self.parameters is None:
            return self.config.goto_xy
        center_x, center_y = self.center_xy
        return (
            center_x + self.radius * math.cos(self.start_angle_rad),
            center_y + self.radius * math.sin(self.start_angle_rad),
        )

    @property
    def trajectory_duration(self) -> float:
        return self.ramp_sec + self.laps * self.period

    @property
    def boundaries(self) -> MissionBoundaries:
        return _trajectory_boundaries(self.config, self.trajectory_duration)

    @property
    def total_sec(self) -> float:
        return self.boundaries.total

    def circle_phase(self, elapsed: float) -> float:
        """Return legacy-compatible phase displacement since CIRCLE entry."""

        return ramped_phase(elapsed, self.omega, self.ramp_sec)

    def _circle_reference(self, elapsed: float) -> np.ndarray:
        phi = self.start_angle_rad + self.circle_phase(elapsed)
        center_x, center_y = self.center_xy
        return np.array(
            [
                center_x + self.radius * math.cos(phi),
                center_y + self.radius * math.sin(phi),
                self.config.hover_altitude,
            ],
            dtype=float,
        )

    def reference(self, time_sec: float) -> tuple[np.ndarray, str]:
        """Return ``(position_xyz, phase_name)`` at one mission time."""

        t = float(time_sec)
        bounds = self.boundaries
        altitude = self.config.hover_altitude
        goto_x, goto_y = self.goto_xy

        if t < bounds.takeoff_end:
            z = altitude * cosine_ease(t / self.config.takeoff_sec)
            return np.array([0.0, 0.0, z], dtype=float), "TAKEOFF"
        if t < bounds.settle1_end:
            return np.array([0.0, 0.0, altitude], dtype=float), "SETTLE1"
        if t < bounds.goto_end:
            amount = cosine_ease((t - bounds.settle1_end) / self.config.goto_sec)
            return np.array(
                [goto_x * amount, goto_y * amount, altitude], dtype=float
            ), "GOTO"
        if t < bounds.settle2_end:
            return np.array([goto_x, goto_y, altitude], dtype=float), "SETTLE2"
        if t < bounds.trajectory_end:
            return self._circle_reference(t - bounds.settle2_end), "CIRCLE"

        if self.legacy_hold:
            # The old scripts returned to the nominal start point even when
            # the ramp left the circle at a different phase.
            return np.array([goto_x, goto_y, altitude], dtype=float), "HOLD"
        return self._circle_reference(self.trajectory_duration), "HOLD"

    def effective_parameters(self) -> dict[str, object]:
        return {
            "center_xy": self.center_xy,
            "radius": self.radius,
            "period": self.period,
            "laps": self.laps,
            "start_angle_deg": math.degrees(self.start_angle_rad),
            "direction": self.direction,
            "ramp_sec": self.ramp_sec,
            "altitude": self.config.hover_altitude,
            "takeoff_sec": self.config.takeoff_sec,
            "settle_sec": self.config.settle_sec,
            "goto_sec": self.config.goto_sec,
            "post_hold_sec": self.config.post_hold_sec,
            "force_floor_start_requested": self.config.force_floor_start,
            "legacy_hold": self.legacy_hold,
        }


@dataclass(frozen=True)
class LissajousMission:
    """Fixed-altitude XY Lissajous reference with a ramped base phase."""

    config: MissionConfig
    parameters: LissajousParameters

    def __post_init__(self) -> None:
        if self.config.type != "lissajous":
            raise ConfigError(
                "LissajousMission requires mission.type='lissajous', "
                f"got {self.config.type!r}"
            )
        amplitude_x, amplitude_y = self.parameters.amplitude_xy
        frequency_x, frequency_y = self.parameters.frequency_ratio
        if amplitude_x < 0.0 or amplitude_y < 0.0:
            raise ConfigError("Lissajous amplitudes must be non-negative")
        if amplitude_x == 0.0 and amplitude_y == 0.0:
            raise ConfigError("Lissajous amplitudes must not both be zero")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in (frequency_x, frequency_y)
        ):
            raise ConfigError("Lissajous frequencies must be positive integers")
        if self.parameters.base_period <= 0.0 or self.parameters.cycles <= 0.0:
            raise ConfigError("Lissajous base_period and cycles must be positive")
        if self.parameters.ramp_sec < 0.0:
            raise ConfigError("Lissajous ramp_sec must be non-negative")

    @classmethod
    def from_experiment(cls, config: ExperimentConfig) -> "LissajousMission":
        return cls(config.mission, config.mission.lissajous)

    @property
    def name(self) -> str:
        return "lissajous"

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.parameters.base_period

    @property
    def trajectory_duration(self) -> float:
        return (
            self.parameters.ramp_sec
            + self.parameters.cycles * self.parameters.base_period
        )

    @property
    def boundaries(self) -> MissionBoundaries:
        return _trajectory_boundaries(self.config, self.trajectory_duration)

    @property
    def total_sec(self) -> float:
        return self.boundaries.total

    def theta(self, elapsed: float) -> float:
        return ramped_phase(elapsed, self.omega, self.parameters.ramp_sec)

    def _path_reference(self, elapsed: float) -> np.ndarray:
        theta = self.theta(elapsed)
        center_x, center_y = self.parameters.center_xy
        amplitude_x, amplitude_y = self.parameters.amplitude_xy
        frequency_x, frequency_y = self.parameters.frequency_ratio
        phase = math.radians(self.parameters.phase_deg)
        return np.array(
            [
                center_x + amplitude_x * math.sin(frequency_x * theta + phase),
                center_y + amplitude_y * math.sin(frequency_y * theta),
                self.config.hover_altitude,
            ],
            dtype=float,
        )

    @property
    def goto_xy(self) -> tuple[float, float]:
        initial = self._path_reference(0.0)
        return float(initial[0]), float(initial[1])

    def reference(self, time_sec: float) -> tuple[np.ndarray, str]:
        t = float(time_sec)
        bounds = self.boundaries
        altitude = self.config.hover_altitude
        goto_x, goto_y = self.goto_xy

        if t < bounds.takeoff_end:
            z = altitude * cosine_ease(t / self.config.takeoff_sec)
            return np.array([0.0, 0.0, z], dtype=float), "TAKEOFF"
        if t < bounds.settle1_end:
            return np.array([0.0, 0.0, altitude], dtype=float), "SETTLE1"
        if t < bounds.goto_end:
            amount = cosine_ease((t - bounds.settle1_end) / self.config.goto_sec)
            return np.array(
                [goto_x * amount, goto_y * amount, altitude], dtype=float
            ), "GOTO"
        if t < bounds.settle2_end:
            return np.array([goto_x, goto_y, altitude], dtype=float), "SETTLE2"
        if t < bounds.trajectory_end:
            return self._path_reference(t - bounds.settle2_end), "LISSAJOUS"
        return self._path_reference(self.trajectory_duration), "HOLD"

    def effective_parameters(self) -> dict[str, object]:
        return {
            "center_xy": self.parameters.center_xy,
            "amplitude_xy": self.parameters.amplitude_xy,
            "frequency_ratio": self.parameters.frequency_ratio,
            "phase_deg": self.parameters.phase_deg,
            "base_period": self.parameters.base_period,
            "cycles": self.parameters.cycles,
            "ramp_sec": self.parameters.ramp_sec,
            "altitude": self.config.hover_altitude,
            "takeoff_sec": self.config.takeoff_sec,
            "settle_sec": self.config.settle_sec,
            "goto_sec": self.config.goto_sec,
            "post_hold_sec": self.config.post_hold_sec,
            "force_floor_start_requested": self.config.force_floor_start,
        }


def build_mission(
    config: ExperimentConfig,
    preset: str = "1",
    *,
    legacy_circle: bool = False,
) -> ReferenceMission:
    """Construct the configured mission without exposing numeric CLI modes."""

    return mission_from_experiment(
        config,
        mission_type=config.mission.type,
        preset=preset,
        legacy_circle_preset=legacy_circle,
    )


def mission_from_experiment(
    config: ExperimentConfig,
    *,
    mission_type: str | None = None,
    preset: str = "1",
    legacy_circle_preset: bool = False,
) -> ReferenceMission:
    """Construct a named mission for the shared evaluation CLI.

    Numeric mode aliases belong to the CLI layer.  Callers pass the canonical
    name and explicitly opt into legacy preset/HOLD behavior for the retained
    circle entrypoints.
    """

    selected = config.mission.type if mission_type is None else mission_type
    if selected != config.mission.type:
        raise ConfigError(
            f"mission type {selected!r} does not match resolved config "
            f"{config.mission.type!r}"
        )
    if selected == "hover":
        return HoverMission.from_experiment(config)
    if selected == "circle":
        if legacy_circle_preset:
            return CircleMission.from_experiment(config, preset)
        return CircleMission.from_parameters(config.mission)
    if selected == "lissajous":
        return LissajousMission.from_experiment(config)
    raise ConfigError(f"unknown mission type: {selected!r}")


__all__ = [
    "CircleMission",
    "HoverMission",
    "LissajousMission",
    "MissionBoundaries",
    "PHASES",
    "ReferenceMission",
    "build_mission",
    "cosine_ease",
    "mission_from_experiment",
    "ramped_phase",
]

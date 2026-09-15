"""Versioned E2E observation construction and transition-owned auxiliary state.

The first 15 values intentionally preserve the historical Crazyflie feature
contract.  This module only owns finite history and the bounded position-error
integral; reading/formatting an observation is pure and never advances either
state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


BASE_OBSERVATION_DIM = 15
ACTION_DIM = 4
BASE_OBSERVATION_SCHEMA_VERSION = "crazyflie_base_observation_v1"
LEGACY_STATE_READER = "legacy_gyro_sensor_v1"
CURRENT_STATE_READER = "current_freejoint_body_rate_v1"
LEGACY_TRACE_SCHEMA_VERSION = "legacy_transition_trace_v1"
TRACE_SCHEMA_VERSION = "policy_transition_trace_v2"
HISTORY_INTEGRAL_SCHEMA_VERSION = "e2e_history_integral_v1"

BASE_FEATURES: tuple[dict[str, Any], ...] = (
    {
        "name": "position_error_x",
        "frame": "world",
        "unit": "m",
        "definition": "body_origin_x-reference_x",
    },
    {
        "name": "position_error_y",
        "frame": "world",
        "unit": "m",
        "definition": "body_origin_y-reference_y",
    },
    {
        "name": "position_error_z",
        "frame": "world",
        "unit": "m",
        "definition": "body_origin_z-reference_z",
    },
    {
        "name": "linear_velocity_x",
        "frame": "world",
        "unit": "m/s",
        "definition": "freejoint translational qvel",
    },
    {
        "name": "linear_velocity_y",
        "frame": "world",
        "unit": "m/s",
        "definition": "freejoint translational qvel",
    },
    {
        "name": "linear_velocity_z",
        "frame": "world",
        "unit": "m/s",
        "definition": "freejoint translational qvel",
    },
    {
        "name": "quaternion_w",
        "frame": "body_to_world",
        "unit": "1",
        "definition": "freejoint quaternion wxyz",
    },
    {
        "name": "quaternion_x",
        "frame": "body_to_world",
        "unit": "1",
        "definition": "freejoint quaternion wxyz",
    },
    {
        "name": "quaternion_y",
        "frame": "body_to_world",
        "unit": "1",
        "definition": "freejoint quaternion wxyz",
    },
    {
        "name": "quaternion_z",
        "frame": "body_to_world",
        "unit": "1",
        "definition": "freejoint quaternion wxyz",
    },
    {
        "name": "angular_velocity_x",
        "frame": "body/identity_imu_site",
        "unit": "rad/s",
        "definition": "selected state reader",
    },
    {
        "name": "angular_velocity_y",
        "frame": "body/identity_imu_site",
        "unit": "rad/s",
        "definition": "selected state reader",
    },
    {
        "name": "angular_velocity_z",
        "frame": "body/identity_imu_site",
        "unit": "rad/s",
        "definition": "selected state reader",
    },
    {
        "name": "sin_yaw_error",
        "frame": "world_heading",
        "unit": "1",
        "definition": "sin(yaw-yaw_reference)",
    },
    {
        "name": "cos_yaw_error",
        "frame": "world_heading",
        "unit": "1",
        "definition": "cos(yaw-yaw_reference)",
    },
)


def observation_dimension(settings: Any | None) -> int:
    """Return the flat policy dimension for typed settings (or legacy None)."""

    if settings is None:
        return BASE_OBSERVATION_DIM
    result = BASE_OBSERVATION_DIM
    history = settings.history
    integral = settings.position_error_integral
    if history.enabled:
        result += int(history.length_steps) * (BASE_OBSERVATION_DIM + ACTION_DIM + 1)
    if integral.enabled:
        result += 3
    return result


def observation_schema(settings: Any | None) -> dict[str, Any]:
    """Return the complete semantic contract used in manifests/checkpoints."""

    if settings is None:
        return {
            "version": BASE_OBSERVATION_SCHEMA_VERSION,
            "dimension": BASE_OBSERVATION_DIM,
            "state_reader": LEGACY_STATE_READER,
            "trace_schema_version": LEGACY_TRACE_SCHEMA_VERSION,
            "base_features": [dict(item) for item in BASE_FEATURES],
            "history": {"enabled": False, "length_steps": 0},
            "position_error_integral": {"enabled": False},
        }
    return {
        "version": str(settings.schema_version),
        "dimension": observation_dimension(settings),
        "state_reader": str(settings.state_reader),
        "trace_schema_version": str(settings.trace_schema_version),
        "base_features": [dict(item) for item in BASE_FEATURES],
        "history": {
            "enabled": bool(settings.history.enabled),
            "length_steps": int(settings.history.length_steps),
            "order": "newest_to_oldest",
            "slot_features": [
                "historical_base_observation[15]",
                "applied_clipped_normalized_action[4]",
                "valid_mask[1]",
            ],
            "reset_padding": {
                "observation": "repeat_final_installed_initial_base_observation",
                "action": [0.0, 0.0, 0.0, 0.0],
                "valid": 0.0,
            },
        },
        "position_error_integral": {
            "enabled": bool(settings.position_error_integral.enabled),
            "definition": "integral(reference_position_world-body_origin_position_world) dt",
            "update": "once_per_policy_transition_using_transition_start_error",
            "clamp_m_s": list(settings.position_error_integral.clamp_m_s),
            "normalization": "componentwise divide by clamp_m_s",
            "feature_order": ["integral_x", "integral_y", "integral_z"],
        },
    }


@dataclass(frozen=True)
class HistoryEntry:
    observation: np.ndarray
    action: np.ndarray


class AuxiliaryObservationState:
    """Per-environment finite-memory state updated exactly once per transition."""

    def __init__(self, settings: Any, policy_dt: float):
        if not np.isfinite(policy_dt) or policy_dt <= 0.0:
            raise ValueError("policy_dt must be finite and positive")
        self.settings = settings
        self.policy_dt = float(policy_dt)
        self._history: deque[HistoryEntry] = deque(
            maxlen=int(settings.history.length_steps)
        )
        self._integral = np.zeros(3, dtype=float)
        self._padding_observation: np.ndarray | None = None
        self._transition_count = 0
        self._clamped_transition_counts = np.zeros(3, dtype=np.int64)
        self._saturated_transition_counts = np.zeros(3, dtype=np.int64)

    @staticmethod
    def _base(value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=float).reshape(-1)
        if result.shape != (BASE_OBSERVATION_DIM,) or not np.all(np.isfinite(result)):
            raise ValueError(
                "base observation must be a finite vector with shape (15,)"
            )
        return result.copy()

    def reset(self, initial_base_observation: np.ndarray) -> None:
        self._padding_observation = self._base(initial_base_observation)
        self._history.clear()
        self._integral.fill(0.0)
        self._transition_count = 0
        self._clamped_transition_counts.fill(0)
        self._saturated_transition_counts.fill(0)

    def compose(self, current_base_observation: np.ndarray) -> np.ndarray:
        """Build an input without mutating history, integral, or counters."""

        current = self._base(current_base_observation)
        if self._padding_observation is None:
            raise RuntimeError("auxiliary observation state must be reset before use")
        chunks: list[np.ndarray] = [current]
        history_config = self.settings.history
        if history_config.enabled:
            entries = tuple(self._history)
            for slot in range(int(history_config.length_steps)):
                if slot < len(entries):
                    entry = entries[slot]
                    chunks.extend((entry.observation, entry.action, np.ones(1)))
                else:
                    chunks.extend(
                        (
                            self._padding_observation,
                            np.zeros(ACTION_DIM),
                            np.zeros(1),
                        )
                    )
        integral_config = self.settings.position_error_integral
        if integral_config.enabled:
            clamp = np.asarray(integral_config.clamp_m_s, dtype=float)
            chunks.append(self._integral / clamp)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def advance(
        self,
        *,
        base_observation_before: np.ndarray,
        applied_action: np.ndarray,
        reference_minus_position_before: np.ndarray,
    ) -> None:
        """Advance once after a completed policy transition."""

        base_before = self._base(base_observation_before)
        action = np.asarray(applied_action, dtype=float).reshape(-1)
        error = np.asarray(reference_minus_position_before, dtype=float).reshape(-1)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError("applied action must be a finite vector with shape (4,)")
        if error.shape != (3,) or not np.all(np.isfinite(error)):
            raise ValueError("position error must be a finite vector with shape (3,)")
        if self.settings.history.enabled:
            self._history.appendleft(HistoryEntry(base_before, action.copy()))
        if self.settings.position_error_integral.enabled:
            clamp = np.asarray(
                self.settings.position_error_integral.clamp_m_s, dtype=float
            )
            candidate = self._integral + self.policy_dt * error
            clipped = np.clip(candidate, -clamp, clamp)
            self._clamped_transition_counts += (candidate != clipped).astype(np.int64)
            self._saturated_transition_counts += np.isclose(
                np.abs(clipped), clamp, rtol=0.0, atol=1e-15
            ).astype(np.int64)
            self._integral = clipped
        self._transition_count += 1

    def diagnostics(self) -> dict[str, Any]:
        denominator = max(self._transition_count, 1)
        return {
            "observation_schema_version": str(self.settings.schema_version),
            "state_reader": str(self.settings.state_reader),
            "trace_schema_version": str(self.settings.trace_schema_version),
            "policy_dt_s": self.policy_dt,
            "history_valid_count": len(self._history),
            "history_capacity": int(self.settings.history.length_steps),
            "position_integral_m_s": self._integral.tolist(),
            "position_integral_normalized": (
                self._integral
                / np.asarray(
                    self.settings.position_error_integral.clamp_m_s, dtype=float
                )
                if self.settings.position_error_integral.enabled
                else np.zeros(3)
            ).tolist(),
            "integral_clamp_transition_count_xyz": self._clamped_transition_counts.tolist(),
            "integral_clamp_ratio_xyz": (
                self._clamped_transition_counts / denominator
            ).tolist(),
            "integral_saturation_transition_count_xyz": self._saturated_transition_counts.tolist(),
            "integral_saturation_ratio_xyz": (
                self._saturated_transition_counts / denominator
            ).tolist(),
            "transition_count": self._transition_count,
        }

    def state_snapshot(self) -> Mapping[str, Any]:
        """Testing aid whose contents do not expose writable internal arrays."""

        return {
            "history": [
                {"observation": item.observation.copy(), "action": item.action.copy()}
                for item in self._history
            ],
            "integral": self._integral.copy(),
            "padding_observation": (
                None
                if self._padding_observation is None
                else self._padding_observation.copy()
            ),
            "transition_count": self._transition_count,
            "clamped_transition_counts": self._clamped_transition_counts.copy(),
            "saturated_transition_counts": self._saturated_transition_counts.copy(),
        }


__all__ = [
    "ACTION_DIM",
    "AuxiliaryObservationState",
    "BASE_FEATURES",
    "BASE_OBSERVATION_DIM",
    "BASE_OBSERVATION_SCHEMA_VERSION",
    "CURRENT_STATE_READER",
    "HISTORY_INTEGRAL_SCHEMA_VERSION",
    "LEGACY_STATE_READER",
    "LEGACY_TRACE_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "observation_dimension",
    "observation_schema",
]

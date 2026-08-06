"""Stable control-interface contracts shared by configuration and runtime code."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_RESIDUAL_SCALE = (0.022, 0.022, 0.0001, 0.3)
ACTION_DIM = 4


@dataclass(frozen=True)
class ObservationContract:
    control_mode: str
    schema: str
    dimension: int


OBSERVATION_CONTRACTS = {
    "residual": ObservationContract("residual", "residual_v1", 13),
    "e2e": ObservationContract("e2e", "e2e_v1", 15),
}


def observation_contract(control_mode: str) -> ObservationContract:
    try:
        return OBSERVATION_CONTRACTS[control_mode]
    except KeyError as exc:
        allowed = ", ".join(sorted(OBSERVATION_CONTRACTS))
        raise ValueError(
            f"control_mode must be one of {{{allowed}}}, got {control_mode!r}"
        ) from exc

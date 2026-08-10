"""Factories that own runtime environment construction."""

from __future__ import annotations

from typing import Any

from .config import ExperimentConfig
from .environment import CrazyflieResidualEnv


class EnvironmentFactory:
    """Create independent Gym environments from one immutable configuration."""

    def __init__(self, config: ExperimentConfig):
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        self.config = config

    def make(
        self, seed: int | None = None, **overrides: Any
    ) -> CrazyflieResidualEnv:
        """Build one environment, with explicit legacy keyword overrides.

        The configured XML path is passed through exactly.  No search, copy,
        relative fallback, or fabricated model is attempted.  A caller may
        override ordinary legacy environment parameters for a specific rollout;
        ``xml_path`` is accepted only when it is an explicit absolute path and
        is validated by ``CrazyflieResidualEnv``.
        """

        effective_seed = self.config.training.seed if seed is None else seed
        return CrazyflieResidualEnv(
            config=self.config,
            seed=effective_seed,
            **overrides,
        )

    def __call__(self) -> CrazyflieResidualEnv:
        """Allow use as a fresh-environment callable (for ``DummyVecEnv``)."""

        return self.make()


__all__ = ["EnvironmentFactory"]

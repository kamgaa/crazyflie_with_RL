"""Deterministic policy evaluation shared by PPO training entrypoints.

The module intentionally imports only the Python standard library at import
time.  NumPy and the MuJoCo-backed environment factory are loaded only when an
evaluation is explicitly executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .physics_version import PHYSICS_MODEL_VERSION, physics_comparison

if TYPE_CHECKING:
    from .config import ExperimentConfig


class EnvironmentFactoryProtocol(Protocol):
    """Small factory surface needed by :class:`PolicyEvaluator`."""

    def make(self, seed: int | None = None, **overrides: Any) -> Any: ...


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate result for the fixed-seed hover evaluation contract."""

    score: float
    disqualifications: int
    mean_episode_length: float
    episode_count: int
    episodes: tuple[dict[str, Any], ...] = ()
    physics_provenance: dict[str, Any] | None = None

    @property
    def mean_error(self) -> float:
        """Compatibility alias for the steady-state position-error score."""

        return self.score

    def as_metrics(self) -> dict[str, Any]:
        """Return JSON-serializable metric values."""

        return {
            "score": self.score,
            "mean_error": self.score,
            "disqualifications": self.disqualifications,
            "mean_episode_length": self.mean_episode_length,
            "episode_count": self.episode_count,
            "physics_model_version": PHYSICS_MODEL_VERSION,
            "episodes": list(self.episodes),
            "physics_provenance": self.physics_provenance,
        }


class PolicyEvaluator:
    """Evaluate a policy with the master branch's deterministic protocol.

    The episode count and seed range are profile-specific: the residual smoke
    profile uses five episodes starting at seed 100, while the active E2E
    profile uses thirty episodes starting at seed 1000.  E2E applies the
    original ``max(1, floor(fraction * length))`` tail.  Residual deliberately
    preserves the smoke script's ``errors[-int(...) :]`` edge case, where a
    zero tail selects the whole short episode.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        environment_factory: EnvironmentFactoryProtocol | None = None,
    ) -> None:
        self.config = config
        self._environment_factory = environment_factory

    @property
    def environment_factory(self) -> EnvironmentFactoryProtocol:
        if self._environment_factory is None:
            # Importing factories may make simulator code reachable, so keep it
            # behind an explicit evaluation call.
            from .factories import EnvironmentFactory

            self._environment_factory = EnvironmentFactory(self.config)
        return self._environment_factory

    def evaluate(self, model: Any | None) -> EvaluationResult:
        """Run all configured episodes and return the unweighted mean score."""

        import numpy as np

        settings = self.config.evaluation
        # Periodic training evaluation must not inherit the training
        # curriculum's moving reset distribution. Existing profiles are
        # unchanged because this override only disables the new opt-in path.
        overrides: dict[str, Any] = {
            "initial_state_randomization_enabled": False,
        }
        payload_curriculum = getattr(
            getattr(getattr(self.config, "environment", None), "payload", None),
            "curriculum",
            None,
        )
        if bool(getattr(payload_curriculum, "enabled", False)):
            # The legacy periodic score remains a payload-free nominal metric.
            # Payload checkpoint selection is handled by the fixed 14-case suite.
            overrides.update(
                {
                    "payload_curriculum_enabled": False,
                    "com_bias_randomize": False,
                    "com_bias_mass": 0.0,
                    "com_bias_offset": (0.0, 0.0),
                }
            )
        environment = self.environment_factory.make(seed=None, **overrides)
        episode_scores: list[float] = []
        episode_lengths: list[int] = []
        disqualifications = 0
        episodes = []

        try:
            for episode_index in range(settings.episode_count):
                observation, _info = environment.reset(
                    seed=settings.seed_start + episode_index
                )
                episode_record = {
                    "seed": settings.seed_start + episode_index,
                    **_info,
                }
                episodes.append(episode_record)
                position_errors: list[float] = []
                tilt_angles_deg: list[float] = []
                done = False

                while not done:
                    if model is None:
                        action = np.zeros(self.config.action_shape[0], dtype=float)
                    else:
                        action = model.predict(
                            observation,
                            deterministic=settings.deterministic,
                        )[0]

                    observation, _reward, terminated, truncated, _step_info = (
                        environment.step(action)
                    )
                    position_errors.append(
                        float(np.linalg.norm(np.asarray(observation)[0:3]))
                    )

                    quaternion = np.asarray(observation)[6:10]
                    cosine_tilt = np.clip(
                        1.0
                        - 2.0
                        * (float(quaternion[1]) ** 2 + float(quaternion[2]) ** 2),
                        -1.0,
                        1.0,
                    )
                    tilt_angles_deg.append(
                        float(np.degrees(np.arccos(cosine_tilt)))
                    )
                    done = bool(terminated or truncated)

                if not position_errors:
                    raise RuntimeError(
                        "evaluation environment ended an episode without a step"
                    )

                episode_length = len(position_errors)
                tail_length = int(episode_length * settings.tail_fraction)
                if self.config.control_mode == "e2e":
                    tail_length = max(1, tail_length)
                # Do not normalize a residual zero tail to one: ``[-0:]`` is
                # the legacy smoke-test behavior and selects the full episode.
                tail_errors = position_errors[-tail_length:]
                tail_tilts = tilt_angles_deg[-tail_length:]
                episode_lengths.append(episode_length)
                episode_scores.append(float(np.mean(tail_errors)))
                if (
                    float(np.max(tail_tilts))
                    > settings.tilt_limit_deg
                ):
                    disqualifications += 1
                if isinstance(_step_info, dict) and _step_info.get(
                    "observation_diagnostics"
                ):
                    episode_record["final_observation_diagnostics"] = dict(
                        _step_info["observation_diagnostics"]
                    )
                    episode_record["final_legacy_reward_terms"] = dict(
                        _step_info.get("legacy_reward_terms", {})
                    )
                    episode_record["final_transition_timing"] = dict(
                        _step_info.get("transition_timing", {})
                    )
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()

        return EvaluationResult(
            score=float(np.mean(episode_scores)),
            disqualifications=disqualifications,
            mean_episode_length=float(np.mean(episode_lengths)),
            episode_count=settings.episode_count,
            episodes=tuple(episodes),
            physics_provenance=physics_comparison(
                PHYSICS_MODEL_VERSION
                if model is None
                else getattr(model, "physics_model_version", None)
            ),
        )


__all__ = ["EvaluationResult", "PolicyEvaluator"]

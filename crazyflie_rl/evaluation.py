"""Deterministic policy evaluation shared by PPO training entrypoints.

The module intentionally imports only the Python standard library at import
time.  NumPy and the MuJoCo-backed environment factory are loaded only when an
evaluation is explicitly executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .config import ExperimentConfig


def position_rmse_metrics(errors: Any) -> dict[str, float | None]:
    """RMSE over sampled 3D errors; xy is horizontal distance, not axis mean.

    No time windows or per-axis normalization are introduced here.
    """
    import numpy as np

    values = np.asarray(errors, dtype=float).reshape((-1, 3))
    if not len(values):
        return {"position_rmse_xy": None, "position_rmse_z": None}
    return {
        "position_rmse_xy": float(np.sqrt(np.mean(np.sum(values[:, :2] ** 2, axis=1)))),
        "position_rmse_z": float(np.sqrt(np.mean(values[:, 2] ** 2))),
    }


class EnvironmentFactoryProtocol(Protocol):
    """Small factory surface needed by :class:`PolicyEvaluator`."""

    def make(self, seed: int | None = None, **overrides: Any) -> Any: ...


@dataclass(frozen=True)
class EvaluationResult:
    """Fixed-seed hover metrics.

    Axis RMSE pools samples across episodes (including early-ended episodes);
    tail RMSE pools exactly the existing per-episode score windows. The legacy
    score remains the unweighted episode mean of tail mean distances.
    """

    score: float
    disqualifications: int
    mean_episode_length: float
    episode_count: int
    position_rmse_xy: float | None = None
    position_rmse_z: float | None = None
    tail_position_rmse_xy: float | None = None
    tail_position_rmse_z: float | None = None

    @property
    def mean_error(self) -> float:
        """Compatibility alias for the steady-state position-error score."""

        return self.score

    def as_metrics(self) -> dict[str, float | int | None]:
        """Return JSON-serializable metric values."""

        return {
            "score": self.score,
            "mean_error": self.score,
            "disqualifications": self.disqualifications,
            "mean_episode_length": self.mean_episode_length,
            "episode_count": self.episode_count,
            "position_rmse_xy": self.position_rmse_xy,
            "position_rmse_z": self.position_rmse_z,
            "tail_position_rmse_xy": self.tail_position_rmse_xy,
            "tail_position_rmse_z": self.tail_position_rmse_z,
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
        environment = self.environment_factory.make(seed=None)
        episode_scores: list[float] = []
        episode_lengths: list[int] = []
        disqualifications = 0
        all_vectors: list[Any] = []
        tail_vectors: list[Any] = []

        try:
            for episode_index in range(settings.episode_count):
                observation, _info = environment.reset(
                    seed=settings.seed_start + episode_index
                )
                position_errors: list[float] = []
                position_vectors: list[Any] = []
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
                    position_vectors.append(np.asarray(observation, dtype=float)[0:3].copy())

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
                all_vectors.extend(position_vectors)
                tail_vectors.extend(position_vectors[-tail_length:])
                episode_lengths.append(episode_length)
                episode_scores.append(float(np.mean(tail_errors)))
                if (
                    float(np.max(tail_tilts))
                    > settings.tilt_limit_deg
                ):
                    disqualifications += 1
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()

        return EvaluationResult(
            score=float(np.mean(episode_scores)),
            disqualifications=disqualifications,
            mean_episode_length=float(np.mean(episode_lengths)),
            episode_count=settings.episode_count,
            **position_rmse_metrics(all_vectors),
            **{f"tail_{key}": value for key, value in position_rmse_metrics(tail_vectors).items()},
        )


__all__ = ["EvaluationResult", "PolicyEvaluator", "position_rmse_metrics"]

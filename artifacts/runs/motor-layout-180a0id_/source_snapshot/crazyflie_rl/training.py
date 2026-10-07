"""Configuration-driven PPO training with run-scoped artifacts.

Importing this module is inert: Stable-Baselines3, NumPy, Torch, Gymnasium,
MuJoCo, environment instances, and artifact directories are all deferred until
an explicit :meth:`PPOTrainer.train` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from .evaluation import EvaluationResult, PolicyEvaluator

if TYPE_CHECKING:
    from .config import ExperimentConfig


class ArtifactManagerProtocol(Protocol):
    tensorboard_dir: Path

    def save_model(
        self,
        model: Any,
        kind: str,
        *,
        timestep: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path: ...

    def write_metrics(self, kind: str, payload: Mapping[str, Any]) -> Path: ...

    def finalize(self, status: str = "completed", **extra: Any) -> None: ...


class EnvironmentFactoryProtocol(Protocol):
    def make(self, seed: int | None = None, **overrides: Any) -> Any: ...


class EvaluatorProtocol(Protocol):
    def evaluate(self, model: Any | None) -> EvaluationResult: ...


@dataclass(frozen=True)
class TrainingOutcome:
    """Paths and evaluations produced by one completed training run."""

    requested_timesteps: int
    actual_timesteps: int
    floor_evaluation: EvaluationResult
    best_evaluation: EvaluationResult | None
    final_evaluation: EvaluationResult | None
    best_model_path: Path | None
    final_model_path: Path


def _result_fields(prefix: str, result: EvaluationResult) -> dict[str, Any]:
    return {
        f"{prefix}_score": result.score,
        f"{prefix}_disqualifications": result.disqualifications,
        f"{prefix}_mean_episode_length": result.mean_episode_length,
        f"{prefix}_episode_count": result.episode_count,
        **{f"{prefix}_{name}": getattr(result, name) for name in (
            "position_rmse_xy", "position_rmse_z",
            "tail_position_rmse_xy", "tail_position_rmse_z",
        )},
    }


def _improvement_percent(policy_score: float, floor_score: float) -> float:
    denominator = max(abs(floor_score), 1e-12)
    return 100.0 * (floor_score - policy_score) / denominator


def _make_reward_diagnostics_callback() -> Any:
    """Defer SB3 imports, as with the existing evaluation callback.

    Scalars describe reward trajectories, not proof of gradient conflict.
    SB3's normal rollout logger dump writes the recorded values to TensorBoard.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    components = (
        "position", "velocity", "tilt", "angular_velocity", "yaw",
        "action", "action_rate", "crash",
    )
    groups = {
        "reward_terms": (*components, "total"),
        "reward_raw": (
            "position_sq", "position_sq_xy", "position_sq_z",
            "velocity_sq", "tilt_error", "angular_velocity_sq",
            "velocity_error_sq", "desired_velocity_sq", "velocity_reward_sq",
            "yaw_error_sq", "action_sq", "action_rate_sq",
        ),
        "reward_costs": ("position_xy", "position_z"),
    }

    class RewardDiagnosticsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._on_rollout_start()

        def _on_rollout_start(self) -> None:
            self.sums = {
                f"{group}/{name}": 0.0
                for group, names in groups.items() for name in names
            }
            self.counts = dict.fromkeys(self.sums, 0)
            self.magnitudes = dict.fromkeys(components, 0.0)

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                for group, names in groups.items():
                    values = info.get(group, {})
                    for name in names:
                        if name not in values:
                            continue
                        value = float(values[name])
                        tag = f"{group}/{name}"
                        self.sums[tag] += value
                        self.counts[tag] += 1
                        if group == "reward_terms" and name != "total":
                            self.magnitudes[name] += abs(value)
            return True

        def _on_rollout_end(self) -> None:
            for tag, total in self.sums.items():
                if self.counts[tag]:
                    self.logger.record(tag, total / self.counts[tag])
            if any(self.counts[f"reward_terms/{name}"] for name in components):
                denominator = sum(self.magnitudes.values()) + 1e-12
                for name, magnitude in self.magnitudes.items():
                    self.logger.record(f"reward_fraction/{name}", magnitude / denominator)

    return RewardDiagnosticsCallback()


class PPOTrainer:
    """Build, evaluate, train, and persist one configured PPO experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        artifact_manager: ArtifactManagerProtocol | None = None,
        environment_factory: EnvironmentFactoryProtocol | None = None,
        evaluator: EvaluatorProtocol | None = None,
    ) -> None:
        self.config = config
        self._artifact_manager = artifact_manager
        self._environment_factory = environment_factory
        self._evaluator = evaluator

    @property
    def artifact_manager(self) -> ArtifactManagerProtocol:
        if self._artifact_manager is None:
            from .artifacts import ArtifactManager

            self._artifact_manager = ArtifactManager.create(self.config)
        return self._artifact_manager

    @property
    def environment_factory(self) -> EnvironmentFactoryProtocol:
        if self._environment_factory is None:
            from .factories import EnvironmentFactory

            self._environment_factory = EnvironmentFactory(self.config)
        return self._environment_factory

    @property
    def evaluator(self) -> EvaluatorProtocol:
        if self._evaluator is None:
            self._evaluator = PolicyEvaluator(
                self.config,
                environment_factory=self.environment_factory,
            )
        return self._evaluator

    def _build_model(self, vector_environment: Any) -> Any:
        """Instantiate PPO while explicitly forwarding every config field."""

        from stable_baselines3 import PPO

        ppo = self.config.training.ppo
        model = PPO(
            policy=ppo.policy,
            env=vector_environment,
            learning_rate=ppo.learning_rate,
            n_steps=ppo.n_steps,
            batch_size=ppo.batch_size,
            n_epochs=ppo.n_epochs,
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            clip_range=ppo.clip_range,
            ent_coef=ppo.ent_coef,
            vf_coef=ppo.vf_coef,
            max_grad_norm=ppo.max_grad_norm,
            target_kl=ppo.target_kl,
            policy_kwargs={
                "log_std_init": ppo.log_std_init,
                "net_arch": list(ppo.net_arch),
            },
            tensorboard_log=str(self.artifact_manager.tensorboard_dir),
            seed=self.config.training.seed,
            device=ppo.device,
            verbose=ppo.verbose,
        )
        from .velocity_reference import observation_contract, velocity_semantics, velocity_reward_semantics
        model.observation_contract = observation_contract(self.config)
        model.velocity_semantics = velocity_semantics(self.config)
        model.velocity_reward_semantics = velocity_reward_semantics(self.config)
        return model

    def _write_floor_metrics(
        self,
        artifacts: ArtifactManagerProtocol,
        result: EvaluationResult,
    ) -> None:
        artifacts.write_metrics(
            "evaluation-floor-step0000000",
            {
                "timestep": 0,
                "control_mode": self.config.control_mode,
                **_result_fields("floor", result),
            },
        )

    def _write_policy_metrics(
        self,
        artifacts: ArtifactManagerProtocol,
        *,
        timestep: int,
        policy_result: EvaluationResult,
        floor_result: EvaluationResult,
        saved_as_best: bool,
        kind: str | None = None,
    ) -> None:
        artifacts.write_metrics(
            kind or f"evaluation-step{timestep:07d}",
            {
                "timestep": timestep,
                "control_mode": self.config.control_mode,
                "saved_as_best": saved_as_best,
                "improvement_percent": _improvement_percent(
                    policy_result.score,
                    floor_result.score,
                ),
                **_result_fields("policy", policy_result),
                **_result_fields("floor", floor_result),
            },
        )

    def _make_evaluation_callback(
        self,
        artifacts: ArtifactManagerProtocol,
    ) -> Any | None:
        interval = self.config.evaluation.evaluation_interval
        if interval <= 0:
            return None

        from stable_baselines3.common.callbacks import BaseCallback

        trainer = self

        class EvaluationCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self.interval = interval
                self.last_evaluation_step = 0
                self.best_score = float("inf")
                self.best_result: EvaluationResult | None = None
                self.best_model_path: Path | None = None

            def _on_step(self) -> bool:
                if self.num_timesteps - self.last_evaluation_step < self.interval:
                    return True

                self.last_evaluation_step = int(self.num_timesteps)
                current = trainer.evaluator.evaluate(self.model)
                floor = trainer.evaluator.evaluate(None)
                saved = False

                # Preserve master exactly: strict score improvement and zero
                # tail-tilt disqualifications.  Beating floor or surviving the
                # full horizon is intentionally not an additional gate.
                if (
                    current.score < self.best_score
                    and current.disqualifications == 0
                ):
                    self.best_score = current.score
                    self.best_result = current
                    self.best_model_path = artifacts.save_model(
                        self.model,
                        "best",
                        timestep=int(self.num_timesteps),
                    )
                    saved = True
                    tag = "WIN*(saved)"
                else:
                    tag = "WIN" if current.score < floor.score else "..."

                trainer._write_policy_metrics(
                    artifacts,
                    timestep=int(self.num_timesteps),
                    policy_result=current,
                    floor_result=floor,
                    saved_as_best=saved,
                )
                print(
                    f"  step {self.num_timesteps:>7d} | "
                    f"{trainer.config.control_mode}={current.score:.4f} "
                    f"| floor={floor.score:.4f} "
                    f"| surv={current.mean_episode_length:.0f} "
                    f"| disq={current.disqualifications}/"
                    f"{current.episode_count} | {tag} "
                    f"({_improvement_percent(current.score, floor.score):+.1f}%)"
                )
                return True

        return EvaluationCallback()

    def train(self, *, total_timesteps: int | None = None) -> TrainingOutcome:
        """Execute one training run and save run-scoped best/final archives."""

        requested_timesteps = (
            self.config.training.total_timesteps
            if total_timesteps is None
            else total_timesteps
        )
        if (
            isinstance(requested_timesteps, bool)
            or not isinstance(requested_timesteps, int)
            or requested_timesteps <= 0
        ):
            raise ValueError("total_timesteps must be a positive integer")

        # Missing static resources should fail before a run directory exists.
        self.config.require_runtime_resources()
        reward = self.config.environment.reward
        print(
            f"[config] position_xy_weight={reward.effective_position_xy_weight:g} "
            f"position_z_weight={reward.effective_position_z_weight:g} "
            f"velocity_weight={reward.velocity_weight:g} "
            f"e2e_velocity={self.config.environment.e2e_velocity} "
            f"action_scale={self.config.environment.residual_scale}"
        )
        artifacts = self.artifact_manager
        vector_environment: Any | None = None

        try:
            from stable_baselines3.common.vec_env import DummyVecEnv

            def make_training_environment():
                env = self.environment_factory.make()
                if getattr(self.config.training, 'record_episodes', False):
                    from .training_observers import EpisodeCSVRecorder
                    env = EpisodeCSVRecorder(env, artifacts.path('metrics', 'training-episodes', '.csv'))
                return env

            vector_environment = DummyVecEnv([make_training_environment])
            model = self._build_model(vector_environment)

            floor_result = self.evaluator.evaluate(None)
            self._write_floor_metrics(artifacts, floor_result)
            floor_label = (
                "PID-only" if self.config.control_mode == "residual"
                else "open-loop-hover"
            )
            print(
                f"[floor] {floor_label} "
                f"(n={floor_result.episode_count} avg) = "
                f"{floor_result.score:.4f} m, "
                f"surv={floor_result.mean_episode_length:.0f}"
            )

            callback = self._make_evaluation_callback(artifacts)
            # SB3 converts a callback list into CallbackList. Keep the original
            # evaluation handle for the unchanged best/final selection below.
            callbacks = [_make_reward_diagnostics_callback()]
            checkpoint_interval = getattr(self.config.training, 'checkpoint_interval', 0)
            if checkpoint_interval:
                from .training_observers import make_periodic_checkpoint_callback
                callbacks.append(make_periodic_checkpoint_callback(artifacts, checkpoint_interval))
            if callback is not None:
                callbacks.append(callback)
            model.learn(
                total_timesteps=int(requested_timesteps),
                callback=callbacks,
                progress_bar=False,
            )
            actual_timesteps = int(
                getattr(model, "num_timesteps", requested_timesteps)
            )

            best_result: EvaluationResult | None = None
            best_model_path: Path | None = None
            final_result: EvaluationResult | None = None

            if callback is None:
                # The residual smoke profile has no periodic callback.  Its
                # single trained-policy evaluation is therefore also the best
                # checkpoint under the run-scoped best/final naming contract.
                final_result = self.evaluator.evaluate(model)
                best_result = final_result
                best_model_path = artifacts.save_model(
                    model,
                    "best",
                    timestep=actual_timesteps,
                )
                self._write_policy_metrics(
                    artifacts,
                    timestep=actual_timesteps,
                    policy_result=final_result,
                    floor_result=floor_result,
                    saved_as_best=True,
                    kind=f"evaluation-trained-step{actual_timesteps:07d}",
                )
                print(
                    f"[learned] {self.config.control_mode}="
                    f"{final_result.score:.4f} m | "
                    f"delta={floor_result.score-final_result.score:+.4f} m "
                    f"({_improvement_percent(final_result.score, floor_result.score):+.1f}%)"
                )
            else:
                best_result = callback.best_result
                best_model_path = callback.best_model_path

            final_model_path = artifacts.save_model(
                model,
                "final",
                timestep=actual_timesteps,
            )
            artifacts.finalize(
                "completed",
                requested_total_timesteps=int(requested_timesteps),
                actual_total_timesteps=actual_timesteps,
                best_model=(
                    str(best_model_path) if best_model_path is not None else None
                ),
                final_model=str(final_model_path),
            )
        except BaseException as exc:
            artifacts.finalize(
                "failed",
                error_type=type(exc).__name__,
                error=str(exc),
                requested_total_timesteps=int(requested_timesteps),
            )
            raise
        finally:
            if vector_environment is not None:
                vector_environment.close()

        print("done.")
        return TrainingOutcome(
            requested_timesteps=int(requested_timesteps),
            actual_timesteps=actual_timesteps,
            floor_evaluation=floor_result,
            best_evaluation=best_result,
            final_evaluation=final_result,
            best_model_path=best_model_path,
            final_model_path=final_model_path,
        )


__all__ = ["PPOTrainer", "TrainingOutcome"]

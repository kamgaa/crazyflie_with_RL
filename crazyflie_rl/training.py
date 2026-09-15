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
from .physics_version import PHYSICS_MODEL_VERSION

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

    def record_policy_initialization(
        self, parent_model: Path, provenance: Mapping[str, Any]
    ) -> None: ...

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
    best_recovery_model_path: Path | None = None
    best_payload_model_path: Path | None = None


def _result_fields(prefix: str, result: EvaluationResult) -> dict[str, Any]:
    return {
        f"{prefix}_score": result.score,
        f"{prefix}_disqualifications": result.disqualifications,
        f"{prefix}_mean_episode_length": result.mean_episode_length,
        f"{prefix}_episode_count": result.episode_count,
        f"{prefix}_evaluation": result.as_metrics(),
    }


def _improvement_percent(policy_score: float, floor_score: float) -> float:
    denominator = max(abs(floor_score), 1e-12)
    return 100.0 * (floor_score - policy_score) / denominator


def _reward_mode(config: Any) -> str:
    """Read reward provenance while retaining lightweight test/fake configs."""

    return str(
        getattr(
            getattr(getattr(config, "environment", None), "reward", None),
            "mode",
            "legacy",
        )
    )


def _initial_state_randomization_enabled(config: Any) -> bool:
    return bool(
        getattr(
            getattr(
                getattr(config, "environment", None),
                "initial_state_randomization",
                None,
            ),
            "enabled",
            False,
        )
    )


def _policy_initialization_required(config: Any) -> bool:
    return bool(
        getattr(
            getattr(config, "training", None), "policy_initialization_required", False
        )
    )


def _payload_curriculum_enabled(config: Any) -> bool:
    return bool(
        getattr(
            getattr(getattr(config, "environment", None), "payload", None),
            "curriculum",
            None,
        )
        and getattr(config.environment.payload.curriculum, "enabled", False)
    )


def _observation_input_expansion_enabled(config: Any) -> bool:
    return bool(
        getattr(getattr(config, "environment", None), "observation", None) is not None
        and tuple(getattr(config, "observation_shape", (15,))) != (15,)
    )


def _recovery_evaluation_settings(config: Any) -> Any | None:
    settings = getattr(getattr(config, "evaluation", None), "recovery", None)
    return settings if bool(getattr(settings, "enabled", False)) else None


def _payload_evaluation_settings(config: Any) -> Any | None:
    settings = getattr(getattr(config, "evaluation", None), "payload", None)
    return settings if bool(getattr(settings, "enabled", False)) else None


class PPOTrainer:
    """Build, evaluate, train, and persist one configured PPO experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        artifact_manager: ArtifactManagerProtocol | None = None,
        environment_factory: EnvironmentFactoryProtocol | None = None,
        evaluator: EvaluatorProtocol | None = None,
        init_policy_from: str | Path | None = None,
    ) -> None:
        self.config = config
        self._artifact_manager = artifact_manager
        self._environment_factory = environment_factory
        self._evaluator = evaluator
        self.init_policy_from = (
            None
            if init_policy_from is None
            else Path(init_policy_from).expanduser().resolve()
        )

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
        # Runtime-only provenance for evaluations performed before this model
        # has an artifact manifest. Saved archives use the manifest field.
        model.physics_model_version = getattr(
            self.config, "physics_model_version", PHYSICS_MODEL_VERSION
        )
        configured_observation_schema = getattr(self.config, "observation_schema", None)
        if configured_observation_schema is not None:
            model.observation_schema = dict(configured_observation_schema)
        return model

    def _initialize_policy_from_donor(
        self, model: Any, donor_path: Path
    ) -> dict[str, Any]:
        """Load a donor archive only long enough to copy its policy tensors."""

        from stable_baselines3 import PPO

        from .warm_start import (
            copy_policy_parameters,
            copy_policy_parameters_with_input_expansion,
        )

        donor = PPO.load(str(donor_path), device=self.config.training.ppo.device)
        if _observation_input_expansion_enabled(self.config):
            return copy_policy_parameters_with_input_expansion(model, donor)
        copy_policy_parameters(model, donor)
        return {
            "strategy": "policy_parameters_only",
            "optimizer_state_copied": False,
            "rollout_buffer_copied": False,
            "initial_timestep": 0,
        }

    @staticmethod
    def _payload_sampling_statistics(vector_environment: Any) -> dict[str, Any] | None:
        """Read the single training environment's cumulative sampler counters."""

        if vector_environment is None:
            return None
        env_method = getattr(vector_environment, "env_method", None)
        if callable(env_method):
            values = env_method("payload_curriculum_statistics")
            if values:
                return dict(values[0])
        environments = getattr(vector_environment, "envs", None)
        if environments:
            method = getattr(environments[0], "payload_curriculum_statistics", None)
            if callable(method):
                return dict(method())
        environment = getattr(vector_environment, "environment", None)
        method = getattr(environment, "payload_curriculum_statistics", None)
        return dict(method()) if callable(method) else None

    def _evaluate_recovery(
        self, model: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Evaluate the opt-in reduced grid with training randomization disabled."""

        from .recovery import (
            evaluate_recovery_case,
            reduced_recovery_cases,
            summarize_recovery,
        )

        settings = _recovery_evaluation_settings(self.config)
        if settings is None:
            raise RuntimeError("recovery evaluation is not enabled")
        environment = self.environment_factory.make(
            seed=self.config.evaluation.seed_start,
            initial_state_randomization_enabled=False,
            payload_curriculum_enabled=False,
            com_bias_randomize=False,
            com_bias_mass=0.0,
            com_bias_offset=(0.0, 0.0),
        )
        try:
            results = [
                evaluate_recovery_case(
                    environment,
                    model,
                    case,
                    seed=self.config.evaluation.seed_start + index,
                    duration_s=settings.duration_s,
                )
                for index, case in enumerate(reduced_recovery_cases())
            ]
        finally:
            environment.close()
        return results, summarize_recovery(
            "training-policy", results, controller_type="e2e_ppo"
        )

    def _evaluate_payload(
        self, model: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run the fixed payload/recovery grid with all training DR disabled."""

        from .payload_evaluation import evaluate_fixed_payload_suite

        settings = _payload_evaluation_settings(self.config)
        if settings is None:
            raise RuntimeError("payload evaluation is not enabled")
        return evaluate_fixed_payload_suite(
            self.environment_factory,
            model,
            seed=int(settings.seed),
            duration_s=float(settings.duration_s),
        )

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
                "reward_mode": _reward_mode(self.config),
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
                "reward_mode": _reward_mode(self.config),
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
        recovery_settings = _recovery_evaluation_settings(self.config)
        recovery_interval = (
            int(recovery_settings.evaluation_interval)
            if recovery_settings is not None
            else 0
        )
        payload_settings = _payload_evaluation_settings(self.config)
        payload_interval = (
            int(payload_settings.evaluation_interval)
            if payload_settings is not None
            else 0
        )
        if interval <= 0 and recovery_interval <= 0 and payload_interval <= 0:
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
                self.last_recovery_evaluation_step = 0
                self.best_recovery_rank: tuple[float, ...] | None = None
                self.best_recovery_summary: dict[str, Any] | None = None
                self.best_recovery_model_path: Path | None = None
                self.last_payload_evaluation_step = 0
                self.best_payload_rank: tuple[float, ...] | None = None
                self.best_payload_summary: dict[str, Any] | None = None
                self.best_payload_model_path: Path | None = None

            def _on_step(self) -> bool:
                nominal_due = (
                    self.interval > 0
                    and self.num_timesteps - self.last_evaluation_step >= self.interval
                )
                recovery_due = (
                    recovery_interval > 0
                    and self.num_timesteps - self.last_recovery_evaluation_step
                    >= recovery_interval
                )
                payload_due = (
                    payload_interval > 0
                    and self.num_timesteps - self.last_payload_evaluation_step
                    >= payload_interval
                )
                if not nominal_due and not recovery_due and not payload_due:
                    return True

                if nominal_due:
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

                if recovery_due:
                    from .recovery import recovery_ranking_key

                    self.last_recovery_evaluation_step = int(self.num_timesteps)
                    results, summary = trainer._evaluate_recovery(self.model)
                    rank = recovery_ranking_key(summary)
                    saved_recovery = False
                    # Nominal hover is a hard gate, not merely another weighted
                    # grid row. The existing nominal ``best`` archive remains
                    # independent from this recovery-selected checkpoint.
                    if bool(summary["nominal_hover_success"]) and (
                        self.best_recovery_rank is None
                        or rank < self.best_recovery_rank
                    ):
                        self.best_recovery_rank = rank
                        self.best_recovery_summary = dict(summary)
                        self.best_recovery_model_path = artifacts.save_model(
                            self.model,
                            "best-recovery",
                            timestep=int(self.num_timesteps),
                            metadata={"recovery_summary": summary},
                        )
                        saved_recovery = True
                    artifacts.write_metrics(
                        f"evaluation-recovery-step{int(self.num_timesteps):07d}",
                        {
                            "timestep": int(self.num_timesteps),
                            "control_mode": trainer.config.control_mode,
                            "reward_mode": _reward_mode(trainer.config),
                            "training_randomization_disabled": True,
                            "saved_as_best_recovery": saved_recovery,
                            "ranking": [
                                "nominal_hover_success_required",
                                "overall_success_rate_descending",
                                "terminated_case_count_ascending",
                                "position_perturbation_success_rate_descending",
                                "attitude_perturbation_success_rate_descending",
                                "mean_maximum_position_error_m_ascending",
                                "mean_recovery_time_s_ascending",
                                "exact_hover_initial_action_l2_ascending",
                            ],
                            "summary": summary,
                            "case_results": results,
                        },
                    )
                    print(
                        f"  recovery {self.num_timesteps:>7d} | "
                        f"nominal={summary['nominal_hover_success']} | "
                        f"success={summary['overall_success_rate']:.3f} | "
                        f"best={'saved' if saved_recovery else 'unchanged'}"
                    )
                if payload_due:
                    from .payload_evaluation import (
                        BEST_PAYLOAD_RANKING,
                        best_payload_ranking_key,
                    )

                    self.last_payload_evaluation_step = int(self.num_timesteps)
                    results, summary = trainer._evaluate_payload(self.model)
                    saved_payload = False
                    rank: tuple[float, ...] | None = None
                    if bool(summary["nominal_success"]):
                        rank = best_payload_ranking_key(summary)
                        if (
                            self.best_payload_rank is None
                            or rank < self.best_payload_rank
                        ):
                            self.best_payload_rank = rank
                            self.best_payload_summary = dict(summary)
                            self.best_payload_model_path = artifacts.save_model(
                                self.model,
                                "best-payload",
                                timestep=int(self.num_timesteps),
                                metadata={"payload_summary": summary},
                            )
                            saved_payload = True
                    artifacts.write_metrics(
                        f"evaluation-payload-step{int(self.num_timesteps):07d}",
                        {
                            "timestep": int(self.num_timesteps),
                            "control_mode": trainer.config.control_mode,
                            "reward_mode": _reward_mode(trainer.config),
                            "training_payload_dr_disabled": True,
                            "training_initial_state_randomization_disabled": True,
                            "saved_as_best_payload": saved_payload,
                            "best_payload_exists": self.best_payload_model_path
                            is not None,
                            "ranking": list(BEST_PAYLOAD_RANKING),
                            "candidate_rank": None if rank is None else list(rank),
                            "no_best_payload_reason": (
                                "nominal_success_gate_failed"
                                if not bool(summary["nominal_success"])
                                else None
                            ),
                            "summary": summary,
                            "case_results": results,
                        },
                    )
                    print(
                        f"  payload {self.num_timesteps:>7d} | "
                        f"nominal={summary['nominal_success']} | "
                        f"success={summary['payload_success_count']}/"
                        f"{summary['payload_case_count']} | "
                        f"complete={summary['payload_completed_full_duration_count']} | "
                        f"best={'saved' if saved_payload else 'unchanged'}"
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

        if (
            _policy_initialization_required(self.config)
            and self.init_policy_from is None
        ):
            raise ValueError(
                "this policy-initialized profile requires --init-policy-from; "
                "a compatible donor must be selected explicitly"
            )
        donor_provenance: Any | None = None
        if self.init_policy_from is not None:
            if _observation_input_expansion_enabled(self.config):
                from .warm_start import validate_observation_expansion_donor

                donor_provenance = validate_observation_expansion_donor(
                    self.init_policy_from, self.config
                )
                parent_step = donor_provenance.saved_timestep
            elif _payload_curriculum_enabled(self.config):
                # Payload DR intentionally starts from the user-selected
                # recovery checkpoint. It must satisfy the inference ABI but
                # need not be a nominal-only donor.
                from .warm_start import validate_e2e_policy_compatibility

                donor_provenance = validate_e2e_policy_compatibility(
                    self.init_policy_from, self.config
                )
                parent_step = donor_provenance.saved_timestep
            else:
                from .warm_start import validate_legacy_e2e_donor

                donor_provenance = validate_legacy_e2e_donor(
                    self.init_policy_from, self.config
                )
                parent_step = donor_provenance.model_timestep
            print(
                "[warm-start] "
                f"strategy={donor_provenance.as_dict().get('strategy', 'policy_parameters_only')} | "
                f"parent={donor_provenance.model_path} | "
                f"parent_step={parent_step} | "
                "new_optimizer=true | initial_timestep=0"
            )

        # Missing static resources should fail before a run directory exists.
        self.config.require_runtime_resources()
        artifacts = self.artifact_manager
        vector_environment: Any | None = None

        try:
            from stable_baselines3.common.vec_env import DummyVecEnv

            if _initial_state_randomization_enabled(
                self.config
            ) or _payload_curriculum_enabled(self.config):
                training_environment = lambda: self.environment_factory.make(
                    track_curriculum_steps=True
                )
            else:
                # Preserve the original callable and construction path for all
                # existing profiles.
                training_environment = self.environment_factory.make
            vector_environment = DummyVecEnv([training_environment])
            model = self._build_model(vector_environment)
            if self.init_policy_from is not None:
                migration = self._initialize_policy_from_donor(
                    model, self.init_policy_from
                )
                initialization_provenance = donor_provenance.as_dict()
                initialization_provenance["parameter_transfer"] = migration
                initialization_provenance["strategy"] = migration["strategy"]
                artifacts.record_policy_initialization(
                    self.init_policy_from,
                    initialization_provenance,
                )

            floor_result = self.evaluator.evaluate(None)
            self._write_floor_metrics(artifacts, floor_result)
            floor_label = (
                "PID-only"
                if self.config.control_mode == "residual"
                else "open-loop-hover"
            )
            print(
                f"[floor] {floor_label} "
                f"(n={floor_result.episode_count} avg) = "
                f"{floor_result.score:.4f} m, "
                f"surv={floor_result.mean_episode_length:.0f}"
            )

            callback = self._make_evaluation_callback(artifacts)
            model.learn(
                total_timesteps=int(requested_timesteps),
                callback=callback,
                progress_bar=False,
            )
            actual_timesteps = int(getattr(model, "num_timesteps", requested_timesteps))
            payload_sampling_statistics = self._payload_sampling_statistics(
                vector_environment
            )
            if _payload_curriculum_enabled(self.config):
                if payload_sampling_statistics is None:
                    raise RuntimeError(
                        "payload curriculum statistics are unavailable from training env"
                    )
                artifacts.write_metrics(
                    "payload-curriculum-sampling",
                    payload_sampling_statistics,
                )

            best_result: EvaluationResult | None = None
            best_model_path: Path | None = None
            best_recovery_model_path: Path | None = None
            best_payload_model_path: Path | None = None
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
                    f"delta={floor_result.score - final_result.score:+.4f} m "
                    f"({_improvement_percent(final_result.score, floor_result.score):+.1f}%)"
                )
            else:
                best_result = callback.best_result
                best_model_path = callback.best_model_path
                best_recovery_model_path = callback.best_recovery_model_path
                best_payload_model_path = callback.best_payload_model_path

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
                best_recovery_model=(
                    str(best_recovery_model_path)
                    if best_recovery_model_path is not None
                    else None
                ),
                best_payload_model=(
                    str(best_payload_model_path)
                    if best_payload_model_path is not None
                    else None
                ),
                best_payload_status=(
                    "selected"
                    if best_payload_model_path is not None
                    else "none: no evaluated candidate passed nominal success"
                ),
                payload_curriculum_statistics=payload_sampling_statistics,
                final_model=str(final_model_path),
            )
        except BaseException as exc:
            artifacts.finalize(
                "failed",
                error_type=type(exc).__name__,
                error=str(exc),
                requested_total_timesteps=int(requested_timesteps),
                payload_curriculum_statistics=self._payload_sampling_statistics(
                    vector_environment
                ),
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
            best_recovery_model_path=best_recovery_model_path,
            best_payload_model_path=best_payload_model_path,
        )


__all__ = ["PPOTrainer", "TrainingOutcome"]

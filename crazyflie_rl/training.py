"""Config-driven PPO training without import-time runtime side effects.

The numerical PPO and evaluation settings come exclusively from the resolved
``ExperimentConfig``. The supplied profiles preserve the active behavior that
previously lived in ``train_ppo_02.py``; legacy ``train_ppo.py`` values are not
consulted.

NumPy, MuJoCo, Torch and Stable-Baselines3 are imported only while executing a
runner method. Importing this module cannot create an environment or model,
start training, or write an artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .config import ExperimentConfig


class ArtifactRunProtocol(Protocol):
    """Run artifact interface used by :class:`ExperimentRunner`.

    ``save_model`` is also the explicit sidecar-manifest update point for each
    best/final model.
    """

    tensorboard_dir: Path

    def save_model(
        self,
        model: Any,
        kind: str,
        timestep: int | None = None,
    ) -> Path: ...

    def write_metrics(self, kind: str, payload: Mapping[str, Any]) -> Path: ...

    def finalize(
        self,
        status: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate returned by the preserved 30-episode evaluation method."""

    mean_error: float
    disqualifications: int
    mean_episode_length: float


def _config_value(config: Mapping[str, Any], dotted_path: str) -> Any:
    """Read one required resolved setting; never substitute a hidden default."""
    value: Any = config
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"missing required resolved setting: {dotted_path}")
        value = value[key]
    return value


class ExperimentRunner:
    """Create, evaluate, train, and persist one configured PPO experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        artifact_run: ArtifactRunProtocol | None = None,
    ) -> None:
        self.config = config
        self._artifact_run = artifact_run

        # ExperimentConfig validates these mode contracts while loading. Keep
        # local copies so the exact runtime values are used consistently in the
        # environment, metrics, model sidecars, and final manifest.
        self.control_mode = config.control_mode
        self.observation_schema = config.observation_schema
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.residual_scale = config.residual_scale
        self.seed = config.seed

    def _new_artifact_run(self) -> ArtifactRunProtocol:
        # ArtifactRun.create writes resolved.yaml/manifest.json, so it must stay
        # behind an explicit training call.
        from .artifacts import ArtifactRun

        return ArtifactRun.create(self.config)

    @property
    def artifact_run(self) -> ArtifactRunProtocol:
        if self._artifact_run is None:
            self._artifact_run = self._new_artifact_run()
        return self._artifact_run

    def make_env(self) -> Any:
        """Construct one environment from required resolved settings."""
        from crazyflie_residual_env import CrazyflieResidualEnv

        environment = _config_value(self.config, "environment")
        return CrazyflieResidualEnv(
            str(self.config.resolve_path("mujoco_xml")),
            policy_hz=float(environment["policy_hz"]),
            episode_sec=float(environment["episode_sec"]),
            residual_scale=self.residual_scale,
            mode=self.control_mode,
            com_bias_mass=float(environment["com_bias_mass"]),
            com_bias_offset=tuple(environment["com_bias_offset"]),
            com_bias_randomize=bool(environment["com_bias_randomize"]),
            pos_perturb=float(environment["pos_perturb"]),
            att_perturb_deg=float(environment["att_perturb_deg"]),
            seed=self.seed,
        )

    def evaluate_policy(self, model: Any | None) -> EvaluationResult:
        """Run the original deterministic, fixed-seed evaluation protocol."""
        import numpy as np

        evaluation = _config_value(self.config, "evaluation")
        episode_count = int(evaluation["n_episodes"])
        seed_start = int(evaluation["seed_start"])
        tail_fraction = float(evaluation["tail_fraction"])
        tilt_limit = float(evaluation["tilt_limit_deg"])
        deterministic = bool(evaluation["deterministic"])

        env = self.make_env()
        errors: list[float] = []
        episode_lengths: list[int] = []
        disqualifications = 0
        try:
            for episode in range(episode_count):
                observation, _ = env.reset(seed=seed_start + episode)
                episode_errors: list[float] = []
                episode_tilts: list[float] = []
                done = False
                while not done:
                    action = (
                        model.predict(observation, deterministic=deterministic)[0]
                        if model is not None
                        else np.zeros(self.action_dim)
                    )
                    observation, _reward, terminated, truncated, _info = env.step(
                        action
                    )
                    episode_errors.append(float(np.linalg.norm(observation[0:3])))
                    quaternion = observation[6:10]
                    tilt = np.degrees(
                        np.arccos(
                            np.clip(
                                1.0
                                - 2.0
                                * (quaternion[1] ** 2 + quaternion[2] ** 2),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    episode_tilts.append(float(tilt))
                    done = bool(terminated or truncated)

                episode_lengths.append(len(episode_errors))
                tail = max(1, int(len(episode_errors) * tail_fraction))
                errors.append(float(np.mean(episode_errors[-tail:])))
                if float(np.max(episode_tilts[-tail:])) > tilt_limit:
                    disqualifications += 1
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

        return EvaluationResult(
            mean_error=float(np.mean(errors)),
            disqualifications=disqualifications,
            mean_episode_length=float(np.mean(episode_lengths)),
        )

    def _validate_environment_contract(self, env: Any) -> None:
        actual_observation_shape = tuple(env.observation_space.shape)
        actual_action_shape = tuple(env.action_space.shape)
        expected_observation_shape = (self.observation_dim,)
        expected_action_shape = (self.action_dim,)
        if actual_observation_shape != expected_observation_shape:
            raise ValueError(
                "Configured observation contract does not match the environment: "
                f"mode={self.control_mode!r}, schema={self.observation_schema!r}, "
                f"configured={expected_observation_shape}, "
                f"environment={actual_observation_shape}"
            )
        if actual_action_shape != expected_action_shape:
            raise ValueError(
                "Configured action contract does not match the environment: "
                f"configured={expected_action_shape}, environment={actual_action_shape}"
            )

        actual_mode = getattr(env, "mode", None)
        if actual_mode != self.control_mode:
            raise ValueError(
                "Configured control mode does not match the environment: "
                f"configured={self.control_mode!r}, environment={actual_mode!r}"
            )
        actual_schema = getattr(env, "observation_schema", None)
        if actual_schema != self.observation_schema:
            raise ValueError(
                "Configured observation schema does not match the environment: "
                f"configured={self.observation_schema!r}, "
                f"environment={actual_schema!r}, mode={self.control_mode!r}"
            )
        actual_scale = tuple(float(value) for value in env.residual_scale)
        if actual_scale != self.residual_scale:
            raise ValueError(
                "Configured residual scale does not match the environment: "
                f"configured={self.residual_scale}, environment={actual_scale}"
            )

    def _save_model(self, model: Any, *, kind: str, timestep: int) -> Path:
        # ArtifactRun.save_model owns compliant naming and immediately updates
        # both the adjacent model sidecar and the run-level manifest.
        return self.artifact_run.save_model(
            model,
            kind=kind,
            timestep=timestep,
        )

    def _record_evaluation(
        self,
        *,
        timestep: int,
        policy_result: EvaluationResult | None,
        floor_result: EvaluationResult,
        saved: bool,
    ) -> None:
        values: dict[str, Any] = {
            "timestep": timestep,
            "control_mode": self.control_mode,
            "floor_mean_error": floor_result.mean_error,
            "floor_mean_episode_length": floor_result.mean_episode_length,
            "saved": saved,
        }
        if policy_result is not None:
            values.update(
                {
                    "policy_mean_error": policy_result.mean_error,
                    "policy_disqualifications": policy_result.disqualifications,
                    "policy_mean_episode_length": policy_result.mean_episode_length,
                    "improvement_percent": 100.0
                    * (floor_result.mean_error - policy_result.mean_error)
                    / floor_result.mean_error,
                }
            )
        self.artifact_run.write_metrics(
            kind=f"evaluation-step{timestep:07d}",
            payload=values,
        )

    def _make_eval_callback(self) -> Any:
        from stable_baselines3.common.callbacks import BaseCallback

        runner = self
        evaluation = _config_value(self.config, "evaluation")
        every_steps = int(evaluation["every_steps"])
        episode_count = int(evaluation["n_episodes"])

        class EvalCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.every_steps = every_steps
                self.last = 0
                self.best = float("inf")

            def _on_step(self) -> bool:
                if self.num_timesteps - self.last >= self.every_steps:
                    self.last = self.num_timesteps
                    current = runner.evaluate_policy(self.model)
                    floor = runner.evaluate_policy(None)
                    saved = False
                    if (
                        current.mean_error < self.best
                        and current.disqualifications == 0
                    ):
                        self.best = current.mean_error
                        runner._save_model(
                            self.model,
                            kind="best",
                            timestep=int(self.num_timesteps),
                        )
                        saved = True
                        tag = "WIN★(saved)"
                    else:
                        tag = (
                            "WIN"
                            if current.mean_error < floor.mean_error
                            else "..."
                        )

                    runner._record_evaluation(
                        timestep=int(self.num_timesteps),
                        policy_result=current,
                        floor_result=floor,
                        saved=saved,
                    )
                    print(
                        f"  step {self.num_timesteps:>7d} | "
                        f"{runner.control_mode}={current.mean_error:.4f} "
                        f"| floor={floor.mean_error:.4f} "
                        f"| surv={current.mean_episode_length:.0f} "
                        f"| disq={current.disqualifications}/{episode_count} | {tag} "
                        f"({100 * (floor.mean_error-current.mean_error) / floor.mean_error:+.1f}%)"
                    )
                return True

        return EvalCallback()

    def _build_model(self, vector_env: Any) -> Any:
        from stable_baselines3 import PPO

        ppo = _config_value(self.config, "training.ppo")
        policy_kwargs = ppo["policy_kwargs"]
        return PPO(
            str(ppo["policy"]),
            vector_env,
            verbose=int(ppo["verbose"]),
            n_steps=int(ppo["n_steps"]),
            batch_size=int(ppo["batch_size"]),
            gamma=float(ppo["gamma"]),
            gae_lambda=float(ppo["gae_lambda"]),
            n_epochs=int(ppo["n_epochs"]),
            learning_rate=float(ppo["learning_rate"]),
            ent_coef=float(ppo["ent_coef"]),
            vf_coef=float(ppo["vf_coef"]),
            max_grad_norm=float(ppo["max_grad_norm"]),
            normalize_advantage=bool(ppo["normalize_advantage"]),
            clip_range=float(ppo["clip_range"]),
            target_kl=float(ppo["target_kl"]),
            device=str(ppo["device"]),
            tensorboard_log=str(self.artifact_run.tensorboard_dir),
            policy_kwargs={
                "log_std_init": float(policy_kwargs["log_std_init"]),
                "net_arch": list(policy_kwargs["net_arch"]),
            },
            seed=self.seed,
        )

    def train(self) -> None:
        """Execute one run and persist best/final models through ArtifactRun."""
        # Fail before creating a run directory or importing the training stack
        # when the real XML resource is unavailable.
        self.config.require_runtime_resources()

        from stable_baselines3.common.vec_env import DummyVecEnv

        artifact_run = self.artifact_run
        vector_env: Any | None = None
        final_model_path: Path | None = None
        try:
            vector_env = DummyVecEnv([self.make_env])
            self._validate_environment_contract(vector_env.envs[0])
            model = self._build_model(vector_env)

            floor = self.evaluate_policy(None)
            self._record_evaluation(
                timestep=0,
                policy_result=None,
                floor_result=floor,
                saved=False,
            )
            evaluation = _config_value(self.config, "evaluation")
            print(
                f"[floor]  PID-only (n={int(evaluation['n_episodes'])} avg) = "
                f"{floor.mean_error:.4f} m, "
                f"surv={floor.mean_episode_length:.0f}"
            )

            total_timesteps = int(
                _config_value(self.config, "training.total_timesteps")
            )
            model.learn(
                total_timesteps=total_timesteps,
                callback=self._make_eval_callback(),
            )
            final_model_path = self._save_model(
                model,
                kind="final",
                timestep=int(model.num_timesteps),
            )
        except BaseException as exc:
            artifact_run.finalize(
                status="failed",
                extra={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        finally:
            if vector_env is not None:
                vector_env.close()

        artifact_run.finalize(
            status="completed",
            extra={
                "final_model": str(final_model_path) if final_model_path else None,
            },
        )
        print("done.")

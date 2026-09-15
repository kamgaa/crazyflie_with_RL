from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import sys

import numpy as np
import pytest

from crazyflie_rl.evaluation import EvaluationResult, PolicyEvaluator
from crazyflie_rl.training import PPOTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _observation(position_error: float, tilt_deg: float = 0.0) -> np.ndarray:
    observation = np.zeros(15, dtype=np.float32)
    observation[0] = position_error
    half_angle = np.deg2rad(tilt_deg) / 2.0
    observation[6:10] = [np.cos(half_angle), np.sin(half_angle), 0.0, 0.0]
    return observation


class ScriptedEnvironment:
    def __init__(self, episodes: list[list[np.ndarray]]) -> None:
        self.episodes = episodes
        self.reset_seeds: list[int] = []
        self.actions: list[np.ndarray] = []
        self.episode_index = -1
        self.step_index = 0
        self.closed = False

    def reset(self, *, seed: int):
        self.episode_index += 1
        self.step_index = 0
        self.reset_seeds.append(seed)
        return np.zeros(15, dtype=np.float32), {}

    def step(self, action: Any):
        self.actions.append(np.asarray(action))
        episode = self.episodes[self.episode_index]
        observation = episode[self.step_index]
        self.step_index += 1
        truncated = self.step_index == len(episode)
        return observation, 0.0, False, truncated, {}

    def close(self) -> None:
        self.closed = True


class SingleEnvironmentFactory:
    def __init__(self, environment: ScriptedEnvironment) -> None:
        self.environment = environment
        self.make_seeds: list[int | None] = []
        self.make_overrides: list[dict[str, Any]] = []

    def make(self, seed: int | None = None, **overrides: Any):
        self.make_seeds.append(seed)
        self.make_overrides.append(dict(overrides))
        return self.environment


def test_policy_evaluator_preserves_seed_tail_and_tilt_contract() -> None:
    # Episode 1 has length 3, so max(1, floor(30%)) selects only its last
    # sample. Episode 2 has length 10, so its last three scores are averaged.
    episodes = [
        [
            _observation(100.0),
            _observation(100.0),
            _observation(9.0, tilt_deg=31.0),
        ],
        [
            _observation(50.0, tilt_deg=80.0),
            *[_observation(50.0) for _ in range(6)],
            _observation(1.0),
            _observation(2.0),
            _observation(3.0),
        ],
    ]
    environment = ScriptedEnvironment(episodes)
    factory = SingleEnvironmentFactory(environment)
    config = SimpleNamespace(
        control_mode="e2e",
        evaluation=SimpleNamespace(
            episode_count=2,
            seed_start=100,
            deterministic=True,
            tail_fraction=0.3,
            tilt_limit_deg=30.0,
        ),
        action_shape=(4,),
    )

    result = PolicyEvaluator(config, factory).evaluate(None)

    assert result.score == pytest.approx((9.0 + 2.0) / 2.0)
    assert result.disqualifications == 1
    assert result.mean_episode_length == pytest.approx(6.5)
    assert result.episode_count == 2
    assert environment.reset_seeds == [100, 101]
    assert factory.make_seeds == [None]
    assert factory.make_overrides == [
        {"initial_state_randomization_enabled": False}
    ]
    assert all(np.array_equal(action, np.zeros(4)) for action in environment.actions)
    assert environment.closed is True


def test_policy_evaluator_preserves_residual_zero_tail_slice_bug() -> None:
    environment = ScriptedEnvironment(
        [[_observation(1.0), _observation(2.0), _observation(9.0)]]
    )
    config = SimpleNamespace(
        control_mode="residual",
        evaluation=SimpleNamespace(
            episode_count=1,
            seed_start=100,
            deterministic=True,
            tail_fraction=0.3,
            tilt_limit_deg=60.0,
        ),
        action_shape=(4,),
    )

    result = PolicyEvaluator(
        config,
        SingleEnvironmentFactory(environment),
    ).evaluate(None)

    # int(3 * 0.3) == 0 and legacy ``errors[-0:]`` means all three values.
    assert result.score == pytest.approx(4.0)


def _ppo_settings() -> SimpleNamespace:
    return SimpleNamespace(
        policy="MlpPolicy",
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        ent_coef=0.003,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.1,
        target_kl=0.03,
        log_std_init=-1.5,
        net_arch=(64, 64),
        device="cpu",
        verbose=0,
    )


class RecordingArtifacts:
    def __init__(self) -> None:
        self.tensorboard_dir = Path("run/tensorboard")
        self.saved: list[tuple[str, int | None, Any]] = []
        self.metrics: list[tuple[str, dict[str, Any]]] = []
        self.finalized: list[tuple[str, dict[str, Any]]] = []
        self.policy_initializations: list[tuple[Path, dict[str, Any]]] = []

    def save_model(
        self,
        model: Any,
        kind: str,
        *,
        timestep: int | None = None,
        metadata: Any = None,
    ) -> Path:
        del metadata
        self.saved.append((kind, timestep, model))
        return Path(f"run/{kind}-{len(self.saved)}.zip")

    def write_metrics(self, kind: str, payload: Any) -> Path:
        self.metrics.append((kind, dict(payload)))
        return Path(f"run/{kind}.json")

    def record_policy_initialization(
        self, parent_model: Path, provenance: Any
    ) -> None:
        self.policy_initializations.append((parent_model, dict(provenance)))

    def finalize(self, status: str = "completed", **extra: Any) -> None:
        self.finalized.append((status, extra))


def test_ppo_builder_forwards_every_configured_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakePPO:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    stable_baselines3 = ModuleType("stable_baselines3")
    stable_baselines3.PPO = FakePPO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines3)

    config = SimpleNamespace(
        training=SimpleNamespace(seed=None, ppo=_ppo_settings()),
    )
    artifacts = RecordingArtifacts()
    vector_environment = object()

    model = PPOTrainer(config, artifact_manager=artifacts)._build_model(
        vector_environment
    )

    assert isinstance(model, FakePPO)
    assert captured == {
        "policy": "MlpPolicy",
        "env": vector_environment,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "ent_coef": 0.003,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
        "policy_kwargs": {"log_std_init": -1.5, "net_arch": [64, 64]},
        "tensorboard_log": str(artifacts.tensorboard_dir),
        "seed": None,
        "device": "cpu",
        "verbose": 0,
    }


def _install_fake_callback_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBaseCallback:
        def __init__(self, verbose: int = 0) -> None:
            self.verbose = verbose
            self.num_timesteps = 0
            self.model: Any = None

    stable_baselines3 = ModuleType("stable_baselines3")
    common = ModuleType("stable_baselines3.common")
    callbacks = ModuleType("stable_baselines3.common.callbacks")
    callbacks.BaseCallback = FakeBaseCallback  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.callbacks", callbacks)


class SequencedEvaluator:
    def __init__(self, results: list[EvaluationResult]) -> None:
        self.results = list(results)
        self.models: list[Any | None] = []

    def evaluate(self, model: Any | None) -> EvaluationResult:
        self.models.append(model)
        return self.results.pop(0)


def _evaluation(score: float, disq: int = 0) -> EvaluationResult:
    return EvaluationResult(score, disq, 800.0, 30)


def test_callback_uses_strict_score_and_zero_disqualification_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_callback_module(monkeypatch)
    # Each callback evaluation consumes policy then floor.
    evaluator = SequencedEvaluator(
        [
            _evaluation(1.0, 1), _evaluation(2.0),
            _evaluation(1.0, 0), _evaluation(2.0),
            _evaluation(1.0, 0), _evaluation(2.0),
            _evaluation(0.9, 0), _evaluation(2.0),
        ]
    )
    artifacts = RecordingArtifacts()
    config = SimpleNamespace(
        control_mode="e2e",
        evaluation=SimpleNamespace(evaluation_interval=10),
    )
    trainer = PPOTrainer(
        config,
        artifact_manager=artifacts,
        evaluator=evaluator,
    )
    callback = trainer._make_evaluation_callback(artifacts)
    assert callback is not None
    model = object()
    callback.model = model

    callback.num_timesteps = 9
    assert callback._on_step() is True
    for timestep in (10, 20, 30, 40):
        callback.num_timesteps = timestep
        assert callback._on_step() is True

    # Disqualified 1.0 is rejected; the first clean 1.0 is saved; its tie is
    # rejected; the strict 0.9 improvement is saved.
    assert [(kind, step) for kind, step, _ in artifacts.saved] == [
        ("best", 20),
        ("best", 40),
    ]
    assert callback.best_score == pytest.approx(0.9)
    assert len(artifacts.metrics) == 4


def test_opt_in_recovery_callback_uses_distinct_best_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_callback_module(monkeypatch)
    artifacts = RecordingArtifacts()
    config = SimpleNamespace(
        control_mode="e2e",
        evaluation=SimpleNamespace(
            evaluation_interval=0,
            recovery=SimpleNamespace(
                enabled=True,
                evaluation_interval=100,
                duration_s=8.0,
            ),
        ),
    )
    trainer = PPOTrainer(config, artifact_manager=artifacts)
    summary = {
        "nominal_hover_success": True,
        "overall_success_rate": 0.5,
        "mean_recovery_time_s": 2.0,
        "mean_maximum_position_error_m": 0.1,
    }
    monkeypatch.setattr(
        trainer,
        "_evaluate_recovery",
        lambda model: ([{"case": "nominal"}], dict(summary)),
    )
    callback = trainer._make_evaluation_callback(artifacts)
    assert callback is not None
    callback.model = object()
    callback.num_timesteps = 100

    assert callback._on_step() is True
    assert [(kind, step) for kind, step, _ in artifacts.saved] == [
        ("best-recovery", 100)
    ]
    assert callback.best_model_path is None
    assert callback.best_recovery_model_path is not None
    assert artifacts.metrics[-1][1]["training_randomization_disabled"] is True


def test_payload_callback_applies_nominal_gate_and_ordered_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_callback_module(monkeypatch)
    artifacts = RecordingArtifacts()
    config = SimpleNamespace(
        control_mode="e2e",
        evaluation=SimpleNamespace(
            evaluation_interval=0,
            payload=SimpleNamespace(
                enabled=True,
                evaluation_interval=100,
                duration_s=8.0,
                seed=1000,
            ),
        ),
    )
    trainer = PPOTrainer(config, artifact_manager=artifacts)
    summaries = iter(
        (
            {
                "nominal_success": False,
                "payload_success_count": 10,
                "payload_case_count": 10,
                "payload_completed_full_duration_count": 10,
                "payload_completed_worst_tail_position_rmse_m": 0.001,
            },
            {
                "nominal_success": True,
                "payload_success_count": 6,
                "payload_case_count": 10,
                "payload_completed_full_duration_count": 8,
                "payload_completed_worst_tail_position_rmse_m": 0.03,
            },
            {
                "nominal_success": True,
                "payload_success_count": 6,
                "payload_case_count": 10,
                "payload_completed_full_duration_count": 9,
                "payload_completed_worst_tail_position_rmse_m": 0.08,
            },
        )
    )
    monkeypatch.setattr(
        trainer,
        "_evaluate_payload",
        lambda model: ([{"model": model}], next(summaries)),
    )
    callback = trainer._make_evaluation_callback(artifacts)
    assert callback is not None
    callback.model = object()

    for timestep in (100, 200, 300):
        callback.num_timesteps = timestep
        assert callback._on_step() is True

    # The nominal-failing candidate is never saved. The third candidate beats
    # the second on full-duration count despite its worse tail RMSE.
    assert [(kind, step) for kind, step, _ in artifacts.saved] == [
        ("best-payload", 200),
        ("best-payload", 300),
    ]
    assert callback.best_model_path is None
    assert callback.best_payload_model_path is not None
    first_metrics = artifacts.metrics[0][1]
    assert first_metrics["best_payload_exists"] is False
    assert first_metrics["no_best_payload_reason"] == "nominal_success_gate_failed"
    assert first_metrics["candidate_rank"] is None
    assert artifacts.metrics[-1][1]["candidate_rank"] == [-6.0, -9.0, 0.08]


def test_recovery_evaluation_factory_disables_training_randomization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crazyflie_rl import recovery
    from crazyflie_rl.config import load_config

    class RecoveryEnvironment:
        closed = False

        def close(self) -> None:
            self.closed = True

    class RecoveryFactory:
        def __init__(self) -> None:
            self.overrides: list[dict[str, Any]] = []
            self.environment = RecoveryEnvironment()

        def make(self, seed: int | None = None, **overrides: Any) -> Any:
            self.overrides.append({"seed": seed, **overrides})
            return self.environment

    def fake_evaluate(
        environment: Any,
        policy: Any,
        case: Any,
        *,
        seed: int,
        duration_s: float,
    ) -> dict[str, Any]:
        del environment, policy, seed, duration_s
        return {
            "category": case.category,
            "name": case.name,
            "tilt_deg": case.tilt_deg,
            "success": True,
            "recovery_time_s": 1.0,
            "maximum_position_error_m": 0.01,
        }

    monkeypatch.setattr(recovery, "evaluate_recovery_case", fake_evaluate)
    config = load_config(
        PROJECT_ROOT / "configs" / "e2e_train_legacy_initial_perturb_v2.yaml"
    )
    factory = RecoveryFactory()
    results, summary = PPOTrainer(
        config,
        environment_factory=factory,
    )._evaluate_recovery(object())

    assert len(results) == 19
    assert summary["nominal_hover_success"] is True
    assert factory.overrides == [
        {
            "seed": config.evaluation.seed_start,
            "initial_state_randomization_enabled": False,
            "payload_curriculum_enabled": False,
            "com_bias_randomize": False,
            "com_bias_mass": 0.0,
            "com_bias_offset": (0.0, 0.0),
        }
    ]
    assert factory.environment.closed is True


class FakeRuntimeConfig(SimpleNamespace):
    def require_runtime_resources(self) -> None:
        self.resources_checked = True


class FakeTrainingEnvironment:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeTrainingFactory:
    def __init__(self) -> None:
        self.environments: list[FakeTrainingEnvironment] = []
        self.overrides: list[dict[str, Any]] = []

    def make(self, seed: int | None = None, **overrides: Any):
        del seed
        self.overrides.append(dict(overrides))
        environment = FakeTrainingEnvironment()
        self.environments.append(environment)
        return environment


def _install_fake_training_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> type:
    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.num_timesteps = 0
            self.learn_call: dict[str, Any] | None = None

        def learn(self, **kwargs: Any):
            self.learn_call = kwargs
            self.num_timesteps = kwargs["total_timesteps"]
            return self

    class FakeDummyVecEnv:
        instances: list[Any] = []

        def __init__(self, factories: list[Any]) -> None:
            self.environment = factories[0]()
            self.closed = False
            self.__class__.instances.append(self)

        def close(self) -> None:
            self.closed = True
            self.environment.close()

    stable_baselines3 = ModuleType("stable_baselines3")
    stable_baselines3.PPO = FakeModel  # type: ignore[attr-defined]
    common = ModuleType("stable_baselines3.common")
    vec_env = ModuleType("stable_baselines3.common.vec_env")
    vec_env.DummyVecEnv = FakeDummyVecEnv  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.vec_env", vec_env)
    return FakeDummyVecEnv


def test_zero_interval_smoke_saves_trained_policy_as_best_and_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_vec_env = _install_fake_training_stack(monkeypatch)
    artifacts = RecordingArtifacts()
    factory = FakeTrainingFactory()
    floor = EvaluationResult(1.2, 0, 800.0, 5)
    trained = EvaluationResult(0.2, 0, 800.0, 5)
    evaluator = SequencedEvaluator([floor, trained])
    config = FakeRuntimeConfig(
        resources_checked=False,
        control_mode="residual",
        training=SimpleNamespace(
            seed=None,
            total_timesteps=30_000,
            ppo=_ppo_settings(),
        ),
        evaluation=SimpleNamespace(evaluation_interval=0),
    )

    outcome = PPOTrainer(
        config,
        artifact_manager=artifacts,
        environment_factory=factory,
        evaluator=evaluator,
    ).train()

    assert config.resources_checked is True
    assert [(kind, step) for kind, step, _ in artifacts.saved] == [
        ("best", 30_000),
        ("final", 30_000),
    ]
    assert outcome.best_evaluation == trained
    assert outcome.final_evaluation == trained
    assert artifacts.finalized[-1][0] == "completed"
    assert fake_vec_env.instances[-1].closed is True


def test_opt_in_training_environment_tracks_absolute_curriculum_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_training_stack(monkeypatch)
    artifacts = RecordingArtifacts()
    factory = FakeTrainingFactory()
    evaluation = EvaluationResult(0.2, 0, 800.0, 1)
    config = FakeRuntimeConfig(
        resources_checked=False,
        control_mode="e2e",
        environment=SimpleNamespace(
            initial_state_randomization=SimpleNamespace(enabled=True)
        ),
        training=SimpleNamespace(
            seed=42,
            total_timesteps=2,
            ppo=_ppo_settings(),
        ),
        evaluation=SimpleNamespace(evaluation_interval=0),
    )
    PPOTrainer(
        config,
        artifact_manager=artifacts,
        environment_factory=factory,
        evaluator=SequencedEvaluator([evaluation, evaluation]),
    ).train()
    assert factory.overrides == [{"track_curriculum_steps": True}]


@pytest.mark.parametrize(
    ("module_name", "expected_profile", "expected_mode"),
    [
        ("train_ppo", "residual_train.yaml", "residual"),
        ("train_ppo_02", "e2e_train.yaml", "e2e"),
    ],
)
def test_training_cli_dry_run_uses_profile_and_override_without_artifacts(
    module_name: str,
    expected_profile: str,
    expected_mode: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = __import__(module_name)
    parser_default = module.build_parser().parse_args([]).config
    assert parser_default.name == expected_profile

    runs_root = PROJECT_ROOT / "artifacts" / "runs"
    before = set(runs_root.iterdir()) if runs_root.is_dir() else set()
    assert module.main(["--dry-run", "--total-timesteps", "7"]) == 0
    after = set(runs_root.iterdir()) if runs_root.is_dir() else set()

    payload = json.loads(capsys.readouterr().out)
    assert payload["environment"]["control_mode"] == expected_mode
    assert payload["training"]["total_timesteps"] == 7
    assert after == before

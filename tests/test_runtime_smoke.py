"""Real MuJoCo/SB3 smoke coverage, skipped only when runtime assets are absent."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from crazyflie_rl.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _require_runtime(config) -> None:
    missing = [
        module
        for module in ("gymnasium", "mujoco", "stable_baselines3", "tensorboard", "torch")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        pytest.skip("runtime packages unavailable: " + ", ".join(missing))
    if not Path(config.paths.mujoco_xml).is_file():
        pytest.skip(f"server MuJoCo XML unavailable: {config.paths.mujoco_xml}")


def test_short_ppo_save_reload_and_headless_evaluation(tmp_path: Path) -> None:
    """Train 64 steps, save best/final, reload final, and evaluate headlessly."""

    config = load_config(ROOT / "configs" / "e2e_train.yaml")
    _require_runtime(config)

    smoke_environment = replace(
        config.environment,
        episode_sec=0.1,
        position_perturbation=0.0,
        attitude_perturbation_deg=0.0,
    )
    smoke_ppo = replace(
        config.training.ppo,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        net_arch=(16, 16),
        verbose=0,
    )
    smoke_config = replace(
        config,
        paths=replace(config.paths, artifact_root=tmp_path / "artifacts"),
        environment=smoke_environment,
        training=replace(config.training, seed=7, total_timesteps=64, ppo=smoke_ppo),
        evaluation=replace(
            config.evaluation,
            episode_count=1,
            seed_start=70,
            tilt_limit_deg=180.0,
            evaluation_interval=32,
        ),
        experiment=replace(
            config.experiment,
            condition="runtime-smoke",
            description="64-step real MuJoCo/SB3 smoke test",
        ),
    )

    from stable_baselines3 import PPO

    from crazyflie_rl.evaluation import PolicyEvaluator
    from crazyflie_rl.training import PPOTrainer

    try:
        outcome = PPOTrainer(smoke_config).train()
    except (FileNotFoundError, ValueError) as exc:
        message = str(exc).lower()
        if "mesh" in message or "texture" in message or "error opening file" in message:
            pytest.skip(f"MuJoCo XML dependency unavailable: {exc}")
        raise

    assert outcome.best_model_path is not None
    assert outcome.best_model_path.is_file()
    assert outcome.final_model_path.is_file()
    assert not outcome.best_model_path.name.endswith(".zip.zip")
    assert not outcome.final_model_path.name.endswith(".zip.zip")

    reloaded = PPO.load(str(outcome.final_model_path), device="cpu")
    evaluation = PolicyEvaluator(smoke_config).evaluate(reloaded)
    assert evaluation.episode_count == 1
    assert evaluation.mean_episode_length > 0

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = list((tmp_path / "artifacts").rglob("events.out.tfevents.*"))
    assert event_files
    events = EventAccumulator(str(event_files[0])).Reload()
    for tag in (
        "reward_terms/position", "reward_terms/total",
        "reward_raw/position_sq", "reward_fraction/position",
    ):
        assert [event.step for event in events.Scalars(tag)] == [32, 64]

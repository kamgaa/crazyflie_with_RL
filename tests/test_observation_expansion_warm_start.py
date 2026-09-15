from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.factories import EnvironmentFactory
from crazyflie_rl.warm_start import (
    copy_policy_parameters_with_input_expansion,
    validate_loaded_observation_schema,
    validate_observation_expansion_donor,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "e2e_train_payload_dr_history_integral_v1.yaml"
DONOR = (
    ROOT
    / "artifacts"
    / "runs"
    / "ppo_e2e_hover_payload-dr-pos6_seed42_20260912-132845"
    / "models"
    / "ppo_e2e_hover_payload-dr-pos6_seed42_best-payload_20260912-132845-04.zip"
)
DONOR_SHA256 = "ab9a01e8ae91471d3c3061aed96da8e67ae8a6ed67654cf4eabfe93e269af85c"


def _fresh_target(config, vector):
    from stable_baselines3 import PPO

    ppo = config.training.ppo
    return PPO(
        policy=ppo.policy,
        env=vector,
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
        seed=config.training.seed,
        device="cpu",
        verbose=0,
    )


def test_selected_donor_expansion_preserves_network_and_fresh_training_state(
    tmp_path: Path,
) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    assert DONOR.is_file()
    config = load_config(PROFILE)
    compatibility = validate_observation_expansion_donor(DONOR, config)
    assert compatibility.saved_timestep == 1_000_000
    assert (
        compatibility.base_compatibility.training_provenance["model_sha256"]
        == DONOR_SHA256
    )
    assert all(compatibility.exact_contract_checks.values())

    factory = EnvironmentFactory(config)
    vector = DummyVecEnv(
        [
            lambda: factory.make(
                seed=42,
                payload_curriculum_enabled=False,
                initial_state_randomization_enabled=False,
                pos_perturb=0.0,
                att_perturb_deg=0.0,
            )
        ]
    )
    try:
        target = _fresh_target(config, vector)
        target.observation_schema = dict(config.observation_schema)
        donor = PPO.load(str(DONOR), device="cpu")
        original_optimizer = target.policy.optimizer
        original_rollout_buffer = target.rollout_buffer
        transfer = copy_policy_parameters_with_input_expansion(target, donor)
        assert transfer["function_preservation"]["passed"] is True
        assert transfer["function_preservation"]["action_mean_max_abs_error"] == 0.0
        assert transfer["function_preservation"]["value_max_abs_error"] == 0.0
        assert target.policy.optimizer is original_optimizer
        assert target.rollout_buffer is original_rollout_buffer
        assert not target.policy.optimizer.state
        assert target.num_timesteps == 0
        parameters = dict(target.policy.named_parameters())
        for key in transfer["expanded_tensor_names"]:
            assert parameters[key].requires_grad
            assert np.count_nonzero(parameters[key].detach().cpu().numpy()[:, 15:]) == 0

        observation = vector.reset()
        auxiliary_before = vector.envs[0]._auxiliary_observation.state_snapshot()
        action_before = target.predict(observation, deterministic=True)[0]
        target.predict(observation, deterministic=True)
        auxiliary_after = vector.envs[0]._auxiliary_observation.state_snapshot()
        assert auxiliary_before["transition_count"] == auxiliary_after[
            "transition_count"
        ]
        np.testing.assert_array_equal(
            auxiliary_before["integral"], auxiliary_after["integral"]
        )
        archive = tmp_path / "expanded_model"
        target.save(str(archive))
        loaded = PPO.load(str(archive) + ".zip", device="cpu")
        validate_loaded_observation_schema(
            loaded, config, model_path=Path(str(archive) + ".zip")
        )
        action_after = loaded.predict(observation, deterministic=True)[0]
        np.testing.assert_array_equal(action_before, action_after)

        loaded.observation_schema = {
            **dict(config.observation_schema),
            "state_reader": "wrong_reader",
        }
        with pytest.raises(ValueError, match="observation schema"):
            validate_loaded_observation_schema(loaded, config)
    finally:
        vector.close()


def test_manifest_records_schema_and_explicit_transfer(tmp_path: Path) -> None:
    from crazyflie_rl.artifacts import ArtifactManager

    config = load_config(PROFILE)
    config = replace(
        config,
        paths=replace(config.paths, artifact_root=tmp_path / "artifacts"),
    )
    manager = ArtifactManager.create(
        config,
        now=datetime(2026, 9, 15, 1, 2, 3, tzinfo=ZoneInfo("Asia/Seoul")),
        command=["python", "train_ppo_02.py", "--config", str(PROFILE)],
    )
    assert manager.manifest["observation_shape"] == [418]
    assert manager.manifest["observation_schema"] == config.observation_schema
    assert manager.manifest["state_reader"] == "current_freejoint_body_rate_v1"
    assert manager.manifest["trace_schema_version"] == "policy_transition_trace_v2"
    manager.record_policy_initialization(
        DONOR,
        {
            "strategy": "explicit_mlp_input_expansion_v1",
            "training_provenance": {"model_sha256": DONOR_SHA256},
            "training_physics_model_version": "rigid_point_payload_v2",
            "cross_physics_evaluation": False,
            "parameter_transfer": {
                "optimizer_state_copied": False,
                "rollout_buffer_copied": False,
                "initial_timestep": 0,
                "zero_initialized_input_columns": "[:, 15:418]",
            },
        },
    )
    initialization = manager.manifest["policy_initialization"]
    assert initialization["strategy"] == "explicit_mlp_input_expansion_v1"
    assert initialization["parent_model_sha256"] == DONOR_SHA256
    assert initialization["optimizer_state_copied"] is False
    assert initialization["initial_timestep"] == 0

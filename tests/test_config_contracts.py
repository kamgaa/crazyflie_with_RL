from __future__ import annotations

import ast
from pathlib import Path

import pytest

from crazyflie_rl.config import MissingResourceError, load_config
from crazyflie_rl.contracts import DEFAULT_RESIDUAL_SCALE


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_environment_source_constants_match_shared_contract() -> None:
    tree = ast.parse(
        (PROJECT_ROOT / "crazyflie_residual_env.py").read_text(encoding="utf-8")
    )
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"DEFAULT_RESIDUAL_SCALE", "OBSERVATION_SCHEMA", "OBSERVATION_DIM"}
    }

    assert assignments["DEFAULT_RESIDUAL_SCALE"] == DEFAULT_RESIDUAL_SCALE
    assert assignments["OBSERVATION_SCHEMA"] == {
        "residual": "residual_v1",
        "e2e": "e2e_v1",
    }
    assert assignments["OBSERVATION_DIM"] == {"residual": 13, "e2e": 15}


@pytest.mark.parametrize(
    ("profile", "mode", "schema", "observation_dim"),
    [
        ("residual_train.yaml", "residual", "residual_v1", 13),
        ("e2e_train.yaml", "e2e", "e2e_v1", 15),
    ],
)
def test_profile_control_contract(
    profile: str,
    mode: str,
    schema: str,
    observation_dim: int,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / profile)

    assert config.control_mode == mode
    assert config.observation_schema == schema
    assert config.observation_dim == observation_dim
    assert config.action_dim == 4
    assert config.residual_scale == DEFAULT_RESIDUAL_SCALE


def test_profiles_share_preserved_training_and_evaluation_settings() -> None:
    residual = load_config(PROJECT_ROOT / "configs" / "residual_train.yaml")
    e2e = load_config(PROJECT_ROOT / "configs" / "e2e_train.yaml")

    assert residual.data["training"] == e2e.data["training"]
    assert residual.data["evaluation"] == e2e.data["evaluation"]
    assert e2e.seed is None

    ppo = e2e.data["training"]["ppo"]
    assert ppo == {
        "policy": "MlpPolicy",
        "verbose": 0,
        "device": "cpu",
        "n_steps": 2048,
        "batch_size": 256,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_epochs": 10,
        "ent_coef": 0.003,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "clip_range": 0.1,
        "target_kl": 0.03,
        "policy_kwargs": {"log_std_init": -1.5, "net_arch": [64, 64]},
    }
    assert e2e.data["training"]["total_timesteps"] == 1_000_000
    assert e2e.data["evaluation"] == {
        "n_episodes": 30,
        "seed_start": 1000,
        "tail_fraction": 0.3,
        "tilt_limit_deg": 30.0,
        "every_steps": 20_000,
        "deterministic": True,
    }


def test_relative_paths_are_resolved_from_project_root() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "e2e_train.yaml")

    assert config.resolve_path("resource_root") == PROJECT_ROOT / "resources"
    assert config.resolve_path("mujoco_xml") == (
        PROJECT_ROOT / "resources" / "mujoco" / "cf21B_500.xml"
    )
    assert config.resolve_path("pretrained_model_root") == (
        PROJECT_ROOT / "resources" / "pretrained_models"
    )
    assert config.resolve_path("artifact_root") == PROJECT_ROOT / "artifacts"


def test_missing_xml_is_reported_without_fabricating_a_path() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "e2e_train.yaml")
    expected_xml = config.resolve_path("mujoco_xml")
    if expected_xml.is_file():
        pytest.skip("real MuJoCo XML has been supplied")

    with pytest.raises(MissingResourceError) as exc_info:
        config.require_runtime_resources()

    message = str(exc_info.value)
    assert str(expected_xml) in message
    assert "mesh/texture" in message
    assert "placeholder" in message

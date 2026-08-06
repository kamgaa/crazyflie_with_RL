from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.model_compat import (
    ModelCompatibilityError,
    manifest_path_for,
    read_model_manifest,
    validate_model_compatibility,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _owner(observation_dim: int, action_dim: int, mode: str, schema: str):
    return SimpleNamespace(
        observation_space=SimpleNamespace(shape=(observation_dim,)),
        action_space=SimpleNamespace(shape=(action_dim,)),
        mode=mode,
        observation_schema=schema,
    )


def test_dimension_and_mode_mismatch_error_is_actionable(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "residual_train.yaml")
    model_path = tmp_path / "legacy-e2e.zip"
    model_path.write_bytes(b"not deserialized by this unit test")
    manifest = {
        "control_mode": "e2e",
        "observation_schema": "e2e_v1",
        "observation_dim": 15,
        "action_dim": 4,
    }
    manifest_path_for(model_path).write_text(json.dumps(manifest), encoding="utf-8")
    model = _owner(15, 4, "e2e", "e2e_v1")
    env = _owner(13, 4, "residual", "residual_v1")

    with pytest.raises(ModelCompatibilityError) as exc_info:
        validate_model_compatibility(model, env, config, model_path)

    message = str(exc_info.value)
    assert "model required observation dimension: 15" in message
    assert "environment observation dimension: 13" in message
    assert "selected control mode: residual" in message
    assert str(model_path) in message
    assert "required config profile: residual_train" in message
    assert "not converted or forced" in message


def test_matching_e2e_contract_passes(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "e2e_train.yaml")
    model_path = tmp_path / "e2e.zip"
    model_path.write_bytes(b"not deserialized by this unit test")
    manifest = {
        "control_mode": "e2e",
        "observation_schema": "e2e_v1",
        "observation_dim": 15,
        "action_dim": 4,
    }
    manifest_path_for(model_path).write_text(json.dumps(manifest), encoding="utf-8")
    model = _owner(15, 4, "e2e", "e2e_v1")
    env = _owner(15, 4, "e2e", "e2e_v1")

    validate_model_compatibility(model, env, config, model_path)


def test_manifest_is_mandatory(tmp_path: Path) -> None:
    model_path = tmp_path / "orphan.zip"
    model_path.write_bytes(b"model")

    with pytest.raises(ModelCompatibilityError, match="manifest is required"):
        read_model_manifest(model_path)


@pytest.mark.parametrize("name", ["ppo_best", "ppo_residual_cf"])
def test_preserved_legacy_models_are_explicitly_e2e_only(name: str) -> None:
    model_path = PROJECT_ROOT / "model" / f"{name}.zip"
    manifest = read_model_manifest(model_path)

    with zipfile.ZipFile(model_path) as archive:
        sb3_data = json.loads(archive.read("data").decode("utf-8"))

    assert manifest["legacy"] is True
    assert manifest["control_mode"] == "e2e"
    assert manifest["observation_schema"] == "e2e_v1"
    assert manifest["observation_dim"] == 15
    assert manifest["action_dim"] == 4
    assert manifest["compatibility"] == "e2e_only"
    assert sb3_data["observation_space"]["_shape"] == [15]
    assert sb3_data["action_space"]["_shape"] == [4]

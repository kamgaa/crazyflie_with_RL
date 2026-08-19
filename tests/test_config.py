from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from crazyflie_rl.config import ConfigError, _foreign_posix_absolute, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SERVER_XML = (ROOT / "resources" / "mujoco" / "cf21B_500.xml").resolve()


def _mutated_base(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))
    mutate(data)
    target = tmp_path / "profile.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def test_base_and_profiles_merge_without_changing_master_contracts() -> None:
    e2e = load_config(CONFIGS / "e2e_train.yaml")
    residual = load_config(CONFIGS / "residual_train.yaml")

    assert e2e.control_mode == "e2e"
    assert residual.control_mode == "residual"
    assert e2e.observation_shape == residual.observation_shape == (15,)
    assert e2e.action_shape == residual.action_shape == (4,)
    assert e2e.paths.mujoco_xml.as_posix() == SERVER_XML
    assert residual.environment.residual_scale == (0.022, 0.022, 0.0001, 0.3)

    assert e2e.training.total_timesteps == 1_000_000
    assert e2e.training.ppo.ent_coef == 0.003
    assert e2e.training.ppo.clip_range == 0.1
    assert e2e.training.ppo.target_kl == 0.03
    assert e2e.evaluation.episode_count == 30
    assert e2e.evaluation.seed_start == 1000

    assert residual.training.total_timesteps == 30_000
    assert residual.training.ppo.ent_coef == 0.0
    assert residual.training.ppo.clip_range == 0.2
    assert residual.training.ppo.target_kl is None
    assert residual.environment.payload.mass == 0.010
    assert residual.evaluation.episode_count == 5
    assert residual.evaluation.seed_start == 100


@pytest.mark.parametrize(
    "profile",
    sorted(path.name for path in CONFIGS.glob("*.yaml")),
)
def test_every_shipped_profile_loads(profile: str) -> None:
    config = load_config(CONFIGS / profile)
    assert config.source_path.name == profile
    assert config.observation_shape == (15,)
    assert config.action_shape == (4,)


def test_protected_xml_default_is_literal_server_path() -> None:
    source = (CONFIGS / "base.yaml").read_text(encoding="utf-8")
    assert f"mujoco_xml: {SERVER_XML}" in source
    assert "resources/mujoco" not in source


def test_posix_server_path_is_preserved_when_native_host_calls_it_relative() -> None:
    preserved = _foreign_posix_absolute(SERVER_XML, native_is_absolute=False)
    assert preserved is not None
    assert preserved.as_posix() == SERVER_XML
    assert _foreign_posix_absolute(SERVER_XML, native_is_absolute=True) is None


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    profile = _mutated_base(
        tmp_path,
        lambda data: data["environment"].__setitem__("policy_hzz", 100),
    )
    with pytest.raises(ConfigError, match="policy_hzz"):
        load_config(profile)


def test_missing_required_key_is_rejected(tmp_path: Path) -> None:
    profile = _mutated_base(
        tmp_path,
        lambda data: data["training"]["ppo"].pop("batch_size"),
    )
    with pytest.raises(ConfigError, match="batch_size"):
        load_config(profile)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data["environment"].__setitem__("control_mode", "absolute"), "control_mode"),
        (lambda data: data["environment"].__setitem__("residual_scale", [1, 2, 3]), "exactly 4"),
        (lambda data: data["environment"].__setitem__("policy_hz", -1), "positive"),
        (lambda data: data["environment"].__setitem__("episode_sec", 0), "positive"),
    ],
)
def test_invalid_environment_values_fail_early(tmp_path: Path, mutate, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(_mutated_base(tmp_path, mutate))


def test_loaded_config_has_no_shared_mutable_sequences() -> None:
    first = load_config(CONFIGS / "e2e_train.yaml")
    second = load_config(CONFIGS / "e2e_train.yaml")

    assert isinstance(first.environment.residual_scale, tuple)
    assert isinstance(first.vehicle.inertia_diagonal, tuple)
    assert first.environment.residual_scale is not second.environment.residual_scale
    with pytest.raises(FrozenInstanceError):
        first.environment.policy_hz = 50  # type: ignore[misc]

    exported = first.resolved_dict()
    exported["environment"]["residual_scale"][0] = 999
    assert first.environment.residual_scale[0] == 0.022


def test_loading_config_has_no_filesystem_or_runtime_side_effects(tmp_path: Path) -> None:
    before = set(tmp_path.rglob("*"))
    config = load_config(CONFIGS / "e2e_train.yaml")
    assert config.profile_name == "e2e_train"
    assert set(tmp_path.rglob("*")) == before

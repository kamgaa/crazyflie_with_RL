from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from crazyflie_rl.config import ConfigError, _foreign_posix_absolute, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PROJECT_XML = (ROOT / "resources" / "mujoco" / "cf21B_500.xml").resolve()


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
    assert e2e.paths.mujoco_xml == PROJECT_XML
    assert residual.environment.residual_scale == (0.022, 0.022, 0.0001, 0.3)
    assert e2e.actuator.enabled
    assert e2e.actuator.model == "cf21b_first_order"

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
    expected_observation_shape = (
        (418,)
        if profile == "e2e_train_payload_dr_history_integral_v1.yaml"
        or profile.startswith("view_payload_history_integral_v1_")
        else (15,)
    )
    assert config.observation_shape == expected_observation_shape
    assert config.action_shape == (4,)
    assert config.actuator.enabled
    assert config.actuator.model == "cf21b_first_order"
    if profile == "cf21b_actuator_torque_poly_eval.yaml":
        assert config.actuator.reaction_torque.model == "paper_polynomial"
    else:
        assert config.actuator.reaction_torque.model == "legacy_ratio"


def test_default_xml_is_repository_relative_and_resolves_from_project_root() -> None:
    source = (CONFIGS / "base.yaml").read_text(encoding="utf-8")
    assert "mujoco_xml: resources/mujoco/cf21B_500.xml" in source
    assert load_config(CONFIGS / "base.yaml").paths.mujoco_xml == PROJECT_XML


def test_foreign_posix_absolute_path_is_preserved_when_inspected_on_windows() -> None:
    server_xml = "/srv/crazyflie/cf21B_500.xml"
    preserved = _foreign_posix_absolute(server_xml, native_is_absolute=False)
    assert preserved is not None
    assert preserved.as_posix() == server_xml
    assert _foreign_posix_absolute(server_xml, native_is_absolute=True) is None


def test_required_actuator_config_uses_first_order_model_and_legacy_yaw_torque() -> (
    None
):
    config = load_config(CONFIGS / "base.yaml")
    actuator = config.actuator

    assert actuator.enabled
    assert actuator.model == "cf21b_first_order"
    assert actuator.time_constant_s == pytest.approx(0.050)
    assert actuator.steady_state_gain_rad_s == pytest.approx(2900.0)
    assert actuator.thrust_polynomial_coefficients == (-0.23, 0.562, -0.043)
    assert actuator.parameter_source == "paper_candidate"
    assert actuator.verification_status == "unverified"
    assert actuator.reaction_torque.model == "legacy_ratio"
    assert actuator.reaction_torque.legacy_ratio_m == pytest.approx(0.00594)
    assert actuator.reset_rpm_mode == "auto"
    assert not actuator.randomization.enabled
    assert actuator.randomization.time_constant_s.min == pytest.approx(0.040)
    assert actuator.randomization.time_constant_s.max == pytest.approx(0.060)


def test_cf21b_actuator_profiles_keep_legacy_yaw_by_default_and_opt_in_to_polynomial() -> (
    None
):
    first_order = load_config(CONFIGS / "cf21b_actuator_eval.yaml")
    torque_poly = load_config(CONFIGS / "cf21b_actuator_torque_poly_eval.yaml")

    assert first_order.actuator.enabled
    assert first_order.actuator.model == "cf21b_first_order"
    assert first_order.actuator.reaction_torque.model == "legacy_ratio"
    assert torque_poly.actuator.enabled
    assert torque_poly.actuator.model == "cf21b_first_order"
    assert torque_poly.actuator.reaction_torque.model == "paper_polynomial"
    assert torque_poly.actuator.reaction_torque.polynomial_coefficients == (
        -3.4,
        8.7,
        2.9,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["actuator"].__setitem__("enabled", False),
            "actuator.enabled must be true",
        ),
        (
            lambda data: data["actuator"].__setitem__("model", "instantaneous"),
            "actuator.model",
        ),
        (
            lambda data: data["actuator"]["thrust_polynomial"].__setitem__(
                "positive_branch_min_ratio", 1.1
            ),
            "must not exceed",
        ),
        (
            lambda data: data["actuator"]["randomization"]["time_constant_s"].update(
                {"min": 0.06, "max": 0.04}
            ),
            "max must be >=",
        ),
        (
            lambda data: data["actuator"]["reaction_torque"].__setitem__(
                "model", "invented_curve"
            ),
            "reaction_torque.model",
        ),
    ],
)
def test_invalid_actuator_values_fail_early(
    tmp_path: Path, mutate, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(_mutated_base(tmp_path, mutate))


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    profile = _mutated_base(
        tmp_path,
        lambda data: data["environment"].__setitem__("policy_hzz", 100),
    )
    with pytest.raises(ConfigError, match="policy_hzz"):
        load_config(profile)


def test_unknown_reward_key_is_rejected(tmp_path: Path) -> None:
    profile = _mutated_base(
        tmp_path,
        lambda data: data["environment"]["reward"].__setitem__(
            "e2e_torque_xy_wieght", 0.1
        ),
    )
    with pytest.raises(ConfigError, match="e2e_torque_xy_wieght"):
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
        (
            lambda data: data["environment"].__setitem__("control_mode", "absolute"),
            "control_mode",
        ),
        (
            lambda data: data["environment"].__setitem__("residual_scale", [1, 2, 3]),
            "exactly 4",
        ),
        (lambda data: data["environment"].__setitem__("policy_hz", -1), "positive"),
        (lambda data: data["environment"].__setitem__("episode_sec", 0), "positive"),
    ],
)
def test_invalid_environment_values_fail_early(
    tmp_path: Path, mutate, message: str
) -> None:
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


def test_loading_config_has_no_filesystem_or_runtime_side_effects(
    tmp_path: Path,
) -> None:
    before = set(tmp_path.rglob("*"))
    config = load_config(CONFIGS / "e2e_train.yaml")
    assert config.profile_name == "e2e_train"
    assert set(tmp_path.rglob("*")) == before

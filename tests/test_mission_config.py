from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from crazyflie_rl.config import ConfigError, dump_resolved_config, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _mutated_base(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))
    mutate(data)
    target = tmp_path / "mission-profile.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def test_unified_view_live_profiles_have_documented_defaults() -> None:
    hover = load_config(CONFIGS / "view_live_hover_eval.yaml")
    circle = load_config(CONFIGS / "view_live_circle_eval.yaml")
    lissajous = load_config(CONFIGS / "view_live_lissajous_eval.yaml")

    assert hover.mission.type == "hover"
    assert hover.mission.hover.target == (0.0, 0.0, 1.0)
    assert hover.mission.hover.yaw_deg == 0.0
    assert hover.mission.hover.duration == 8.0

    assert circle.mission.type == "circle"
    assert circle.mission.hover_altitude == 1.0
    assert circle.mission.goto_xy == (1.0, 0.0)
    assert circle.mission.circle.center_xy == (0.5, 0.0)
    assert circle.mission.circle.radius == 0.5
    assert circle.mission.circle.period == 10.0
    assert circle.mission.circle.laps == 2.0
    assert circle.mission.circle.start_angle_deg == 0.0
    assert circle.mission.circle.direction == "ccw"
    assert circle.mission.circle.clockwise is False
    assert circle.mission.circle.ramp_sec == 2.0

    assert lissajous.mission.type == "lissajous"
    assert lissajous.mission.hover_altitude == 1.0
    assert lissajous.mission.goto_xy == (0.5, 0.0)
    assert lissajous.mission.lissajous.center_xy == (0.0, 0.0)
    assert lissajous.mission.lissajous.amplitude_xy == (0.5, 0.5)
    assert lissajous.mission.lissajous.frequency_ratio == (1, 2)
    assert lissajous.mission.lissajous.phase_deg == 90.0
    assert lissajous.mission.lissajous.base_period == 10.0
    assert lissajous.mission.lissajous.cycles == 2.0
    assert lissajous.mission.lissajous.ramp_sec == 2.0


def test_nested_mission_profile_merge_is_recursive_and_immutable(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        (CONFIGS / "base.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    profile = tmp_path / "override.yaml"
    profile.write_text(
        "\n".join(
            [
                "extends: base.yaml",
                "mission:",
                "  type: lissajous",
                "  lissajous:",
                "    amplitude_xy: [0.8, 0.4]",
                "    cycles: 3.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(profile)
    params = config.mission.lissajous
    assert params.amplitude_xy == (0.8, 0.4)
    assert params.cycles == 3.0
    assert params.frequency_ratio == (1, 2)
    assert params.base_period == 10.0
    with pytest.raises(FrozenInstanceError):
        params.cycles = 4.0  # type: ignore[misc]

    base = load_config(CONFIGS / "base.yaml")
    assert base.mission.lissajous.amplitude_xy == (0.5, 0.5)
    assert base.mission.lissajous.cycles == 2.0


def test_resolved_config_serializes_typed_mission_parameters(tmp_path: Path) -> None:
    config = load_config(CONFIGS / "view_live_lissajous_eval.yaml")
    target = dump_resolved_config(config, tmp_path / "resolved.yaml")
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))

    assert payload["mission"]["hover"]["target"] == [0.0, 0.0, 1.0]
    assert payload["mission"]["circle"]["center_xy"] == [0.5, 0.0]
    assert payload["mission"]["circle"]["direction"] == "ccw"
    assert payload["mission"]["lissajous"]["frequency_ratio"] == [1, 2]
    assert payload["mission"]["lissajous"]["cycles"] == 2.0


def test_existing_circle_profiles_keep_legacy_flat_preset_contract() -> None:
    config = load_config(CONFIGS / "residual_circle_eval.yaml")

    assert config.mission.type == "circle"
    assert config.mission.goto_xy == (1.0, 0.0)
    assert config.mission.circle_ramp_sec == 2.0
    assert config.mission.number_of_laps == 2.0
    assert config.mission.circle_preset("1").radius == 0.5
    assert config.mission.circle_preset("2").radius == 1.0
    assert config.mission.circle_preset("3").period == 5.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data["mission"].__setitem__("type", "spiral"), "mission.type"),
        (lambda data: data["mission"]["hover"].__setitem__("duration", 0), "duration"),
        (lambda data: data["mission"]["hover"].__setitem__("target", [0, 0, -1]), "altitude"),
        (lambda data: data["mission"]["circle"].__setitem__("radius", 0), "radius"),
        (lambda data: data["mission"]["circle"].__setitem__("period", 0), "period"),
        (lambda data: data["mission"]["circle"].__setitem__("laps", 0), "laps"),
        (lambda data: data["mission"]["circle"].__setitem__("direction", "left"), "direction"),
        (lambda data: data["mission"]["circle"].__setitem__("ramp_sec", -1), "ramp_sec"),
        (lambda data: data["mission"]["lissajous"].__setitem__("amplitude_xy", [-1, 1]), "non-negative"),
        (lambda data: data["mission"]["lissajous"].__setitem__("amplitude_xy", [0, 0]), "both be zero"),
        (lambda data: data["mission"]["lissajous"].__setitem__("frequency_ratio", [0, 2]), "integer"),
        (lambda data: data["mission"]["lissajous"].__setitem__("frequency_ratio", [1.0, 2]), "integer"),
        (lambda data: data["mission"]["lissajous"].__setitem__("base_period", 0), "base_period"),
        (lambda data: data["mission"]["lissajous"].__setitem__("cycles", 0), "cycles"),
        (lambda data: data["mission"]["lissajous"].__setitem__("ramp_sec", -1), "ramp_sec"),
        (lambda data: data["mission"].__setitem__("hover_altitude", -0.1), "hover_altitude"),
    ],
)
def test_invalid_mission_parameters_fail_early(
    tmp_path: Path, mutate, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(_mutated_base(tmp_path, mutate))


def test_zero_ramp_is_a_valid_instantaneous_ramp(tmp_path: Path) -> None:
    profile = _mutated_base(
        tmp_path,
        lambda data: (
            data["mission"].__setitem__("circle_ramp_sec", 0.0),
            data["mission"]["circle"].__setitem__("ramp_sec", 0.0),
            data["mission"]["lissajous"].__setitem__("ramp_sec", 0.0),
        ),
    )
    config = load_config(profile)
    assert config.mission.circle_ramp_sec == 0.0
    assert config.mission.circle.ramp_sec == 0.0
    assert config.mission.lissajous.ramp_sec == 0.0


def test_clockwise_property_tracks_validated_direction(tmp_path: Path) -> None:
    profile = _mutated_base(
        tmp_path,
        lambda data: data["mission"]["circle"].__setitem__("direction", "cw"),
    )
    assert load_config(profile).mission.circle.clockwise is True

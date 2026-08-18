from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.missions import (
    CircleMission,
    HoverMission,
    LissajousMission,
    mission_from_experiment,
    ramped_phase,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_hover_mission_uses_target_duration_and_yaw_metadata() -> None:
    config = load_config(CONFIGS / "view_live_hover_eval.yaml")
    mission = HoverMission.from_experiment(config)

    reference, phase = mission.reference(123.0)

    np.testing.assert_array_equal(reference, [0.0, 0.0, 1.0])
    assert phase == "HOVER"
    assert mission.total_sec == 8.0
    assert mission.effective_parameters()["yaw_deg"] == 0.0


def test_explicit_circle_start_matches_formula_and_goto_boundary() -> None:
    config = load_config(CONFIGS / "view_live_circle_eval.yaml")
    params = replace(
        config.mission.circle,
        center_xy=(0.2, -0.3),
        radius=0.8,
        start_angle_deg=30.0,
    )
    mission_config = replace(config.mission, circle=params)
    mission = CircleMission.from_parameters(mission_config)
    expected = np.array(
        [
            0.2 + 0.8 * math.cos(math.radians(30.0)),
            -0.3 + 0.8 * math.sin(math.radians(30.0)),
            1.0,
        ]
    )

    before, before_phase = mission.reference(mission.boundaries.settle2_end - 1e-9)
    start, start_phase = mission.reference(mission.boundaries.settle2_end)

    np.testing.assert_allclose(before, expected, atol=1e-12)
    np.testing.assert_allclose(start, expected, atol=1e-12)
    assert before_phase == "SETTLE2"
    assert start_phase == "CIRCLE"


def test_explicit_circle_clockwise_and_counterclockwise_are_opposites() -> None:
    config = load_config(CONFIGS / "view_live_circle_eval.yaml")
    ccw = CircleMission.from_parameters(config.mission)
    cw_params = replace(config.mission.circle, direction="cw")
    cw = CircleMission.from_parameters(replace(config.mission, circle=cw_params))
    elapsed = 0.5

    ccw_position = ccw._circle_reference(elapsed)
    cw_position = cw._circle_reference(elapsed)

    assert ccw_position[1] > ccw.center_xy[1]
    assert cw_position[1] < cw.center_xy[1]
    assert ccw_position[0] == pytest.approx(cw_position[0])


def test_explicit_circle_holds_its_actual_last_reference() -> None:
    config = load_config(CONFIGS / "view_live_circle_eval.yaml")
    mission = CircleMission.from_parameters(config.mission)

    hold, phase = mission.reference(mission.boundaries.trajectory_end)
    expected = mission._circle_reference(mission.trajectory_duration)

    np.testing.assert_allclose(hold, expected, atol=1e-12)
    assert phase == "HOLD"
    assert mission.legacy_hold is False


def test_factory_keeps_legacy_circle_and_selects_all_named_missions() -> None:
    hover_config = load_config(CONFIGS / "view_live_hover_eval.yaml")
    circle_config = load_config(CONFIGS / "residual_circle_eval.yaml")
    lissajous_config = load_config(CONFIGS / "view_live_lissajous_eval.yaml")

    assert isinstance(
        mission_from_experiment(hover_config, mission_type="hover"), HoverMission
    )
    legacy = mission_from_experiment(
        circle_config,
        mission_type="circle",
        preset="1",
        legacy_circle_preset=True,
    )
    assert isinstance(legacy, CircleMission)
    assert legacy.legacy_hold is True
    assert isinstance(
        mission_from_experiment(lissajous_config, mission_type="lissajous"),
        LissajousMission,
    )


def test_lissajous_initial_reference_and_goto_boundary_are_continuous() -> None:
    config = load_config(CONFIGS / "view_live_lissajous_eval.yaml")
    mission = LissajousMission.from_experiment(config)
    expected = np.array([0.5, 0.0, 1.0])

    before, before_phase = mission.reference(mission.boundaries.settle2_end - 1e-9)
    start, start_phase = mission.reference(mission.boundaries.settle2_end)

    np.testing.assert_allclose(before, expected, atol=1e-12)
    np.testing.assert_allclose(start, expected, atol=1e-12)
    assert before_phase == "SETTLE2"
    assert start_phase == "LISSAJOUS"


def test_lissajous_formula_applies_frequency_ratio_and_phase() -> None:
    config = load_config(CONFIGS / "view_live_lissajous_eval.yaml")
    mission = LissajousMission.from_experiment(config)
    elapsed = 3.25
    theta = mission.theta(elapsed)
    reference = mission._path_reference(elapsed)

    assert reference[0] == pytest.approx(0.5 * math.sin(theta + math.pi / 2.0))
    assert reference[1] == pytest.approx(0.5 * math.sin(2.0 * theta))
    assert reference[2] == 1.0


def test_lissajous_complete_path_reaches_configured_xy_range() -> None:
    config = load_config(CONFIGS / "view_live_lissajous_eval.yaml")
    mission = LissajousMission.from_experiment(config)
    samples = np.asarray(
        [
            mission._path_reference(elapsed)
            for elapsed in np.linspace(0.0, mission.trajectory_duration, 20_001)
        ]
    )

    assert samples[:, 0].min() == pytest.approx(-0.5, abs=2e-4)
    assert samples[:, 0].max() == pytest.approx(0.5, abs=2e-4)
    assert samples[:, 1].min() == pytest.approx(-0.5, abs=2e-4)
    assert samples[:, 1].max() == pytest.approx(0.5, abs=2e-4)


def test_lissajous_ramp_starts_with_near_zero_reference_speed() -> None:
    config = load_config(CONFIGS / "view_live_lissajous_eval.yaml")
    mission = LissajousMission.from_experiment(config)
    step = 1e-4
    speed = np.linalg.norm(
        (mission._path_reference(step) - mission._path_reference(0.0)) / step
    )

    assert speed < 1e-5


def test_lissajous_hold_uses_actual_final_reference() -> None:
    config = load_config(CONFIGS / "view_live_lissajous_eval.yaml")
    mission = LissajousMission.from_experiment(config)

    hold, phase = mission.reference(mission.boundaries.trajectory_end)

    np.testing.assert_allclose(
        hold,
        mission._path_reference(mission.trajectory_duration),
        atol=1e-12,
    )
    assert phase == "HOLD"


def test_zero_ramp_uses_instantaneous_constant_angular_speed() -> None:
    assert ramped_phase(1.25, 2.0, 0.0) == pytest.approx(2.5)


def test_public_mission_factory_defaults_to_explicit_circle() -> None:
    config = load_config(CONFIGS / "view_live_circle_eval.yaml")
    from crazyflie_rl.missions import build_mission

    mission = build_mission(config)

    assert isinstance(mission, CircleMission)
    assert mission.legacy_hold is False
    assert mission.center_xy == config.mission.circle.center_xy


def test_public_core_rejects_negative_ramp_and_noninteger_frequency() -> None:
    with pytest.raises(Exception, match="non-negative"):
        ramped_phase(1.0, 2.0, -0.1)

    config = load_config(CONFIGS / "view_live_lissajous_eval.yaml")
    invalid = replace(
        config.mission.lissajous,
        frequency_ratio=(1.5, 2),  # type: ignore[arg-type]
    )
    with pytest.raises(Exception, match="positive integers"):
        LissajousMission(config.mission, invalid)

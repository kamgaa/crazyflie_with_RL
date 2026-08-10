from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crazyflie_rl.config import ConfigError, load_config
from crazyflie_rl.eval_cli import EvaluationRunner
from crazyflie_rl.missions import CircleMission, cosine_ease


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture(scope="module")
def circle_config():
    return load_config(CONFIGS / "residual_circle_eval.yaml")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 1.0)],
)
def test_cosine_ease_preserves_clamping_and_endpoints(value: float, expected: float) -> None:
    assert cosine_ease(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("preset", "circle_end", "total"),
    [("1", 34.0, 36.0), ("2", 30.0, 32.0), ("3", 24.0, 26.0)],
)
def test_legacy_phase_boundaries(
    circle_config, preset: str, circle_end: float, total: float
) -> None:
    mission = CircleMission.from_experiment(circle_config, preset)
    assert mission.boundaries.takeoff_end == 4.0
    assert mission.boundaries.settle1_end == 6.0
    assert mission.boundaries.goto_end == 10.0
    assert mission.boundaries.settle2_end == 12.0
    assert mission.boundaries.circle_end == circle_end
    assert mission.total_sec == total


def test_reference_values_and_boundary_phase_tags(circle_config) -> None:
    mission = CircleMission.from_experiment(circle_config, "1")
    cases = (
        (0.0, [0.0, 0.0, 0.0], "TAKEOFF"),
        (2.0, [0.0, 0.0, 0.5], "TAKEOFF"),
        (4.0, [0.0, 0.0, 1.0], "SETTLE1"),
        (6.0, [0.0, 0.0, 1.0], "GOTO"),
        (8.0, [0.5, 0.0, 1.0], "GOTO"),
        (10.0, [1.0, 0.0, 1.0], "SETTLE2"),
        (12.0, [1.0, 0.0, 1.0], "CIRCLE"),
        (34.0, [1.0, 0.0, 1.0], "HOLD"),
        (36.0, [1.0, 0.0, 1.0], "HOLD"),
    )
    for time_sec, expected_position, expected_phase in cases:
        position, phase = mission.reference(time_sec)
        np.testing.assert_allclose(position, expected_position, atol=1e-12)
        assert phase == expected_phase


def test_circle_ramp_and_constant_phase_match_master_formula(circle_config) -> None:
    mission = CircleMission.from_experiment(circle_config, "1")
    omega = 2.0 * math.pi / 10.0
    assert mission.circle_phase(0.0) == 0.0
    assert mission.circle_phase(1.0) == pytest.approx(omega * (0.5 - 1.0 / math.pi))
    assert mission.circle_phase(2.0) == pytest.approx(omega)
    assert mission.circle_phase(7.0) == pytest.approx(omega * 6.0)

    position, phase = mission.reference(13.0)
    phi = omega * (0.5 - 1.0 / math.pi)
    expected = np.array([0.5 + 0.5 * math.cos(phi), 0.5 * math.sin(phi), 1.0])
    np.testing.assert_allclose(position, expected, atol=1e-12)
    assert phase == "CIRCLE"


def test_circle_to_hold_reference_jump_is_intentionally_preserved(circle_config) -> None:
    mission = CircleMission.from_experiment(circle_config, "1")
    circle_elapsed = (
        circle_config.mission.circle_ramp_sec
        + circle_config.mission.number_of_laps * mission.preset.period
    )
    end_phase = mission.circle_phase(circle_elapsed)
    center_x, center_y = mission.center_xy
    circle_end_limit = np.array(
        [
            center_x + mission.preset.radius * math.cos(end_phase),
            center_y + mission.preset.radius * math.sin(end_phase),
            circle_config.mission.hover_altitude,
        ]
    )
    hold, phase = mission.reference(mission.boundaries.circle_end)

    # For preset 1 the ramp adds 36 degrees beyond two constant-speed laps,
    # so returning HOLD to (1,0,1) is a real discontinuity in master.
    assert math.degrees(end_phase % (2.0 * math.pi)) == pytest.approx(36.0)
    np.testing.assert_allclose(hold, [1.0, 0.0, 1.0])
    assert np.linalg.norm(circle_end_limit - hold) == pytest.approx(
        2.0 * mission.preset.radius * math.sin(math.radians(18.0))
    )
    assert phase == "HOLD"


def test_mission_and_config_are_frozen_and_unknown_preset_is_explicit(circle_config) -> None:
    mission = CircleMission.from_experiment(circle_config, "1")
    with pytest.raises(FrozenInstanceError):
        mission.preset = circle_config.mission.circle_presets[1]  # type: ignore[misc]
    with pytest.raises(ConfigError, match="unknown circle preset"):
        CircleMission.from_experiment(circle_config, "missing")


def test_headless_circle_ignores_legacy_eight_second_truncation(circle_config) -> None:
    mission_config = replace(circle_config.mission, force_floor_start=False)
    config = replace(circle_config, mission=mission_config)

    class FakeEnvironment:
        def __init__(self) -> None:
            self.dt_phys = 0.01
            self.substeps = 1
            self.pos_des = np.array([0.0, 0.0, 1.0])
            self.dist_torque_body = np.zeros(3)
            self.steps = 0

        @staticmethod
        def _observation() -> np.ndarray:
            return np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            )

        def reset(self, seed=None):
            assert seed == 42
            return self._observation(), {}

        def step(self, action):
            np.testing.assert_array_equal(action, np.zeros(4))
            self.steps += 1
            return self._observation(), 0.0, False, self.steps >= 800, {}

    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.config = config
    runner.artifacts = SimpleNamespace()
    runner.headless = True
    runner.realtime = False
    runner.camera_tracking = False
    runner.seed = 42
    runner.circle = CircleMission.from_experiment(config, "1")

    trace = runner._rollout(FakeEnvironment(), None, "floor", "floor (PID)")

    assert trace.sample_count == 3600
    assert trace.time_sec[-1] == pytest.approx(35.99)
    assert trace.truncated_at == pytest.approx(7.99)
    assert trace.terminated_at is None
    assert trace.phases.count("TAKEOFF") == 400
    assert trace.phases.count("SETTLE1") == 200
    assert trace.phases.count("GOTO") == 400
    assert trace.phases.count("SETTLE2") == 200
    assert trace.phases.count("CIRCLE") == 2200
    assert trace.phases.count("HOLD") == 200

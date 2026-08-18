from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from crazyflie_rl.config import ConfigError, load_config
import crazyflie_rl.eval_cli as eval_cli
from crazyflie_rl.eval_cli import (
    EvaluationRunner,
    RolloutTrace,
    apply_runtime_overrides,
    build_parser,
    format_run_summary,
    mission_condition,
    normalize_direction,
    normalize_mode,
    prompt_runtime_options,
    run_evaluation_cli,
    trace_metrics,
)
from crazyflie_rl.missions import mission_from_experiment


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_view_live_help_needs_no_runtime_dependency_imports() -> None:
    script = r'''
import builtins
import importlib.util
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
heavy = {"gymnasium", "matplotlib", "mujoco", "numpy", "stable_baselines3", "torch", "yaml"}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in heavy:
        raise AssertionError(f"--help imported runtime dependency {name!r}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location("_view_live_help", root / "view_live.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.main(["--help"])
except SystemExit as exc:
    if exc.code != 0:
        raise
else:
    raise AssertionError("argparse --help did not exit")
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--mode" in completed.stdout
    assert "--amplitude-x" in completed.stdout


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", "hover"),
        ("hover", "hover"),
        ("2", "circle"),
        ("circle", "circle"),
        ("3", "lissajous"),
        ("lissajous", "lissajous"),
    ],
)
def test_numeric_and_named_modes_are_normalized(value: str, expected: str) -> None:
    assert normalize_mode(value) == expected


def test_unknown_mode_is_explicit() -> None:
    with pytest.raises(Exception, match="unknown mode"):
        normalize_mode("orbit")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("cw", "cw"),
        ("clockwise", "cw"),
        ("ccw", "ccw"),
        ("counterclockwise", "ccw"),
    ],
)
def test_direction_aliases_are_normalized(value: str, expected: str) -> None:
    assert normalize_direction(value) == expected


def test_non_tty_unified_invocation_requires_mode(capsys: pytest.CaptureFixture[str]) -> None:
    stream = SimpleNamespace(isatty=lambda: False)
    with pytest.raises(SystemExit) as caught:
        run_evaluation_cli(
            default_config=CONFIGS / "e2e_hover_eval.yaml",
            description="test",
            argv=[],
            unified=True,
            stdin=stream,
        )
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "--mode is required when stdin is not a TTY" in error
    assert "python view_live.py --mode hover --headless" in error


def test_parser_exposes_all_runtime_override_groups() -> None:
    parser = build_parser(CONFIGS / "e2e_hover_eval.yaml", "test")
    destinations = {action.dest for action in parser._actions}
    assert {
        "mode",
        "hover_target",
        "yaw_deg",
        "duration",
        "center",
        "radius",
        "circle_period",
        "laps",
        "altitude",
        "ramp_sec",
        "start_angle_deg",
        "direction",
        "amplitude_x",
        "amplitude_y",
        "frequency_x",
        "frequency_y",
        "phase_deg",
        "base_period",
        "cycles",
        "takeoff_sec",
        "settle_sec",
        "goto_sec",
        "post_hold_sec",
        "seed",
        "position_perturbation",
        "attitude_perturbation_deg",
        "force_floor_start",
    } <= destinations


def test_hover_runtime_overrides_update_mission_and_environment() -> None:
    config = load_config(CONFIGS / "view_live_hover_eval.yaml")
    args = build_parser(config.source_path, "test").parse_args(
        [
            "--mode",
            "1",
            "--hover-target",
            "0.2",
            "-0.1",
            "1.3",
            "--yaw-deg",
            "90",
            "--duration",
            "4.5",
            "--seed",
            "7",
            "--position-perturbation",
            "0.02",
            "--attitude-perturbation-deg",
            "3",
        ]
    )
    effective = apply_runtime_overrides(config, args, "hover")

    assert effective.mission.hover.target == pytest.approx((0.2, -0.1, 1.3))
    assert effective.mission.hover.duration == 4.5
    assert effective.environment.position_target == pytest.approx((0.2, -0.1, 1.3))
    assert effective.environment.yaw_target == pytest.approx(np.pi / 2.0)
    assert effective.environment.episode_sec == 4.5
    assert effective.environment.position_perturbation == 0.02
    assert effective.environment.attitude_perturbation_deg == 3.0
    assert effective.evaluation.seed_start == 7


def test_circle_runtime_overrides_include_direction_and_common_phase_values() -> None:
    config = load_config(CONFIGS / "view_live_circle_eval.yaml")
    args = build_parser(config.source_path, "test").parse_args(
        [
            "--mode",
            "circle",
            "--center",
            "1",
            "-2",
            "--radius",
            "0.8",
            "--circle-period",
            "6",
            "--laps",
            "3",
            "--start-angle-deg",
            "45",
            "--direction",
            "clockwise",
            "--ramp-sec",
            "0",
            "--altitude",
            "1.4",
            "--takeoff-sec",
            "2",
            "--settle-sec",
            "1",
            "--goto-sec",
            "3",
            "--post-hold-sec",
            "0",
            "--force-floor-start",
        ]
    )
    effective = apply_runtime_overrides(config, args, "circle")

    assert effective.mission.circle.center_xy == pytest.approx((1.0, -2.0))
    assert effective.mission.circle.radius == 0.8
    assert effective.mission.circle.period == 6.0
    assert effective.mission.circle.laps == 3.0
    assert effective.mission.circle.start_angle_deg == 45.0
    assert effective.mission.circle.direction == "cw"
    assert effective.mission.circle.ramp_sec == 0.0
    assert effective.mission.hover_altitude == 1.4
    assert effective.mission.takeoff_sec == 2.0
    assert effective.mission.settle_sec == 1.0
    assert effective.mission.goto_sec == 3.0
    assert effective.mission.post_hold_sec == 0.0
    assert effective.mission.force_floor_start is True
    expected_offset = 0.8 / np.sqrt(2.0)
    assert effective.mission.goto_xy == pytest.approx(
        (1.0 + expected_offset, -2.0 + expected_offset)
    )


def test_lissajous_runtime_override_validation_and_condition() -> None:
    config = load_config(CONFIGS / "view_live_lissajous_eval.yaml")
    parser = build_parser(config.source_path, "test")
    args = parser.parse_args(
        [
            "--mode",
            "3",
            "--center",
            "0",
            "0",
            "--amplitude-x",
            "0.8",
            "--amplitude-y",
            "0.4",
            "--frequency-x",
            "1",
            "--frequency-y",
            "2",
            "--phase-deg",
            "90",
            "--base-period",
            "10",
            "--cycles",
            "2",
        ]
    )
    effective = apply_runtime_overrides(config, args, "lissajous")
    mission = mission_from_experiment(effective, mission_type="lissajous")
    condition = mission_condition(effective, mission)
    assert condition.startswith("cx0-cy0-Ax0p8-Ay0p4-a1-b2-ph90-T10-cycles2")
    assert effective.mission.goto_xy == pytest.approx((0.8, 0.0))

    invalid = parser.parse_args(
        ["--mode", "lissajous", "--amplitude-x", "0", "--amplitude-y", "0"]
    )
    with pytest.raises(ConfigError, match="must not both be zero"):
        apply_runtime_overrides(config, invalid, "lissajous")


def test_runtime_seed_and_hover_altitude_must_be_nonnegative() -> None:
    config = load_config(CONFIGS / "view_live_hover_eval.yaml")
    parser = build_parser(config.source_path, "test")
    with pytest.raises(ConfigError, match="seed must be non-negative"):
        apply_runtime_overrides(
            config, parser.parse_args(["--seed", "-1"]), "hover"
        )
    with pytest.raises(ConfigError, match="altitude must be non-negative"):
        apply_runtime_overrides(
            config,
            parser.parse_args(["--hover-target", "0", "0", "-0.1"]),
            "hover",
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("hover", "x0-y0-z1-yaw0-dur8-p0-a0-floorReq0"),
        (
            "circle",
            "cx0p5-cy0-r0p5-T10-laps2-z1-sa0-ccw-ramp2-"
            "to4-set2-goto4-hold2-floorReq1-p0-a0",
        ),
        (
            "lissajous",
            "cx0-cy0-Ax0p5-Ay0p5-a1-b2-ph90-T10-cycles2-z1-ramp2-"
            "to4-set2-goto4-hold2-floorReq1-p0-a0",
        ),
    ],
)
def test_each_mode_artifact_condition_records_effective_path(
    mode: str, expected: str
) -> None:
    config = load_config(CONFIGS / f"view_live_{mode}_eval.yaml")
    mission = mission_from_experiment(config, mission_type=mode)
    assert mission_condition(config, mission) == expected


def test_summary_includes_effective_mission_and_runtime_options() -> None:
    config = load_config(CONFIGS / "view_live_lissajous_eval.yaml")
    mission = mission_from_experiment(config, mission_type="lissajous")
    summary = format_run_summary(
        config,
        mission,
        policy="both",
        model=Path("model/ppo_best.zip"),
        headless=True,
        realtime=False,
        camera_tracking=True,
    )
    for expected in (
        "Mode              : Lissajous",
        "Ramp              : 2 s",
        "Takeoff           : 4 s",
        "Settle            : 2 s",
        "GOTO              : 4 s",
        "Post-HOLD         : 2 s",
        "Force floor request: True",
        "Display           : headless",
        "Realtime          : disabled",
        "Camera tracking   : enabled",
        "Position perturb. : 0 m",
        "Attitude perturb. : 0 deg",
    ):
        assert expected in summary


def test_interactive_hover_blank_answers_use_profile_defaults() -> None:
    config = load_config(CONFIGS / "view_live_hover_eval.yaml")
    args = build_parser(config.source_path, "test").parse_args([])
    prompts: list[str] = []

    def blank(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    prompted = prompt_runtime_options(args, config, "hover", input_fn=blank)
    effective = apply_runtime_overrides(config, prompted, "hover")

    assert effective.mission.hover == config.mission.hover
    assert effective.evaluation.seed_start == config.evaluation.seed_start
    assert effective.environment.position_perturbation == 0.0
    assert effective.environment.attitude_perturbation_deg == 0.0
    assert prompted.policy == "both"
    assert prompted.headless is False
    assert prompted.no_realtime is False
    assert "Hover altitude [1.0 m]: " in prompts


def test_interactive_hover_accepts_user_values_and_common_options() -> None:
    config = load_config(CONFIGS / "view_live_hover_eval.yaml")
    args = build_parser(config.source_path, "test").parse_args([])
    answers = iter(
        ("0.3", "", "1.2", "45", "5", "9", "0.1", "2", "floor", "y", "n")
    )
    prompted = prompt_runtime_options(
        args, config, "hover", input_fn=lambda _prompt: next(answers)
    )
    effective = apply_runtime_overrides(config, prompted, "hover")

    assert effective.mission.hover.target == pytest.approx((0.3, 0.0, 1.2))
    assert effective.environment.yaw_target == pytest.approx(np.pi / 4.0)
    assert effective.mission.hover.duration == 5.0
    assert effective.evaluation.seed_start == 9
    assert effective.environment.position_perturbation == 0.1
    assert effective.environment.attitude_perturbation_deg == 2.0
    assert prompted.policy == "floor"
    assert prompted.headless is True
    assert prompted.no_realtime is True


def test_interactive_circle_cli_values_survive_mode_selection_and_enter() -> None:
    config = load_config(CONFIGS / "view_live_circle_eval.yaml")
    args = build_parser(config.source_path, "test").parse_args(
        [
            "--center",
            "0.25",
            "-0.4",
            "--radius",
            "0.8",
            "--circle-period",
            "7",
            "--laps",
            "1.5",
            "--altitude",
            "1.3",
            "--ramp-sec",
            "0.75",
            "--start-angle-deg",
            "30",
            "--direction",
            "clockwise",
            "--takeoff-sec",
            "2.5",
            "--settle-sec",
            "0.5",
            "--goto-sec",
            "3.5",
            "--post-hold-sec",
            "1.25",
            "--seed",
            "17",
            "--position-perturbation",
            "0.07",
            "--attitude-perturbation-deg",
            "4",
            "--no-force-floor-start",
        ]
    )
    prompts: list[str] = []

    def enter(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    # This mirrors the unified TTY path: CLI parsing happens first, then the
    # user chooses Circle, then Enter accepts every displayed default.
    prompted = prompt_runtime_options(args, config, "circle", input_fn=enter)
    effective = apply_runtime_overrides(config, prompted, "circle")

    assert effective.mission.circle.center_xy == pytest.approx((0.25, -0.4))
    assert effective.mission.circle.radius == 0.8
    assert effective.mission.circle.period == 7.0
    assert effective.mission.circle.laps == 1.5
    assert effective.mission.circle.start_angle_deg == 30.0
    assert effective.mission.circle.direction == "cw"
    assert effective.mission.circle.ramp_sec == 0.75
    assert effective.mission.hover_altitude == 1.3
    assert effective.mission.takeoff_sec == 2.5
    assert effective.mission.settle_sec == 0.5
    assert effective.mission.goto_sec == 3.5
    assert effective.mission.post_hold_sec == 1.25
    assert effective.mission.force_floor_start is False
    assert effective.evaluation.seed_start == 17
    assert effective.environment.position_perturbation == 0.07
    assert effective.environment.attitude_perturbation_deg == 4.0
    assert "Circle radius [0.8 m]: " in prompts
    assert "Seed [17]: " in prompts
    assert "Initial position perturbation [0.07 m]: " in prompts
    assert "Force floor start [y/N]: " in prompts


@pytest.mark.parametrize(
    ("mode", "cli", "expected"),
    [
        (
            "hover",
            [
                "--hover-target",
                "0.2",
                "-0.3",
                "1.4",
                "--yaw-deg",
                "35",
                "--duration",
                "6",
            ],
            {
                "target": (0.2, -0.3, 1.4),
                "yaw_deg": 35.0,
                "duration": 6.0,
            },
        ),
        (
            "lissajous",
            [
                "--center",
                "0.1",
                "-0.2",
                "--amplitude-x",
                "0.7",
                "--amplitude-y",
                "0.35",
                "--frequency-x",
                "3",
                "--frequency-y",
                "2",
                "--phase-deg",
                "45",
                "--base-period",
                "12",
                "--cycles",
                "1.5",
                "--altitude",
                "1.6",
                "--ramp-sec",
                "1.25",
            ],
            {
                "center_xy": (0.1, -0.2),
                "amplitude_xy": (0.7, 0.35),
                "frequency_ratio": (3, 2),
                "phase_deg": 45.0,
                "base_period": 12.0,
                "cycles": 1.5,
                "ramp_sec": 1.25,
                "altitude": 1.6,
            },
        ),
    ],
)
def test_interactive_mode_specific_cli_values_are_enter_defaults(
    mode: str, cli: list[str], expected: dict[str, object]
) -> None:
    config = load_config(CONFIGS / f"view_live_{mode}_eval.yaml")
    args = build_parser(config.source_path, "test").parse_args(cli)
    prompted = prompt_runtime_options(
        args, config, mode, input_fn=lambda _prompt: ""
    )
    effective = apply_runtime_overrides(config, prompted, mode)

    if mode == "hover":
        assert effective.mission.hover.target == pytest.approx(expected["target"])
        assert effective.mission.hover.yaw_deg == expected["yaw_deg"]
        assert effective.mission.hover.duration == expected["duration"]
    else:
        lissajous = effective.mission.lissajous
        assert lissajous.center_xy == pytest.approx(expected["center_xy"])
        assert lissajous.amplitude_xy == pytest.approx(expected["amplitude_xy"])
        assert lissajous.frequency_ratio == expected["frequency_ratio"]
        assert lissajous.phase_deg == expected["phase_deg"]
        assert lissajous.base_period == expected["base_period"]
        assert lissajous.cycles == expected["cycles"]
        assert lissajous.ramp_sec == expected["ramp_sec"]
        assert effective.mission.hover_altitude == expected["altitude"]


@dataclass(frozen=True)
class _FakeMission:
    name: str
    total_sec: float = 0.3

    def reference(self, time_sec: float):
        return np.array([time_sec, -time_sec, 1.0]), self.name.upper()

    def effective_parameters(self):
        return {"name": self.name, "center_xy": [0.0, 0.0]}


class _FakeEnvironment:
    dt_phys = 0.1
    substeps = 1

    def __init__(self) -> None:
        self.pos_des = np.zeros(3)
        self.dist_torque_body = np.zeros(3)
        self.references: list[np.ndarray] = []
        self.steps = 0
        self.closed = False

    @staticmethod
    def observation() -> np.ndarray:
        result = np.zeros(15)
        result[6] = 1.0
        result[14] = 1.0
        return result

    def reset(self, seed=None):
        assert seed == 42
        return self.observation(), {}

    def step(self, action):
        np.testing.assert_array_equal(action, np.zeros(4))
        self.references.append(self.pos_des.copy())
        self.steps += 1
        return self.observation(), 0.0, self.steps == 2, self.steps == 3, {}

    def close(self) -> None:
        self.closed = True


class _FakeFactory:
    def __init__(self, env: _FakeEnvironment) -> None:
        self.env = env

    def make(self, **overrides):
        assert overrides == {}
        return self.env


@pytest.mark.parametrize("mode", ["hover", "circle", "lissajous"])
def test_runner_uses_generic_mission_and_passes_each_reference(mode: str) -> None:
    config = load_config(CONFIGS / f"view_live_{mode}_eval.yaml")
    env = _FakeEnvironment()
    runner = EvaluationRunner(
        config,
        SimpleNamespace(),
        headless=True,
        realtime=False,
        camera_tracking=True,
        mission=_FakeMission(mode),
        environment_factory=_FakeFactory(env),
    )
    trace = runner.run(None, "floor", "floor")

    # Hover preserves termination semantics; trajectories preserve the legacy
    # behavior of recording the guard and completing the requested mission.
    expected_steps = 2 if mode == "hover" else 3
    assert trace.sample_count == expected_steps
    assert trace.terminated_at == pytest.approx(0.1)
    assert trace.truncated_at == (pytest.approx(0.2) if mode != "hover" else None)
    assert len(env.references) == expected_steps
    np.testing.assert_allclose(env.references[0], [0.0, 0.0, 1.0])
    assert env.closed is True


def test_force_floor_start_fallback_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIGS / "view_live_circle_eval.yaml")
    env = _FakeEnvironment()
    monkeypatch.setattr(
        EvaluationRunner,
        "_force_floor_start",
        staticmethod(lambda _env: (False, "RuntimeError: forced failure")),
    )
    runner = EvaluationRunner(
        config,
        SimpleNamespace(),
        headless=True,
        realtime=False,
        camera_tracking=False,
        mission=_FakeMission("circle"),
        environment_factory=_FakeFactory(env),
    )

    trace = runner.run(None, "floor", "floor")
    metrics = trace_metrics(trace, 0.3)

    assert trace.force_floor_start_requested is True
    assert trace.force_floor_start_applied is False
    assert metrics["force_floor_start_error"] == "RuntimeError: forced failure"


def test_force_floor_start_restores_reset_state_when_forward_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qpos = np.array([0.4, -0.2, 0.8, 0.9, 0.1, 0.2, 0.3])
    qvel = np.arange(6, dtype=float)
    original_qpos = qpos.copy()
    original_qvel = qvel.copy()
    calls = 0

    def fake_forward(_model, _data):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("forward failed")

    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(mj_forward=fake_forward),
    )
    env = SimpleNamespace(
        model=object(),
        data=SimpleNamespace(qpos=qpos, qvel=qvel),
    )

    applied, error = EvaluationRunner._force_floor_start(env)

    assert applied is False
    assert error == "RuntimeError: forward failed"
    assert calls == 2
    np.testing.assert_array_equal(env.data.qpos, original_qpos)
    np.testing.assert_array_equal(env.data.qvel, original_qvel)


def test_rollout_step_error_returns_partial_trace() -> None:
    config = load_config(CONFIGS / "view_live_hover_eval.yaml")

    class FailingEnvironment(_FakeEnvironment):
        def step(self, action):
            if self.steps:
                raise RuntimeError("simulated step failure")
            return super().step(action)

    env = FailingEnvironment()
    runner = EvaluationRunner(
        config,
        SimpleNamespace(),
        headless=True,
        realtime=False,
        camera_tracking=False,
        mission=_FakeMission("hover"),
        environment_factory=_FakeFactory(env),
    )

    trace = runner.run(None, "floor", "floor")

    assert trace.sample_count == 1
    assert trace.error == "RuntimeError: simulated step failure"
    assert env.closed is True


def test_rollout_reference_error_returns_partial_trace() -> None:
    config = load_config(CONFIGS / "view_live_hover_eval.yaml")

    class FailingReferenceMission:
        name = "hover"
        total_sec = 0.3

        def __init__(self) -> None:
            self.calls = 0

        def reference(self, time_sec):
            if self.calls:
                raise RuntimeError("simulated reference failure")
            self.calls += 1
            return np.array([time_sec, -time_sec, 1.0]), "HOVER"

        @staticmethod
        def effective_parameters():
            return {"name": "hover"}

    env = _FakeEnvironment()
    runner = EvaluationRunner(
        config,
        SimpleNamespace(),
        headless=True,
        realtime=False,
        camera_tracking=False,
        mission=FailingReferenceMission(),
        environment_factory=_FakeFactory(env),
    )

    trace = runner.run(None, "floor", "floor")

    assert trace.sample_count == 1
    assert trace.error == "RuntimeError: simulated reference failure"
    assert env.closed is True


def test_trace_metrics_include_rmse_phase_and_boundary_contract() -> None:
    trace = RolloutTrace(
        policy="residual",
        label="policy",
        time_sec=np.array([0.0, 0.1, 0.2]),
        position=np.zeros((3, 3)),
        attitude_deg=np.zeros((3, 3)),
        reference_position=np.zeros((3, 3)),
        position_error=np.array([0.1, 0.2, 0.3]),
        phases=("GOTO", "LISSAJOUS", "LISSAJOUS"),
        training_boundary_crossed_at=0.1,
        guard_boundary_crossed_at=None,
        terminated_at=None,
        truncated_at=0.2,
        diverged_at=None,
    )
    metrics = trace_metrics(trace, 0.3)

    assert metrics["completed_sample_count"] == 3
    assert metrics["position_rmse"] == pytest.approx(np.sqrt(14.0 / 300.0))
    assert metrics["mean_position_error"] == pytest.approx(0.2)
    assert metrics["max_position_error"] == pytest.approx(0.3)
    assert metrics["tail_mean_position_error"] == pytest.approx(0.3)
    assert metrics["trajectory_phase_rmse"] == pytest.approx(np.sqrt(0.065))
    assert metrics["phases"]["LISSAJOUS"]["count"] == 2
    assert metrics["training_boundary_crossed_at"] == 0.1
    assert metrics["truncated_at"] == 0.2


@pytest.mark.parametrize("explicit_config", [True, False])
def test_unified_fake_rollout_saves_plot_metrics_runtime_config_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_config: bool
) -> None:
    source = load_config(CONFIGS / "view_live_hover_eval.yaml")
    xml = tmp_path / "model.xml"
    xml.write_text("<mujoco/>", encoding="utf-8")
    config = replace(
        source,
        paths=replace(
            source.paths,
            mujoco_xml=xml,
            artifact_root=tmp_path / "artifacts",
        ),
    )

    class CompletingEnvironment(_FakeEnvironment):
        def step(self, action):
            np.testing.assert_array_equal(action, np.zeros(4))
            self.references.append(self.pos_des.copy())
            self.steps += 1
            return self.observation(), 0.0, False, self.steps == 3, {}

    env = CompletingEnvironment()
    factory = _FakeFactory(env)
    eval_cli._ensure_runtime_imports()
    loaded_profiles: list[Path] = []

    def fake_load_config(path):
        loaded_profiles.append(Path(path))
        return config

    monkeypatch.setattr(eval_cli, "load_config", fake_load_config)
    original_runner = eval_cli.EvaluationRunner

    def runner_with_fake_factory(*args, **kwargs):
        return original_runner(*args, **kwargs, environment_factory=factory)

    def fake_hover_plot(path, **_kwargs):
        path.write_bytes(b"fake-png")
        return path

    monkeypatch.setattr(eval_cli, "EvaluationRunner", runner_with_fake_factory)
    monkeypatch.setattr(eval_cli, "save_hover_trace", fake_hover_plot)

    selection = (
        ["--config", str(CONFIGS / "e2e_hover_eval.yaml")]
        if explicit_config
        else ["--mode", "hover"]
    )
    result = run_evaluation_cli(
        default_config=source.source_path,
        description="test",
        argv=[
            *selection,
            "--duration",
            "0.3",
            "--policy",
            "floor",
            "--headless",
            "--no-realtime",
        ],
        unified=True,
        mode_profiles={"hover": CONFIGS / "view_live_hover_eval.yaml"},
    )
    assert result == 0
    assert loaded_profiles[0].name == (
        "e2e_hover_eval.yaml" if explicit_config else "view_live_hover_eval.yaml"
    )

    run_dirs = list((tmp_path / "artifacts" / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (
        "ppo_e2e_hover_x0-y0-z1-yaw0-dur0p3-p0-a0-floorReq0_seed42"
        in run_dir.name
    )
    assert len(list((run_dir / "plots").glob("*.png"))) == 1
    runtime_configs = list((run_dir / "config").glob("*runtime-resolved*.yaml"))
    assert len(runtime_configs) == 1

    metrics_path = next((run_dir / "metrics").glob("*.json"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["status"] == "completed"
    assert metrics["effective_parameters"]["duration"] == 0.3
    assert metrics["runtime_parameters"]["policy"] == "floor"
    assert metrics["control_mode"] == "e2e"
    assert metrics["selected_policy"] == "floor"
    assert metrics["position_perturbation"] == 0.0
    assert metrics["attitude_perturbation_deg"] == 0.0
    assert metrics["policies"]["floor"]["sample_count"] == 3
    assert metrics["policies"]["floor"]["plot"].startswith("plots/")

    manifest_path = next((run_dir / "manifests").glob("*manifest*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["condition"] == "x0-y0-z1-yaw0-dur0p3-p0-a0-floorReq0"
    assert manifest["runtime_config"]["parameters"]["mode"] == "hover"
    assert manifest["result"]["effective_parameters"]["duration"] == 0.3
    assert (
        manifest["result"]["policy_outcomes"]["floor"][
            "force_floor_start_applied"
        ]
        is False
    )


def test_model_zip_fallback_uses_actual_path_in_all_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = load_config(CONFIGS / "view_live_hover_eval.yaml")
    xml = tmp_path / "model.xml"
    xml.write_text("<mujoco/>", encoding="utf-8")
    model_argument = tmp_path / "policy"
    model_archive = tmp_path / "policy.zip"
    model_archive.write_bytes(b"fake-model")
    config = replace(
        source,
        paths=replace(
            source.paths,
            mujoco_xml=xml,
            artifact_root=tmp_path / "artifacts",
        ),
    )

    class CompletingEnvironment(_FakeEnvironment):
        def step(self, action):
            self.references.append(self.pos_des.copy())
            self.steps += 1
            return self.observation(), 0.0, False, self.steps == 3, {}

    class FakePolicy:
        @staticmethod
        def predict(_observation, deterministic=True):
            assert deterministic is True
            return np.zeros(4), None

    env = CompletingEnvironment()
    factory = _FakeFactory(env)
    eval_cli._ensure_runtime_imports()
    monkeypatch.setattr(eval_cli, "load_config", lambda _path: config)
    original_runner = eval_cli.EvaluationRunner
    monkeypatch.setattr(
        eval_cli,
        "EvaluationRunner",
        lambda *args, **kwargs: original_runner(
            *args, **kwargs, environment_factory=factory
        ),
    )
    monkeypatch.setattr(eval_cli, "_load_policy", lambda _path, _config: FakePolicy())

    def fake_hover_plot(path, **_kwargs):
        path.write_bytes(b"fake-png")
        return path

    monkeypatch.setattr(eval_cli, "save_hover_trace", fake_hover_plot)

    assert run_evaluation_cli(
        default_config=source.source_path,
        description="test",
        argv=[
            "--mode",
            "hover",
            "--duration",
            "0.3",
            "--policy",
            "residual",
            "--model",
            str(model_argument),
            "--headless",
            "--no-realtime",
        ],
        unified=True,
        mode_profiles={"hover": CONFIGS / "view_live_hover_eval.yaml"},
    ) == 0

    run_dir = next((tmp_path / "artifacts" / "runs").iterdir())
    metrics = json.loads(
        next((run_dir / "metrics").glob("*.json")).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        next((run_dir / "manifests").glob("*manifest*.json")).read_text(
            encoding="utf-8"
        )
    )
    actual = str(model_archive.resolve())
    assert metrics["model"] == actual
    assert metrics["runtime_parameters"]["model"] == actual
    assert manifest["runtime_config"]["parameters"]["model"] == actual
    assert manifest["result"]["input_model"] == actual
    assert actual in capsys.readouterr().out


def test_preflight_resource_failure_writes_runtime_metrics_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = load_config(CONFIGS / "view_live_hover_eval.yaml")
    config = replace(
        source,
        paths=replace(
            source.paths,
            mujoco_xml=tmp_path / "missing.xml",
            artifact_root=tmp_path / "artifacts",
        ),
    )
    eval_cli._ensure_runtime_imports()
    monkeypatch.setattr(eval_cli, "load_config", lambda _path: config)

    with pytest.raises(FileNotFoundError, match="Configured MuJoCo XML"):
        run_evaluation_cli(
            default_config=source.source_path,
            description="test",
            argv=[
                "--mode",
                "hover",
                "--policy",
                "floor",
                "--headless",
                "--no-realtime",
            ],
            unified=True,
            mode_profiles={"hover": CONFIGS / "view_live_hover_eval.yaml"},
        )

    run_dir = next((tmp_path / "artifacts" / "runs").iterdir())
    assert len(list((run_dir / "config").glob("*runtime-resolved*.yaml"))) == 1
    metrics = json.loads(
        next((run_dir / "metrics").glob("*.json")).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        next((run_dir / "manifests").glob("*manifest*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert metrics["status"] == "failed"
    assert "Configured MuJoCo XML" in metrics["error"]
    assert manifest["status"] == "failed"
    assert "Configured MuJoCo XML" in manifest["result"]["error"]


def test_hover_and_tracking_plots_render_with_custom_ranges(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from crazyflie_rl.plotting import save_hover_trace, save_tracking_trace

    time_sec = np.linspace(0.0, 1.0, 6)
    reference = np.column_stack(
        (0.8 * time_sec, 0.4 * np.sin(2.0 * np.pi * time_sec), np.full(6, 2.5))
    )
    position = reference + np.array([0.02, -0.01, 0.03])
    attitude = np.column_stack((30.0 * time_sec, -25.0 * time_sec, 90.0 * time_sec))
    error = np.linalg.norm(position - reference, axis=1)

    hover_path = save_hover_trace(
        tmp_path / "hover.png",
        tag="hover x0-y0-z2p5",
        time_sec=time_sec,
        position=position,
        attitude_deg=attitude,
        hover_altitude=2.5,
        reference_position=reference,
        position_error=error,
    )
    tracking_path = save_tracking_trace(
        tmp_path / "lissajous.png",
        tag="lissajous Ax0p8-Ay0p4",
        time_sec=time_sec,
        position=position,
        attitude_deg=attitude,
        reference_position=reference,
        position_error=error,
        mission_name="lissajous",
        mission_parameters={"center_xy": (0.0, 0.0)},
        phases=("GOTO", "GOTO", "LISSAJOUS", "LISSAJOUS", "HOLD", "HOLD"),
    )

    assert hover_path.read_bytes().startswith(b"\x89PNG")
    assert tracking_path.read_bytes().startswith(b"\x89PNG")

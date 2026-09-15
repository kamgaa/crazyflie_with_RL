from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from crazyflie_rl.attitude import (
    ATTITUDE_AXIS_CHOICES,
    attitude_axis_vector,
    named_attitude_quaternion_wxyz,
)
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
    resolve_path_preset,
    run_evaluation_cli,
    trace_metrics,
)
from crazyflie_rl.missions import mission_from_experiment


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PATH_PROFILES = {
    "circle": {
        "small": CONFIGS / "view_live_circle_eval.yaml",
        "wide": CONFIGS / "view_live_circle_wide_eval.yaml",
    },
    "lissajous": {
        "figure8": CONFIGS / "view_live_lissajous_eval.yaml",
        "clover": CONFIGS / "view_live_lissajous_clover_eval.yaml",
    },
}


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
    assert "--path-preset" in completed.stdout
    assert "--period" in completed.stdout
    assert "--amplitude-x" in completed.stdout
    assert "latest-best" in completed.stdout


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
        "path_preset",
        "period",
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
        "attitude_axis",
        "force_floor_start",
    } <= destinations
    assert parser.parse_args(["--model", "latest-best"]).model == Path(
        "latest-best"
    )
    assert parser.parse_args(["--policy", "ppo"]).policy == "ppo"


@pytest.mark.parametrize("attitude_axis", ATTITUDE_AXIS_CHOICES)
def test_parser_accepts_each_attitude_axis(attitude_axis: str) -> None:
    parser = build_parser(CONFIGS / "e2e_hover_eval.yaml", "test")

    args = parser.parse_args(["--attitude-axis", attitude_axis])

    assert args.attitude_axis == attitude_axis


def test_parser_rejects_invalid_attitude_axis() -> None:
    parser = build_parser(CONFIGS / "e2e_hover_eval.yaml", "test")

    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["--attitude-axis", "yaw_plus"])

    assert caught.value.code == 2


@pytest.mark.parametrize("attitude_axis", ATTITUDE_AXIS_CHOICES)
def test_named_attitude_axes_are_unit_vectors(attitude_axis: str) -> None:
    assert np.linalg.norm(attitude_axis_vector(attitude_axis)) == pytest.approx(1.0)


def test_roll_minus_twenty_degree_quaternion_uses_wxyz_convention() -> None:
    quaternion = named_attitude_quaternion_wxyz("roll_minus", 20.0)

    np.testing.assert_allclose(
        quaternion,
        [np.cos(np.deg2rad(10.0)), -np.sin(np.deg2rad(10.0)), 0.0, 0.0],
        rtol=0.0,
        atol=1e-15,
    )
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    assert quaternion[0] >= 0.0


@pytest.mark.parametrize(
    ("attitude_axis", "expected"),
    [
        ("diagonal_pp", [1.0, 1.0, 0.0]),
        ("diagonal_pm", [1.0, -1.0, 0.0]),
        ("diagonal_mp", [-1.0, 1.0, 0.0]),
        ("diagonal_mm", [-1.0, -1.0, 0.0]),
    ],
)
def test_diagonal_attitude_axes_are_normalized(
    attitude_axis: str, expected: list[float]
) -> None:
    np.testing.assert_allclose(
        attitude_axis_vector(attitude_axis),
        np.asarray(expected, dtype=float) / np.sqrt(2.0),
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize("attitude_axis", ATTITUDE_AXIS_CHOICES)
def test_zero_degree_perturbation_is_identity_for_every_axis(
    attitude_axis: str,
) -> None:
    assert named_attitude_quaternion_wxyz(attitude_axis, 0.0) == (
        1.0,
        0.0,
        0.0,
        0.0,
    )


def test_omitting_attitude_axis_preserves_legacy_parser_default() -> None:
    args = build_parser(CONFIGS / "e2e_hover_eval.yaml", "test").parse_args(
        ["--attitude-perturbation-deg", "20"]
    )

    assert args.attitude_axis is None


@pytest.mark.parametrize(
    ("name", "policy", "position", "attitude", "model"),
    [
        ("recovery-ppo-nominal", "ppo", 0.0, 0.0, "latest-final"),
        ("recovery-ppo-tilt30", "ppo", 0.0, 30.0, "latest-final"),
        ("recovery-pid-tilt30", "floor", 0.0, 30.0, None),
        ("recovery-both-tilt30", "both", 0.0, 30.0, "latest-final"),
        ("recovery-ppo-position20cm", "ppo", 0.20, 0.0, "latest-final"),
    ],
)
def test_named_evaluation_preset_contracts(
    name: str,
    policy: str,
    position: float,
    attitude: float,
    model: str | None,
) -> None:
    raw = ["--eval-preset", name]
    args = build_parser(CONFIGS / "e2e_hover_eval.yaml", "test").parse_args(raw)
    merged = eval_cli._apply_evaluation_preset(args, raw)

    assert merged.config == CONFIGS / "e2e_train_initial_perturb.yaml"
    assert merged.mode == "hover"
    assert merged.policy == policy
    assert merged.duration == 8.0
    assert merged.position_perturbation == pytest.approx(position)
    assert merged.attitude_perturbation_deg == pytest.approx(attitude)
    assert merged.seed == 1000
    assert merged.force_floor_start is False
    assert merged.headless is False
    assert merged.no_realtime is False
    assert merged.no_camera is False
    assert merged.model == (Path(model) if model is not None else None)


def test_explicit_cli_values_override_named_evaluation_preset(tmp_path: Path) -> None:
    explicit_model = tmp_path / "manual.zip"
    raw = [
        "--eval-preset",
        "recovery-ppo-tilt30",
        "--config",
        str(CONFIGS / "e2e_hover_eval.yaml"),
        "--mode",
        "circle",
        "--policy",
        "floor",
        "--duration",
        "2",
        "--position-perturbation",
        "0.07",
        "--attitude-perturbation-deg",
        "12",
        "--seed",
        "9",
        "--model",
        str(explicit_model),
        "--force-floor-start",
        "--headless",
        "--no-realtime",
        "--no-camera",
    ]
    args = build_parser(CONFIGS / "e2e_hover_eval.yaml", "test").parse_args(raw)
    merged = eval_cli._apply_evaluation_preset(args, raw)

    assert merged.config == CONFIGS / "e2e_hover_eval.yaml"
    assert merged.mode == "circle"
    assert merged.policy == "floor"
    assert merged.duration == 2.0
    assert merged.position_perturbation == pytest.approx(0.07)
    assert merged.attitude_perturbation_deg == pytest.approx(12.0)
    assert merged.seed == 9
    assert merged.model == explicit_model
    assert merged.force_floor_start is True
    assert merged.headless is True
    assert merged.no_realtime is True
    assert merged.no_camera is True


def test_list_presets_prints_names_without_mode_or_runtime_imports() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "view_live.py"), "--list-presets"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    for name in eval_cli.EVALUATION_PRESETS:
        assert name in completed.stdout


def test_view_live_attitude_axis_headless_cli_smoke(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    yaml = pytest.importorskip("yaml")

    source = load_config(CONFIGS / "e2e_train_legacy_initial_perturb_v2.yaml")
    artifact_root = tmp_path / "artifacts"
    smoke_config = tmp_path / "smoke.yaml"
    smoke_config.write_text(
        yaml.safe_dump(
            {
                "extends": os.path.relpath(source.source_path, tmp_path),
                "paths": {
                    "mujoco_xml": str(source.paths.mujoco_xml),
                    "project_root": str(source.paths.project_root),
                    "artifact_root": str(artifact_root),
                    "legacy_model_root": str(source.paths.legacy_model_root),
                    "legacy_tensorboard_root": str(
                        source.paths.legacy_tensorboard_root
                    ),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "view_live.py"),
            "--config",
            str(smoke_config),
            "--mode",
            "hover",
            "--policy",
            "floor",
            "--duration",
            "0.01",
            "--position-perturbation",
            "0",
            "--attitude-perturbation-deg",
            "20",
            "--attitude-axis",
            "roll_minus",
            "--headless",
            "--no-realtime",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Attitude perturb. : 20 deg" in completed.stdout
    assert "Attitude axis     : roll_minus" in completed.stdout
    assert "Initial axis      : [-1, 0, 0]" in completed.stdout
    assert "Initial quaternion:" in completed.stdout

    run_dir = next((artifact_root / "runs").iterdir())
    metrics = json.loads(
        next((run_dir / "metrics").glob("*.json")).read_text(encoding="utf-8")
    )
    assert metrics["attitude_axis"] == "roll_minus"
    assert metrics["initial_axis_xyz"] == pytest.approx([-1.0, 0.0, 0.0])
    assert metrics["initial_quaternion_wxyz"] == pytest.approx(
        named_attitude_quaternion_wxyz("roll_minus", 20.0)
    )
    floor_metrics = metrics["policies"]["floor"]
    assert floor_metrics["attitude_axis"] == "roll_minus"
    assert floor_metrics["trace_csv"].endswith(".csv")

    trace_csv = run_dir / floor_metrics["trace_csv"]
    with trace_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["attitude_axis"] == "roll_minus"
    assert float(rows[0]["initial_axis_x"]) == pytest.approx(-1.0)
    assert float(rows[0]["initial_quaternion_w"]) == pytest.approx(
        np.cos(np.deg2rad(10.0))
    )


@pytest.mark.parametrize(
    ("selected", "expected_keys", "expected_modes"),
    [
        ("floor", ["floor"], {"floor": "residual"}),
        ("residual", ["residual"], {"residual": "e2e"}),
        (
            "both",
            ["floor", "residual"],
            {"floor": "residual", "residual": "e2e"},
        ),
    ],
)
def test_initial_perturb_profile_honors_unified_policy_selection(
    selected: str,
    expected_keys: list[str],
    expected_modes: dict[str, str],
) -> None:
    config = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")
    assert [
        key for key, _label in eval_cli._policy_specs(config, selected, unified=True)
    ] == expected_keys
    assert eval_cli._expected_policy_control_modes(
        config, selected, unified=True
    ) == expected_modes


def test_viewer_immutably_disables_curriculum_but_keeps_cli_perturbations() -> None:
    source = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")
    args = build_parser(source.source_path, "test").parse_args(
        [
            "--position-perturbation",
            "0.12",
            "--attitude-perturbation-deg",
            "30",
        ]
    )
    overridden = apply_runtime_overrides(source, args, "hover")
    effective = eval_cli._disable_training_initial_state_randomization(overridden)

    assert source.environment.initial_state_randomization.enabled is True
    assert overridden.environment.initial_state_randomization.enabled is True
    assert effective.environment.initial_state_randomization.enabled is False
    assert effective.environment.position_perturbation == pytest.approx(0.12)
    assert effective.environment.attitude_perturbation_deg == pytest.approx(30.0)


def test_explicit_thirty_degree_viewer_perturbation_has_exact_total_tilt() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from crazyflie_rl.factories import EnvironmentFactory

    source = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")
    args = build_parser(source.source_path, "test").parse_args(
        ["--position-perturbation", "0", "--attitude-perturbation-deg", "30"]
    )
    effective = eval_cli._disable_training_initial_state_randomization(
        apply_runtime_overrides(source, args, "hover")
    )
    environment = EnvironmentFactory(effective).make(
        seed=1000,
        initial_state_randomization_enabled=False,
        exact_attitude_perturbation=True,
    )
    try:
        observation, _info = environment.reset(seed=1000)
        quaternion = np.asarray(observation[6:10], dtype=float)
        quaternion /= np.linalg.norm(quaternion)
        tilt_deg = np.degrees(
            np.arccos(
                np.clip(
                    1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2),
                    -1.0,
                    1.0,
                )
            )
        )
        assert tilt_deg == pytest.approx(30.0, abs=1e-5)
    finally:
        environment.close()


def test_unified_floor_and_e2e_start_from_identical_nominal_hover_state() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from crazyflie_rl.factories import EnvironmentFactory

    source = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")
    args = build_parser(source.source_path, "test").parse_args(
        ["--position-perturbation", "0", "--attitude-perturbation-deg", "0"]
    )
    effective = eval_cli._disable_training_initial_state_randomization(
        apply_runtime_overrides(source, args, "hover")
    )
    factory = EnvironmentFactory(effective)
    floor = factory.make(
        seed=1000,
        mode="residual",
        initial_state_randomization_enabled=False,
        exact_attitude_perturbation=True,
    )
    policy = factory.make(
        seed=1000,
        initial_state_randomization_enabled=False,
        exact_attitude_perturbation=True,
    )
    try:
        floor_observation, _floor_info = floor.reset(seed=1000)
        policy_observation, _policy_info = policy.reset(seed=1000)
        np.testing.assert_array_equal(floor_observation, policy_observation)
        np.testing.assert_array_equal(floor.data.qpos, policy.data.qpos)
        np.testing.assert_array_equal(floor.data.qvel, policy.data.qvel)
        np.testing.assert_array_equal(floor.data.xpos, policy.data.xpos)
        np.testing.assert_array_equal(floor.data.xquat, policy.data.xquat)
        np.testing.assert_array_equal(floor.data.sensordata, policy.data.sensordata)
        np.testing.assert_array_equal(floor._last_f_cmd, policy._last_f_cmd)
        np.testing.assert_array_equal(floor._last_f, policy._last_f)
        np.testing.assert_array_equal(floor._last_omega, policy._last_omega)
        np.testing.assert_array_equal(floor._actuator.omega, policy._actuator.omega)
        np.testing.assert_array_equal(floor.data.qpos[0:3], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(floor.data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(floor.data.qvel, np.zeros_like(floor.data.qvel))
        hover_thrust = effective.vehicle.mass * effective.vehicle.gravity / 4.0
        np.testing.assert_allclose(floor._last_f_cmd, hover_thrust, atol=1e-12)
        np.testing.assert_allclose(floor._last_f, hover_thrust, atol=1e-12)
    finally:
        floor.close()
        policy.close()


def test_omitted_axis_preserves_the_seeded_legacy_reset_result_exactly() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from crazyflie_rl.factories import EnvironmentFactory

    source = load_config(CONFIGS / "e2e_train_legacy_initial_perturb_v2.yaml")
    config = replace(
        source,
        environment=replace(
            source.environment,
            episode_sec=0.01,
            position_perturbation=0.05,
            attitude_perturbation_deg=20.0,
        ),
        mission=replace(
            source.mission,
            hover=replace(source.mission.hover, duration=0.01),
        ),
    )
    legacy_environment = EnvironmentFactory(config).make(
        initial_state_randomization_enabled=False,
        exact_attitude_perturbation=True,
    )
    try:
        expected_observation, _info = legacy_environment.reset(
            seed=config.evaluation.seed_start
        )
    finally:
        legacy_environment.close()

    policy_observations: list[np.ndarray] = []

    class CapturingPolicy:
        @staticmethod
        def predict(observation, deterministic=True):
            assert deterministic is True
            policy_observations.append(np.asarray(observation).copy())
            return np.zeros(4, dtype=np.float32), None

    runner = EvaluationRunner(
        config,
        SimpleNamespace(),
        headless=True,
        realtime=False,
        camera_tracking=False,
        mission=mission_from_experiment(config, mission_type="hover"),
        disable_initial_state_randomization=True,
        exact_attitude_perturbation=True,
    )

    trace = runner.run(CapturingPolicy(), "residual", "PPO")

    np.testing.assert_array_equal(policy_observations[0], expected_observation)
    assert trace.attitude_axis is None
    assert trace.initial_quaternion_wxyz is None


def test_floor_and_ppo_share_selected_initial_attitude_and_actuator_state() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")

    source = load_config(CONFIGS / "e2e_train_legacy_initial_perturb_v2.yaml")
    config = replace(
        source,
        environment=replace(
            source.environment,
            episode_sec=0.01,
            position_perturbation=0.0,
            attitude_perturbation_deg=20.0,
        ),
        mission=replace(
            source.mission,
            hover=replace(source.mission.hover, duration=0.01),
        ),
    )
    mission = mission_from_experiment(config, mission_type="hover")
    runner = EvaluationRunner(
        config,
        SimpleNamespace(),
        headless=True,
        realtime=False,
        camera_tracking=False,
        mission=mission,
        disable_initial_state_randomization=True,
        exact_attitude_perturbation=True,
        attitude_axis="roll_minus",
    )

    class ZeroPolicy:
        @staticmethod
        def predict(_observation, deterministic=True):
            assert deterministic is True
            return np.zeros(4, dtype=np.float32), None

    floor = runner.run(None, "floor", "PID floor", control_mode="residual")
    ppo = runner.run(ZeroPolicy(), "residual", "PPO")

    expected_quaternion = named_attitude_quaternion_wxyz("roll_minus", 20.0)
    np.testing.assert_array_equal(
        floor.initial_position_xyz_m, ppo.initial_position_xyz_m
    )
    np.testing.assert_array_equal(
        floor.initial_quaternion_wxyz, ppo.initial_quaternion_wxyz
    )
    np.testing.assert_allclose(
        floor.initial_quaternion_wxyz,
        expected_quaternion,
        rtol=0.0,
        atol=1e-15,
    )
    assert floor.initial_actuator_state == ppo.initial_actuator_state
    assert floor.actuator == ppo.actuator
    assert floor.initial_axis_xyz == pytest.approx([-1.0, 0.0, 0.0])
    assert ppo.initial_axis_xyz == pytest.approx([-1.0, 0.0, 0.0])


def _write_saved_best_model(
    artifact_root: Path,
    *,
    run_name: str,
    control_mode: str,
    model_name: str,
    modified_ns: int,
) -> Path:
    """Create the minimal completed-artifact shape used by latest-best."""

    run_dir = artifact_root / "runs" / run_name
    model_path = run_dir / "models" / model_name
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    os.utime(model_path, ns=(modified_ns, modified_ns))
    manifest_path = run_dir / "manifests" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "control_mode": control_mode,
                "models": {
                    "best": {
                        "path": model_path.relative_to(run_dir).as_posix(),
                    },
                    "final": None,
                },
            }
        ),
        encoding="utf-8",
    )
    return model_path.resolve()


def test_latest_best_selects_newest_matching_control_mode(tmp_path: Path) -> None:
    residual_source = load_config(CONFIGS / "residual_train.yaml")
    e2e_source = load_config(CONFIGS / "e2e_train.yaml")
    artifact_root = tmp_path / "artifacts"
    residual_config = replace(
        residual_source,
        paths=replace(residual_source.paths, artifact_root=artifact_root),
    )
    e2e_config = replace(
        e2e_source,
        paths=replace(e2e_source.paths, artifact_root=artifact_root),
    )

    older_residual = _write_saved_best_model(
        artifact_root,
        run_name="ppo_residual_hover_old",
        control_mode="residual",
        model_name="old_best.zip",
        modified_ns=1_000_000_000,
    )
    newest_residual = _write_saved_best_model(
        artifact_root,
        run_name="ppo_residual_hover_new",
        control_mode="residual",
        model_name="new_best.zip",
        modified_ns=2_000_000_000,
    )
    newest_e2e = _write_saved_best_model(
        artifact_root,
        run_name="ppo_e2e_hover_newest",
        control_mode="e2e",
        model_name="e2e_best.zip",
        modified_ns=3_000_000_000,
    )

    assert eval_cli._model_path("latest-best", residual_config) == newest_residual
    assert eval_cli._model_path(Path("LATEST-BEST"), e2e_config) == newest_e2e
    assert eval_cli._model_path("latest-best", residual_config) != older_residual


def test_latest_best_never_falls_back_to_final_model(tmp_path: Path) -> None:
    source = load_config(CONFIGS / "residual_train.yaml")
    artifact_root = tmp_path / "artifacts"
    config = replace(
        source,
        paths=replace(source.paths, artifact_root=artifact_root),
    )
    run_dir = artifact_root / "runs" / "ppo_residual_final_only"
    final_model = run_dir / "models" / "final.zip"
    final_model.parent.mkdir(parents=True)
    final_model.write_bytes(b"final")
    manifest_path = run_dir / "manifests" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "control_mode": "residual",
                "models": {
                    "best": None,
                    "final": {"path": "models/final.zip"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="No saved best PPO model"):
        eval_cli._model_path("latest-best", config)


def _write_final_manifest(
    artifact_root: Path,
    *,
    run_name: str,
    profile: str,
    created_at: str,
    command: str = "train_ppo_02.py",
    control_mode: str = "e2e",
    observation_shape: list[int] | None = None,
    action_shape: list[int] | None = None,
) -> Path:
    run_dir = artifact_root / "runs" / run_name
    model_path = run_dir / "models" / "final.zip"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"final-model")
    manifest_path = run_dir / "manifests" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_name,
                "status": "completed",
                "created_at": created_at,
                "command": [command, "--config", f"configs/{profile}.yaml"],
                "config_profile": profile,
                "control_mode": control_mode,
                "observation_shape": observation_shape or [15],
                "action_shape": action_shape or [4],
                "models": {
                    "best": None,
                    "final": {
                        "kind": "final",
                        "path": "models/final.zip",
                        "timestep": 1_001_472,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return model_path.resolve()


def test_latest_final_requires_training_profile_mode_shape_and_final_record(
    tmp_path: Path,
) -> None:
    source = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")
    artifact_root = tmp_path / "artifacts"
    config = replace(source, paths=replace(source.paths, artifact_root=artifact_root))
    older = _write_final_manifest(
        artifact_root,
        run_name="matching-old",
        profile=source.profile_name,
        created_at="2026-09-03T12:00:00+09:00",
    )
    newer = _write_final_manifest(
        artifact_root,
        run_name="matching-new",
        profile=source.profile_name,
        created_at="2026-09-03T13:00:00+09:00",
    )
    _write_final_manifest(
        artifact_root,
        run_name="recovery-newest",
        profile=source.profile_name,
        created_at="2026-09-03T16:00:00+09:00",
        command="evaluate_recovery.py",
    )
    _write_final_manifest(
        artifact_root,
        run_name="wrong-profile",
        profile="e2e_train_lyapunov",
        created_at="2026-09-03T15:00:00+09:00",
    )
    _write_final_manifest(
        artifact_root,
        run_name="wrong-shape",
        profile=source.profile_name,
        created_at="2026-09-03T14:00:00+09:00",
        observation_shape=[13],
    )

    selected, provenance = eval_cli._latest_final_model_selection(config)

    assert selected == newer
    assert selected != older
    assert provenance["config_profile"] == source.profile_name
    assert provenance["control_mode"] == "e2e"
    assert provenance["observation_shape"] == [15]
    assert provenance["action_shape"] == [4]
    assert provenance["final_timestep"] == 1_001_472


def test_latest_final_refuses_ambiguous_newest_provenance(tmp_path: Path) -> None:
    source = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")
    artifact_root = tmp_path / "artifacts"
    config = replace(source, paths=replace(source.paths, artifact_root=artifact_root))
    first = _write_final_manifest(
        artifact_root,
        run_name="same-time-a",
        profile=source.profile_name,
        created_at="2026-09-03T13:00:00+09:00",
    )
    second = _write_final_manifest(
        artifact_root,
        run_name="same-time-b",
        profile=source.profile_name,
        created_at="2026-09-03T13:00:00+09:00",
    )

    with pytest.raises(RuntimeError, match="Pass an explicit --model path") as caught:
        eval_cli._latest_final_model_selection(config)
    assert str(first) in str(caught.value)
    assert str(second) in str(caught.value)


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


@pytest.mark.parametrize(
    ("mode", "selector", "key", "profile"),
    [
        ("circle", "1", "small", "view_live_circle_eval.yaml"),
        ("circle", "circle-wide", "wide", "view_live_circle_wide_eval.yaml"),
        (
            "lissajous",
            "1",
            "figure8",
            "view_live_lissajous_eval.yaml",
        ),
        (
            "lissajous",
            "clover",
            "clover",
            "view_live_lissajous_clover_eval.yaml",
        ),
    ],
)
def test_path_presets_accept_numeric_and_named_keys(
    mode: str, selector: str, key: str, profile: str
) -> None:
    resolved = resolve_path_preset(mode, selector, PATH_PROFILES)
    assert resolved is not None
    assert resolved[0] == key
    assert resolved[1].name == profile


def test_predefined_profiles_keep_geometry_out_of_the_basic_cli() -> None:
    small = load_config(CONFIGS / "view_live_circle_eval.yaml")
    wide = load_config(CONFIGS / "view_live_circle_wide_eval.yaml")
    figure8 = load_config(CONFIGS / "view_live_lissajous_eval.yaml")
    clover = load_config(CONFIGS / "view_live_lissajous_clover_eval.yaml")

    assert small.mission.circle.radius == 0.5
    assert wide.mission.circle.radius == 0.8
    assert wide.mission.circle.center_xy == pytest.approx((0.0, 0.0))
    assert figure8.mission.lissajous.frequency_ratio == (1, 2)
    assert clover.mission.lissajous.frequency_ratio == (3, 2)
    assert clover.mission.lissajous.amplitude_xy == pytest.approx((0.55, 0.4))


@pytest.mark.parametrize(
    ("answers", "expected_mode", "expected_key", "expected_prompts"),
    [
        (
            ["1", "5"],
            "hover",
            None,
            ["Select mode [1-3]: ", "Hover duration [8.0 s]: "],
        ),
        (
            ["2", "2", "7", "3"],
            "circle",
            "wide",
            [
                "Select mode [1-3]: ",
                "Select path preset [small]: ",
                "Seconds per lap [10.0 s]: ",
                "Number of laps [2.0]: ",
            ],
        ),
        (
            ["3", "2", "8", "4"],
            "lissajous",
            "clover",
            [
                "Select mode [1-3]: ",
                "Select path preset [figure8]: ",
                "Seconds per cycle [10.0 s]: ",
                "Number of cycles [2.0]: ",
            ],
        ),
    ],
)
def test_interactive_flow_only_asks_mode_path_speed_and_repetitions(
    answers: list[str],
    expected_mode: str,
    expected_key: str | None,
    expected_prompts: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    remaining = iter(answers)
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(remaining)

    mode = eval_cli._interactive_mode(answer)
    key = eval_cli._interactive_path_preset(mode, PATH_PROFILES, answer)
    profile = (
        CONFIGS / "view_live_hover_eval.yaml"
        if key is None
        else resolve_path_preset(mode, key, PATH_PROFILES)[1]
    )
    config = load_config(profile)
    args = build_parser(profile, "test").parse_args([])
    args.path_preset = key
    prompted = prompt_runtime_options(args, config, mode, input_fn=answer)

    assert mode == expected_mode
    assert key == expected_key
    assert prompts == expected_prompts
    assert next(remaining, None) is None
    assert prompted.policy == "both"
    capsys.readouterr()


def test_explicit_path_preset_is_the_tty_default_after_mode_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = build_parser(CONFIGS / "view_live_hover_eval.yaml", "test").parse_args(
        ["--path-preset", "wide"]
    )
    answers = iter(("2", ""))
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    mode = eval_cli._interactive_mode(answer)
    selected = eval_cli._interactive_path_preset(
        mode,
        PATH_PROFILES,
        answer,
        default=args.path_preset,
    )

    assert mode == "circle"
    assert selected == "wide"
    assert prompts == [
        "Select mode [1-3]: ",
        "Select path preset [wide]: ",
    ]
    capsys.readouterr()


def test_compact_condition_only_contains_path_speed_and_repetitions() -> None:
    config = load_config(CONFIGS / "view_live_circle_wide_eval.yaml")
    parser = build_parser(config.source_path, "test")
    args = parser.parse_args(
        ["--period", "7.5", "--laps", "3", "--altitude", "1.7"]
    )
    effective = apply_runtime_overrides(config, args, "circle")
    mission = mission_from_experiment(effective, mission_type="circle")

    assert (
        mission_condition(effective, mission, path_preset="wide")
        == "c-wide-T7p5-L3"
    )


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
        unified=True,
    )
    for expected in (
        "Mode              : Lissajous",
        "Floor controller  : RESIDUAL (PID)",
        "PPO controller    : E2E",
        "Selected controls : PID floor=RESIDUAL, learned PPO=E2E",
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


def test_summary_includes_selected_axis_vector_and_initial_quaternion() -> None:
    source = load_config(CONFIGS / "view_live_hover_eval.yaml")
    config = replace(
        source,
        environment=replace(source.environment, attitude_perturbation_deg=20.0),
    )
    mission = mission_from_experiment(config, mission_type="hover")

    summary = format_run_summary(
        config,
        mission,
        policy="floor",
        model=None,
        unified=True,
        attitude_axis="roll_minus",
    )

    assert "Attitude perturb. : 20 deg" in summary
    assert "Attitude axis     : roll_minus" in summary
    assert "Initial axis      : [-1, 0, 0]" in summary
    assert "Initial quaternion: [0.984808, -0.173648, 0, 0]" in summary


def test_plot_subtitle_records_selected_attitude_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIGS / "view_live_hover_eval.yaml")
    mission = mission_from_experiment(config, mission_type="hover")
    trace = RolloutTrace(
        policy="floor",
        label="PID floor",
        time_sec=np.array([0.0]),
        position=np.array([[0.0, 0.0, 1.0]]),
        attitude_deg=np.array([[-20.0, 0.0, 0.0]]),
        reference_position=np.array([[0.0, 0.0, 1.0]]),
        position_error=np.array([0.0]),
        phases=("HOVER",),
        training_boundary_crossed_at=None,
        guard_boundary_crossed_at=None,
        terminated_at=None,
        truncated_at=None,
        diverged_at=None,
        attitude_axis="roll_minus",
        initial_axis_xyz=np.array([-1.0, 0.0, 0.0]),
        initial_position_xyz_m=np.array([0.0, 0.0, 1.0]),
        initial_quaternion_wxyz=np.asarray(
            named_attitude_quaternion_wxyz("roll_minus", 20.0)
        ),
    )
    recorded: dict[str, str] = {}

    def fake_hover_plot(path, **kwargs):
        recorded["single_tag"] = kwargs["tag"]
        return path

    def fake_policy_plot(path, **kwargs):
        recorded["comparison_tag"] = kwargs["tag"]
        return path

    eval_cli._ensure_runtime_imports()
    monkeypatch.setattr(eval_cli, "save_hover_trace", fake_hover_plot)
    monkeypatch.setattr(eval_cli, "save_policy_trace", fake_policy_plot)
    artifacts = SimpleNamespace(
        path=lambda _group, _kind, _suffix: tmp_path / "trace.png",
        run_dir=tmp_path,
    )

    eval_cli._save_trace(
        artifacts,
        config,
        trace,
        mission,
        title_condition="hover-test",
    )
    eval_cli._save_policy_report(
        artifacts,
        trace,
        mission,
        title_condition="hover-test",
    )

    assert "Attitude axis: roll_minus" in recorded["single_tag"]
    assert "Attitude axis: roll_minus" in recorded["comparison_tag"]


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
    assert prompts == ["Hover duration [8.0 s]: "]


def test_interactive_hover_only_asks_duration_and_keeps_advanced_cli_values() -> None:
    config = load_config(CONFIGS / "view_live_hover_eval.yaml")
    args = build_parser(config.source_path, "test").parse_args(
        [
            "--hover-target",
            "0.3",
            "0",
            "1.2",
            "--yaw-deg",
            "45",
            "--seed",
            "9",
            "--position-perturbation",
            "0.1",
            "--attitude-perturbation-deg",
            "2",
            "--policy",
            "floor",
            "--headless",
            "--no-realtime",
        ]
    )
    prompts: list[str] = []
    prompted = prompt_runtime_options(
        args,
        config,
        "hover",
        input_fn=lambda prompt: prompts.append(prompt) or "5",
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
    assert prompts == ["Hover duration [8.0 s]: "]


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
    assert prompts == ["Seconds per lap [7.0 s]: ", "Number of laps [1.5]: "]


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
        self._last_f = np.full(4, 0.01)

    @staticmethod
    def observation() -> np.ndarray:
        result = np.zeros(15)
        result[3:6] = [0.1, -0.2, 0.3]
        result[6] = 1.0
        result[10:13] = [1.0, -2.0, 3.0]
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
        self.overrides: list[dict[str, object]] = []

    def make(self, **overrides):
        self.overrides.append(dict(overrides))
        if "mode" in overrides:
            self.env.mode = overrides["mode"]
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
    np.testing.assert_allclose(trace.control_input, np.zeros((expected_steps, 4)))
    np.testing.assert_allclose(trace.motor_thrust, np.full((expected_steps, 4), 0.01))
    np.testing.assert_allclose(
        trace.linear_velocity,
        np.tile([0.1, -0.2, 0.3], (expected_steps, 1)),
    )
    np.testing.assert_allclose(
        trace.angular_velocity,
        np.tile([1.0, -2.0, 3.0], (expected_steps, 1)),
    )
    assert trace.control_mode == config.control_mode
    assert runner.factory.overrides == [{}]
    assert env.closed is True


def test_omitted_attitude_axis_does_not_touch_the_legacy_reset_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIGS / "view_live_hover_eval.yaml")
    env = _FakeEnvironment()
    received_observations: list[np.ndarray] = []

    class CapturingPolicy:
        @staticmethod
        def predict(observation, deterministic=True):
            assert deterministic is True
            received_observations.append(np.asarray(observation).copy())
            return np.zeros(4), None

    monkeypatch.setattr(
        EvaluationRunner,
        "_apply_initial_attitude_axis",
        staticmethod(
            lambda *_args, **_kwargs: pytest.fail(
                "legacy reset must not be rewritten when --attitude-axis is omitted"
            )
        ),
    )
    runner = EvaluationRunner(
        config,
        SimpleNamespace(),
        headless=True,
        realtime=False,
        camera_tracking=False,
        mission=_FakeMission("hover"),
        environment_factory=_FakeFactory(env),
    )

    trace = runner.run(CapturingPolicy(), "residual", "PPO")

    np.testing.assert_array_equal(received_observations[0], env.observation())
    assert trace.attitude_axis is None
    assert trace.initial_quaternion_wxyz is None


def test_runner_records_optional_bldc_signals_and_sampled_parameters() -> None:
    config = load_config(CONFIGS / "cf21b_actuator_eval.yaml")

    class ActuatorSignalEnvironment(_FakeEnvironment):
        wrench_command_reference = (
            "body frame; torque about the nominal allocator origin "
            "(not the payload-shifted combined CoM)"
        )

        def actuator_snapshot(self):
            return {
                "enabled": True,
                "model": "cf21b_first_order",
                "sampled_time_constant_s": [0.05] * 4,
                "sampled_steady_state_gain_rad_s": [2900.0] * 4,
            }

        def step(self, action):
            self._last_f_cmd = np.full(4, 0.11)
            self._last_f = np.full(4, 0.08)
            self._last_motor_cmd = np.full(4, 0.6)
            self._last_omega = np.full(4, 1740.0)
            self._last_q_actual = np.array([0.001, -0.001, 0.001, -0.001])
            self._last_wrench_cmd = np.array([0.0, 0.0, 0.0, 0.44])
            self._last_wrench_actual = np.array([0.0, 0.0, 0.0, 0.32])
            self._last_allocation_error = (
                self._last_wrench_cmd - self._last_wrench_actual
            )
            self.references.append(self.pos_des.copy())
            self.steps += 1
            return self.observation(), 0.0, False, self.steps == 2, {}

    env = ActuatorSignalEnvironment()
    runner = EvaluationRunner(
        config,
        SimpleNamespace(),
        headless=True,
        realtime=False,
        camera_tracking=False,
        mission=_FakeMission("hover", total_sec=0.2),
        environment_factory=_FakeFactory(env),
    )

    trace = runner.run(None, "floor", "floor")
    metrics = trace_metrics(trace, 0.3)

    np.testing.assert_allclose(trace.motor_thrust_command, 0.11)
    np.testing.assert_allclose(trace.motor_thrust, 0.08)
    np.testing.assert_allclose(trace.motor_command, 0.6)
    np.testing.assert_allclose(trace.motor_omega_rad_s, 1740.0)
    np.testing.assert_allclose(trace.reaction_torque_nm[0], [0.001, -0.001, 0.001, -0.001])
    assert trace.actuator is not None
    assert trace.actuator["sampled_time_constant_s"] == [0.05] * 4
    assert metrics["motor_thrust_command_n_mean"] == pytest.approx([0.11] * 4)
    assert metrics["motor_omega_rad_s_max"] == pytest.approx([1740.0] * 4)
    assert metrics["wrench_actual_final"] == pytest.approx([0.0, 0.0, 0.0, 0.32])
    assert metrics["wrench_command_final"] == pytest.approx(
        [0.0, 0.0, 0.0, 0.44]
    )
    assert metrics["wrench_command_order"] == [
        "tau_x_cmd",
        "tau_y_cmd",
        "tau_z_cmd",
        "Fz_cmd",
    ]
    assert metrics["wrench_command_units"] == ["N*m", "N*m", "N*m", "N"]
    assert metrics["wrench_command_stage"] == "before_allocation"
    assert "nominal allocator origin" in metrics["wrench_command_reference"]
    assert metrics["actuator"]["model"] == "cf21b_first_order"


def test_unified_residual_profile_keeps_both_rollouts_in_residual_mode() -> None:
    config = load_config(CONFIGS / "residual_hover_eval.yaml")

    class FreshFactory:
        def __init__(self) -> None:
            self.overrides: list[dict[str, object]] = []

        def make(self, **overrides):
            self.overrides.append(dict(overrides))
            env = _FakeEnvironment()
            env.mode = str(overrides.get("mode", config.control_mode))
            return env

    factory = FreshFactory()
    runner = EvaluationRunner(
        config,
        SimpleNamespace(),
        headless=True,
        realtime=False,
        camera_tracking=False,
        mission=_FakeMission("hover"),
        environment_factory=factory,
    )

    floor = runner.run(
        None,
        "floor",
        "floor (PID)",
        control_mode="residual",
    )
    ppo = runner.run(None, "residual", "residual (PID+RL)")

    assert factory.overrides == [{"mode": "residual"}, {}]
    assert floor.control_mode == "residual"
    assert ppo.control_mode == "residual"


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
        control_input=np.array(
            [[0.0, -0.2, 0.4, -0.6], [0.1, -0.3, 0.2, -0.7], [0.2, 0.1, -0.5, 0.8]]
        ),
        motor_thrust=np.array(
            [[0.01, 0.02, np.nan, 0.04], [0.02, 0.03, np.nan, 0.05], [0.03, 0.04, np.nan, 0.06]]
        ),
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
    assert metrics["control_input_abs_max"] == pytest.approx([0.2, 0.3, 0.5, 0.8])
    assert metrics["motor_thrust_n_min"] == [0.01, 0.02, None, 0.04]
    assert metrics["motor_thrust_n_max"] == [0.03, 0.04, None, 0.06]
    assert metrics["motor_thrust_n_mean"][0:2] == pytest.approx([0.02, 0.03])
    assert metrics["motor_thrust_n_mean"][2] is None
    assert metrics["motor_thrust_n_mean"][3] == pytest.approx(0.05)


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
    assert metrics["policies"]["floor"]["control_mode"] == "residual"
    assert metrics["policy_control_modes"] == {"floor": "residual"}
    assert metrics["policies"]["floor"]["plot"].startswith("plots/")
    assert factory.overrides == [
        {"initial_state_randomization_enabled": False, "mode": "residual"}
    ]

    manifest_path = next((run_dir / "manifests").glob("*manifest*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["condition"] == "x0-y0-z1-yaw0-dur0p3-p0-a0-floorReq0"
    assert manifest["runtime_config"]["parameters"]["mode"] == "hover"
    assert manifest["runtime_config"]["parameters"]["policy_control_modes"] == {
        "floor": "residual"
    }
    assert manifest["result"]["effective_parameters"]["duration"] == 0.3
    assert (
        manifest["result"]["policy_outcomes"]["floor"][
            "force_floor_start_applied"
        ]
        is False
    )


def test_simplified_unified_run_uses_pid_floor_and_saves_two_policy_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = load_config(CONFIGS / "view_live_circle_wide_eval.yaml")
    xml = tmp_path / "model.xml"
    xml.write_text("<mujoco/>", encoding="utf-8")
    model = tmp_path / "policy.zip"
    model.write_bytes(b"fake-model")
    config = replace(
        source,
        paths=replace(
            source.paths,
            mujoco_xml=xml,
            artifact_root=tmp_path / "artifacts",
        ),
        mission=replace(
            source.mission,
            takeoff_sec=0.1,
            settle_sec=0.0,
            goto_sec=0.1,
            post_hold_sec=0.0,
            circle=replace(
                source.mission.circle,
                period=0.1,
                laps=1.0,
                ramp_sec=0.0,
            ),
        ),
    )

    class SignalEnvironment(_FakeEnvironment):
        wrench_command_reference = (
            "body frame; torque about the nominal allocator origin "
            "(not the payload-shifted combined CoM)"
        )

        def step(self, action):
            action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
            self._last_f = 0.01 + 0.002 * action
            # Deliberately unrelated to normalized action so the assertion below
            # proves that plotting receives the environment's actual control-path
            # signal instead of reconstructing a wrench in the plotting layer.
            self._last_wrench_cmd = (
                np.array([0.001, -0.002, 0.00003, 0.41])
                if self.mode == "residual"
                else np.array([0.004, -0.005, 0.00006, 0.43])
            )
            self.references.append(self.pos_des.copy())
            self.steps += 1
            return self.observation(), 0.0, False, self.steps == 3, {}

    class FreshFactory:
        calls: list[dict[str, object]] = []

        @classmethod
        def make(cls, **overrides):
            cls.calls.append(dict(overrides))
            env = SignalEnvironment()
            env.mode = str(overrides.get("mode", config.control_mode))
            return env

    class FakePolicy:
        @staticmethod
        def predict(_observation, deterministic=True):
            assert deterministic is True
            return np.array([2.0, -2.0, 0.5, 0.0]), None

    eval_cli._ensure_runtime_imports()
    loaded_profiles: list[Path] = []

    def fake_load_config(path):
        loaded_profiles.append(Path(path))
        return config

    monkeypatch.setattr(eval_cli, "load_config", fake_load_config)
    original_runner = eval_cli.EvaluationRunner
    monkeypatch.setattr(
        eval_cli,
        "EvaluationRunner",
        lambda *args, **kwargs: original_runner(
            *args, **kwargs, environment_factory=FreshFactory()
        ),
    )
    monkeypatch.setattr(eval_cli, "_load_policy", lambda _path, _config: FakePolicy())
    plotted: list[object] = []

    def fake_policy_plot(path, **kwargs):
        trace = kwargs["rollout"]
        assert path.name == ("floor.png" if trace.policy == "floor" else "ppo.png")
        if trace.policy == "floor":
            np.testing.assert_allclose(trace.control_input, 0.0)
            np.testing.assert_allclose(
                trace.wrench_command,
                np.tile([0.001, -0.002, 0.00003, 0.41], (trace.sample_count, 1)),
            )
        else:
            np.testing.assert_allclose(
                trace.control_input[0], [1.0, -1.0, 0.5, 0.0]
            )
            np.testing.assert_allclose(
                trace.wrench_command,
                np.tile([0.004, -0.005, 0.00006, 0.43], (trace.sample_count, 1)),
            )
        assert "nominal allocator origin" in trace.wrench_command_reference
        np.testing.assert_allclose(trace.linear_velocity[0], [0.1, -0.2, 0.3])
        np.testing.assert_allclose(trace.angular_velocity[0], [1.0, -2.0, 3.0])
        assert kwargs["motor_unit"] == "N"
        plotted.append(trace)
        path.write_bytes(b"policy-png")
        return path

    monkeypatch.setattr(eval_cli, "save_policy_trace", fake_policy_plot)

    assert run_evaluation_cli(
        default_config=CONFIGS / "view_live_hover_eval.yaml",
        description="test",
        argv=[
            "--mode",
            "circle",
            "--path-preset",
            "2",
            "--period",
            "0.1",
            "--laps",
            "1",
            "--policy",
            "both",
            "--model",
            str(model),
            "--headless",
            "--no-realtime",
        ],
        unified=True,
        mode_profiles={"circle": CONFIGS / "view_live_circle_eval.yaml"},
        path_profiles=PATH_PROFILES,
        always_compare=True,
    ) == 0

    assert loaded_profiles[0].name == "view_live_circle_wide_eval.yaml"
    assert [trace.policy for trace in plotted] == ["floor", "residual"]
    assert [trace.control_mode for trace in plotted] == ["residual", "e2e"]
    assert FreshFactory.calls == [
        {"initial_state_randomization_enabled": False, "mode": "residual"},
        {"initial_state_randomization_enabled": False},
    ]
    run_dir = next((tmp_path / "artifacts" / "runs").iterdir())
    assert {path.name for path in (run_dir / "plots").glob("*.png")} == {
        "floor.png",
        "ppo.png",
    }
    metrics = json.loads(
        next((run_dir / "metrics").glob("*.json")).read_text(encoding="utf-8")
    )
    assert metrics["effective_condition"] == "c-wide-T0p1-L1"
    assert metrics["selected_policy"] == "both"
    assert metrics["path_preset"] == "wide"
    assert set(metrics["policies"]) == {"floor", "residual"}
    assert metrics["policy_control_modes"] == {
        "floor": "residual",
        "residual": "e2e",
    }
    assert metrics["runtime_parameters"]["policy_control_modes"] == {
        "floor": "residual",
        "residual": "e2e",
    }
    assert metrics["policies"]["floor"]["control_mode"] == "residual"
    assert metrics["policies"]["residual"]["control_mode"] == "e2e"
    assert metrics["policies"]["floor"]["plot"] == "plots/floor.png"
    assert metrics["policies"]["residual"]["plot"] == "plots/ppo.png"


def test_unified_partial_floor_failure_still_runs_ppo_and_keeps_both_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = load_config(CONFIGS / "view_live_hover_eval.yaml")
    xml = tmp_path / "model.xml"
    xml.write_text("<mujoco/>", encoding="utf-8")
    model = tmp_path / "policy.zip"
    model.write_bytes(b"fake-model")
    config = replace(
        source,
        paths=replace(
            source.paths,
            mujoco_xml=xml,
            artifact_root=tmp_path / "artifacts",
        ),
    )
    calls: list[tuple[str, str | None]] = []

    class FakeRunner:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, _policy, policy_key, label, *, control_mode=None):
            calls.append((policy_key, control_mode))
            samples = 2
            return RolloutTrace(
                policy=policy_key,
                label=label,
                time_sec=np.array([0.0, 0.1]),
                position=np.zeros((samples, 3)),
                attitude_deg=np.zeros((samples, 3)),
                reference_position=np.zeros((samples, 3)),
                position_error=np.zeros(samples),
                phases=("HOVER",) * samples,
                training_boundary_crossed_at=None,
                guard_boundary_crossed_at=None,
                terminated_at=None,
                truncated_at=None,
                diverged_at=None,
                error=("RuntimeError: floor fault" if policy_key == "floor" else None),
                control_input=np.zeros((samples, 4)),
                motor_thrust=np.full((samples, 4), 0.01),
                linear_velocity=np.zeros((samples, 3)),
                angular_velocity=np.zeros((samples, 3)),
                control_mode=control_mode or config.control_mode,
            )

    def fake_policy_plot(path, **_kwargs):
        path.write_bytes(b"partial-policy-png")
        return path

    eval_cli._ensure_runtime_imports()
    monkeypatch.setattr(eval_cli, "load_config", lambda _path: config)
    monkeypatch.setattr(eval_cli, "EvaluationRunner", FakeRunner)
    monkeypatch.setattr(eval_cli, "_load_policy", lambda _path, _config: object())
    monkeypatch.setattr(eval_cli, "save_policy_trace", fake_policy_plot)

    with pytest.raises(RuntimeError, match="floor fault"):
        run_evaluation_cli(
            default_config=source.source_path,
            description="test",
            argv=[
                "--mode",
                "hover",
                "--model",
                str(model),
                "--headless",
                "--no-realtime",
            ],
            unified=True,
            mode_profiles={"hover": CONFIGS / "view_live_hover_eval.yaml"},
            always_compare=True,
        )

    assert calls == [("floor", "residual"), ("residual", None)]
    run_dir = next((tmp_path / "artifacts" / "runs").iterdir())
    assert {path.name for path in (run_dir / "plots").glob("*.png")} == {
        "floor.png",
        "ppo.png",
    }
    metrics = json.loads(
        next((run_dir / "metrics").glob("*.json")).read_text(encoding="utf-8")
    )
    assert metrics["status"] == "failed"
    assert metrics["policies"]["floor"]["error"] == "RuntimeError: floor fault"
    assert metrics["policies"]["floor"]["plot"] == "plots/floor.png"
    assert metrics["policies"]["residual"]["plot"] == "plots/ppo.png"


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

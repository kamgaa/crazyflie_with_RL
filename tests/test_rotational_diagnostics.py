from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.factories import EnvironmentFactory
from crazyflie_rl.recovery import RecoveryCase, set_recovery_case
from crazyflie_rl.rotational_diagnostics import (
    PhysicsSubstepRecorder,
    estimate_actuator_delay_s,
    first_zero_crossing_s,
    gyroscopic_torque_xyz_nm,
    integrate_axis_energy,
    integrate_total_energy,
    reconstruct_three_axis_torques,
    rotational_power,
    run_cli,
    run_rotational_rollout,
    validate_rotational_diagnostic_contract,
)
from crazyflie_rl.warm_start import (
    E2EPolicyCompatibility,
    validate_e2e_policy_compatibility,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "e2e_train_legacy_initial_perturb_v2.yaml"


class _ZeroPolicy:
    observation_space = SimpleNamespace(shape=(15,))
    action_space = SimpleNamespace(shape=(4,))

    def predict(self, _observation, deterministic=True):
        assert deterministic is True
        return np.zeros(4, dtype=np.float32), None


def _runtime() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")


def _model_artifact(tmp_path: Path, *, kind: str = "best-recovery") -> Path:
    config = load_config(PROFILE)
    run_dir = tmp_path / "run"
    model_path = run_dir / "models" / "model.zip"
    manifest_path = run_dir / "manifests" / "manifest.json"
    model_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir()
    model_path.write_bytes(b"diagnostic test archive")
    resolved = config.resolved_dict()
    record = {
        "kind": kind,
        "path": model_path.relative_to(run_dir).as_posix(),
        "timestep": 700_000,
    }
    manifest = {
        "run_id": "rotational-diagnostic-test",
        "created_at": "2026-09-08T00:00:00+09:00",
        "status": "completed",
        "control_mode": "e2e",
        "reward_mode": resolved["environment"]["reward"]["mode"],
        "condition": "test",
        "config_profile": config.profile_name,
        "observation_shape": list(config.observation_shape),
        "action_shape": list(config.action_shape),
        "residual_scale": list(config.environment.residual_scale),
        "actuator": resolved["actuator"],
        "resolved_config": resolved,
        "model_history": [record],
        "models": {kind: record},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return model_path


def _compatibility() -> E2EPolicyCompatibility:
    return E2EPolicyCompatibility(
        model_path=Path("model.zip"),
        manifest_path=Path("manifest.json"),
        training_provenance={},
        requested_runtime_config={},
        compatibility_warnings=(),
        checkpoint_kind="best-recovery",
        saved_timestep=700_000,
    )


def test_three_axis_torque_reconstruction() -> None:
    environment = SimpleNamespace(
        _last_wrench_cmd=np.array([0.1, -0.2, 0.3, 0.4]),
        _last_wrench_allocated=np.array([0.08, -0.17, 0.25, 0.4]),
        _last_wrench_actual=np.array([0.06, -0.14, 0.21, 0.35]),
    )

    result = reconstruct_three_axis_torques(environment)

    np.testing.assert_array_equal(
        result["raw_commanded_torque_xyz_nm"], [0.1, -0.2, 0.3]
    )
    np.testing.assert_array_equal(
        result["allocated_requested_torque_xyz_nm"], [0.08, -0.17, 0.25]
    )
    np.testing.assert_array_equal(
        result["actual_motor_torque_xyz_nm"], [0.06, -0.14, 0.21]
    )
    np.testing.assert_allclose(
        result["command_actual_torque_error_xyz_nm"], [0.04, -0.06, 0.09]
    )


def test_axis_power_and_total_power() -> None:
    axis_power, total = rotational_power([2.0, -3.0, 4.0], [-5.0, 6.0, 0.5])

    np.testing.assert_array_equal(axis_power, [-10.0, -18.0, 2.0])
    assert total == pytest.approx(-26.0)


def test_energy_integration() -> None:
    power = np.array(
        [
            [1.0, -2.0, 0.0],
            [-3.0, 4.0, -5.0],
            [2.0, -1.0, 6.0],
        ]
    )

    positive, dissipated = integrate_axis_energy(power, 0.25)
    total_positive, total_dissipated = integrate_total_energy(
        np.sum(power, axis=1), 0.25
    )

    np.testing.assert_array_equal(positive, [0.75, 1.0, 1.5])
    np.testing.assert_array_equal(dissipated, [0.75, 0.75, 1.25])
    assert total_positive == pytest.approx(1.75)
    assert total_dissipated == pytest.approx(1.25)


def test_gyroscopic_torque() -> None:
    result = gyroscopic_torque_xyz_nm([1.0, 2.0, 3.0], np.diag([2.0, 3.0, 4.0]))

    np.testing.assert_array_equal(result, [6.0, -6.0, 2.0])


def test_actuator_delay_estimation() -> None:
    command = np.random.default_rng(42).normal(size=300)
    actual = np.zeros_like(command)
    actual[3:] = command[:-3]

    assert estimate_actuator_delay_s(command, actual, 0.002) == pytest.approx(0.006)
    assert estimate_actuator_delay_s(np.ones(20), np.ones(20), 0.002) is None


def test_zero_crossing_detection() -> None:
    time = np.array([0.0, 0.1, 0.2, 0.3])

    assert first_zero_crossing_s(time, [-1.0, -0.5, 0.5, 1.0]) == pytest.approx(
        0.15
    )
    assert first_zero_crossing_s(time, [0.0, 0.4, -0.4, -0.5]) == pytest.approx(
        0.15
    )
    assert first_zero_crossing_s(time, [-1.0, -0.8, -0.5, -0.1]) is None


def test_physics_substep_alignment_and_piecewise_constant_policy_command() -> None:
    _runtime()
    config = load_config(PROFILE)
    environment = EnvironmentFactory(config).make(
        seed=1000, initial_state_randomization_enabled=False
    )
    recorder = PhysicsSubstepRecorder(environment)
    try:
        set_recovery_case(
            environment,
            RecoveryCase(
                "attitude",
                "roll_minus_20deg",
                20.0,
                (-1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            seed=1000,
        )
        previous = environment.set_physics_substep_observer(recorder)
        action = np.array([0.3, -0.2, 0.1, 0.4], dtype=np.float32)
        recorder.begin_policy_step(action, policy_step_index=0)
        _observation, _reward, terminated, truncated, info = environment.step(action)
        recorder.finish_policy_step(
            terminated=terminated,
            truncated=truncated,
            termination_reasons=info.get("termination_reasons", []),
        )
        environment.set_physics_substep_observer(previous)

        assert len(recorder.samples) == 5
        np.testing.assert_allclose(
            [sample["time_s"] for sample in recorder.samples],
            [0.002, 0.004, 0.006, 0.008, 0.010],
        )
        assert [sample["physics_substep"] for sample in recorder.samples] == [
            1,
            2,
            3,
            4,
            5,
        ]
        for sample in recorder.samples:
            np.testing.assert_array_equal(
                sample["normalized_action_tau_xyz"], action[:3]
            )
            assert sample["normalized_action_fz"] == pytest.approx(action[3])
    finally:
        environment.set_physics_substep_observer(None)
        environment.close()


def test_terminated_rollout_is_recorded_at_last_physics_substep() -> None:
    _runtime()
    config = load_config(PROFILE)
    environment = EnvironmentFactory(config).make(
        seed=1000, initial_state_randomization_enabled=False
    )
    try:
        samples, summary = run_rotational_rollout(
            environment,
            _ZeroPolicy(),
            attitude_axis="roll_plus",
            attitude_perturbation_deg=90.0,
            duration_s=0.02,
            seed=1000,
            contract={},
            compatibility=_compatibility(),
        )
    finally:
        environment.close()

    assert len(samples) == 5
    assert samples[-1]["terminated"] is True
    assert samples[-1]["termination_reasons"] == ["excessive_tilt"]
    assert summary["termination_reason"] == "excessive_tilt"
    assert summary["success"] is False


@pytest.mark.parametrize("kind", ["best", "best-recovery"])
def test_best_and_best_recovery_checkpoints_are_allowed(
    tmp_path: Path, kind: str
) -> None:
    _runtime()
    model = _model_artifact(tmp_path, kind=kind)
    config = load_config(PROFILE)
    compatibility = validate_e2e_policy_compatibility(model, config)
    environment = EnvironmentFactory(config).make(
        seed=1000, initial_state_randomization_enabled=False
    )
    try:
        contract = validate_rotational_diagnostic_contract(
            config, environment, _ZeroPolicy(), compatibility
        )
    finally:
        environment.close()

    assert contract["checkpoint_kind"] == kind
    assert contract["allocation_matrix_exact_match"] is True
    assert contract["actuator_configuration_exact_match"] is True


def test_rotational_response_cli_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "diagnose_rotational_response.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--attitude-axis" in result.stdout
    assert "--attitude-perturbation-deg" in result.stdout


def test_rotational_response_cli_writes_short_physics_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime()
    model = _model_artifact(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(
        "crazyflie_rl.rotational_diagnostics._load_policy", lambda _path: _ZeroPolicy()
    )

    result = run_cli(
        [
            "--config",
            str(PROFILE),
            "--model",
            str(model),
            "--attitude-axis",
            "roll_minus",
            "--attitude-perturbation-deg",
            "20",
            "--duration",
            "0.01",
            "--seed",
            "1000",
            "--output-dir",
            str(output),
            "--no-plots",
        ]
    )

    assert result == 0
    summary = json.loads((output / "rotational_response_summary.json").read_text())
    assert summary["physics_sample_count"] == 5
    assert summary["contract"]["checkpoint_kind"] == "best-recovery"
    lines = (output / "rotational_response_trace.csv").read_text().splitlines()
    assert len(lines) == 6
    assert "requested_motor_thrust_0_n" in lines[0]

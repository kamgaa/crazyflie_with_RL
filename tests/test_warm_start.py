from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import train_ppo_02
from crazyflie_rl.config import load_config
from crazyflie_rl.training import PPOTrainer
from crazyflie_rl.warm_start import (
    copy_policy_parameters,
    validate_e2e_policy_compatibility,
    validate_legacy_e2e_donor,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "e2e_train_legacy_initial_perturb_v2.yaml"


def _donor_artifact(
    tmp_path: Path, *, reward_mode: str = "legacy", observation_shape=(15,), action_shape=(4,)
) -> Path:
    config = load_config(PROFILE)
    run_dir = tmp_path / "artifacts" / "runs" / "nominal-training-run"
    model_path = run_dir / "models" / "donor_final.zip"
    manifest_path = run_dir / "manifests" / "manifest.json"
    model_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir()
    model_path.write_bytes(b"test archive")
    resolved = copy.deepcopy(config.resolved_dict())
    resolved["environment"]["initial_state_randomization"]["enabled"] = False
    resolved["environment"]["reward"]["mode"] = reward_mode
    record = {
        "kind": "final",
        "path": model_path.relative_to(run_dir).as_posix(),
        "timestep": 1_001_472,
    }
    manifest = {
        "run_id": run_dir.name,
        "created_at": "2026-09-01T00:00:00+09:00",
        "status": "completed",
        "command": ["python", "train_ppo_02.py", "--config", "configs/e2e_train.yaml"],
        "control_mode": "e2e",
        "reward_mode": reward_mode,
        "mission": "hover",
        "condition": "nominal",
        "config_profile": "e2e_train",
        "observation_shape": list(observation_shape),
        "action_shape": list(action_shape),
        "residual_scale": list(config.environment.residual_scale),
        "actuator": resolved["actuator"],
        "resolved_config": resolved,
        "model_history": [record],
        "models": {"best": None, "final": record},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return model_path


def test_manifest_validated_legacy_e2e_donor(tmp_path: Path) -> None:
    model = _donor_artifact(tmp_path)
    provenance = validate_legacy_e2e_donor(model, load_config(PROFILE))
    assert provenance.model_path == model.resolve()
    assert provenance.control_mode == "e2e"
    assert provenance.reward_mode == "legacy"
    assert provenance.reward_mode_source == "manifest.reward_mode"
    assert provenance.initial_state_randomization_enabled is False
    assert provenance.observation_shape == (15,)
    assert provenance.action_shape == (4,)
    assert provenance.model_kind == "final"
    assert provenance.model_timestep == 1_001_472


def test_pre_reward_mode_manifest_is_audited_as_historical_legacy(
    tmp_path: Path,
) -> None:
    model = _donor_artifact(tmp_path)
    manifest_path = model.parent.parent / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["reward_mode"]
    del manifest["resolved_config"]["environment"]["reward"]["mode"]
    del manifest["resolved_config"]["environment"]["initial_state_randomization"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    provenance = validate_legacy_e2e_donor(model, load_config(PROFILE))
    assert provenance.reward_mode == "legacy"
    assert provenance.reward_mode_source == "historical_pre_mode_schema"
    assert provenance.initial_state_randomization_enabled is False
    assert provenance.initial_state_randomization_source == (
        "historical_pre_randomization_schema"
    )


def test_diagnostic_compatibility_allows_historical_legacy_best(
    tmp_path: Path,
) -> None:
    model = _donor_artifact(tmp_path)
    manifest_path = model.parent.parent / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["model_history"][0]
    record["kind"] = "best"
    record["timestep"] = 980_000
    manifest["models"] = {"best": copy.deepcopy(record), "final": None}
    del manifest["reward_mode"]
    reward = manifest["resolved_config"]["environment"]["reward"]
    del reward["mode"]
    del reward["lyapunov"]
    del reward["e2e_torque_xy_weight"]
    del reward["e2e_torque_yaw_weight"]
    manifest["resolved_config"]["training"]["seed"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    compatibility = validate_e2e_policy_compatibility(
        model, load_config(PROFILE)
    )

    assert compatibility.checkpoint_kind == "best"
    assert compatibility.saved_timestep == 980_000
    assert compatibility.training_provenance["reward_mode"] == "legacy"
    assert (
        compatibility.training_provenance["reward_mode_source"]
        == "historical_pre_mode_schema"
    )
    assert any(
        "historical_pre_mode_schema" in warning
        for warning in compatibility.compatibility_warnings
    )
    assert any(
        "checkpoint kind is 'best'" in warning
        for warning in compatibility.compatibility_warnings
    )
    assert compatibility.training_provenance["historical_missing_fields"]


def test_diagnostic_compatibility_allows_modern_legacy_final(
    tmp_path: Path,
) -> None:
    model = _donor_artifact(tmp_path)
    compatibility = validate_e2e_policy_compatibility(
        model, load_config(PROFILE)
    )

    assert compatibility.checkpoint_kind == "final"
    assert compatibility.training_provenance["reward_mode"] == "legacy"
    assert compatibility.training_provenance["reward_mode_source"] == (
        "manifest.reward_mode"
    )
    assert compatibility.requested_runtime_config["reward_mode"] == "legacy"
    assert {
        "training_provenance",
        "requested_runtime_config",
        "compatibility_warnings",
        "checkpoint_kind",
        "saved_timestep",
    }.issubset(compatibility.as_dict())


def test_diagnostic_compatibility_allows_lyapunov_final(
    tmp_path: Path,
) -> None:
    model = _donor_artifact(tmp_path, reward_mode="lyapunov")
    manifest_path = model.parent.parent / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reward = manifest["resolved_config"]["environment"]["reward"]
    reward["e2e_torque_xy_weight"] = 0.1
    reward["e2e_torque_yaw_weight"] = 0.02
    reward["lyapunov"]["potential_shaping_enabled"] = True
    reward["lyapunov"]["decay_penalty_enabled"] = True
    manifest["resolved_config"]["training"]["total_timesteps"] = 50_000
    manifest["resolved_config"]["training"]["seed"] = 7
    manifest["model_history"][0]["timestep"] = 40_000
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    compatibility = validate_e2e_policy_compatibility(
        model, load_config(PROFILE)
    )

    assert compatibility.training_provenance["reward_mode"] == "lyapunov"
    assert compatibility.checkpoint_kind == "final"
    assert any(
        "reward mode 'lyapunov' differs" in warning
        for warning in compatibility.compatibility_warnings
    )
    assert any(
        "torque-XY reward weight" in warning
        for warning in compatibility.compatibility_warnings
    )
    assert any(
        "configured total_timesteps 50000 differs" in warning
        for warning in compatibility.compatibility_warnings
    )
    assert any(
        "saved checkpoint timestep 40000 differs" in warning
        for warning in compatibility.compatibility_warnings
    )
    assert any(
        "training seed 7 differs" in warning
        for warning in compatibility.compatibility_warnings
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("control_mode", "control mode"),
        ("observation_shape", "observation shape"),
        ("action_shape", "action shape"),
        ("action_scale", "action scale"),
        ("actuator_model", "actuator model"),
        ("physics_hz", "physics rate"),
        ("policy_hz", "control rate"),
    ],
)
def test_diagnostic_compatibility_rejects_inference_contract_mismatch(
    tmp_path: Path, mutation: str, message: str
) -> None:
    model = _donor_artifact(tmp_path)
    manifest_path = model.parent.parent / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "control_mode":
        manifest["control_mode"] = "residual"
    elif mutation == "observation_shape":
        manifest["observation_shape"] = [13]
    elif mutation == "action_shape":
        manifest["action_shape"] = [3]
    elif mutation == "action_scale":
        manifest["residual_scale"][0] = 999.0
    elif mutation == "actuator_model":
        manifest["actuator"]["model"] = "incompatible_actuator"
    elif mutation == "physics_hz":
        manifest["resolved_config"]["vehicle"]["physics_hz"] = 250.0
    else:
        manifest["resolved_config"]["environment"]["policy_hz"] = 50.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_e2e_policy_compatibility(model, load_config(PROFILE))


def test_nonlegacy_donor_is_rejected(tmp_path: Path) -> None:
    model = _donor_artifact(tmp_path, reward_mode="lyapunov")
    with pytest.raises(ValueError, match="reward mode.*not 'legacy'"):
        validate_legacy_e2e_donor(model, load_config(PROFILE))


@pytest.mark.parametrize(
    ("field", "message"),
    [("action_scale", "action scale"), ("actuator", "actuator configuration")],
)
def test_donor_action_and_actuator_provenance_mismatch_is_rejected(
    tmp_path: Path, field: str, message: str
) -> None:
    model = _donor_artifact(tmp_path)
    manifest_path = model.parent.parent / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "action_scale":
        manifest["residual_scale"][0] = 999.0
    else:
        manifest["actuator"]["time_constant_s"] = 999.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_legacy_e2e_donor(model, load_config(PROFILE))


@pytest.mark.parametrize(
    ("observation_shape", "action_shape", "message"),
    [((13,), (4,), "observation shape"), ((15,), (3,), "action shape")],
)
def test_donor_space_provenance_mismatch_is_rejected(
    tmp_path: Path,
    observation_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    message: str,
) -> None:
    model = _donor_artifact(
        tmp_path,
        observation_shape=observation_shape,
        action_shape=action_shape,
    )
    with pytest.raises(ValueError, match=message):
        validate_legacy_e2e_donor(model, load_config(PROFILE))


class _FakePolicy:
    def __init__(self, value: float, optimizer_state: dict[str, object]) -> None:
        self.value = value
        self.optimizer = SimpleNamespace(state=optimizer_state)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weight": np.array([self.value])}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool) -> None:
        assert strict is True
        self.value = float(state["weight"][0])


def _fake_model(
    value: float,
    *,
    observation_shape: tuple[int, ...] = (15,),
    action_shape: tuple[int, ...] = (4,),
    timesteps: int = 0,
    optimizer_state: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        policy=_FakePolicy(value, optimizer_state or {}),
        observation_space=SimpleNamespace(shape=observation_shape),
        action_space=SimpleNamespace(shape=action_shape),
        num_timesteps=timesteps,
    )


def test_policy_only_copy_keeps_fresh_optimizer_and_timestep() -> None:
    target = _fake_model(1.0, optimizer_state={})
    donor = _fake_model(7.0, timesteps=1_000_000, optimizer_state={"old": object()})
    target_optimizer = target.policy.optimizer

    copy_policy_parameters(target, donor)

    assert target.policy.value == 7.0
    assert target.policy.optimizer is target_optimizer
    assert target.policy.optimizer.state == {}
    assert target.num_timesteps == 0


@pytest.mark.parametrize(
    ("observation_shape", "action_shape", "message"),
    [((13,), (4,), "observation space"), ((15,), (3,), "action space")],
)
def test_policy_copy_rejects_runtime_space_mismatch(
    observation_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    message: str,
) -> None:
    target = _fake_model(1.0)
    donor = _fake_model(
        2.0,
        observation_shape=observation_shape,
        action_shape=action_shape,
    )
    with pytest.raises(ValueError, match=message):
        copy_policy_parameters(target, donor)


def test_recovery_profile_refuses_training_without_policy_donor() -> None:
    config = load_config(PROFILE)
    with pytest.raises(ValueError, match="requires --init-policy-from"):
        PPOTrainer(config).train(total_timesteps=1)


def test_training_cli_exposes_policy_initialization_option() -> None:
    args = train_ppo_02.build_parser().parse_args(
        ["--init-policy-from", "/tmp/model.zip"]
    )
    assert args.init_policy_from == Path("/tmp/model.zip")

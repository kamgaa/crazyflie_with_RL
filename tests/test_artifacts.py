from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from crazyflie_rl.artifacts import ArtifactManager, sanitize_component, stable_float
from crazyflie_rl.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class FakeSB3Model:
    def save(self, path: str) -> None:
        Path(f"{path}.zip").write_bytes(b"fake model")


def _config(tmp_path: Path):
    original = load_config(ROOT / "configs" / "e2e_train.yaml")
    return replace(
        original,
        paths=replace(original.paths, artifact_root=tmp_path / "artifacts"),
        training=replace(original.training, seed=42),
    )


def test_stable_float_and_special_character_sanitization() -> None:
    assert stable_float(0.5) == "0p5"
    assert stable_float(-0.025) == "m0p025"
    assert sanitize_component("rho0.5 / T10:+ test") == "rho0.5-T10-test"


def test_fixed_timestamp_layout_names_and_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    now = datetime(2026, 8, 7, 15, 30, 12, tzinfo=ZoneInfo("Asia/Seoul"))
    run = ArtifactManager.create(config, now=now, command=["python", "train_ppo_02.py"])

    expected_stem = "ppo_e2e_hover_nominal_seed42"
    assert run.timestamp == "20260807-153012"
    assert run.experiment_stem == expected_stem
    assert run.run_dir.name == f"{expected_stem}_20260807-153012"
    assert {item.name for item in run.run_dir.iterdir()} == {
        "models", "tensorboard", "plots", "metrics", "config", "manifests"
    }

    assert run.manifest_path.name == f"{expected_stem}_manifest_20260807-153012.json"
    resolved = run.path("config", "resolved-config", ".yaml")
    assert resolved.is_file()
    assert resolved.name == f"{expected_stem}_resolved-config_20260807-153012.yaml"

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    for key in (
        "experiment_name", "condition", "timestamp", "timezone", "git", "command",
        "control_mode", "observation_shape", "action_shape", "residual_scale",
        "payload", "actuator", "seed", "ppo", "mujoco_xml", "resolved_config", "models", "versions",
    ):
        assert key in manifest
    assert manifest["timezone"] == "Asia/Seoul"
    assert manifest["observation_shape"] == [15]
    assert manifest["action_shape"] == [4]
    assert manifest["actuator"]["enabled"] is True
    assert manifest["actuator"]["model"] == "cf21b_first_order"


def test_best_final_names_and_no_double_zip(tmp_path: Path) -> None:
    run = ArtifactManager.create(
        _config(tmp_path),
        now=datetime(2026, 8, 7, 15, 30, 12, tzinfo=ZoneInfo("Asia/Seoul")),
    )
    best = run.save_model(FakeSB3Model(), "best", timestep=20_000)
    best_payload = run.save_model(
        FakeSB3Model(), "best-payload", timestep=25_000
    )
    final = run.save_model(FakeSB3Model(), "final", timestep=30_720)

    assert "_best_20260807-153012.zip" in best.name
    assert "_best-payload_20260807-153012.zip" in best_payload.name
    assert "_final_20260807-153012.zip" in final.name
    assert not best.name.endswith(".zip.zip")
    assert not final.name.endswith(".zip.zip")
    assert best.is_file() and best_payload.is_file() and final.is_file()
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["models"]["best"]["path"] == best.relative_to(run.run_dir).as_posix()
    assert manifest["models"]["best-payload"]["path"] == best_payload.relative_to(
        run.run_dir
    ).as_posix()
    assert manifest["models"]["final"]["path"] == final.relative_to(run.run_dir).as_posix()


def test_best_recovery_is_separate_and_policy_parent_is_recorded(
    tmp_path: Path,
) -> None:
    run = ArtifactManager.create(
        _config(tmp_path),
        now=datetime(2026, 8, 7, 15, 30, 12, tzinfo=ZoneInfo("Asia/Seoul")),
    )
    parent = tmp_path / "parent.zip"
    parent.write_bytes(b"parent")
    run.record_policy_initialization(
        parent,
        {
            "control_mode": "e2e",
            "reward_mode": "legacy",
            "training_provenance": {"model_sha256": "abc123"},
            "cross_physics_evaluation": True,
        },
    )
    recovery = run.save_model(
        FakeSB3Model(),
        "best-recovery",
        timestep=100_000,
        metadata={"overall_success_rate": 0.5},
    )

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["models"]["best"] is None
    assert manifest["models"]["best-recovery"]["path"] == recovery.relative_to(
        run.run_dir
    ).as_posix()
    initialization = manifest["policy_initialization"]
    assert initialization["strategy"] == "policy_parameters_only"
    assert initialization["parent_model"] == str(parent.resolve())
    assert initialization["parent_model_sha256"] == "abc123"
    assert initialization["optimizer_state_copied"] is False
    assert initialization["rollout_buffer_copied"] is False
    assert initialization["initial_timestep"] == 0
    assert initialization["curriculum_initial_step"] == 0
    assert initialization["cross_physics_evaluation"] is True


def test_caller_metadata_cannot_replace_required_provenance(tmp_path: Path) -> None:
    run = ArtifactManager.create(
        _config(tmp_path),
        now=datetime(2026, 8, 7, 15, 30, 12, tzinfo=ZoneInfo("Asia/Seoul")),
    )
    saved = run.save_model(
        FakeSB3Model(),
        "best",
        timestep=20_000,
        metadata={"path": "forged.zip", "kind": "final"},
    )
    run.finalize("completed", timestamp="forged", models={})

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    record = manifest["models"]["best"]
    assert record["path"] == saved.relative_to(run.run_dir).as_posix()
    assert record["kind"] == "best"
    assert record["metadata"] == {"path": "forged.zip", "kind": "final"}
    assert manifest["timestamp"] == "20260807-153012"
    assert manifest["models"]["best"] == record
    assert manifest["result"]["timestamp"] == "forged"
    assert manifest["result"]["models"] == {}


def test_run_and_repeated_best_use_collision_suffixes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    now = datetime(2026, 8, 7, 15, 30, 12, tzinfo=ZoneInfo("Asia/Seoul"))
    first = ArtifactManager.create(config, now=now)
    second = ArtifactManager.create(config, now=now)
    assert second.run_dir.name == f"{first.run_dir.name}-01"

    best_1 = first.save_model(FakeSB3Model(), "best", timestep=20_000)
    best_2 = first.save_model(FakeSB3Model(), "best", timestep=40_000)
    assert best_1 != best_2
    assert best_2.stem.endswith("-01")
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["model_history"]) == 2
    assert manifest["models"]["best"]["timestep"] == 40_000


def test_runtime_resolved_config_uses_run_timestamp_and_updates_manifest(
    tmp_path: Path,
) -> None:
    run = ArtifactManager.create(
        _config(tmp_path),
        now=datetime(2026, 8, 7, 15, 30, 12, tzinfo=ZoneInfo("Asia/Seoul")),
    )
    runtime = {
        "mode": "lissajous",
        "center_xy": (0.0, 0.0),
        "model": Path("model/ppo_best.zip"),
        "headless": True,
    }

    path = run.write_runtime_config(runtime)

    assert path.name.endswith("_runtime-resolved_20260807-153012.yaml")
    text = path.read_text(encoding="utf-8")
    assert "mode: lissajous" in text
    assert "model: model\\ppo_best.zip" in text or "model: model/ppo_best.zip" in text
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime_config"]["path"] == path.relative_to(
        run.run_dir
    ).as_posix()
    assert manifest["runtime_config"]["parameters"]["center_xy"] == [0.0, 0.0]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run.write_runtime_config(runtime)

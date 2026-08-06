from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

from crazyflie_rl.artifacts import ArtifactRun
from crazyflie_rl.config import ExperimentConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake SB3 archive")


def _temporary_config(tmp_path: Path) -> ExperimentConfig:
    original = load_config(PROJECT_ROOT / "configs" / "e2e_train.yaml")
    data = deepcopy(original.data)
    data["paths"]["artifact_root"] = str(tmp_path / "artifacts")
    return ExperimentConfig(
        data=data,
        source_path=original.source_path,
        project_root=original.project_root,
    )


def test_run_layout_manifest_and_compliant_filenames(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)
    run = ArtifactRun.create(config)

    assert re.search(
        r"mode-e2e__condition-nominal__seed-unset__date-\d{8}",
        run.run_dir.name,
    )
    assert {path.name for path in run.run_dir.iterdir()} == {
        "models",
        "tensorboard",
        "plots",
        "metrics",
        "config",
        "manifest.json",
    }
    assert (run.run_dir / "config" / "resolved.yaml").is_file()

    model_path = run.save_model(FakeModel(), "best", timestep=20_000)
    metrics_path = run.write_metrics("evaluation-step0020000", {"score": 1.0})
    run.finalize("completed")

    for path in (model_path, metrics_path):
        assert "mode-e2e" in path.name
        assert "condition-nominal" in path.name
        assert "seed-unset" in path.name
        assert re.search(r"date-\d{8}", path.name)

    sidecar = model_path.with_suffix(".manifest.json")
    model_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    assert model_manifest["control_mode"] == "e2e"
    assert model_manifest["observation_schema"] == "e2e_v1"
    assert model_manifest["observation_dim"] == 15
    assert model_manifest["action_dim"] == 4
    assert model_manifest["residual_scale"] == [0.022, 0.022, 0.0001, 0.3]

    run_manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["status"] == "completed"
    assert run_manifest["models"]["best"]["timestep"] == 20_000
    assert run_manifest["seed"] is None

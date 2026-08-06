"""Run-scoped artifact layout and reproducibility manifests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from .config import ExperimentConfig


_SAFE_COMPONENT = re.compile(r"[^a-zA-Z0-9._-]+")


def _slug(value: object) -> str:
    result = _SAFE_COMPONENT.sub("-", str(value).strip()).strip("-._")
    return result or "unspecified"


def _git_commit(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass
class ArtifactRun:
    """Own every newly generated file for one immutable experiment identity."""

    config: ExperimentConfig
    run_dir: Path
    tag: str
    created_at: datetime
    manifest: dict[str, Any]

    @classmethod
    def create(cls, config: ExperimentConfig) -> "ArtifactRun":
        now = datetime.now(timezone.utc)
        seed_tag = "unset" if config.seed is None else f"{config.seed:04d}"
        tag = (
            f"mode-{_slug(config.control_mode)}"
            f"__condition-{_slug(config.condition)}"
            f"__seed-{seed_tag}"
            f"__date-{now:%Y%m%d}"
        )
        base_run_id = f"{tag}__time-{now:%H%M%SZ}"
        runs_root = config.resolve_path("artifact_root") / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        run_dir = runs_root / base_run_id
        suffix = 1
        while run_dir.exists():
            run_dir = runs_root / f"{base_run_id}-{suffix:02d}"
            suffix += 1
        for child in ("models", "tensorboard", "plots", "metrics", "config"):
            (run_dir / child).mkdir(parents=True, exist_ok=False)

        manifest: dict[str, Any] = {
            "run_id": run_dir.name,
            "created_at_utc": now.isoformat(),
            "status": "created",
            "entrypoint": "train_ppo_02.py",
            "git_commit": _git_commit(config.project_root),
            "config_profile": config.profile_name,
            "config_source": str(config.source_path),
            "resolved_config": "config/resolved.yaml",
            "control_mode": config.control_mode,
            "observation_schema": config.observation_schema,
            "observation_dim": config.observation_dim,
            "action_dim": config.action_dim,
            "residual_scale": list(config.residual_scale),
            "seed": config.seed,
            "condition": config.condition,
            "ppo": deepcopy(config.data["training"]["ppo"]),
            "total_timesteps": config.data["training"]["total_timesteps"],
            "evaluation": deepcopy(config.data["evaluation"]),
            "models": {},
            "metrics": {},
        }
        run = cls(config, run_dir, tag, now, manifest)
        run._write_resolved_config()
        run._write_manifest()
        return run

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def artifact_path(self, group: str, kind: str, suffix: str) -> Path:
        if group not in {"models", "plots", "metrics"}:
            raise ValueError(f"unsupported artifact group: {group!r}")
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return self.run_dir / group / f"{_slug(kind)}__{self.tag}{suffix}"

    def save_model(
        self,
        model: Any,
        kind: str,
        timestep: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self.artifact_path("models", f"ppo-{kind}", ".zip")
        model.save(str(path))
        if not path.is_file():
            appended = Path(f"{path}.zip")
            if appended.is_file():  # compatibility with custom save implementations
                path = appended
            else:
                raise RuntimeError(f"model save did not create the expected file: {path}")

        record: dict[str, Any] = {
            "kind": kind,
            "path": path.relative_to(self.run_dir).as_posix(),
            "timestep": timestep,
            "control_mode": self.config.control_mode,
            "observation_schema": self.config.observation_schema,
            "observation_dim": self.config.observation_dim,
            "action_dim": self.config.action_dim,
            "residual_scale": list(self.config.residual_scale),
            "config_profile": self.config.profile_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            record.update(deepcopy(dict(metadata)))

        sidecar = path.with_suffix(".manifest.json")
        _write_json(sidecar, record)
        self.manifest["models"][kind] = record
        self._write_manifest()
        return path

    def write_metrics(self, kind: str, payload: Mapping[str, Any]) -> Path:
        path = self.artifact_path("metrics", kind, ".json")
        _write_json(path, dict(payload))
        self.manifest["metrics"][kind] = path.relative_to(self.run_dir).as_posix()
        self._write_manifest()
        return path

    def finalize(
        self,
        status: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.manifest["status"] = status
        self.manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if extra:
            self.manifest.update(deepcopy(dict(extra)))
        self._write_manifest()

    def _write_resolved_config(self) -> None:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise RuntimeError("PyYAML is required to write resolved.yaml") from exc
        target = self.run_dir / "config" / "resolved.yaml"
        target.write_text(
            yaml.safe_dump(
                self.config.resolved_dict(),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

    def _write_manifest(self) -> None:
        _write_json(self.manifest_path, self.manifest)

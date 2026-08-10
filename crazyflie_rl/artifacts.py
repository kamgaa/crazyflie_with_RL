"""Run-scoped artifact layout, deterministic naming, and manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from .config import ExperimentConfig, dump_resolved_config


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_GROUPS = {
    "models": "models",
    "tensorboard": "tensorboard",
    "plots": "plots",
    "metrics": "metrics",
    "config": "config",
    "manifests": "manifests",
}


def stable_float(value: float) -> str:
    """Return a compact filename-safe decimal representation."""

    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise ValueError(f"filename float must be finite, got {value!r}")
    text = format(number, ".12g")
    if "e" in text.lower():
        mantissa, exponent = re.split("[eE]", text)
        text = f"{mantissa}e{int(exponent)}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def sanitize_component(value: object) -> str:
    """Sanitize one semantic filename component without hiding emptiness."""

    if isinstance(value, float):
        text = stable_float(value)
    else:
        text = str(value).strip()
    text = text.replace("+", "-")
    result = re.sub(r"-+", "-", _UNSAFE.sub("-", text)).strip("-._")
    if not result:
        raise ValueError(f"filename component is empty after sanitization: {value!r}")
    return result


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=project_root, check=True, capture_output=True,
            text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_metadata(project_root: Path) -> dict[str, Any]:
    dirty = _git_value(project_root, "status", "--porcelain")
    return {
        "commit_sha": _git_value(project_root, "rev-parse", "HEAD"),
        "branch": _git_value(project_root, "branch", "--show-current"),
        "dirty": None if dirty is None else bool(dirty),
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_versions() -> dict[str, str | None]:
    """Collect versions without importing heavyweight runtime libraries."""

    return {
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "mujoco": _package_version("mujoco"),
        "gymnasium": _package_version("gymnasium"),
        "stable_baselines3": _package_version("stable-baselines3"),
        "torch": _package_version("torch"),
    }


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    # The run manifest is the one deliberately updated lifecycle record.  Its
    # path is run-unique; user-owned or prior-run files are never replaced.
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass
class ArtifactManager:
    """Own all outputs produced during one experiment invocation."""

    config: ExperimentConfig
    timestamp: str
    created_at: datetime
    experiment_stem: str
    run_dir: Path
    command: tuple[str, ...]
    seed: int | None
    condition: str
    mission: str
    manifest: dict[str, Any]

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        *,
        now: datetime | None = None,
        command: list[str] | tuple[str, ...] | None = None,
        condition: str | None = None,
        mission: str | None = None,
        seed: int | None = None,
    ) -> "ArtifactManager":
        zone = ZoneInfo(config.experiment.timezone)
        created_at = datetime.now(zone) if now is None else now
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=zone)
        else:
            created_at = created_at.astimezone(zone)
        timestamp = created_at.strftime("%Y%m%d-%H%M%S")
        run_seed = config.training.seed if seed is None else seed
        seed_text = "unset" if run_seed is None else str(run_seed)
        run_condition = condition or config.experiment.condition
        run_mission = mission or config.mission.type
        stem = "_".join(
            (
                "ppo",
                sanitize_component(config.control_mode),
                sanitize_component(run_mission),
                sanitize_component(run_condition),
                f"seed{sanitize_component(seed_text)}",
            )
        )

        runs_root = config.paths.artifact_root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        base_name = f"{stem}_{timestamp}"
        run_dir = runs_root / base_name
        collision = 1
        while run_dir.exists():
            run_dir = runs_root / f"{base_name}-{collision:02d}"
            collision += 1
        run_dir.mkdir(parents=False, exist_ok=False)
        for group in _GROUPS.values():
            (run_dir / group).mkdir(exist_ok=False)

        actual_command = tuple(command if command is not None else sys.argv)
        manager = cls(
            config=config,
            timestamp=timestamp,
            created_at=created_at,
            experiment_stem=stem,
            run_dir=run_dir,
            command=actual_command,
            seed=run_seed,
            condition=run_condition,
            mission=run_mission,
            manifest={},
        )
        resolved_path = manager.path("config", "resolved-config", ".yaml")
        dump_resolved_config(config, resolved_path)
        manager.manifest = manager._initial_manifest(resolved_path)
        _write_json_new(manager.manifest_path, manager.manifest)
        return manager

    @property
    def manifest_path(self) -> Path:
        return self.path("manifests", "manifest", ".json")

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard"

    def path(self, group: str, kind: str, suffix: str) -> Path:
        try:
            directory = self.run_dir / _GROUPS[group]
        except KeyError as exc:
            raise ValueError(f"unknown artifact group: {group!r}") from exc
        extension = suffix if suffix.startswith(".") else f".{suffix}"
        if extension == ".zip.zip":
            extension = ".zip"
        return directory / (
            f"{self.experiment_stem}_{sanitize_component(kind)}_"
            f"{self.timestamp}{extension}"
        )

    def ensure_available(self, path: Path) -> Path:
        if path.exists() or Path(f"{path}.zip").exists():
            raise FileExistsError(f"refusing to overwrite artifact: {path}")
        return path

    @staticmethod
    def _collision_safe_path(path: Path) -> Path:
        candidate = path
        index = 1
        while candidate.exists() or Path(f"{candidate}.zip").exists():
            candidate = path.with_name(f"{path.stem}-{index:02d}{path.suffix}")
            index += 1
        return candidate

    def save_model(
        self,
        model: Any,
        kind: str,
        *,
        timestep: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        if kind not in {"best", "final"}:
            raise ValueError("model kind must be 'best' or 'final'")
        # A callback may discover several progressively better checkpoints.
        # Preserve every one with a suffix instead of silently replacing the
        # earlier archive; manifest.models[kind] always points at the latest.
        target = self._collision_safe_path(self.path("models", kind, ".zip"))
        save_argument = target.with_suffix("")
        model.save(str(save_argument))
        if not target.is_file():
            if save_argument.is_file():
                save_argument.rename(target)
            else:
                raise RuntimeError(f"model save did not create expected archive: {target}")
        double_zip = Path(f"{target}.zip")
        if double_zip.exists():
            raise RuntimeError(f"model backend created forbidden double extension: {double_zip}")

        record: dict[str, Any] = {
            "kind": kind,
            "path": target.relative_to(self.run_dir).as_posix(),
            "timestep": timestep,
        }
        if metadata:
            # Caller data is namespaced so it cannot falsify the archive path,
            # kind, or timestep recorded by the manager.
            record["metadata"] = dict(metadata)
        sidecar = self._collision_safe_path(
            self.path("manifests", f"{kind}-model", ".json")
        )
        _write_json_new(sidecar, record)
        self.manifest["model_history"].append(record)
        self.manifest["models"][kind] = record
        self._write_manifest()
        return target

    def write_metrics(self, kind: str, payload: Mapping[str, Any]) -> Path:
        path = self.ensure_available(self.path("metrics", kind, ".json"))
        _write_json_new(path, dict(payload))
        self.manifest["metrics"][kind] = path.relative_to(self.run_dir).as_posix()
        self._write_manifest()
        return path

    def finalize(self, status: str = "completed", **extra: Any) -> None:
        self.manifest["status"] = sanitize_component(status)
        self.manifest["finished_at"] = datetime.now(
            ZoneInfo(self.config.experiment.timezone)
        ).isoformat()
        if extra:
            # Lifecycle result details must not be able to replace fixed
            # provenance such as timestamp, git, resolved config, or models.
            self.manifest.setdefault("result", {}).update(extra)
        self._write_manifest()

    def _initial_manifest(self, resolved_path: Path) -> dict[str, Any]:
        env = self.config.environment
        payload = env.payload
        return {
            "schema_version": 1,
            "run_id": self.run_dir.name,
            "experiment_name": self.config.experiment.name,
            "condition": self.condition,
            "mission": self.mission,
            "description": self.config.experiment.description,
            "timestamp": self.timestamp,
            "created_at": self.created_at.isoformat(),
            "timezone": self.config.experiment.timezone,
            "status": "created",
            "git": _git_metadata(self.config.paths.project_root),
            "command": list(self.command),
            "control_mode": self.config.control_mode,
            "observation_shape": list(self.config.observation_shape),
            "action_shape": list(self.config.action_shape),
            "residual_scale": list(env.residual_scale),
            "payload": {
                "randomize": payload.randomize,
                "mass": payload.mass,
                "offset": list(payload.offset),
                "randomization_limits": asdict(payload.randomization_limits),
            },
            "seed": self.seed,
            "ppo": asdict(self.config.training.ppo),
            "mujoco_xml": str(self.config.paths.mujoco_xml),
            "config_profile": self.config.profile_name,
            "resolved_config_path": resolved_path.relative_to(self.run_dir).as_posix(),
            "resolved_config": self.config.resolved_dict(),
            "models": {"best": None, "final": None},
            "model_history": [],
            "metrics": {},
            "versions": runtime_versions(),
        }

    def _write_manifest(self) -> None:
        _write_json_replace(self.manifest_path, self.manifest)

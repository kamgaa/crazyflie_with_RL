"""Strict model/environment compatibility checks for SB3 policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import ExperimentConfig


class ModelCompatibilityError(ValueError):
    """Raised instead of coercing a policy across incompatible control contracts."""


def normalize_model_path(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        return path
    if path.suffix.lower() != ".zip":
        zipped = path.with_suffix(".zip")
        if zipped.is_file():
            return zipped
    raise FileNotFoundError(f"model file does not exist: {path}")


def manifest_path_for(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.suffix.lower() == ".zip":
        return path.with_suffix(".manifest.json")
    return Path(f"{path}.manifest.json")


def read_model_manifest(model_path: str | Path) -> dict[str, Any]:
    normalized = normalize_model_path(model_path)
    path = manifest_path_for(normalized)
    if not path.is_file():
        raise ModelCompatibilityError(
            "Model manifest is required for strict loading and was not found.\n"
            f"- model path: {normalized}\n"
            f"- expected manifest: {path}\n"
            "The model will not be auto-converted or attached to an inferred control mode."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelCompatibilityError(f"invalid model manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelCompatibilityError(f"model manifest root must be an object: {path}")
    return payload


def _space_shape(owner: Any, name: str) -> tuple[int, ...] | None:
    space = getattr(owner, name, None)
    shape = getattr(space, "shape", None)
    if shape is None:
        return None
    return tuple(int(value) for value in shape)


def _flat_dimension(shape: tuple[int, ...] | None) -> int | None:
    if shape is None or len(shape) != 1:
        return None
    return shape[0]


def _compatibility_message(
    *,
    model_path: Path,
    config: ExperimentConfig,
    model_observation_dim: int | None,
    environment_observation_dim: int | None,
    model_action_dim: int | None,
    environment_action_dim: int | None,
    reasons: list[str],
) -> str:
    details = "\n".join(f"- {reason}" for reason in reasons)
    return (
        "Model compatibility check failed; the policy was not converted or forced to run.\n"
        f"- model required observation dimension: {model_observation_dim}\n"
        f"- environment observation dimension: {environment_observation_dim}\n"
        f"- model action dimension: {model_action_dim}\n"
        f"- environment action dimension: {environment_action_dim}\n"
        f"- selected control mode: {config.control_mode}\n"
        f"- model path: {model_path}\n"
        f"- required config profile: {config.profile_name}\n"
        f"Reasons:\n{details}"
    )


def validate_model_compatibility(
    model: Any,
    env: Any,
    config: ExperimentConfig,
    model_path: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate manifest, SB3 spaces, environment spaces, mode, and schema."""

    normalized = normalize_model_path(model_path)
    model_manifest = dict(manifest or read_model_manifest(normalized))
    model_obs_dim = _flat_dimension(_space_shape(model, "observation_space"))
    model_action_dim = _flat_dimension(_space_shape(model, "action_space"))
    env_obs_dim = _flat_dimension(_space_shape(env, "observation_space"))
    env_action_dim = _flat_dimension(_space_shape(env, "action_space"))

    reasons: list[str] = []
    manifest_checks = {
        "control_mode": config.control_mode,
        "observation_schema": config.observation_schema,
        "observation_dim": config.observation_dim,
        "action_dim": config.action_dim,
    }
    for field, expected in manifest_checks.items():
        actual = model_manifest.get(field)
        if actual != expected:
            reasons.append(
                f"manifest {field} is {actual!r}, but profile requires {expected!r}"
            )

    if model_obs_dim != config.observation_dim:
        reasons.append(
            f"model observation space is {model_obs_dim}, profile requires {config.observation_dim}"
        )
    if env_obs_dim != config.observation_dim:
        reasons.append(
            f"environment observation space is {env_obs_dim}, profile requires {config.observation_dim}"
        )
    if model_action_dim != config.action_dim:
        reasons.append(
            f"model action space is {model_action_dim}, profile requires {config.action_dim}"
        )
    if env_action_dim != config.action_dim:
        reasons.append(
            f"environment action space is {env_action_dim}, profile requires {config.action_dim}"
        )

    env_mode = getattr(env, "mode", config.control_mode)
    if env_mode != config.control_mode:
        reasons.append(
            f"environment mode is {env_mode!r}, profile requires {config.control_mode!r}"
        )
    env_schema = getattr(env, "observation_schema", config.observation_schema)
    if env_schema != config.observation_schema:
        reasons.append(
            f"environment schema is {env_schema!r}, profile requires {config.observation_schema!r}"
        )

    if reasons:
        raise ModelCompatibilityError(
            _compatibility_message(
                model_path=normalized,
                config=config,
                model_observation_dim=model_obs_dim,
                environment_observation_dim=env_obs_dim,
                model_action_dim=model_action_dim,
                environment_action_dim=env_action_dim,
                reasons=reasons,
            )
        )


def load_and_validate_policy(
    model_path: str | Path,
    env: Any,
    config: ExperimentConfig,
    *,
    device: str = "auto",
) -> Any:
    """Load an SB3 PPO policy only after strict metadata and space validation."""

    normalized = normalize_model_path(model_path)
    manifest = read_model_manifest(normalized)

    # Reject obvious mode/schema/dimension mismatches before deserializing weights.
    preliminary_reasons: list[str] = []
    expected = {
        "control_mode": config.control_mode,
        "observation_schema": config.observation_schema,
        "observation_dim": config.observation_dim,
        "action_dim": config.action_dim,
    }
    for field, required in expected.items():
        if manifest.get(field) != required:
            preliminary_reasons.append(
                f"manifest {field} is {manifest.get(field)!r}, profile requires {required!r}"
            )
    if preliminary_reasons:
        env_obs_dim = _flat_dimension(_space_shape(env, "observation_space"))
        env_action_dim = _flat_dimension(_space_shape(env, "action_space"))
        raise ModelCompatibilityError(
            _compatibility_message(
                model_path=normalized,
                config=config,
                model_observation_dim=manifest.get("observation_dim"),
                environment_observation_dim=env_obs_dim,
                model_action_dim=manifest.get("action_dim"),
                environment_action_dim=env_action_dim,
                reasons=preliminary_reasons,
            )
        )

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError("stable-baselines3 is required to load PPO policies") from exc

    model = PPO.load(str(normalized), device=device)
    validate_model_compatibility(model, env, config, normalized, manifest)
    model.set_env(env)
    return model

"""Manifest-validated, policy-only PPO warm-start support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .physics_version import (
    PHYSICS_MODEL_VERSION,
    manifest_physics_version,
    physics_comparison,
)


_HISTORICAL_LEGACY_REWARD_FIELDS = {
    "position_weight",
    "velocity_weight",
    "tilt_weight",
    "angular_velocity_weight",
    "yaw_weight",
    "action_weight",
    "action_rate_weight",
    "crash_penalty",
}


@dataclass(frozen=True)
class E2EPolicyCompatibility:
    """Inference compatibility plus non-blocking training provenance notes."""

    model_path: Path
    manifest_path: Path
    training_provenance: dict[str, Any]
    requested_runtime_config: dict[str, Any]
    compatibility_warnings: tuple[str, ...]
    checkpoint_kind: str
    saved_timestep: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            **physics_comparison(self.training_provenance.get("physics_model_version")),
            "model_path": str(self.model_path),
            "manifest_path": str(self.manifest_path),
            "training_provenance": self.training_provenance,
            "requested_runtime_config": self.requested_runtime_config,
            "compatibility_warnings": list(self.compatibility_warnings),
            "checkpoint_kind": self.checkpoint_kind,
            "saved_timestep": self.saved_timestep,
        }


@dataclass(frozen=True)
class PolicyDonorProvenance:
    """Audited provenance for one compatible nominal legacy E2E model."""

    model_path: Path
    manifest_path: Path
    run_id: str
    created_at: str
    status: str
    mission: str
    condition: str
    config_profile: str
    control_mode: str
    reward_mode: str
    reward_mode_source: str
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    action_scale: tuple[float, ...]
    actuator_model: str
    actuator_config_matches_target: bool
    initial_state_randomization_enabled: bool
    initial_state_randomization_source: str
    model_kind: str
    model_timestep: int | None
    model_sha256: str
    git_branch: str
    git_commit_sha: str
    git_dirty: bool | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_path"] = str(self.model_path)
        payload["manifest_path"] = str(self.manifest_path)
        payload["observation_shape"] = list(self.observation_shape)
        payload["action_shape"] = list(self.action_shape)
        payload["action_scale"] = list(self.action_scale)
        return payload


@dataclass(frozen=True)
class ObservationExpansionCompatibility:
    """Validated 15D donor contract for an explicit wider-input migration."""

    base_compatibility: E2EPolicyCompatibility
    donor_observation_schema: dict[str, Any]
    target_observation_schema: dict[str, Any]
    exact_contract_checks: dict[str, bool]

    @property
    def model_path(self) -> Path:
        return self.base_compatibility.model_path

    @property
    def saved_timestep(self) -> int | None:
        return self.base_compatibility.saved_timestep

    def as_dict(self) -> dict[str, Any]:
        result = self.base_compatibility.as_dict()
        result.update(
            {
                "strategy": "explicit_mlp_input_expansion_v1",
                "donor_observation_schema": self.donor_observation_schema,
                "target_observation_schema": self.target_observation_schema,
                "exact_contract_checks": dict(self.exact_contract_checks),
            }
        )
        return result


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read policy artifact manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"policy artifact manifest must contain a mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_for_model(
    manifest: Mapping[str, Any], run_dir: Path, model_path: Path
) -> Mapping[str, Any] | None:
    records: list[Mapping[str, Any]] = []
    history = manifest.get("model_history", ())
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        records.extend(item for item in history if isinstance(item, Mapping))
    models = manifest.get("models", {})
    if isinstance(models, Mapping):
        records.extend(item for item in models.values() if isinstance(item, Mapping))
    for record in records:
        relative_path = record.get("path")
        if not isinstance(relative_path, str):
            continue
        if (run_dir / relative_path).resolve() == model_path:
            return record
    return None


def _manifest_for_model(
    model_path: Path,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    if model_path.parent.name != "models":
        raise ValueError(
            "policy model must be the original model inside a training artifact "
            f"run's models directory: {model_path}"
        )
    run_dir = model_path.parent.parent
    manifests_dir = run_dir / "manifests"
    matches: list[tuple[Path, Mapping[str, Any], Mapping[str, Any]]] = []
    for path in sorted(manifests_dir.glob("*.json")):
        try:
            manifest = _load_manifest(path)
        except ValueError:
            continue
        record = _record_for_model(manifest, run_dir, model_path)
        if record is not None:
            matches.append((path, manifest, record))
    if not matches:
        raise ValueError(
            f"no training manifest in the policy run records this model: {model_path}"
        )
    if len(matches) != 1:
        paths = ", ".join(str(item[0]) for item in matches)
        raise ValueError(f"policy provenance is ambiguous across manifests: {paths}")
    return matches[0]


def _shape(value: Any, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"policy manifest {field} is missing or invalid")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"policy manifest {field} is invalid") from exc


def _float_tuple(value: Any, field: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"policy manifest {field} is missing or invalid")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"policy manifest {field} is invalid") from exc


def _artifact_reward_mode(
    manifest: Mapping[str, Any], resolved_reward: Mapping[str, Any]
) -> tuple[str, str]:
    if isinstance(manifest.get("reward_mode"), str):
        return str(manifest["reward_mode"]), "manifest.reward_mode"
    if isinstance(resolved_reward.get("mode"), str):
        return (
            str(resolved_reward["mode"]),
            "resolved_config.environment.reward.mode",
        )
    if _HISTORICAL_LEGACY_REWARD_FIELDS.issubset(resolved_reward):
        # Reward mode did not exist in this schema. The weighted-error reward
        # was the only executable reward, so this is an artifact-schema fact,
        # not a filename-based guess.
        return "legacy", "historical_pre_mode_schema"
    return "", "unverifiable"


def _optional_shape(value: Any, field: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    return _shape(value, field)


def _required_finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"policy manifest {field} is missing or invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"policy manifest {field} is missing or invalid") from exc
    if not math.isfinite(result):
        raise ValueError(f"policy manifest {field} is missing or invalid")
    return result


def _optional_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_e2e_policy_compatibility(
    model: str | Path, target_config: Any
) -> E2EPolicyCompatibility:
    """Validate only the contracts required to execute an E2E policy.

    Reward design, training duration, seed, and checkpoint selection describe
    how a policy was obtained; they do not change its inference tensor or plant
    contract. Differences in those fields are therefore retained as warnings.
    """

    model_path = Path(model).expanduser().resolve()
    if not model_path.is_file():
        raise ValueError(f"policy model does not exist: {model_path}")
    manifest_path, manifest, record = _manifest_for_model(model_path)
    resolved = manifest.get("resolved_config", {})
    if not isinstance(resolved, Mapping):
        resolved = {}
    resolved_environment = resolved.get("environment", {})
    if not isinstance(resolved_environment, Mapping):
        resolved_environment = {}
    resolved_reward = resolved_environment.get("reward", {})
    if not isinstance(resolved_reward, Mapping):
        resolved_reward = {}
    resolved_vehicle = resolved.get("vehicle", {})
    if not isinstance(resolved_vehicle, Mapping):
        resolved_vehicle = {}
    resolved_training = resolved.get("training", {})
    if not isinstance(resolved_training, Mapping):
        resolved_training = {}

    control_mode = str(
        manifest.get("control_mode", resolved_environment.get("control_mode", ""))
    )
    observation_shape = _optional_shape(
        manifest.get("observation_shape"), "observation_shape"
    )
    action_shape = _optional_shape(manifest.get("action_shape"), "action_shape")
    action_scale = _float_tuple(
        manifest.get("residual_scale", resolved_environment.get("residual_scale")),
        "residual_scale/action scale",
    )
    actuator = manifest.get("actuator", resolved.get("actuator", {}))
    if not isinstance(actuator, Mapping):
        raise ValueError("policy manifest actuator provenance is missing or invalid")
    actuator_model = str(actuator.get("model", ""))
    physics_hz = _required_finite_float(
        resolved_vehicle.get("physics_hz"), "resolved_config.vehicle.physics_hz"
    )
    policy_hz = _required_finite_float(
        resolved_environment.get("policy_hz"),
        "resolved_config.environment.policy_hz",
    )

    target_observation_shape = tuple(target_config.observation_shape)
    target_action_shape = tuple(target_config.action_shape)
    target_action_scale = tuple(
        float(item) for item in target_config.environment.residual_scale
    )
    errors: list[str] = []
    if str(target_config.control_mode) != "e2e":
        errors.append(
            f"requested runtime control mode is {target_config.control_mode!r}, not 'e2e'"
        )
    if control_mode != "e2e":
        errors.append(f"training control mode is {control_mode!r}, not 'e2e'")
    if observation_shape is not None and observation_shape != target_observation_shape:
        errors.append(
            f"observation shape {observation_shape} != {target_observation_shape}"
        )
    if action_shape is not None and action_shape != target_action_shape:
        errors.append(f"action shape {action_shape} != {target_action_shape}")
    if action_scale != target_action_scale:
        errors.append(f"action scale {action_scale} != {target_action_scale}")
    if actuator_model != str(target_config.actuator.model):
        errors.append(
            f"actuator model {actuator_model!r} != {target_config.actuator.model!r}"
        )
    if not math.isclose(
        physics_hz,
        float(target_config.vehicle.physics_hz),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        errors.append(
            f"physics rate {physics_hz} Hz != {target_config.vehicle.physics_hz} Hz"
        )
    if not math.isclose(
        policy_hz,
        float(target_config.environment.policy_hz),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        errors.append(
            f"control rate {policy_hz} Hz != {target_config.environment.policy_hz} Hz"
        )
    if errors:
        raise ValueError(f"incompatible E2E policy {model_path}: " + "; ".join(errors))

    reward_mode, reward_mode_source = _artifact_reward_mode(manifest, resolved_reward)
    training_lyapunov = resolved_reward.get("lyapunov")
    if not isinstance(training_lyapunov, Mapping):
        training_lyapunov = None
    torque_xy = _optional_finite_float(resolved_reward.get("e2e_torque_xy_weight"))
    torque_yaw = _optional_finite_float(resolved_reward.get("e2e_torque_yaw_weight"))
    configured_total_timesteps = _optional_int(resolved_training.get("total_timesteps"))
    training_seed = _optional_int(resolved_training.get("seed", manifest.get("seed")))
    checkpoint_kind = str(record.get("kind", "unknown")) or "unknown"
    saved_timestep = _optional_int(record.get("timestep"))

    runtime_resolved = target_config.resolved_dict()
    runtime_reward = runtime_resolved["environment"]["reward"]
    runtime_total_timesteps = int(target_config.training.total_timesteps)
    runtime_seed = target_config.training.seed
    warnings: list[str] = []
    missing_fields: list[str] = []

    if "reward_mode" not in manifest:
        missing_fields.append("manifest.reward_mode")
    if "mode" not in resolved_reward:
        missing_fields.append("resolved_config.environment.reward.mode")
    if training_lyapunov is None:
        missing_fields.append("resolved_config.environment.reward.lyapunov")
    if torque_xy is None:
        missing_fields.append("resolved_config.environment.reward.e2e_torque_xy_weight")
    if torque_yaw is None:
        missing_fields.append(
            "resolved_config.environment.reward.e2e_torque_yaw_weight"
        )
    if observation_shape is None:
        missing_fields.append("manifest.observation_shape")
    if action_shape is None:
        missing_fields.append("manifest.action_shape")
    if configured_total_timesteps is None:
        missing_fields.append("resolved_config.training.total_timesteps")
    if training_seed is None:
        missing_fields.append("resolved_config.training.seed")
    if checkpoint_kind == "unknown":
        missing_fields.append("model_record.kind")
    if saved_timestep is None:
        missing_fields.append("model_record.timestep")

    if reward_mode_source == "historical_pre_mode_schema":
        warnings.append(
            "legacy reward provenance inferred as historical_pre_mode_schema: "
            "this artifact predates the reward-mode field"
        )
    elif reward_mode_source == "unverifiable":
        warnings.append("training reward mode is absent and cannot be verified")
    if reward_mode and reward_mode != str(runtime_reward["mode"]):
        warnings.append(
            f"training reward mode {reward_mode!r} differs from requested runtime "
            f"mode {runtime_reward['mode']!r}"
        )
    if training_lyapunov is None:
        warnings.append(
            "Lyapunov settings are absent from training provenance; the runtime "
            "settings are not used as evidence about training"
        )
    elif dict(training_lyapunov) != dict(runtime_reward["lyapunov"]):
        warnings.append("training and requested-runtime Lyapunov settings differ")
    runtime_torque_xy = float(runtime_reward["e2e_torque_xy_weight"])
    runtime_torque_yaw = float(runtime_reward["e2e_torque_yaw_weight"])
    if torque_xy is None:
        warnings.append("training torque-XY reward weight is absent")
    elif torque_xy != runtime_torque_xy:
        warnings.append(
            f"training torque-XY reward weight {torque_xy} differs from runtime "
            f"weight {runtime_torque_xy}"
        )
    if torque_yaw is None:
        warnings.append("training torque-yaw reward weight is absent")
    elif torque_yaw != runtime_torque_yaw:
        warnings.append(
            f"training torque-yaw reward weight {torque_yaw} differs from runtime "
            f"weight {runtime_torque_yaw}"
        )
    if configured_total_timesteps is None:
        warnings.append("configured training total_timesteps is absent")
    elif configured_total_timesteps != runtime_total_timesteps:
        warnings.append(
            f"training configured total_timesteps {configured_total_timesteps} differs "
            f"from runtime config value {runtime_total_timesteps}"
        )
    if saved_timestep is None:
        warnings.append("saved checkpoint timestep is absent")
    elif (
        configured_total_timesteps is not None
        and saved_timestep != configured_total_timesteps
    ):
        warnings.append(
            f"saved checkpoint timestep {saved_timestep} differs from training "
            f"configured total_timesteps {configured_total_timesteps}"
        )
    if training_seed is None:
        warnings.append("training seed is absent")
    elif runtime_seed is None:
        warnings.append(
            f"training seed is {training_seed}, while requested runtime training seed is absent"
        )
    elif training_seed != int(runtime_seed):
        warnings.append(
            f"training seed {training_seed} differs from requested runtime seed "
            f"{runtime_seed}"
        )
    if checkpoint_kind == "best":
        warnings.append(
            "checkpoint kind is 'best'; it is accepted for deterministic inference"
        )
    elif checkpoint_kind not in {"final", "best"}:
        warnings.append(
            f"checkpoint kind is {checkpoint_kind!r}; it is accepted for inference"
        )
    if missing_fields:
        warnings.append(
            "historical manifest/provenance fields are missing: "
            + ", ".join(missing_fields)
        )

    try:
        model_sha256 = _sha256(model_path)
    except OSError as exc:
        raise ValueError(f"cannot read policy model file {model_path}: {exc}") from exc
    training_provenance = {
        "physics_model_version": manifest_physics_version(manifest),
        "manifest_path": str(manifest_path.resolve()),
        "run_id": str(manifest.get("run_id", model_path.parent.parent.name)),
        "created_at": str(manifest.get("created_at", "")),
        "status": str(manifest.get("status", "")),
        "config_profile": str(manifest.get("config_profile", "")),
        "control_mode": control_mode,
        "reward_mode": reward_mode or None,
        "reward_mode_source": reward_mode_source,
        "lyapunov": dict(training_lyapunov) if training_lyapunov is not None else None,
        "e2e_torque_xy_weight": torque_xy,
        "e2e_torque_yaw_weight": torque_yaw,
        "configured_total_timesteps": configured_total_timesteps,
        "training_seed": training_seed,
        "observation_shape": (
            list(observation_shape) if observation_shape is not None else None
        ),
        "action_shape": list(action_shape) if action_shape is not None else None,
        "action_scale": list(action_scale),
        "actuator_model": actuator_model,
        "physics_hz": physics_hz,
        "policy_hz": policy_hz,
        "historical_missing_fields": missing_fields,
        "model_sha256": model_sha256,
    }
    requested_runtime_config = {
        "physics_model_version": PHYSICS_MODEL_VERSION,
        "source_path": str(target_config.source_path),
        "config_profile": target_config.profile_name,
        "control_mode": target_config.control_mode,
        "reward_mode": runtime_reward["mode"],
        "lyapunov": runtime_reward["lyapunov"],
        "e2e_torque_xy_weight": runtime_torque_xy,
        "e2e_torque_yaw_weight": runtime_torque_yaw,
        "configured_total_timesteps": runtime_total_timesteps,
        "training_seed": runtime_seed,
        "observation_shape": list(target_observation_shape),
        "action_shape": list(target_action_shape),
        "action_scale": list(target_action_scale),
        "actuator_model": target_config.actuator.model,
        "physics_hz": float(target_config.vehicle.physics_hz),
        "policy_hz": float(target_config.environment.policy_hz),
    }
    if manifest_physics_version(manifest) != PHYSICS_MODEL_VERSION:
        warnings.append(
            "cross-physics evaluation: training="
            + manifest_physics_version(manifest)
            + "; runtime="
            + PHYSICS_MODEL_VERSION
        )
    return E2EPolicyCompatibility(
        model_path=model_path,
        manifest_path=manifest_path.resolve(),
        training_provenance=training_provenance,
        requested_runtime_config=requested_runtime_config,
        compatibility_warnings=tuple(warnings),
        checkpoint_kind=checkpoint_kind,
        saved_timestep=saved_timestep,
    )


def validate_legacy_e2e_donor(
    model: str | Path, target_config: Any
) -> PolicyDonorProvenance:
    """Require a completed, nominal legacy-E2E training artifact.

    Provenance is intentionally taken from the artifact manifest and resolved
    config, never inferred from a model filename.
    """

    compatibility = validate_e2e_policy_compatibility(model, target_config)
    model_path = compatibility.model_path
    manifest_path, manifest, record = _manifest_for_model(model_path)
    resolved = manifest.get("resolved_config", {})
    resolved_environment = (
        resolved.get("environment", {}) if isinstance(resolved, Mapping) else {}
    )
    resolved_reward = (
        resolved_environment.get("reward", {})
        if isinstance(resolved_environment, Mapping)
        else {}
    )
    initial_state = (
        resolved_environment.get("initial_state_randomization", {})
        if isinstance(resolved_environment, Mapping)
        else {}
    )
    control_mode = str(
        manifest.get(
            "control_mode",
            resolved_environment.get("control_mode", "")
            if isinstance(resolved_environment, Mapping)
            else "",
        )
    )
    reward_mode, reward_mode_source = _artifact_reward_mode(
        manifest,
        resolved_reward if isinstance(resolved_reward, Mapping) else {},
    )
    randomization_enabled = bool(
        initial_state.get("enabled", False)
        if isinstance(initial_state, Mapping)
        else False
    )
    initial_state_randomization_source = (
        "resolved_config.environment.initial_state_randomization.enabled"
        if isinstance(initial_state, Mapping) and "enabled" in initial_state
        else "historical_pre_randomization_schema"
    )
    observation_shape = _shape(manifest.get("observation_shape"), "observation_shape")
    action_shape = _shape(manifest.get("action_shape"), "action_shape")
    action_scale = _float_tuple(
        manifest.get(
            "residual_scale",
            resolved_environment.get("residual_scale")
            if isinstance(resolved_environment, Mapping)
            else None,
        ),
        "residual_scale/action scale",
    )
    actuator = manifest.get("actuator", {})
    if not isinstance(actuator, Mapping):
        raise ValueError("donor manifest actuator provenance is missing or invalid")
    actuator_model = str(actuator.get("model", ""))
    command = manifest.get("command", ())
    command_parts = (
        [str(item) for item in command]
        if isinstance(command, Sequence) and not isinstance(command, (str, bytes))
        else []
    )

    target_actuator = target_config.resolved_dict()["actuator"]
    errors: list[str] = []
    if manifest.get("status") != "completed":
        errors.append(f"run status is {manifest.get('status')!r}, not 'completed'")
    if not any(Path(part).name == "train_ppo_02.py" for part in command_parts):
        errors.append("artifact command is not E2E PPO training")
    if control_mode != "e2e":
        errors.append(f"control mode is {control_mode!r}, not 'e2e'")
    if reward_mode != "legacy":
        errors.append(f"reward mode is {reward_mode!r}, not 'legacy'")
    if randomization_enabled:
        errors.append("initial-state randomization was enabled")
    if str(manifest.get("mission", "")) != "hover":
        errors.append("artifact mission is not nominal hover")
    if str(manifest.get("condition", "")) != "nominal":
        errors.append("artifact condition is not nominal")
    if observation_shape != tuple(target_config.observation_shape):
        errors.append(
            f"observation shape {observation_shape} != "
            f"{tuple(target_config.observation_shape)}"
        )
    if action_shape != tuple(target_config.action_shape):
        errors.append(
            f"action shape {action_shape} != {tuple(target_config.action_shape)}"
        )
    target_scale = tuple(
        float(item) for item in target_config.environment.residual_scale
    )
    if action_scale != target_scale:
        errors.append(f"action scale {action_scale} != {target_scale}")
    if dict(actuator) != target_actuator:
        errors.append("actuator configuration differs from the target profile")
    if errors:
        details = "; ".join(errors)
        raise ValueError(f"incompatible policy donor {model_path}: {details}")

    timestep_value = record.get("timestep")
    timestep = int(timestep_value) if timestep_value is not None else None
    git = manifest.get("git", {})
    if not isinstance(git, Mapping):
        git = {}
    return PolicyDonorProvenance(
        model_path=model_path,
        manifest_path=manifest_path.resolve(),
        run_id=str(manifest.get("run_id", model_path.parent.parent.name)),
        created_at=str(manifest.get("created_at", "")),
        status=str(manifest.get("status", "")),
        mission=str(manifest.get("mission", "")),
        condition=str(manifest.get("condition", "")),
        config_profile=str(manifest.get("config_profile", "")),
        control_mode=control_mode,
        reward_mode=reward_mode,
        reward_mode_source=reward_mode_source,
        observation_shape=observation_shape,
        action_shape=action_shape,
        action_scale=action_scale,
        actuator_model=actuator_model,
        actuator_config_matches_target=True,
        initial_state_randomization_enabled=randomization_enabled,
        initial_state_randomization_source=initial_state_randomization_source,
        model_kind=str(record.get("kind", "")),
        model_timestep=timestep,
        model_sha256=_sha256(model_path),
        git_branch=str(git.get("branch", "")),
        git_commit_sha=str(git.get("commit_sha", "")),
        git_dirty=(bool(git["dirty"]) if isinstance(git.get("dirty"), bool) else None),
    )


def validate_observation_expansion_donor(
    model: str | Path, target_config: Any
) -> ObservationExpansionCompatibility:
    """Validate a historical 15D E2E donor for the explicit MLP expansion.

    The ordinary compatibility validator deliberately rejects different input
    shapes.  Here it is first run against a temporary *legacy-observation*
    view of the requested config, then the stricter plant and architecture
    contracts needed for a semantically safe first-layer expansion are checked.
    """

    from .observation import (
        BASE_OBSERVATION_DIM,
        BASE_OBSERVATION_SCHEMA_VERSION,
        LEGACY_STATE_READER,
        observation_schema,
    )

    target_settings = getattr(target_config.environment, "observation", None)
    if target_settings is None:
        raise ValueError(
            "explicit observation expansion requires environment.observation"
        )
    if tuple(target_config.observation_shape) == (BASE_OBSERVATION_DIM,):
        raise ValueError("target observation is not wider than the 15D donor input")
    legacy_environment = replace(target_config.environment, observation=None)
    legacy_target = replace(target_config, environment=legacy_environment)
    base = validate_e2e_policy_compatibility(model, legacy_target)

    manifest_path, manifest, _record = _manifest_for_model(base.model_path)
    del manifest_path
    resolved = manifest.get("resolved_config", {})
    if not isinstance(resolved, Mapping):
        raise ValueError("donor manifest resolved_config is missing or invalid")
    donor_environment = resolved.get("environment", {})
    donor_training = resolved.get("training", {})
    donor_ppo = (
        donor_training.get("ppo", {}) if isinstance(donor_training, Mapping) else {}
    )
    if not isinstance(donor_environment, Mapping) or not isinstance(donor_ppo, Mapping):
        raise ValueError("donor environment/PPO provenance is missing or invalid")

    donor_schema_value = manifest.get("observation_schema")
    if donor_schema_value is None:
        if (
            tuple(base.training_provenance.get("observation_shape") or ())
            != (BASE_OBSERVATION_DIM,)
            or "observation" in donor_environment
        ):
            raise ValueError(
                "missing donor observation schema cannot be identified as the "
                "historical 15D base contract"
            )
        donor_schema = observation_schema(None)
        donor_schema_source = "historical_15d_manifest_contract"
    elif isinstance(donor_schema_value, Mapping):
        donor_schema = dict(donor_schema_value)
        donor_schema_source = "manifest.observation_schema"
    else:
        raise ValueError("donor manifest observation_schema is invalid")
    if donor_schema.get("version") != BASE_OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            "donor first-15 semantics are unsupported: observation schema is "
            f"{donor_schema.get('version')!r}"
        )
    if donor_schema.get("state_reader") != LEGACY_STATE_READER:
        raise ValueError(
            "donor first-15 semantics are unsupported: state reader is "
            f"{donor_schema.get('state_reader')!r}"
        )
    donor_schema["provenance_source"] = donor_schema_source

    target_resolved = target_config.resolved_dict()
    exact_checks = {
        "physics_model_version": (
            manifest_physics_version(manifest)
            == str(target_config.physics_model_version)
        ),
        "vehicle": dict(resolved.get("vehicle", {}))
        == dict(target_resolved["vehicle"]),
        "actuator": dict(manifest.get("actuator", resolved.get("actuator", {})))
        == dict(target_resolved["actuator"]),
        "action_scale": tuple(
            float(item) for item in donor_environment.get("residual_scale", ())
        )
        == tuple(float(item) for item in target_config.environment.residual_scale),
        "policy_type": str(donor_ppo.get("policy", ""))
        == str(target_config.training.ppo.policy)
        == "MlpPolicy",
        "net_arch": tuple(int(item) for item in donor_ppo.get("net_arch", ()))
        == tuple(target_config.training.ppo.net_arch),
    }
    failures = [name for name, passed in exact_checks.items() if not passed]
    if failures:
        raise ValueError(
            "observation-expansion donor contract mismatch: " + ", ".join(failures)
        )
    return ObservationExpansionCompatibility(
        base_compatibility=base,
        donor_observation_schema=donor_schema,
        target_observation_schema=dict(target_config.observation_schema),
        exact_contract_checks=exact_checks,
    )


def discover_legacy_e2e_donors(target_config: Any) -> list[PolicyDonorProvenance]:
    """Return compatible current best/final model records, newest first."""

    runs_root = Path(target_config.paths.artifact_root) / "runs"
    discovered: list[PolicyDonorProvenance] = []
    if not runs_root.is_dir():
        return discovered
    for manifest_path in sorted(runs_root.glob("*/manifests/*.json")):
        try:
            manifest = _load_manifest(manifest_path)
        except ValueError:
            continue
        models = manifest.get("models", {})
        if not isinstance(models, Mapping):
            continue
        run_dir = manifest_path.parent.parent
        for kind in ("final", "best"):
            record = models.get(kind)
            if not isinstance(record, Mapping) or not isinstance(
                record.get("path"), str
            ):
                continue
            candidate = run_dir / str(record["path"])
            try:
                provenance = validate_legacy_e2e_donor(candidate, target_config)
            except ValueError:
                continue
            discovered.append(provenance)
    return sorted(
        discovered,
        key=lambda item: (item.created_at, item.model_kind == "final"),
        reverse=True,
    )


def copy_policy_parameters(target_model: Any, donor_model: Any) -> None:
    """Copy policy tensors only, retaining the fresh target optimizer/state."""

    import numpy as np

    def compatible_space(target: Any, donor: Any) -> bool:
        if type(target) is not type(donor):
            return False
        if tuple(target.shape) != tuple(donor.shape):
            return False
        if getattr(target, "dtype", None) != getattr(donor, "dtype", None):
            return False
        for bound in ("low", "high"):
            target_bound = getattr(target, bound, None)
            donor_bound = getattr(donor, bound, None)
            if (target_bound is None) != (donor_bound is None):
                return False
            if target_bound is not None and not np.array_equal(
                target_bound, donor_bound
            ):
                return False
        return True

    if not compatible_space(
        target_model.observation_space, donor_model.observation_space
    ):
        raise ValueError(
            "donor observation space does not match the new PPO observation space"
        )
    if not compatible_space(target_model.action_space, donor_model.action_space):
        raise ValueError("donor action space does not match the new PPO action space")
    if int(getattr(target_model, "num_timesteps", -1)) != 0:
        raise ValueError("policy initialization requires a fresh PPO at timestep 0")
    target_policy = target_model.policy
    target_optimizer = getattr(target_policy, "optimizer", None)
    target_policy.load_state_dict(donor_model.policy.state_dict(), strict=True)
    if getattr(target_policy, "optimizer", None) is not target_optimizer:
        raise RuntimeError("policy initialization unexpectedly replaced the optimizer")
    if int(getattr(target_model, "num_timesteps", -1)) != 0:
        raise RuntimeError("policy initialization unexpectedly changed PPO timesteps")


def copy_policy_parameters_with_input_expansion(
    target_model: Any,
    donor_model: Any,
    *,
    base_observation_dim: int = 15,
    numeric_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Copy a Flatten-MLP policy while zero-initializing new input columns.

    Only the actor and critic first Linear weights may differ in shape.  Every
    other policy tensor, including output heads and ``log_std``, is copied
    exactly.  The target PPO optimizer, rollout buffer, and timestep remain
    those of the freshly constructed model.
    """

    import numpy as np
    import torch
    from stable_baselines3.common.torch_layers import FlattenExtractor

    if int(getattr(target_model, "num_timesteps", -1)) != 0:
        raise ValueError("policy expansion requires a fresh PPO at timestep 0")
    if tuple(donor_model.observation_space.shape) != (base_observation_dim,):
        raise ValueError(
            "donor observation space must be exactly the supported 15D base input"
        )
    target_input_dim = int(np.prod(target_model.observation_space.shape))
    if target_input_dim <= base_observation_dim:
        raise ValueError("target observation must be wider than the donor input")
    for bound in ("low", "high"):
        donor_bound = np.asarray(getattr(donor_model.action_space, bound))
        target_bound = np.asarray(getattr(target_model.action_space, bound))
        if not np.array_equal(donor_bound, target_bound):
            raise ValueError(f"donor and target action-space {bound} bounds differ")
    if type(target_model.action_space) is not type(donor_model.action_space):
        raise ValueError("donor and target action-space types differ")

    target_policy = target_model.policy
    donor_policy = donor_model.policy
    if type(target_policy) is not type(donor_policy):
        raise ValueError("donor and target policy classes differ")
    if not isinstance(
        target_policy.features_extractor, FlattenExtractor
    ) or not isinstance(donor_policy.features_extractor, FlattenExtractor):
        raise ValueError("only the verified SB3 FlattenExtractor path is supported")
    if type(target_policy.action_dist) is not type(donor_policy.action_dist):
        raise ValueError("donor and target action distribution classes differ")
    target_policy_modules = tuple(
        type(module).__name__ for module in target_policy.mlp_extractor.modules()
    )
    donor_policy_modules = tuple(
        type(module).__name__ for module in donor_policy.mlp_extractor.modules()
    )
    if target_policy_modules != donor_policy_modules:
        raise ValueError("donor and target MLP module/activation structures differ")

    actor_key = "mlp_extractor.policy_net.0.weight"
    critic_key = "mlp_extractor.value_net.0.weight"
    expanded_keys = {actor_key, critic_key}
    donor_state = donor_policy.state_dict()
    target_state = target_policy.state_dict()
    if set(donor_state) != set(target_state):
        missing = sorted(set(donor_state) ^ set(target_state))
        raise ValueError(f"donor/target policy tensor keys differ: {missing}")
    copied_state: dict[str, Any] = {}
    copied_tensor_names: list[str] = []
    for name, target_tensor in target_state.items():
        donor_tensor = donor_state[name]
        if name in expanded_keys:
            if (
                donor_tensor.ndim != 2
                or target_tensor.ndim != 2
                or donor_tensor.shape[0] != target_tensor.shape[0]
                or donor_tensor.shape[1] != base_observation_dim
                or target_tensor.shape[1] != target_input_dim
            ):
                raise ValueError(
                    f"unsupported first-layer shapes for {name}: "
                    f"donor={tuple(donor_tensor.shape)}, "
                    f"target={tuple(target_tensor.shape)}"
                )
            expanded = torch.zeros_like(target_tensor)
            expanded[:, :base_observation_dim].copy_(donor_tensor)
            copied_state[name] = expanded
        else:
            if donor_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"unsupported non-input tensor shape mismatch for {name}: "
                    f"donor={tuple(donor_tensor.shape)}, "
                    f"target={tuple(target_tensor.shape)}"
                )
            copied_state[name] = donor_tensor.detach().clone()
        copied_tensor_names.append(name)

    optimizer = getattr(target_policy, "optimizer", None)
    if optimizer is None:
        raise ValueError("fresh target policy has no optimizer")
    if len(optimizer.state) != 0:
        raise ValueError("fresh target optimizer already has state")
    rollout_buffer = getattr(target_model, "rollout_buffer", None)
    optimizer_identity = id(optimizer)
    rollout_buffer_identity = id(rollout_buffer)
    target_policy.load_state_dict(copied_state, strict=True)
    if (
        id(target_policy.optimizer) != optimizer_identity
        or target_policy.optimizer.state
    ):
        raise RuntimeError("input expansion copied or replaced optimizer state")
    if id(getattr(target_model, "rollout_buffer", None)) != rollout_buffer_identity:
        raise RuntimeError("input expansion replaced the fresh rollout buffer")
    if int(getattr(target_model, "num_timesteps", -1)) != 0:
        raise RuntimeError("input expansion changed PPO timesteps")

    loaded = target_policy.state_dict()
    for name in expanded_keys:
        if not torch.equal(loaded[name][:, :base_observation_dim], donor_state[name]):
            raise RuntimeError(f"donor input columns were not preserved for {name}")
        if not torch.count_nonzero(loaded[name][:, base_observation_dim:]).item() == 0:
            raise RuntimeError(f"new input columns are not zero for {name}")
        parameter = dict(target_policy.named_parameters())[name]
        if not parameter.requires_grad:
            raise RuntimeError(f"expanded first layer is not trainable: {name}")

    base = np.linspace(-0.75, 0.75, base_observation_dim, dtype=np.float32)
    extras = np.linspace(
        0.9, -0.9, target_input_dim - base_observation_dim, dtype=np.float32
    )
    donor_tensor = torch.as_tensor(base[None, :], device=donor_policy.device)
    target_tensor = torch.as_tensor(
        np.concatenate((base, extras))[None, :], device=target_policy.device
    )
    with torch.no_grad():
        donor_distribution = donor_policy.get_distribution(donor_tensor).distribution
        target_distribution = target_policy.get_distribution(target_tensor).distribution
        donor_value = donor_policy.predict_values(donor_tensor)
        target_value = target_policy.predict_values(target_tensor)
    action_mean_error = float(
        torch.max(torch.abs(donor_distribution.mean - target_distribution.mean)).item()
    )
    action_std_error = float(
        torch.max(
            torch.abs(donor_distribution.stddev - target_distribution.stddev)
        ).item()
    )
    value_error = float(torch.max(torch.abs(donor_value - target_value)).item())
    maximum_error = max(action_mean_error, action_std_error, value_error)
    if maximum_error > numeric_tolerance:
        raise RuntimeError(
            "expanded network does not preserve the donor function for equal "
            f"first-15 input: maximum error {maximum_error}"
        )
    return {
        "strategy": "explicit_mlp_input_expansion_v1",
        "base_input_dimension": base_observation_dim,
        "target_input_dimension": target_input_dim,
        "expanded_tensor_names": sorted(expanded_keys),
        "copied_tensor_names": copied_tensor_names,
        "actor_input_weight_copy": f"[:, :{base_observation_dim}]",
        "critic_input_weight_copy": f"[:, :{base_observation_dim}]",
        "zero_initialized_input_columns": (
            f"[:, {base_observation_dim}:{target_input_dim}]"
        ),
        "all_new_input_weights_trainable": True,
        "optimizer_state_copied": False,
        "rollout_buffer_copied": False,
        "initial_timestep": 0,
        "function_preservation": {
            "numeric_tolerance": numeric_tolerance,
            "action_mean_max_abs_error": action_mean_error,
            "action_std_max_abs_error": action_std_error,
            "value_max_abs_error": value_error,
            "passed": True,
        },
    }


def validate_loaded_observation_schema(
    model: Any, target_config: Any, *, model_path: str | Path | None = None
) -> None:
    """Reject semantic checkpoint/profile mismatches beyond tensor shape."""

    from .observation import BASE_OBSERVATION_SCHEMA_VERSION

    expected = dict(target_config.observation_schema)
    actual = getattr(model, "observation_schema", None)
    label = "loaded PPO" if model_path is None else str(model_path)
    if expected.get("version") == BASE_OBSERVATION_SCHEMA_VERSION and actual is None:
        # Historical 15D archives predate the explicit schema field.
        return
    if not isinstance(actual, Mapping):
        raise ValueError(
            f"model {label} lacks the required observation schema metadata; "
            "shape-only fallback is forbidden for this profile"
        )
    if dict(actual) != expected:
        raise ValueError(
            f"model {label} observation schema does not match the requested "
            f"profile: model={dict(actual)!r}, requested={expected!r}"
        )


__all__ = [
    "E2EPolicyCompatibility",
    "ObservationExpansionCompatibility",
    "PolicyDonorProvenance",
    "copy_policy_parameters",
    "copy_policy_parameters_with_input_expansion",
    "discover_legacy_e2e_donors",
    "validate_e2e_policy_compatibility",
    "validate_legacy_e2e_donor",
    "validate_loaded_observation_schema",
    "validate_observation_expansion_donor",
]

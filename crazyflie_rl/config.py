"""Load and validate experiment profiles without importing simulation packages."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from .contracts import ACTION_DIM, observation_contract


class ConfigError(ValueError):
    """Raised when an experiment profile violates a project contract."""


class MissingResourceError(FileNotFoundError):
    """Raised when a configured, non-generated resource is unavailable."""


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "PyYAML is required to read experiment profiles; install project dependencies first"
        ) from exc
    return yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    yaml = _yaml_module()
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in configuration file {path}: {exc}") from exc
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return content


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"cyclic configuration inheritance: {chain}")

    current = _read_yaml(path)
    extends = current.pop("extends", None)
    if extends is None:
        return current
    if not isinstance(extends, str) or not extends.strip():
        raise ConfigError(f"extends must be a non-empty relative path in {path}")

    parent = (path.parent / extends).resolve()
    base = _load_with_extends(parent, (*stack, path))
    return _deep_merge(base, current)


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key!r} must be a mapping")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class ExperimentConfig(Mapping[str, Any]):
    """Merged experiment data plus deterministic project-relative path resolution."""

    data: dict[str, Any]
    source_path: Path
    project_root: Path

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def profile_name(self) -> str:
        return self.source_path.stem

    @property
    def control_mode(self) -> str:
        return str(self.data["experiment"]["control_mode"])

    @property
    def observation_schema(self) -> str:
        return str(self.data["experiment"]["observation_schema"])

    @property
    def observation_dim(self) -> int:
        return int(self.data["experiment"]["observation_dim"])

    @property
    def action_dim(self) -> int:
        return int(self.data["experiment"]["action_dim"])

    @property
    def residual_scale(self) -> tuple[float, float, float, float]:
        values = self.data["environment"]["residual_scale"]
        return tuple(float(item) for item in values)  # type: ignore[return-value]

    @property
    def seed(self) -> int | None:
        value = self.data["experiment"]["seed"]
        return None if value is None else int(value)

    @property
    def condition(self) -> str:
        return str(self.data["experiment"]["condition"])

    def resolve_path(self, key: str) -> Path:
        paths = _require_mapping(self.data, "paths")
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"paths.{key} must be a non-empty path string")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def resolved_dict(self) -> dict[str, Any]:
        resolved = deepcopy(self.data)
        paths = _require_mapping(self.data, "paths")
        resolved["paths"] = {
            key: str(self.resolve_path(key))
            for key in paths
        }
        resolved["config_profile"] = self.profile_name
        resolved["config_source"] = str(self.source_path)
        return resolved

    def require_runtime_resources(self) -> None:
        xml_path = self.resolve_path("mujoco_xml")
        if not xml_path.is_file():
            raise MissingResourceError(
                "Required MuJoCo XML is missing: "
                f"{xml_path}. Place cf21B_500.xml and every referenced mesh/texture "
                f"under {self.resolve_path('resource_root')} without inventing paths or "
                "placeholder assets, then run validation again."
            )


def _validate(data: dict[str, Any], source_path: Path) -> None:
    paths = _require_mapping(data, "paths")
    required_paths = {
        "resource_root",
        "mujoco_xml",
        "pretrained_model_root",
        "artifact_root",
    }
    missing_paths = sorted(required_paths.difference(paths))
    if missing_paths:
        raise ConfigError(f"missing path settings: {', '.join(missing_paths)}")

    experiment = _require_mapping(data, "experiment")
    mode = experiment.get("control_mode")
    if not isinstance(mode, str):
        raise ConfigError("experiment.control_mode must be a string")
    try:
        contract = observation_contract(mode)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    if experiment.get("observation_schema") != contract.schema:
        raise ConfigError(
            f"{mode} requires observation_schema={contract.schema!r}, got "
            f"{experiment.get('observation_schema')!r}"
        )
    if experiment.get("observation_dim") != contract.dimension:
        raise ConfigError(
            f"{mode} requires observation_dim={contract.dimension}, got "
            f"{experiment.get('observation_dim')!r}"
        )
    if experiment.get("action_dim") != ACTION_DIM:
        raise ConfigError(
            f"all control modes require action_dim={ACTION_DIM}, got "
            f"{experiment.get('action_dim')!r}"
        )
    seed = experiment.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ConfigError(
            f"experiment.seed must be null or a non-negative integer, got {seed!r}"
        )
    condition = experiment.get("condition")
    if not isinstance(condition, str) or not condition.strip():
        raise ConfigError("experiment.condition must be a non-empty string")

    environment = _require_mapping(data, "environment")
    scale = environment.get("residual_scale")
    if not isinstance(scale, list) or len(scale) != ACTION_DIM:
        raise ConfigError(
            "environment.residual_scale must be a 4-value list ordered as "
            "[tau_x, tau_y, tau_z, F_z]"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in scale
    ):
        raise ConfigError("environment.residual_scale values must be finite and positive")
    required_environment = {
        "policy_hz",
        "episode_sec",
        "com_bias_randomize",
        "com_bias_mass",
        "com_bias_offset",
        "att_perturb_deg",
        "pos_perturb",
    }
    missing_environment = sorted(required_environment.difference(environment))
    if missing_environment:
        raise ConfigError(
            f"missing environment settings: {', '.join(missing_environment)}"
        )
    if not isinstance(environment["com_bias_randomize"], bool):
        raise ConfigError("environment.com_bias_randomize must be a boolean")
    offset = environment["com_bias_offset"]
    if not isinstance(offset, list) or len(offset) != 2:
        raise ConfigError("environment.com_bias_offset must be a 2-value list")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in offset
    ):
        raise ConfigError("environment.com_bias_offset values must be finite numbers")
    for key in (
        "policy_hz",
        "episode_sec",
        "com_bias_mass",
        "att_perturb_deg",
        "pos_perturb",
    ):
        value = environment[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ConfigError(f"environment.{key} must be finite and non-negative")
    if float(environment["policy_hz"]) <= 0.0 or float(environment["episode_sec"]) <= 0.0:
        raise ConfigError("environment policy_hz and episode_sec must be positive")

    training = _require_mapping(data, "training")
    _positive_int(training.get("total_timesteps"), "training.total_timesteps")
    ppo = _require_mapping(training, "ppo")
    required_ppo = {
        "policy",
        "verbose",
        "device",
        "n_steps",
        "batch_size",
        "gamma",
        "gae_lambda",
        "n_epochs",
        "learning_rate",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "normalize_advantage",
        "clip_range",
        "target_kl",
        "policy_kwargs",
    }
    missing_ppo = sorted(required_ppo.difference(ppo))
    if missing_ppo:
        raise ConfigError(f"missing training.ppo settings: {', '.join(missing_ppo)}")
    for key in ("n_steps", "batch_size", "n_epochs"):
        _positive_int(ppo.get(key), f"training.ppo.{key}")
    if isinstance(ppo["verbose"], bool) or not isinstance(ppo["verbose"], int):
        raise ConfigError("training.ppo.verbose must be an integer")
    if not isinstance(ppo["policy"], str) or not ppo["policy"].strip():
        raise ConfigError("training.ppo.policy must be a non-empty string")
    if not isinstance(ppo["device"], str) or not ppo["device"].strip():
        raise ConfigError("training.ppo.device must be a non-empty string")
    if not isinstance(ppo["normalize_advantage"], bool):
        raise ConfigError("training.ppo.normalize_advantage must be a boolean")
    for key in (
        "gamma",
        "gae_lambda",
        "learning_rate",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "clip_range",
        "target_kl",
    ):
        value = ppo[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ConfigError(f"training.ppo.{key} must be finite and non-negative")
    policy_kwargs = _require_mapping(ppo, "policy_kwargs")
    if set(policy_kwargs) != {"log_std_init", "net_arch"}:
        raise ConfigError(
            "training.ppo.policy_kwargs must contain exactly log_std_init and net_arch"
        )
    if not isinstance(policy_kwargs["net_arch"], list) or not policy_kwargs["net_arch"]:
        raise ConfigError("training.ppo.policy_kwargs.net_arch must be a non-empty list")
    for index, width in enumerate(policy_kwargs["net_arch"]):
        _positive_int(width, f"training.ppo.policy_kwargs.net_arch[{index}]")
    log_std = policy_kwargs["log_std_init"]
    if (
        isinstance(log_std, bool)
        or not isinstance(log_std, (int, float))
        or not math.isfinite(float(log_std))
    ):
        raise ConfigError("training.ppo.policy_kwargs.log_std_init must be finite")

    evaluation = _require_mapping(data, "evaluation")
    for key in ("n_episodes", "seed_start", "every_steps"):
        value = evaluation.get(key)
        if key == "seed_start":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(f"evaluation.{key} must be a non-negative integer")
        else:
            _positive_int(value, f"evaluation.{key}")
    for key in ("tail_fraction", "tilt_limit_deg"):
        value = evaluation.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ConfigError(f"evaluation.{key} must be finite and positive")
    if float(evaluation["tail_fraction"]) > 1.0:
        raise ConfigError("evaluation.tail_fraction must not exceed 1.0")
    if not isinstance(evaluation.get("deterministic"), bool):
        raise ConfigError("evaluation.deterministic must be a boolean")

    if source_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigError(f"configuration file must use .yaml or .yml: {source_path}")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a profile, recursively merge ``extends``, and enforce mode contracts."""

    source_path = Path(path).expanduser().resolve()
    data = _load_with_extends(source_path)
    _validate(data, source_path)

    # Profiles live in <project>/configs. Keeping this convention explicit makes
    # all configured relative paths independent of the caller's working directory.
    project_root = source_path.parent.parent.resolve()
    config = ExperimentConfig(
        data=data,
        source_path=source_path,
        project_root=project_root,
    )
    resource_root = config.resolve_path("resource_root")
    for key in ("mujoco_xml", "pretrained_model_root"):
        resolved = config.resolve_path(key)
        if not resolved.is_relative_to(resource_root):
            raise ConfigError(
                f"paths.{key} must be inside paths.resource_root: "
                f"{resolved} is outside {resource_root}"
            )
    artifact_root = config.resolve_path("artifact_root")
    if artifact_root == resource_root or artifact_root.is_relative_to(resource_root):
        raise ConfigError(
            "paths.artifact_root must be separate from the static resource root"
        )
    return config

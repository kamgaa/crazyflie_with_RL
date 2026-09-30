"""Read-only checkpoint provenance and frozen observation preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from .eval_cli import _load_policy

OBSERVATION_CONTRACT = 'position_error,absolute_velocity,wxyz,body_omega,sin_yaw_error,cos_yaw_error'
VELOCITY_SLICE = slice(3, 6)
VELOCITY_INPUTS = ('absolute', 'error')


def policy_raw_observation(observation, reference_velocity, mode='absolute'):
    """Evaluation intervention before frozen normalization; never mutate raw state."""
    if mode not in VELOCITY_INPUTS:
        raise ValueError(f'unknown velocity input: {mode}')
    if mode == 'absolute':
        return observation  # Preserve the original inference path and dtype.
    reference_velocity = np.asarray(reference_velocity)
    if np.shape(observation) != (15,) or reference_velocity.shape != (3,) or not np.all(np.isfinite(reference_velocity)):
        raise ValueError('expected raw 15D observation and finite world reference velocity')
    value = observation.copy()
    value[VELOCITY_SLICE] = observation[VELOCITY_SLICE] - reference_velocity
    return value


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_value(value):
    label, separator, path = value.partition('=')
    if not separator or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', label) or not path:
        raise ValueError('expected LABEL=PATH; label may contain letters, digits, _ and -')
    return label, path


def labeled(values):
    result = {}
    for value in values:
        label, path = label_value(value)
        if label in result:
            raise ValueError(f'duplicate label: {label}')
        result[label] = path
    return result


def training_manifest(checkpoint, explicit=None):
    """Discover only sidecars or a run manifest actually referencing this file."""
    if explicit:
        path = Path(explicit).expanduser().resolve(strict=True)
        return path, json.loads(path.read_text())
    sidecar = checkpoint.with_suffix('.manifest.json')
    if sidecar.is_file():
        return sidecar, json.loads(sidecar.read_text())
    run = checkpoint.parent.parent
    for path in sorted((run / 'manifests').glob('*manifest*.json')):
        data = json.loads(path.read_text())
        records = list(data.get('models', {}).values()) + data.get('model_history', [])
        if any(isinstance(r, dict) and r.get('path') and
               (run / r['path']).resolve() == checkpoint for r in records):
            return path, data
    return None, None


def validate_metadata(data, config):
    """Check each available source; absence is not a successful verification."""
    env = data.get('resolved_config', {}).get('environment', {})
    for source in (data, env):
        mode = source.get('control_mode')
        if mode is not None and mode != 'e2e':
            raise ValueError(f'checkpoint control_mode mismatch: {mode}')
        for key, expected in (('observation_shape', [15]), ('action_shape', [4])):
            if key in source and list(source[key]) != expected:
                raise ValueError(f'checkpoint {key} mismatch')
        for key in ('residual_scale', 'action_scale'):
            if key in source and not np.array_equal(source[key], config.environment.residual_scale):
                raise ValueError(f'checkpoint action scale mismatch: {source[key]}')
        if source.get('observation_contract', OBSERVATION_CONTRACT) != OBSERVATION_CONTRACT:
            raise ValueError('unsupported checkpoint observation contract')
        if source.get('velocity_input', 'absolute') != 'absolute':
            raise ValueError('checkpoint velocity input must be absolute, not error')
    wrappers = data.get('wrappers', [])
    if not isinstance(wrappers, list) or any(w not in ('Monitor', 'DummyVecEnv', 'VecNormalize') for w in wrappers):
        raise ValueError('unsupported or unspecified checkpoint wrapper transformation')


@dataclass
class FrozenPolicy:
    model: object
    provenance: dict
    normalization_path: Path | None
    normalizer: object = None

    def bind(self, env):
        self.normalizer = None
        if self.normalization_path:
            if sha256(self.normalization_path) != self.provenance['normalization']['sha256']:
                raise ValueError('normalization statistics changed after model inspection')
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            # The wrapper only supplies frozen normalize_obs. We never reset or
            # step this vector wrapper, so it cannot reset a completed episode.
            self.normalizer = VecNormalize.load(str(self.normalization_path), DummyVecEnv([lambda: env]))
            self.normalizer.training = False
            self.normalizer.norm_reward = False
            if self.normalizer.observation_space.shape != (15,):
                raise ValueError('VecNormalize observation shape mismatch')
            if self.normalizer.norm_obs:
                rms = self.normalizer.obs_rms
                if isinstance(rms, dict) or np.shape(rms.mean) != (15,) or not np.all(np.isfinite(rms.mean)):
                    raise ValueError('invalid VecNormalize observation statistics')
                if np.shape(rms.var) != (15,) or not np.all(np.isfinite(rms.var)) or np.any(rms.var < 0):
                    raise ValueError('invalid VecNormalize observation variance')
            self.provenance['normalization']['frozen'] = True

    def predict(self, observation):
        value = self.normalizer.normalize_obs(observation) if self.normalizer else observation
        return self.model.predict(value, deterministic=True)[0]


def load_frozen_policy(label, checkpoint, config, *, manifest=None, normalization=None):
    if checkpoint.lower() in ('latest', 'latest-best'):
        raise ValueError('explicit checkpoint zip path required; latest is not supported')
    path = Path(checkpoint).expanduser().resolve(strict=True)
    if path.suffix.lower() != '.zip' or not path.is_file():
        raise ValueError(f'explicit .zip checkpoint required: {path}')
    digest = sha256(path)
    mp, md = training_manifest(path, manifest)
    md = md or {}
    validate_metadata(md, config)
    for key in ('sha256', 'checkpoint_sha256'):
        if key in md and md[key] != digest:
            raise ValueError('training manifest checkpoint SHA256 mismatch')
    with zipfile.ZipFile(path) as archive:
        saved = json.loads(archive.read('data'))
    validate_metadata(saved, config)
    # PPO excludes VecNormalize itself from its zip. A non-null original obs is
    # positive evidence of its use; normalize_advantage is unrelated.
    norm_meta = md.get('normalization', {})
    if isinstance(norm_meta, bool):
        norm_meta = {'enabled': norm_meta}
    if not isinstance(norm_meta, dict):
        raise ValueError('normalization metadata must be an object or boolean')
    needed = (saved.get('_last_original_obs') is not None or
              norm_meta.get('enabled') is True or norm_meta.get('norm_obs') is True or
              'VecNormalize' in md.get('wrappers', []))
    if normalization is None:
        candidate = norm_meta.get('path')
        if candidate:
            normalization = str((mp.parent / candidate).resolve())
        elif needed:
            raise ValueError(f'{label}: saved observation normalization statistics required; use --normalization {label}=PATH')
        elif (md.get('command') or [None])[0] in ('train_ppo_02.py', 'train_ppo.py') and '_last_original_obs' in saved:
            normalization = 'none'
            norm_source = 'repository training command + null PPO _last_original_obs (no VecNormalize evidence)'
        elif norm_meta.get('enabled') is False:
            normalization = 'none'
            norm_source = 'manifest declaration'
        else:
            raise ValueError(f'{label}: preprocessing is unknown; provide --normalization {label}=none or a saved VecNormalize path')
    else:
        norm_source = 'explicit CLI declaration'
    if needed and normalization == 'none':
        raise ValueError(f'{label}: normalized checkpoint requires saved statistics; none is incompatible')
    norm_path = None if normalization == 'none' else Path(normalization).expanduser().resolve(strict=True)
    model = _load_policy(path, config)
    if not (np.array_equal(model.action_space.low, -np.ones(4)) and
            np.array_equal(model.action_space.high, np.ones(4))):
        raise ValueError('checkpoint action bounds must be [-1, 1]')
    model.policy.set_training_mode(False)
    scale_known = any(k in source for source in (md, md.get('resolved_config', {}).get('environment', {}), saved)
                      for k in ('residual_scale', 'action_scale'))
    provenance = {
        'label': label, 'path': str(path), 'sha256': digest,
        'training_manifest_path': str(mp) if mp else None,
        'training_manifest_sha256': sha256(mp) if mp else None,
        'training_manifest': md or None,
        'observation_shape': list(model.observation_space.shape), 'action_shape': list(model.action_space.shape),
        'observation_contract': OBSERVATION_CONTRACT,
        'action_scale_verification': 'matched_available_metadata' if scale_known else 'unknown_no_training_scale_metadata',
        'applied_action_scale': list(config.environment.residual_scale),
        'normalization': {'path': str(norm_path) if norm_path else None,
                          'sha256': sha256(norm_path) if norm_path else None,
                          'source': 'saved VecNormalize statistics' if norm_path else norm_source,
                          'limitation': 'PPO zip alone cannot prove absence of custom external preprocessing'},
    }
    return FrozenPolicy(model, provenance, norm_path)

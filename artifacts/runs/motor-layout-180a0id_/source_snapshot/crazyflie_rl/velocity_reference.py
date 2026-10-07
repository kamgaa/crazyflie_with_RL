"""Stateless E2E velocity reference and versioned policy input contract."""

ABSOLUTE_CONTRACT = 'position_error,absolute_velocity,wxyz,body_omega,sin_yaw_error,cos_yaw_error'
POSITION_VELOCITY_CONTRACT = 'position_error,position_generated_velocity_error,wxyz,body_omega,sin_yaw_error,cos_yaw_error'
VELOCITY_SLICE = slice(3, 6)


def velocity_mode(settings, channel):
    """Independent observation/reward selection, with the legacy coupled fallback."""
    if channel not in ('observation', 'reward'):
        raise ValueError(f'unknown velocity channel: {channel}')
    return (getattr(settings, f'{channel}_mode', None) or settings.mode) if settings else 'absolute'


def _settings(config):
    return getattr(getattr(config, 'environment', None), 'e2e_velocity', None)


def _semantics(settings, mode):
    if mode == 'absolute':
        return {'mode': 'absolute'}
    return {'mode': 'position_error', 'position_gain': settings.position_gain,
            'max_speed': settings.max_speed}


def velocity_semantics(config):
    """Policy INPUT contract only. Reward changes do not invalidate a policy."""
    settings = _settings(config)
    return _semantics(settings, velocity_mode(settings, 'observation'))


def velocity_reward_semantics(config):
    settings = _settings(config)
    return _semantics(settings, velocity_mode(settings, 'reward'))


def desired_velocity_semantics(config):
    settings = _settings(config)
    return _semantics(settings, 'position_error' if _reference_enabled(settings) else 'absolute')


def _reference_enabled(settings):
    return settings is not None and (settings.mode == 'position_error' or any(
        velocity_mode(settings, channel) == 'position_error' for channel in ('observation', 'reward')))


def observation_contract(config):
    return (ABSOLUTE_CONTRACT if velocity_semantics(config)['mode'] == 'absolute'
            else POSITION_VELOCITY_CONTRACT)


def desired_velocity(position_error, settings=None):
    import numpy as np
    error = np.asarray(position_error, dtype=float)
    if error.shape != (3,) or not np.all(np.isfinite(error)):
        raise ValueError('position error must be a finite world-frame 3-vector')
    if not _reference_enabled(settings):
        return np.zeros(3)
    value = -settings.position_gain * error
    norm = np.linalg.norm(value)
    if norm > settings.max_speed:
        value *= settings.max_speed / norm
    return value


def validate_velocity_metadata(data, config):
    """Absent metadata is legacy absolute; never accept it for the new contract."""
    source = data.get('resolved_config', {}).get('environment', {}).get('e2e_velocity')
    if source:
        mode = source.get('observation_mode') or source.get('mode', 'absolute')
        source = ({'mode': 'absolute'} if mode == 'absolute' else
                  {'mode': mode, 'position_gain': source['position_gain'], 'max_speed': source['max_speed']})
    saved = data.get('velocity_semantics', source) or {'mode': 'absolute'}
    if saved.get('mode') == 'absolute':
        saved = {'mode': 'absolute'}
    if saved != velocity_semantics(config):
        raise ValueError(f'checkpoint velocity semantics mismatch: saved={saved}, requested={velocity_semantics(config)}')
    if data.get('observation_contract', observation_contract(config)) != observation_contract(config):
        raise ValueError('checkpoint observation contract mismatch')

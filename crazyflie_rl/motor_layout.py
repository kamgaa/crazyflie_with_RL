"""Motor boundary conventions; engine and PPO remain native FLU / world Z-up.

Directions are BODY REACTION torque signs along native +Z, not hinge speeds.
User propeller CW/CCW is a physical assumption, not a rendering command.
"""
from __future__ import annotations

import numpy as np

LAYOUTS = ('legacy', 'user_frd')
S = np.diag([1., -1., -1.])
P = np.eye(4)[::-1].copy()  # f_native = P @ f_user
USER_REACTION_FRD = np.array([-1., 1., -1., 1.])


def reaction_directions(layout, legacy):
    if layout not in LAYOUTS:
        raise ValueError(f'unknown reaction_torque_layout {layout!r}; expected {LAYOUTS}')
    values = np.asarray(legacy, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError('motor_direction must contain four finite native +Z reaction signs')
    # Preserve legacy (including explicitly configured old directions) exactly.
    return tuple(map(float, values if layout == 'legacy' else S[2, 2] * (P @ USER_REACTION_FRD)))


def user_from_native(values, axis=-1):
    values = np.asarray(values)
    if values.shape[axis] != 4:
        raise ValueError('motor axis must have length four')
    return np.take(values, [3, 2, 1, 0], axis=axis).copy()


def native_from_user(values, axis=-1):
    return user_from_native(values, axis=axis)


def body_frd_from_native(values):
    return np.asarray(values) @ S.T


def world_from_frd(rotation_world_from_native):
    return np.asarray(rotation_world_from_native) @ S.T


def exposed_motor_index(number, layout):
    if number not in (1, 2, 3, 4) or layout not in LAYOUTS:
        raise ValueError('motor number must be 1..4 and layout must be explicit')
    return number-1 if layout == 'legacy' else 4-number


def exposed_values(values, layout):
    return np.asarray(values).copy() if layout == 'legacy' else user_from_native(values)


def layout_metadata(env):
    layout = getattr(env, 'reaction_torque_layout', 'legacy')
    return dict(reaction_torque_layout=layout, world_frame='right-handed Z-up',
        native_body_frame='FLU', policy_observation_frame='unchanged mixed world/native FLU',
        exposed_motor_numbering='native 1..4' if layout == 'legacy' else 'user FRD 1..4',
        legacy_csv_motor_arrays='native order, unchanged', user_csv_motor_arrays='explicit user_ prefix',
        motor_direction_meaning='body reaction torque sign along native +Z',
        motor_direction_native=env.motor_direction.tolist(), S_native_to_frd=S.tolist(),
        P_native_from_user=P.tolist(), native_index_to_user_id=[4, 3, 2, 1],
        payload_offset_input_frame='native FLU body, never world',
        payload_offset_native_m=env._com_off3.tolist(),
        payload_offset_user_frd_m=body_frd_from_native(env._com_off3).tolist(),
        propeller_spin='assumed opposite body reaction; no hinge animation or rotor dynamics change')


def user_motor_signals(row):
    """Additional columns, never relabel or overwrite existing native columns.

    Reaction torque scalars are about native +Z even in user motor order;
    explicit *_frd columns carry the FRD +Z sign instead.
    """
    keys = ('motor_thrust_unclipped', 'motor_thrust_command', 'motor_thrust_nominal',
        'motor_thrust_actual', 'motor_reaction_nominal', 'motor_reaction_actual',
        'motor_effectiveness', 'motor_command', 'motor_omega', 'allocator_clipped',
        'allocator_lower', 'allocator_upper', 'allocator_lower_margin_n', 'allocator_upper_margin_n',
        'esc_lower', 'esc_upper', 'effective_max_thrust_n', 'effective_max_margin_n',
        'allocator_efficiency', 'plant_efficiency', 'reaction_torque_nm',
        'motor_thrust_before_effectiveness', 'motor_thrust_applied', 'motor_thrust',
        'reaction_torque_before_effectiveness_nm')
    extra = {'user_'+k: user_from_native(row[k]) for k in keys if k in row}
    for key in ('motor_reaction_actual', 'motor_reaction_nominal', 'reaction_torque_nm'):
        if key in row: extra['user_'+key+'_frd'] = -user_from_native(row[key])
    return extra

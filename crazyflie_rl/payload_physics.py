"""Rigid point payload composition and plant-only diagnostics.

Spatial wrenches here are [torque xyz, force xyz], expressed in body axes.
No function updates the controller, allocator, sampling or motor dynamics.
"""

import itertools
import numpy as np

from .controllers import rotmat_from_quat_wxyz


def bounded_equilibrium(matrix, target, bounds, inequalities=None, limits=None):
    """Small bounded linear feasibility problem by nullspace vertex enumeration.

    NumPy-only: at most eight variables (four thrusts and four actions).
    The bounded feasible polytope has a vertex if it is nonempty.
    """
    a, b = np.asarray(matrix), np.asarray(target)
    x = np.linalg.lstsq(a, b, rcond=None)[0]
    if np.max(np.abs(a @ x - b)) > 1e-9:
        return None
    _, s, vt = np.linalg.svd(a, full_matrices=True)
    rank = (
        np.count_nonzero(s > max(a.shape) * s[0] * np.finfo(float).eps)
        if len(s) and s[0]
        else 0
    )
    null = vt[rank:].T
    lo, hi = np.asarray(bounds).T
    g = np.vstack([np.eye(len(x)), -np.eye(len(x))])
    h = np.r_[hi, -lo]
    if inequalities is not None and len(inequalities):
        g = np.vstack([g, inequalities])
        h = np.r_[h, limits]
    if np.all(g @ x <= h + 1e-10):
        return x
    if null.shape[1] == 0:
        return None
    gn, hn = g @ null, h - g @ x
    for selected in itertools.combinations(range(len(h)), null.shape[1]):
        sub = gn[list(selected)]
        if np.linalg.matrix_rank(sub) < null.shape[1]:
            continue
        z = np.linalg.solve(sub, hn[list(selected)])
        if np.all(gn @ z <= hn + 1e-10):
            return x + null @ z
    return None


def parallel_axis(displacement):
    d = np.asarray(displacement, dtype=float)
    return (d @ d) * np.eye(3) - np.outer(d, d)


def inertia_body(model, bid):
    rotation = rotmat_from_quat_wxyz(model.body_iquat[bid])
    return rotation @ np.diag(model.body_inertia[bid]) @ rotation.T


def compose_point_payload(mass, com, inertia, payload_mass, attachment):
    total = mass + payload_mass
    center = (mass * com + payload_mass * attachment) / total
    tensor = inertia + mass * payload_mass / total * parallel_axis(attachment - com)
    return total, center, (tensor + tensor.T) / 2


def principal_axes(tensor):
    import mujoco

    values, axes = np.linalg.eigh(tensor)
    if values[0] <= 0 or values[2] > values[0] + values[1] + 1e-14:
        raise ValueError(
            "payload inertia must be positive and obey triangle inequalities"
        )
    # A principal frame must be a proper rotation, not a reflection.
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, axes.ravel())
    if quaternion[0] < 0:
        quaternion *= -1
    return values, quaternion


def subtree_properties(env):
    """Whole vehicle with children counted once; locked-joint inertia only."""
    m, d, root = env.model, env.data, env.drone_bid
    ids = [root]
    for i in range(root + 1, m.nbody):
        if int(m.body_parentid[i]) in ids:
            ids.append(i)
    rotation = d.xmat[root].reshape(3, 3)
    positions = np.array([rotation.T @ (d.xipos[i] - d.xpos[root]) for i in ids])
    masses = m.body_mass[ids]
    total = float(masses.sum())
    center = masses @ positions / total
    tensor = np.zeros((3, 3))
    for i, p, mass in zip(ids, positions, masses):
        axes = rotation.T @ d.ximat[i].reshape(3, 3)
        tensor += axes @ np.diag(m.body_inertia[i]) @ axes.T + mass * parallel_axis(
            p - center
        )
    return total, center, tensor, ids


def motor_geometry(env, center):
    """XML transmission geometry at current pose, relative to a body-frame COM."""
    m, d = env.model, env.data
    rotation = d.xmat[env.drone_bid].reshape(3, 3)
    origin = d.xpos[env.drone_bid]
    matrices = []
    for actuators in (env.act_force, env.act_torque):
        columns = []
        for aid in actuators:
            sid = m.actuator_trnid[aid, 0]
            axes = rotation.T @ d.site_xmat[sid].reshape(3, 3)
            lever = rotation.T @ (d.site_xpos[sid] - origin) - center
            force = axes @ m.actuator_gear[aid, :3]
            torque = axes @ m.actuator_gear[aid, 3:] + np.cross(lever, force)
            columns.append(np.r_[torque, force])
        matrices.append(np.column_stack(columns))
    return matrices


def wrench_snapshot(env):
    """Motor, applied external, and gravity wrenches with explicit references."""
    m, d, root = env.model, env.data, env.drone_bid
    rotation = d.xmat[root].reshape(3, 3)
    total, center, tensor, ids = subtree_properties(env)
    force_map, torque_map = motor_geometry(env, center)
    motor = (
        force_map @ d.actuator_force[env.act_force]
        + torque_map @ d.actuator_force[env.act_torque]
    )
    body_force, body_torque = motor_geometry(env, m.body_ipos[root])
    motor_body = (
        body_force @ d.actuator_force[env.act_force]
        + body_torque @ d.actuator_force[env.act_torque]
    )
    external = np.zeros(6)
    for i in ids:
        force = rotation.T @ d.xfrc_applied[i, :3]
        torque = rotation.T @ d.xfrc_applied[i, 3:]
        lever = rotation.T @ (d.xipos[i] - d.xpos[root]) - center
        external += np.r_[torque + np.cross(lever, force), force]
    return {
        "wrench_order": "torque_xyz_force_xyz",
        "frame": "body",
        "nominal_allocator_wrench_order": "torque_xyz_collective_thrust",
        "nominal_allocator_requested_wrench": np.asarray(
            env._last_wrench_allocated, dtype=float
        ).tolist(),
        "nominal_allocator_actual_wrench": np.asarray(
            env._last_wrench_actual, dtype=float
        ).tolist(),
        "motor_wrench_vehicle_com_body": motor.tolist(),
        "motor_wrench_vehicle_com_world": np.r_[
            rotation @ motor[:3], rotation @ motor[3:]
        ].tolist(),
        "motor_wrench_drone_com_body": motor_body.tolist(),
        "external_applied_wrench_vehicle_com_body": external.tolist(),
        "external_applied_wrench_vehicle_com_world": np.r_[
            rotation @ external[:3], rotation @ external[3:]
        ].tolist(),
        "external_applied_wrench_drone_com_body": np.r_[
            rotation.T @ d.xfrc_applied[root, 3:], rotation.T @ d.xfrc_applied[root, :3]
        ].tolist(),
        "generalized_external_force": d.qfrc_applied.tolist(),
        "generalized_constraint_force": d.qfrc_constraint.tolist(),
        "contact_count": int(d.ncon),
        "gravity_wrench_vehicle_com_body": np.r_[
            np.zeros(3), rotation.T @ (total * m.opt.gravity)
        ].tolist(),
        "vehicle_com_body_m": center.tolist(),
        "vehicle_locked_inertia_body_kg_m2": tensor.tolist(),
        "child_joint_note": "locked-joint approximation; excludes relative rotor kinetic energy; contacts are separate",
    }


def static_hover(env):
    """Level, zero-rate equilibrium with bounded linear certification.

    For nonlinear reaction torque a failed bounded search is indeterminate,
    not proof of infeasibility. Reachability is through the unchanged clipped
    nominal allocator at its zero-error, zero-integrator command baseline.
    """
    total, center, _, _ = subtree_properties(env)
    force_map, torque_map = motor_geometry(env, center)
    # Body axes are assumed level with world axes for this static check.
    target = np.r_[np.zeros(3), -total * env.model.opt.gravity]
    actuator = env.actuator_model
    lower = max(0.0, env.thrust_min)
    upper = min(env.thrust_max, actuator._mapping_thrust_max)
    # Steady-state motor command <= 1 also bounds attainable actual thrust.
    upper = np.minimum(
        upper, actuator.thrust_from_omega(actuator.steady_state_gain_rad_s)
    )
    upper = np.broadcast_to(upper, (4,)).copy()

    def reaction(f):
        return actuator._reaction_torque(f, actuator.inverse_thrust(f), np.zeros(4))

    linear = actuator.reaction_torque_model == "legacy_ratio"
    matrix = force_map + torque_map @ np.diag(reaction(np.ones(4))) if linear else None
    balance = lambda f: force_map @ f + torque_map @ reaction(f) - target
    bounds = list(zip(np.full(4, lower), upper))
    unbounded = np.linalg.lstsq(matrix, target, rcond=None)[0] if linear else None
    if linear:
        f = bounded_equilibrium(matrix, target, bounds)
        status = "feasible" if f is not None else "infeasible"
    else:
        # For planar rotors, force and roll/pitch balance leave one free
        # parameter. Bracket the nonlinear yaw balance without a SciPy dependency.
        rows = [0, 1, 3, 4, 5]
        a = force_map[rows]
        b = target[rows]
        start = bounded_equilibrium(a, b, bounds)
        f = None
        _, s, vt = np.linalg.svd(a, full_matrices=True)
        rank = np.count_nonzero(s > 1e-12)
        if start is not None and rank == 3 and np.max(np.abs(torque_map[rows])) < 1e-12:
            direction = vt[-1]
            left, right = -np.inf, np.inf
            for i, value in enumerate(direction):
                if abs(value) > 1e-12:
                    ends = sorted(
                        ((lower - start[i]) / value, (upper[i] - start[i]) / value)
                    )
                    left, right = max(left, ends[0]), min(right, ends[1])
            grid = np.linspace(left, right, 129)

            def yaw(t):
                return balance(np.clip(start + t * direction, lower, upper))[2]

            for l, r in zip(grid[:-1], grid[1:]):
                yl, yr = yaw(l), yaw(r)
                if abs(yl) < 1e-10:
                    f = np.clip(start + l * direction, lower, upper)
                    break
                if yl * yr <= 0 or abs(yr) < 1e-10:
                    for _ in range(50):
                        mid = (l + r) / 2
                        ym = yaw(mid)
                        if yl * ym <= 0:
                            r = mid
                        else:
                            l, yl = mid, ym
                    f = np.clip(start + (l + r) / 2 * direction, lower, upper)
                    break
        if f is not None and np.max(np.abs(balance(f))) > 1e-9:
            f = None
        status = "feasible" if f is not None else "indeterminate"

    if f is not None:
        actual = actuator.thrust_from_omega(
            np.minimum(actuator.inverse_thrust(f), actuator.steady_state_gain_rad_s)
        )
        if np.max(np.abs(actual - f)) > 1e-9:
            # A tiny positive branch dead zone is not a continuous interval.
            # Do not certify an equilibrium the actuator cannot actually hold.
            f, status = None, "indeterminate"

    baseline = np.array([0.0, 0.0, 0.0, env.mass * env.gravity])
    mapping = env.B_pinv @ np.diag(env.residual_scale)
    offset = env.B_pinv @ baseline
    reachable, action, reachable_f = None, None, None
    if status == "infeasible":
        reachable = False
    elif linear and status == "feasible":
        reachable = False
        # Enumerate allocator clipping regimes. This also finds alternate
        # physical equilibria, rather than only testing one LP solution.
        for regime in itertools.product((0, -1, 1), repeat=4):
            eq = [np.r_[row, np.zeros(4)] for row in matrix]
            rhs = list(target)
            inequalities, limits = [], []
            for i, state in enumerate(regime):
                fi = np.eye(4)[i]
                if state == 0:
                    eq.append(np.r_[fi, -mapping[i]])
                    rhs.append(offset[i])
                else:
                    # Includes steady-state ESC/RPM saturation, not just the
                    # allocator's nominal thrust clipping limits.
                    bound = lower if state == -1 else upper[i]
                    eq.append(np.r_[fi, np.zeros(4)])
                    rhs.append(bound)
                    inequalities.append(np.r_[np.zeros(4), -state * mapping[i]])
                    limits.append(state * (offset[i] - bound))
            candidate = bounded_equilibrium(
                np.array(eq), rhs, bounds + [(-1, 1)] * 4, inequalities, limits
            )
            if candidate is not None:
                reachable, reachable_f, action = True, candidate[:4], candidate[4:]
                break
    elif f is not None:
        candidate = bounded_equilibrium(mapping, f - offset, [(-1, 1)] * 4)
        if candidate is not None:
            reachable, action, reachable_f = True, candidate, f
    return {
        "assumption": "level attitude, zero velocity/rates, no external wrench/contact, steady-state motors",
        "physical_status": status,
        "physically_feasible": True
        if status == "feasible"
        else False
        if status == "infeasible"
        else None,
        "equilibrium_motor_thrust_n": None if f is None else f.tolist(),
        "unbounded_equilibrium_motor_thrust_n": None
        if unbounded is None
        else unbounded.tolist(),
        "unbounded_equilibrium_limit_violation_n": None
        if unbounded is None
        else np.maximum(np.maximum(lower - unbounded, unbounded - upper), 0).tolist(),
        "equilibrium_residual_torque_force": None if f is None else balance(f).tolist(),
        "total_mass_kg": total,
        "vehicle_com_body_m": center.tolist(),
        "actual_thrust_limits_n": [[lower, float(u)] for u in upper],
        "policy_allocator_reachable": reachable,
        "reachable_equilibrium_motor_thrust_n": None
        if reachable_f is None
        else reachable_f.tolist(),
        "equilibrium_normalized_action": None if action is None else action.tolist(),
        "reachability_baseline_wrench": baseline.tolist(),
        "reachability_scope": "E2E baseline; residual conditional on zero-error/zero-integrator PID baseline",
        "reaction_torque_model": actuator.reaction_torque_model,
    }

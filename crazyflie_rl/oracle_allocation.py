"""Evaluation-only efficiency oracle and isolated, engine-based static audit.

No plant, PPO, action bias, motor limits or integral rules are changed. B0 is
always retained; only its inverse used for allocation can be replaced.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .payload_motor_eval import RecordedFaultEnv
from .interactive_eval import rotor_mapping


def efficiency_matrix(matrix, efficiency):
    eta = np.asarray(efficiency, dtype=float)
    if eta.shape != (4,) or not np.all(np.isfinite(eta)) or np.any((eta < 0) | (eta > 1)):
        raise ValueError('efficiency must have four finite values in [0,1]')
    return np.asarray(matrix) * eta[None, :]


def plant_geometry(env):
    """Rotor wrench about drone body origin, expressed in drone body axes.

    Unlike allocator B0, this uses the actual XML sites/gears. Keep the small
    0.035355 vs 0.03536 m difference explicit, never repair it silently.
    """
    model, data, bid = env.model, env.data, env.drone_bid
    rotation = data.xmat[bid].reshape(3, 3)
    origin = data.xpos[bid]
    force, torque, reaction = [], [], []
    for fa, ta in zip(env.act_force, env.act_torque):
        fs, ts = (int(model.actuator_trnid[a, 0]) for a in (fa, ta))
        for a in (fa, ta):
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_SITE:
                raise ValueError('static audit requires the current site actuators')
            if model.actuator_ctrllimited[a] or model.actuator_forcelimited[a]:
                raise ValueError('unexpected engine actuator bounds; audit them before running')
        fr = rotation.T @ data.site_xmat[fs].reshape(3, 3)
        tr = rotation.T @ data.site_xmat[ts].reshape(3, 3)
        f = fr @ model.actuator_gear[fa, :3]
        moment = np.cross(rotation.T @ (data.site_xpos[fs]-origin), f) + fr @ model.actuator_gear[fa, 3:]
        np.testing.assert_allclose(tr @ model.actuator_gear[ta, :3], 0, atol=1e-15)
        force.append(f); torque.append(moment); reaction.append(tr @ model.actuator_gear[ta, 3:])
    force, torque, reaction = np.array(force).T, np.array(torque).T, np.array(reaction).T
    np.testing.assert_allclose(force, np.tile([[0.], [0.], [1.]], (1, 4)), atol=1e-14)
    act = env.actuator_model
    if act.reaction_torque_model != 'legacy_ratio' or act.include_rotor_acceleration_torque:
        raise ValueError('linear static audit requires the current constant reaction-torque ratio')
    plant = np.vstack((torque + reaction * (env.motor_direction*env.torque_coefficient)[None, :], force[2]))
    return dict(force=force, force_moment=torque, reaction_axis=reaction, matrix=plant)


def mass_geometry(env):
    """Include every descendant, including the four 1e-6 kg prop bodies."""
    model, data, bid = env.model, env.data, env.drone_bid
    ids = []
    for i in range(1, model.nbody):
        parent = i
        while parent not in (0, bid): parent = int(model.body_parentid[parent])
        if parent == bid: ids.append(i)
    masses = model.body_mass[ids]
    world_com = np.sum(masses[:, None]*data.xipos[ids], axis=0)/masses.sum()
    rotation = data.xmat[bid].reshape(3, 3)
    com = rotation.T @ (world_com-data.xpos[bid])
    np.testing.assert_allclose(world_com, data.subtree_com[bid], atol=1e-14)
    np.testing.assert_allclose(masses.sum(), model.body_subtreemass[bid], atol=1e-15)
    return dict(total_mass_kg=float(masses.sum()), composite_com_body_m=com,
                body_mass_kg=float(model.body_mass[bid]), body_ipos_m=model.body_ipos[bid].copy(),
                descendant_bodies=[dict(name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i),
                    mass_kg=float(model.body_mass[i]), com_body_m=rotation.T@(data.xipos[i]-data.xpos[bid])) for i in ids])


class OracleAllocationEnv(RecordedFaultEnv):
    """Replace only the current pseudoinverse; preserve B and all parent paths."""
    def __init__(self, *args, allocator_mode='existing', **kwargs):
        if allocator_mode not in ('existing', 'oracle'): raise ValueError(allocator_mode)
        self.allocator_mode = allocator_mode
        super().__init__(*args, **kwargs)
        self.B0 = self.B.copy()
        self.B0_pinv = self.B_pinv.copy()
        self.allocator_efficiency = np.ones(4)
        self.allocator_matrix = self.B0.copy()
        self.geometry = None

    def sync_allocator(self):
        eta = self.motor_effectiveness if self.allocator_mode == 'oracle' else np.ones(4)
        if not np.array_equal(eta, self.allocator_efficiency):
            self.allocator_efficiency = eta.copy()
            self.allocator_matrix = efficiency_matrix(self.B0, eta)
            # Exact original numerical path before faults and after restoration.
            self.B_pinv = (self.B0_pinv.copy() if np.all(eta == 1)
                           else np.linalg.pinv(self.allocator_matrix))

    def _apply_control(self, wrench):
        self.sync_allocator()
        if self.geometry is None: self.geometry = plant_geometry(self)
        super()._apply_control(wrench)
        self.physics_rows[-1].update(self.allocation_diagnostics())

    def allocation_diagnostics(self):
        g = self.geometry
        eta = self.motor_effectiveness.copy()
        static = g['matrix'] @ (eta*self._last_f_cmd)
        actual = np.r_[g['force_moment'] @ self._last_f + g['reaction_axis'] @ self._last_q_actual,
                       (g['force'] @ self._last_f)[2]]
        predicted_b0 = self.B0 @ (eta*self._last_f_cmd)
        desired = self._last_wrench_cmd.copy()
        return dict(allocator_mode=self.allocator_mode, allocator_efficiency=self.allocator_efficiency.copy(),
            plant_efficiency=eta, desired_wrench=desired,
            baseline_unclipped_motor_command=self.B0_pinv @ desired,
            allocator_unclipped_motor_command=self.B_pinv @ desired,
            allocator_clipped_motor_command=self._last_f_cmd.copy(),
            allocator_predicted_wrench=self.allocator_matrix @ self._last_f_cmd,
            static_rotor_wrench_xml=static, actual_rotor_wrench_xml=actual,
            allocation_residual_xml=desired-static, actuator_response_residual_xml=static-actual,
            total_rotor_residual_xml=desired-actual,
            static_rotor_wrench_b0=predicted_b0, allocation_residual_b0=desired-predicted_b0,
            geometry_wrench_difference=static-predicted_b0)


def steady_output(env, nominal_command):
    act = env.actuator_model
    rpm_target = act.inverse_thrust(nominal_command)
    esc = np.clip(rpm_target/act.steady_state_gain_rad_s, 0, 1)
    omega = act.steady_state_gain_rad_s*esc
    return act.thrust_from_omega(omega), esc, omega


def static_hover(env, efficiency):
    """Solve in an isolated reset env. Computed trim never reaches rollouts."""
    eta = np.asarray(efficiency, dtype=float)
    if eta.shape != (4,) or np.any(eta <= 0): raise ValueError('invertible positive efficiency required')
    g, mass = plant_geometry(env), mass_geometry(env)
    R = env.data.xmat[env.drone_bid].reshape(3, 3)
    gravity_force = mass['total_mass_kg']*(R.T @ env.model.opt.gravity)
    gravity_moment = np.cross(mass['composite_com_body_m'], gravity_force)
    required = np.r_[-gravity_moment, -gravity_force[2]]
    np.testing.assert_allclose(gravity_force[:2], 0, atol=1e-14)
    actual = np.linalg.solve(g['matrix'], required)
    nominal = np.linalg.solve(efficiency_matrix(g['matrix'], eta), required)
    lo, _, _ = steady_output(env, np.full(4, env.thrust_min))
    hi, _, _ = steady_output(env, np.full(4, env.thrust_max))
    steady, esc, omega = steady_output(env, nominal)
    feasible = bool(np.all(nominal >= lo-1e-12) and np.all(nominal <= hi+1e-12)
                    and np.allclose(steady, nominal, atol=1e-12, rtol=1e-12))
    bias = np.array([0., 0., 0., env.mass*env.gravity])
    actions = {}
    for mode in ('existing', 'oracle'):
        used = env.B if mode == 'existing' else efficiency_matrix(env.B, eta)
        command = used @ nominal
        action = (command-bias)/env.residual_scale
        actions[mode] = dict(required_policy_wrench=command, required_action=action,
                             expressible=bool(np.all(np.abs(action) <= 1+1e-12)))
    force_residual = g['force']@actual + gravity_force
    moment_residual = g['force_moment']@actual + g['reaction_axis']@(env.motor_direction*env.torque_coefficient*actual) + gravity_moment
    # Clone dynamic state, sharing this isolated model read-only. No integration
    # or custom gravity torque; MuJoCo computes gravity for all descendant mass.
    probe = mujoco.MjData(env.model)
    probe.qpos[:] = env.data.qpos; probe.qvel[:] = 0
    probe.ctrl[env.act_force] = eta*steady
    probe.ctrl[env.act_torque] = eta*steady*env.motor_direction*env.torque_coefficient
    mujoco.mj_forward(env.model, probe)
    nominal_b0_solution = np.linalg.solve(efficiency_matrix(env.B, eta), required)
    result = dict(**mass, efficiency=eta, rotors=rotor_mapping(env), actuator=env.actuator_snapshot(),
        moment_reference='drone body origin, body frame; wrench order [Mx,My,Mz,Fz], units [Nm,Nm,Nm,N]',
        B0_allocator=env.B.copy(), B0_plant_xml=g['matrix'], matrix_difference=g['matrix']-env.B,
        gravity_force_body_n=gravity_force, gravity_moment_about_body_origin_nm=gravity_moment,
        required_rotor_wrench=required, required_actual_thrust_n=actual, required_nominal_thrust_n=nominal,
        nominal_lower_n=lo, nominal_upper_n=hi, nominal_lower_margin_n=nominal-lo,
        allocator_command_bounds_n=[env.thrust_min,env.thrust_max], esc_input_bounds=[0.,1.],
        steady_omega_at_esc1_rad_s=env.actuator_model.steady_state_gain_rad_s.copy(),
        thrust_mapping_max_speed_ratio=env.actuator_model.max_ratio,
        thrust_mapping_reference_speed_rad_s=env.actuator_model.omega_reference_rad_s,
        nominal_upper_margin_n=hi-nominal, nominal_upper_utilization=nominal/hi,
        actual_upper_n=eta*hi, actual_upper_margin_n=eta*hi-actual,
        bottleneck_motor=int(np.argmin(hi-nominal)+1), feasible_static_equilibrium=feasible,
        nominal_command_using_allocator_geometry=nominal_b0_solution,
        allocator_geometry_equilibrium_residual=g['matrix']@(eta*nominal_b0_solution)-required,
        required_esc=esc, required_omega_rad_s=omega, required_rpm=omega*60/(2*np.pi),
        force_balance_residual_n=force_residual, moment_balance_residual_nm=moment_residual,
        policy_action_scale=env.residual_scale.copy(), policy_hover_bias=bias,
        physical_wrench_action=(required-bias)/env.residual_scale,
        physical_wrench_in_policy_range=bool(np.all(np.abs((required-bias)/env.residual_scale) <= 1+1e-12)),
        equilibrium_commands=actions,
        probe_uses='isolated MjData with steady clipped/inverted actuator outputs, zero external torque',
        probe_linear_acceleration_m_s2=probe.qacc[:3].copy(),
        probe_angular_acceleration_rad_s2=probe.qacc[3:6].copy(),
        probe_all_generalized_acceleration=probe.qacc.copy(),
        external_torque_applied_nm=probe.xfrc_applied[env.drone_bid, 3:6].copy())
    if feasible:
        np.testing.assert_allclose(probe.qacc[:6], 0, atol=1e-8)
    return result

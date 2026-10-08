"""Isolated command-path arithmetic and motor probes; never alter the plant."""
from dataclasses import asdict
import numpy as np
from .actuators import Cf21bFirstOrderActuatorModel
from .fault_estimation_eval import write_json
from .oracle_eval import flat_csv


def audit(directory, model, config):
    kwargs = dict(model.actuator_kwargs)
    act = Cf21bFirstOrderActuatorModel(**kwargs)
    changed = Cf21bFirstOrderActuatorModel(**dict(kwargs, thrust_max=.2943))
    commands = np.linspace(0, config.vehicle.thrust_max, 101)
    command_rows = []; errors = []
    for command in commands:
        speed = act.inverse_thrust(command)
        esc = np.clip(speed/act.steady_state_gain_rad_s, 0, 1)
        force = act.thrust_from_omega(esc*act.steady_state_gain_rad_s)
        other = changed.inverse_thrust(command)
        errors.append(float(np.max(np.abs(other-speed))))
        command_rows.append(dict(command_n=float(command), esc=float(esc[0]),
            steady_omega_rad_s=float(speed[0]), rpm=float(speed[0]*60/(2*np.pi)),
            steady_nominal_thrust_n=float(force[0]), reaction_magnitude_nm=float(force[0]*act.legacy_ratio_m)))
    esc_rows = []
    for u in np.linspace(0, 1, 101):
        w = u*act.steady_state_gain_rad_s
        ratio = float(w[0]/act.omega_reference_rad_s)
        esc_rows.append(dict(esc=float(u), omega_rad_s=float(w[0]), rpm=float(w[0]*60/(2*np.pi)),
            raw_polynomial_n=float(act._thrust_polynomial(ratio)),
            nominal_thrust_n=float(act.thrust_from_omega(w)[0]),
            isolated_raised_cap_thrust_n=float(changed.thrust_from_omega(w)[0])))
    probes = []
    for command in (0., .05, .10638945, .15, .20):
        a = Cf21bFirstOrderActuatorModel(**kwargs)
        a.reset(airborne=False)
        for _ in range(round(2/a.dt)): out = a.apply(np.full(4, command))
        probes.append(dict(command_n=command, duration_s=2., steady_thrust_n=out.f_actual,
            omega_rad_s=out.omega, esc=out.motor_command,
            max_force_error_n=float(np.max(np.abs(out.f_actual-command)))))
    # A mathematical extrapolation is disclosed, never sent to any runtime model.
    a3,a2,a1 = act.thrust_polynomial_coefficients
    roots = np.roots([a3,a2,a1,-.2943])
    root = min(float(v.real) for v in roots if abs(v.imag)<1e-10 and v.real>act.positive_branch_min_ratio)
    top = command_rows[-1]
    result = dict(manufacturer_sources=[
        dict(url='https://www.bitcraze.io/products/crazyflie-2-1-brushless/', claim='up to 30 grams thrust each'),
        dict(url='https://www.bitcraze.io/2024/08/the-optimized-crazyflie-2-1-brushless-motors/',
             claim='30 gram-force at 4 V with 55-35 propeller; peak current 1.8 A, power 7.2 W')],
        fetched_date='2026-10-07', manufacturer_30gf_n=.030*9.81,
        loaded_vehicle=asdict(config.vehicle), loaded_actuator=asdict(config.actuator),
        source_functions=['environment.CrazyflieResidualEnv._apply_control',
            'actuators.Cf21bFirstOrderActuatorModel.apply/inverse_thrust/thrust_from_omega/_advance/_reaction_torque',
            'interactive_eval.InteractiveEnv._apply_control', 'oracle_allocation.OracleAllocationEnv.sync_allocator'],
        path=['wrench=[tau_x,tau_y,tau_z,T], native FLU body origin',
              'pinv(B0 @ diag(eta_used)) @ wrench', 'clip per-rotor command to [0,0.20] N',
              '48-iteration inverse cubic on [0.0791,1] -> omega target',
              'u=clip(omega_target/2900,0,1)',
              'omega_next=exp(-0.002/0.05)*omega+(1-exp(-0.002/0.05))*2900*u',
              'f_nom=clip(-0.23*r^3+0.562*r^2-0.043*r,0,min(thrust_max,P(1))); r=clip(omega/2900,0,1); zero for r<=0.0791',
              'q_nom=direction*0.00594*f_nom, no rotor-acceleration torque',
              'f_actual=eta_true*f_nom; q_actual=eta_true*q_nom, once, at plant only'],
        hover_bias_n=config.vehicle.mass*config.vehicle.gravity,
        action_scale=config.environment.residual_scale, allocator_bounds_n=[0.,.20], esc_bounds=[0.,1.],
        nominal_max_omega_at_esc1_rad_s=2900., nominal_max_rpm_at_esc1=2900*60/(2*np.pi),
        tau_s=.05, polynomial_coefficients_n=kwargs['thrust_polynomial_coefficients'],
        polynomial_raw_at_r1_n=float(act._branch_max_thrust), forward_inverse_cap_n=float(act._mapping_thrust_max),
        currently_reachable_steady_upper_n=top['steady_nominal_thrust_n'],
        command_020_operating_point=top,
        cap_meaning='Configured operational limit used BOTH by allocator clipping and forward/inverse motor maps; not a manufacturer-calibrated motor limit.',
        inverse_unchanged_below_old_cap_max_omega_error_rad_s=max(errors),
        same_esc_below_old_cap_unchanged=True,
        raised_cap_only_upper_n=float(changed.thrust_from_omega(2900)[0]),
        hypothetical_30gf_extrapolation=dict(ratio=root,omega_rad_s=root*2900,
            rpm=root*2900*60/(2*np.pi),required_esc_at_current_gain=root,
            outside_configured_range=True,used_in_flight=False),
        interpretation='Raising cap alone leaves inverse commands <=0.20 N unchanged, but changes clipping and forward map above the old plateau. No ESC normalization by thrust_max. With cap 0.2943, r<=1 and u<=1 still limit output to 0.289 N.',
        other_limits='No battery voltage/current/thermal model; RPM gain is the fixed nominal value. XML actuator limits inspected separately in plant_geometry.',
        estimator_match='Exactly same actuator class, coefficients, bounds, dt and nominal initialization; independent state driven by delivered ESC history. No actual motor state copied.',
        future_work='Measure RPM/thrust/reaction torque/ESC curves across voltage and load, validate gain/time constant and valid speed range, separate allocator operating cap from calibrated forward curve if desired, retest inverse/lag/estimator/model contracts. One 30gf point is insufficient to refit the curve.',
        isolated_response_probes=probes, full_curve_refit_performed=False, flight_model_unchanged=True)
    np.testing.assert_allclose(errors,0,atol=0)
    assert max(p['max_force_error_n'] for p in probes)<1e-10
    flat_csv(directory/'motor_command_steady_curve.csv',command_rows)
    flat_csv(directory/'esc_steady_curve.csv',esc_rows)
    write_json(directory/'motor_limit_audit.json',result)
    return result

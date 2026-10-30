"""Actual-target metrics; physical end-of-step state, never virtual-target scoring."""
import numpy as np

def segment(rows,circle=False):
    if not rows:return None
    a=lambda keys:np.asarray([[r[k] for k in keys] for r in rows],dtype=np.float64)
    rms=lambda x:float(np.sqrt(np.mean(np.square(x))))
    e=a(['ex','ey','ez']);vel=a(['vx','vy','vz']);om=a(['wx','wy','wz'])
    ang=a(['roll_rad','pitch_rad','yaw_rad']);raw=a([f'raw_action_{i}' for i in range(4)]);act=a([f'action_{i}' for i in range(4)])
    force=a([f'thrust_{i}' for i in range(4)]);torque=a([f'actual_{i}' for i in range(3)])
    alloc=a([f'allocation_{i}' for i in range(4)]);motor=a([f'motor_rpm_{i}' for i in range(4)])
    d=dict(samples=len(rows),sample_interval_s=[rows[0]['time_s'],rows[-1]['time_s']],duration_s=len(rows)*.01,
        position_rmse_m=rms(np.linalg.norm(e,axis=1)),xyz_rmse_m=np.sqrt((e*e).mean(0)).tolist(),bias_xyz_m=e.mean(0).tolist(),std_xyz_m=e.std(0).tolist(),
        max_position_error_m=float(np.linalg.norm(e,axis=1).max()),max_tilt_deg=max(r['tilt_deg'] for r in rows),
        speed_rms_m_s=rms(np.linalg.norm(vel,axis=1)),body_omega_rms_rad_s=rms(np.linalg.norm(om,axis=1)),omega_axis_rms_rad_s=np.sqrt((om*om).mean(0)).tolist(),
        attitude_rms_rad=np.sqrt((ang*ang).mean(0)).tolist(),attitude_std_rad=ang.std(0).tolist(),
        action_clipping_axes=(np.abs(raw)>1).mean(0).tolist(),action_clipping_fraction=float((np.abs(raw)>1).mean()),
        applied_saturation_axes=(np.abs(act)>=1-1e-6).mean(0).tolist(),allocation_clipping_fraction=float(((alloc<0)|(alloc>.289)).mean()),
        allocation_clipped_step_fraction=float(((alloc<0)|(alloc>.289)).any(1).mean()),
        thrust_mean_N=force.mean(0).tolist(),thrust_std_N=force.std(0).tolist(),thrust_peak_to_peak_N=np.ptp(force,axis=0).tolist(),
        motor_rpm_mean=motor.mean(0).tolist(),motor_rpm_std=motor.std(0).tolist(),torque_std_Nm=torque.std(0).tolist(),
        integral_frozen_fraction=float(np.mean([r['integral_frozen'] for r in rows])),
        integral_xy_limit_fraction=float(np.mean([r['xy_at_limit'] for r in rows])),integral_z_limit_fraction=float(np.mean([r['z_at_limit'] for r in rows])),
        reward_sum=float(sum(r['reward'] for r in rows)),reward_mean=float(np.mean([r['reward'] for r in rows])))
    if len(rows)>1:d['applied_action_delta_rms']=np.sqrt(np.mean(np.diff(act,axis=0)**2,axis=0)).tolist()
    if circle:
        lag=a(['phase_lag_deg'])[:,0];z=a(['radial_error_m'])[:,0]
        d.update(radial_rmse_m=rms(z),radial_mean_m=float(z.mean()),phase_lag_mean_deg=float(lag.mean()),phase_lag_rms_deg=rms(lag))
    return d

def summarize(rows,task,horizon,period):
    last=rows[-1];complete=not last['terminated'] and len(rows)==round(horizon*100)
    end_reason=[k.replace('end_','') for k in ['end_min_altitude','end_max_altitude','end_max_tilt','end_max_position_error'] if last[k]]
    if not end_reason:end_reason=['time_limit' if last['truncated'] else 'interrupted']
    tail_start=horizon-(10 if period is None else period)
    tail=[r for r in rows if r['time_s']>tail_start+1e-9] if complete else []
    return dict(completed=complete,duration_s=last['time_s'],termination_reasons=end_reason,full=segment(rows,period is not None),
        tail=segment(tail,period is not None),tail_definition=('last10s' if period is None else 'last_reference_lap'),tail_requested_interval_s=[tail_start,horizon],
        final_xi_used_m=[last['xi_used_'+x] for x in 'xyz'],final_xi_before_episode_reset_m=[last['xi_interval_end_'+x] for x in 'xyz'],
        state_after_final_step_m=[last['xi_next_'+x] for x in 'xyz'])

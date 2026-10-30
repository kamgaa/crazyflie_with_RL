import numpy as np

def section(rows):
 if not rows:return None
 def a(keys):return np.array([[r[k] for k in keys] for r in rows])
 def rms(x,axis=None):return np.sqrt(np.mean(x*x,axis=axis))
 e=a(['ex','ey','ez']);v=a(['vx','vy','vz']);w=a(['wx','wy','wz']);ang=a(['roll_rad','pitch_rad','yaw_rad']);u=a([f'raw_action_{i}' for i in range(4)])
 return dict(samples=len(rows),interval_s=[rows[0]['time_s'],rows[-1]['time_s']],position_rmse_m=float(rms(np.linalg.norm(e,axis=1))),axis_rmse_m=rms(e,0).tolist(),bias_xyz_m=e.mean(0).tolist(),std_xyz_m=e.std(0).tolist(),max_position_error_m=float(np.linalg.norm(e,axis=1).max()),speed_rms_m_s=float(rms(np.linalg.norm(v,axis=1))),omega_rms_rad_s=float(rms(np.linalg.norm(w,axis=1))),omega_axis_rms_rad_s=rms(w,0).tolist(),attitude_rms_rad=rms(ang,0).tolist(),max_tilt_deg=max(x['tilt_deg'] for x in rows),max_abs_yaw_deg=float(np.max(np.abs(ang[:,2]))*180/np.pi),action_clipping_axes=(np.abs(u)>1).mean(0).tolist(),action_clipping_fraction=float((np.abs(u)>1).mean()),allocation_clipping_fraction=float(np.mean([r['allocation_clipping_fraction'] for r in rows])),thrust_mean_user_N=a([f'user_actual_thrust_N_{i}' for i in range(1,5)]).mean(0).tolist())

def fault_metrics(rows,completed,onset=5.):
 post=[r for r in rows if r['time_s']>onset];pre=[r for r in rows if r['time_s']<=onset];tail=[r for r in rows if r['time_s']>20] if completed else []
 outside=[i for i,r in enumerate(post) if r['position_error_m']>.01];idx=outside[-1]+1 if outside else 0
 settling=post[idx]['time_s']-onset if completed and idx<len(post) and rows[-1]['time_s']-post[idx]['time_s']>=1 else None
 return dict(prefault=section(pre),postfault=section(post),tail20_30=section(tail),postfault_duration_s=max(0,rows[-1]['time_s']-onset),first_failure_after_fault_s=max(0,rows[-1]['time_s']-onset) if rows[-1]['terminated'] else None,settling_1cm_for_final1s_after_fault_s=settling,settling_definition='first postfault sample after which norm(position error)<=1cm until end with>=1s hold; completed only')

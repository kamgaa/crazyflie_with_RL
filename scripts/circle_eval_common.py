"""Reference and metrics for the two-lap, position-only circle comparison."""
import importlib.util,math,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def reference(t,period,radius=1.,center=(0.,0.,1.)):
 phase=2*math.pi*np.asarray(t)/period;w=2*math.pi/period
 pos=np.stack((radius*np.cos(phase),radius*np.sin(phase),np.zeros_like(phase)),axis=-1)+np.asarray(center)
 vel=np.stack((-radius*w*np.sin(phase),radius*w*np.cos(phase),np.zeros_like(phase)),axis=-1)
 acc=np.stack((-radius*w*w*np.cos(phase),-radius*w*w*np.sin(phase),np.zeros_like(phase)),axis=-1)
 return pos,vel,acc

def load_pid(run,plan):
 p=Path(run)/'source_snapshot/original_controllers.py'
 spec=importlib.util.spec_from_file_location('circle_original_controller',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m.CascadePID(.01,vehicle=SimpleNamespace(**plan['pid']['vehicle']),settings=SimpleNamespace(**plan['pid']['settings']))

def wrench_to_action(wrench,scale=(.0075,.0075,.001,.5),mass=.04338,g=9.81):
 return (np.asarray(wrench)-np.array([0.,0.,0.,mass*g]))/np.asarray(scale)

def segment_metrics(rows):
 if not rows:return None
 a=lambda keys:np.array([[r[k] for k in keys] for r in rows],dtype=float)
 e=a(['ex','ey','ez']);v=a(['vx','vy','vz']);ve=a(['vex','vey','vez']);omega=a(['wx','wy','wz'])
 rms=lambda z:float(np.sqrt(np.mean(np.square(z))))
 return dict(samples=len(rows),duration_s=len(rows)*.01,first_sample_s=rows[0]['time_s'],last_sample_s=rows[-1]['time_s'],
  position_rmse_m=rms(np.linalg.norm(e,axis=1)),axis_rmse_m=np.sqrt((e*e).mean(0)).tolist(),max_position_error_m=float(np.linalg.norm(e,axis=1).max()),
  mean_error_vector_m=e.mean(0).tolist(),mean_error_m=float(np.linalg.norm(e,axis=1).mean()),
  radial_error_rmse_m=rms([r['radial_error_m'] for r in rows]),mean_radial_error_m=float(np.mean([r['radial_error_m'] for r in rows])),
  tangential_error_rmse_m=rms([r['tangential_error_m'] for r in rows]),phase_lag_mean_deg=float(np.mean([r['phase_lag_deg'] for r in rows])),
  phase_lag_rmse_deg=rms([r['phase_lag_deg'] for r in rows]),speed_mean_m_s=float(np.linalg.norm(v,axis=1).mean()),
  velocity_tracking_rmse_m_s=rms(np.linalg.norm(ve,axis=1)),body_omega_rms_rad_s=rms(np.linalg.norm(omega,axis=1)),
  omega_axes_rms_rad_s=np.sqrt((omega*omega).mean(0)).tolist(),roll_rms_deg=rms([r['roll_deg'] for r in rows]),
  max_tilt_deg=max(r['tilt_deg'] for r in rows),z_peak_to_peak_m=float(np.ptp(e[:,2])),max_abs_z_error_m=float(np.abs(e[:,2]).max()),
  action_clipping_fraction=float(np.mean([r['action_clipping_fraction'] for r in rows])),allocation_clipping_fraction=float(np.mean([r['allocation_clipping_fraction'] for r in rows])),
  motor_thrust_mean_N=a([f'thrust_{i}' for i in range(4)]).mean(0).tolist())

def trajectory_metrics(rows,period,laps=2):
 result=segment_metrics(rows);end=rows[-1]
 result.update(completed_two_reference_laps=len(rows)==round(period*laps*100) and not end['terminated'],
  requested_duration_s=period*laps,reference_laps_elapsed=end['time_s']/period,terminated=bool(end['terminated']),truncated=bool(end['truncated']),
  reasons=[k for k in ['min_altitude','max_altitude','max_tilt','max_position_error'] if end['end_'+k]])
 if not result['reasons'] and end['truncated']:result['reasons']=['time_limit']
 result['laps']={}
 for lap in range(laps):
  seq=[r for r in rows if lap*period < r['time_s'] <= (lap+1)*period+1e-8]
  result['laps'][str(lap+1)]=dict(complete=len(seq)==round(period*100) and not any(r['terminated'] for r in seq),metrics=segment_metrics(seq))
 pos=np.array([[1.,0.]]+[[r['x'],r['y']] for r in rows]);ph=np.unwrap(np.arctan2(pos[:,1],pos[:,0]))
 result['measured_azimuth_revolutions']=(ph[-1]-ph[0])/(2*math.pi)
 result['azimuth_interpretation']='net unwrapped azimuth change about center; not proof of geometric path completion'
 return result

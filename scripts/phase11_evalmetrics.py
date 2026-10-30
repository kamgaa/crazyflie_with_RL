import numpy as np
from phase10_evalmetrics import augment_row,summarize as previous

def extra_metrics(rows,offset,completed):
    # Saved t=0 state is known: horizontal, velocity zero, fixed target plus offset.
    init=dict(rows[0]);init.update(time_s=0.,ex=offset[0],ey=offset[1],ez=offset[2],vx=0.,vy=0.,vz=0.,roll_rad=0.,pitch_rad=0.,yaw_rad=0.,wx=0.,wy=0.,wz=0.,tilt_deg=0.)
    seq=[init]+rows;result={}
    for name,part in [('full0',seq),('early0_5',[r for r in seq if r['time_s']<=5])]:
        e=np.array([[r[k] for k in ('ex','ey','ez')] for r in part]);v=np.array([[r[k] for k in ('vx','vy','vz')] for r in part]);a=np.array([[r[k] for k in ('roll_rad','pitch_rad','yaw_rad')] for r in part]);om=np.array([[r[k] for k in ('wx','wy','wz')] for r in part])
        result[name+'_interval_s']=[part[0]['time_s'],part[-1]['time_s']]
        result[name+'_position_rmse_m']=float(np.sqrt(np.mean(np.sum(e*e,axis=1))))
        result[name+'_axis_rmse_m']=np.sqrt(np.mean(e*e,0)).tolist()
        result[name+'_axis_max_abs_error_m']=np.max(np.abs(e),0).tolist()
        result[name+'_attitude_max_abs_rad']=np.max(np.abs(a),0).tolist()
        result[name+'_max_tilt_deg']=max(r['tilt_deg'] for r in part)
        result[name+'_speed_rms_m_s']=float(np.sqrt(np.mean(np.sum(v*v,axis=1))))
        result[name+'_omega_rms_rad_s']=float(np.sqrt(np.mean(np.sum(om*om,axis=1))))
        axes=[j for j,x in enumerate(offset) if abs(x)>1e-10]
        result[name+'_return_axis']=axes[0] if len(axes)==1 else None
        result[name+'_overshoot_m']=float(max(0,np.max(-np.sign(offset[axes[0]])*e[:,axes[0]]))) if len(axes)==1 else None
        result[name+'_cross_axis_max_m']={('xyz'[j]):float(np.max(np.abs(e[:,j]))) for j in range(3) if len(axes)==1 and j!=axes[0]}
    e=np.array([[r[k] for k in ('ex','ey','ez')] for r in seq]);times=np.array([r['time_s'] for r in seq]);outside=np.flatnonzero(np.linalg.norm(e,axis=1)>.01)
    start=int(outside[-1]+1) if len(outside) else 0
    result['settling_time_1cm_1s_s']=float(times[start]) if completed and start<len(times) and times[-1]-times[start]>=1.-1e-9 else None
    result['settling_definition']='first time norm(error)<=0.01m for >=1s AND never exits before completion; failures N/A'
    # Float32 world coordinates are the source; local error subtraction cannot restore lost precision.
    if 'world_x' in rows[0]:
        world=np.array([[r['world_'+a] for a in 'xyz'] for r in rows],dtype=np.float32)
        result['position_precision']=dict(source_dtype='float32',csv_format='Python float repr round-trip; no intentional decimal rounding',world_abs_max_m=np.max(np.abs(world),0).astype(float).tolist(),max_world_float32_ulp_m=np.max(np.abs(np.spacing(world)),0).astype(float).tolist())
    return result

def summarize(rows,horizon):return previous(rows,horizon)

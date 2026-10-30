"""Offline measured trajectory metrics, in SI units; policy sampling 100 Hz."""
import math
import numpy as np
FS=100.
def rms(x,axis=None):return np.sqrt(np.mean(np.square(x),axis=axis))
def spectral(x):
    x=np.asarray(x,dtype=float);n=len(x)
    if n<4:return None
    centered=x-x.mean();window=np.hanning(n)
    psd=np.abs(np.fft.rfft(centered*window))**2/(FS*np.sum(window**2))
    if n%2==0:psd[1:-1]*=2
    else:psd[1:]*=2
    freq=np.fft.rfftfreq(n,1/FS);df=FS/n;idx=np.argmax(psd[1:])+1
    band=(freq>=10)&(freq<=20)
    return dict(frequency_hz=freq.tolist(),psd=psd.tolist(),peak_frequency_hz=float(freq[idx]),
        band_10_20_power=float(psd[band].sum()*df),psd_integrated_power=float(psd.sum()*df),
        rms_raw=float(rms(x)),rms_mean_removed=float(rms(centered)),peak_to_peak=float(np.ptp(x)),
        mean=float(x.mean()),samples=n,fs_hz=FS,resolution_hz=df,detrend='constant: remove segment mean',
        window='symmetric Hann, numpy.hanning(N)',normalization='one-sided PSD / (fs * sum(window**2)); band sum * df')
def augment_row(row):
    w,x,y,z=[row[k] for k in ['qw','qx','qy','qz']]
    row['roll_rad']=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    row['pitch_rad']=math.asin(max(-1,min(1,2*(w*y-z*x))))
    row['yaw_rad']=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    row['roll_deg']=math.degrees(row['roll_rad']);row['pitch_deg']=math.degrees(row['pitch_rad'])
    row['total_thrust_N']=sum(row['thrust_'+str(i)] for i in range(4))
    for i in range(4):
        row[f'motor_rpm_{i}']=row[f'omega_{i}']*60/(2*math.pi)
        row[f'action_clip_{i}']=float(abs(row[f'raw_action_{i}'])>1)
    return row
def summarize(rows,horizon):
    def cols(keys,seq=rows):return np.array([[r[k] for k in keys] for r in seq],float)
    e=cols(['ex','ey','ez']);v=cols(['vx','vy','vz']);w=cols(['wx','wy','wz'])
    raw=cols([f'raw_action_{i}' for i in range(4)]);act=cols([f'action_{i}' for i in range(4)])
    complete=len(rows)==int(horizon*FS) and not rows[-1]['terminated']
    tail_s=2 if horizon==8 else 10;tail=rows[-int(tail_s*FS):] if complete else []
    result=dict(horizon_s=horizon,completed=complete,completed_8s=complete if horizon==8 else None,
        completed_30s=complete if horizon==30 else None,duration_s=len(rows)/FS,steps=len(rows),
        position_rmse_m=float(rms(np.linalg.norm(e,axis=1))),xyz_rmse_m=rms(e,axis=0).tolist(),
        max_position_error_m=float(np.linalg.norm(e,axis=1).max()),max_abs_z_error_m=float(np.abs(e[:,2]).max()),
        z_std_m=float(e[:,2].std()),z_peak_to_peak_m=float(np.ptp(e[:,2])),
        max_tilt_deg=max(x['tilt_deg'] for x in rows),roll_rms_rad=float(rms(cols(['roll_rad']))),
        pitch_rms_rad=float(rms(cols(['pitch_rad']))),speed_rms_m_s=float(rms(np.linalg.norm(v,axis=1))),
        body_omega_rms_rad_s=float(rms(np.linalg.norm(w,axis=1))),omega_axes_rms_rad_s=rms(w,axis=0).tolist(),
        raw_action_mean=raw.mean(0).tolist(),raw_action_rms=rms(raw,0).tolist(),raw_action_min=raw.min(0).tolist(),raw_action_max=raw.max(0).tolist(),
        applied_action_mean=act.mean(0).tolist(),applied_action_rms=rms(act,0).tolist(),
        action_clipping_axes=(np.abs(raw)>1).mean(0).tolist(),action_clipping_fraction=float((np.abs(raw)>1).mean()),
        raw_action_delta_rms=rms(np.diff(raw,axis=0),0).tolist() if len(rows)>1 else None,
        applied_action_delta_rms=rms(np.diff(act,axis=0),0).tolist() if len(rows)>1 else None,
        allocation_clipping_fraction=float(np.mean([x['allocation_clipping_fraction'] for x in rows])),
        tail_duration_s=tail_s if tail else None,tail_start_s=horizon-tail_s if tail else None,
        tail_position_mean_m=None,tail_error_vector_m=None,tail_speed_mean_m_s=None,tail_position_rmse_m=None,
        tail_z_mean_m=None,tail_z_std_m=None,tail_z_peak_to_peak_m=None,tail_body_omega_rms_rad_s=None,
        tail_omega_axes_rms_rad_s=None,tail_roll_rms_rad=None,tail_pitch_rms_rad=None)
    if tail:
        te=cols(['ex','ey','ez'],tail);tv=cols(['vx','vy','vz'],tail);tw=cols(['wx','wy','wz'],tail)
        result.update(tail_position_mean_m=float(np.linalg.norm(te,axis=1).mean()),tail_error_vector_m=te.mean(0).tolist(),
            tail_speed_mean_m_s=float(np.linalg.norm(tv,axis=1).mean()),tail_position_rmse_m=float(rms(np.linalg.norm(te,axis=1))),
            tail_z_mean_m=float(te[:,2].mean()),tail_z_std_m=float(te[:,2].std()),tail_z_peak_to_peak_m=float(np.ptp(te[:,2])),
            tail_body_omega_rms_rad_s=float(rms(np.linalg.norm(tw,axis=1))),tail_omega_axes_rms_rad_s=rms(tw,0).tolist(),
            tail_roll_rms_rad=float(rms(cols(['roll_rad'],tail))),tail_pitch_rms_rad=float(rms(cols(['pitch_rad'],tail))))
    initial=rows[:200];ie=cols(['ex','ey','ez'],initial)
    result.update(initial_2s_actual_duration_s=len(initial)/FS,initial_2s_position_rmse_m=float(rms(np.linalg.norm(ie,axis=1))))
    spectra={}
    for label,seq in [('full',rows),('initial_2s',initial),('tail',tail)]:
        if not seq:continue
        spectra[label]={k:spectral([x[k] for x in seq]) for k in ['roll_rad','wx','ez','vz']}
        for k,sp in spectra[label].items():
            if sp:
                result[f'{label}_{k}_peak_hz']=sp['peak_frequency_hz']
                result[f'{label}_{k}_band_10_20_power']=sp['band_10_20_power']
        spectra[label]['interval_s']=[seq[0]['time_s'],seq[-1]['time_s']]
    return result,spectra


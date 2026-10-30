"""Additional offline diagnostics only; existing phase9 telemetry and dynamics unchanged."""
import numpy as np
from phase9_evalmetrics import augment_row,summarize as prior_summary
from phase8_evalmetrics import spectral,rms

def summarize(rows,horizon):
    result,spectra=prior_summary(rows,horizon)
    def cols(keys,seq):return np.array([[r[k] for k in keys] for r in seq],float)
    tail=rows[-1000:] if result['completed'] and horizon==30 else []
    for label,seq in [('full',rows),('tail',tail)]:
        result[label+'_interval_s']=[seq[0]['time_s'],seq[-1]['time_s']] if seq else None
        groups={'position_error_m':['ex','ey','ez'],'attitude_rad':['roll_rad','pitch_rad','yaw_rad'],'omega_rad_s':['wx','wy','wz'],'rpm':[f'motor_rpm_{i}' for i in range(4)],'thrust_N':[f'thrust_{i}' for i in range(4)],'raw_action':[f'raw_action_{i}' for i in range(4)],'applied_action':[f'action_{i}' for i in range(4)],'requested_wrench':[f'request_{i}' for i in range(4)],'actual_wrench':[f'actual_{i}' for i in range(4)]}
        for name,keys in groups.items():
            arr=cols(keys,seq) if seq else None
            for stat,fun in [('mean',lambda a:a.mean(0)),('std',lambda a:a.std(0)),('rms',lambda a:rms(a,0)),('ptp',lambda a:np.ptp(a,axis=0)),('min',lambda a:a.min(0)),('max',lambda a:a.max(0))]:
                result[f'{label}_{name}_{stat}']=fun(arr).tolist() if seq else None
        for name,keys,test in [('raw_action_exceedance',[f'raw_action_{i}' for i in range(4)],lambda a:np.abs(a)>1),('applied_action_saturation',[f'action_{i}' for i in range(4)],lambda a:np.abs(a)>=1-1e-6),('allocation_clipping',[f'allocation_{i}' for i in range(4)],lambda a:(a<0)|(a>.289)),('rotor_command_bound',[f'command_{i}' for i in range(4)],lambda a:(a<=1e-7)|(a>=.289-1e-7)),('actual_thrust_bound',[f'thrust_{i}' for i in range(4)],lambda a:(a<=1e-7)|(a>=.289-1e-7))]:
            values=test(cols(keys,seq)) if seq else None
            result[f'{label}_{name}_axes']=values.mean(0).tolist() if seq else None
            result[f'{label}_{name}_fraction']=float(values.mean()) if seq else None
        if seq:
            spectra.setdefault(label,{})
            for k in ['roll_rad','pitch_rad','wx','wy','wz','ez','vz']:spectra[label][k]=spectral([r[k] for r in seq])
    return result,spectra

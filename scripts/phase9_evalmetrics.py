import numpy as np
from phase8_evalmetrics import augment_row as old_augment,summarize as old_summary,rms

def augment_row(row):
 old_augment(row)
 e=np.array([row[k] for k in ['ex','ey','ez']]);vd=-4*e;vd*=min(1.,1.5/max(np.linalg.norm(vd),1e-30))
 for i,axis in enumerate('xyz'):
  row['vdes_'+axis]=float(vd[i]);row['verr_'+axis]=row['v'+axis]-float(vd[i])
 return row

def summarize(rows,horizon):
 result,spectra=old_summary(rows,horizon)
 def cols(keys,seq):return np.array([[r[k] for k in keys] for r in seq])
 result['velocity_error_rms_m_s']=float(rms(np.linalg.norm(cols(['verr_x','verr_y','verr_z'],rows),axis=1)))
 tail=rows[-(200 if horizon==8 else 1000):] if result['completed'] else []
 for k in ['tail_speed_rms_m_s','tail_velocity_error_rms_m_s','tail_raw_action_delta_rms','tail_applied_action_delta_rms','tail_raw_action_rate_rms_s_inv','tail_applied_action_rate_rms_s_inv']:result[k]=None
 if tail:
  result['tail_speed_rms_m_s']=float(rms(np.linalg.norm(cols(['vx','vy','vz'],tail),axis=1)))
  result['tail_velocity_error_rms_m_s']=float(rms(np.linalg.norm(cols(['verr_x','verr_y','verr_z'],tail),axis=1)))
  for field,label in [('raw_action','raw'),('action','applied')]:
   d=np.diff(cols([f'{field}_{i}' for i in range(4)],tail),axis=0)
   result[f'tail_{label}_action_delta_rms']=rms(d,0).tolist();result[f'tail_{label}_action_rate_rms_s_inv']=rms(d/.01,0).tolist()
 return result,spectra

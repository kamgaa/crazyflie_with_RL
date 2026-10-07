"""Independent nominal metrics and shared initialization checks; no simulation."""
from pathlib import Path
import json
import numpy as np
from crazyflie_rl.integral_eval import read_columns
from crazyflie_rl.oracle_recovery_artifacts import vector
D=Path(__file__).resolve().parent
records=json.loads((D/'summary.json').read_text())
manifest=json.loads((D/'manifest.json').read_text())
checks={}
for r in records:
 if r['group']!='nominal': continue
 f=read_columns(D/(r['key']+'.csv'));t=f.time_post
 e=vector(f,'position')-vector(f,'reference_post');v=vector(f,'velocity');w=vector(f,'omega');q=vector(f,'quaternion',4)
 np.testing.assert_allclose(e,vector(f,'position_error_world'),atol=1e-15)
 np.testing.assert_allclose(vector(f,'policy_raw_velocity'),vector(f,'velocity_before'),rtol=1e-7,atol=1e-7)
 init=manifest['runs'][r['key']]['initial_snapshot']
 ie=np.array(init['position'])-init['reference'];t=np.r_[0.,t]
 ee=np.vstack((ie,e));vv=np.vstack((init['velocity'],v));ww=np.vstack((init['omega'],w));qq=np.vstack((init['quaternion'],q))
 yaw0=np.radians(manifest['runs'][r['key']]['initial_yaw_native_deg']);yaw=np.r_[yaw0,f.yaw_error_rad]
 latencies={}
 for axis,sl in [('xy',slice(0,2)),('z',slice(2,3)),('3d',slice(0,3)),('joint',slice(0,3))]:
  ok=(np.linalg.norm(ee[:,sl],axis=1)<=.005)&(np.linalg.norm(vv[:,sl],axis=1)<=.02)
  if axis=='joint':
   tilt=np.degrees(np.arccos(np.clip(1-2*(qq[:,1]**2+qq[:,2]**2),-1,1)))
   ok &= (tilt<=5)&(np.abs(np.degrees(yaw))<=5)&(np.linalg.norm(ww,axis=1)<=.10)
  bad=np.flatnonzero(~ok);start=bad[-1]+1 if len(bad) else 0
  value=float(t[start]) if r['completed'] and start<len(t) and 8-t[start]>=1-1e-9 else None
  expected=r['recovery_position_attitude_s' if axis=='joint' else 'recovery_'+axis+'_s']
  assert value==expected,(r['key'],axis,value,expected)
  latencies[axis]=value
 if r['completed']:
  tail=e[f.time_post>6+1e-9];mu=tail.mean(0);var=np.mean((tail-mu)**2,axis=0);sq=np.mean(tail**2,axis=0)
  m=r['tail'];np.testing.assert_allclose(mu,np.r_[m['mean_error_xy_m'],m['mean_error_z_m']],atol=1e-13)
  np.testing.assert_allclose([np.sqrt(sq[:2].sum()),np.sqrt(sq[2])],[m['rmse_xy_m'],m['rmse_z_m']],atol=1e-13)
  np.testing.assert_allclose([np.sqrt(var[:2].sum()),np.sqrt(var[2])],[m['sway_xy_rms_m'],m['sway_z_rms_m']],atol=1e-13)
 else: assert r['tail'] is None
 checks[r['key']]=dict(recovery_recomputed_s=latencies,same_time_errors=True,absolute_velocity_observation=True,tail_bias_variance=True)
 if r['layout']=='user_frd':
  other=r['key'].replace('-user_frd','-legacy');old=manifest['runs'][other]['initial_snapshot']
  for name in ('position','quaternion','velocity','omega','qpos','qvel','observation','reference','_last_omega','_last_f'):
   np.testing.assert_array_equal(init[name],old[name])
  raw=read_columns(D/(other+'.csv'))
  np.testing.assert_array_equal(vector(f,'action',4)[0],vector(raw,'action',4)[0])
  checks[r['key']]['same_initial_physical_state_motor_speed_observation_and_first_action']=True
(D/'nominal_csv_verification.json').write_text(json.dumps(checks,indent=2))
print('independently verified',len(checks),'nominal CSVs')

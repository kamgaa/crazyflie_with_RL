import csv, hashlib, json
from pathlib import Path
import numpy as np
root=Path(Path('/tmp/crazyflie-ab-training-comparison-dir').read_text())
completion=json.loads((root/'completion.json').read_text())
evaldir=Path(completion['evaluation_dir'])
manifest=json.loads((evaldir/'manifest.json').read_text())
summary=json.loads((evaldir/'summary.json').read_text())
assert manifest['status']=='completed'
assert manifest['initial_snapshots_equal'] and manifest['reference_sequences_equal'] and manifest['checkpoint_hashes_unchanged']
checks=[]
for r in summary:
 with (evaldir/f"{r['case']}-{r['label']}.csv").open() as f:rows=list(csv.DictReader(f))
 def vector(key,n=3):return np.array([[float(row[f'{key}_{i}']) for i in range(n)] for row in rows])
 pos,ref,vel,des=map(vector,['position','reference_post','velocity','desired_velocity'])
 err=pos-ref
 expected=-4*err
 scale=np.minimum(1.,1.5/np.maximum(np.linalg.norm(expected,axis=1),1e-30))
 np.testing.assert_allclose(des,expected*scale[:,None],rtol=1e-12,atol=1e-12)
 np.testing.assert_allclose(vector('internal_velocity_error'),vel-des,rtol=1e-12,atol=1e-12)
 metrics={'position_rmse_xy':np.sqrt(np.mean(np.sum(err[:,:2]**2,axis=1))),
          'position_rmse_z':np.sqrt(np.mean(err[:,2]**2)),
          'position_rmse_total':np.sqrt(np.mean(np.sum(err**2,axis=1))),
          'actual_speed_rms_m_s':np.sqrt(np.mean(np.sum(vel**2,axis=1))),
          'internal_velocity_error_rmse_m_s':np.sqrt(np.mean(np.sum((vel-des)**2,axis=1)))}
 for k,v in metrics.items():np.testing.assert_allclose(r[k],v,rtol=1e-12,atol=1e-12)
 clip=np.abs(vector('motor_thrust_unclipped',4)-vector('motor_thrust_command',4))>1e-12
 np.testing.assert_array_equal(clip,vector('motor_allocation_clipped',4).astype(bool))
 np.testing.assert_allclose(np.mean(np.any(clip,axis=1)),r['motor_allocation_clipped_fraction_any_motor'])
 assert r['trial_count']==1 and r['completed_count']==int(r['completed'])
 if not r['completed']:
  assert r['partial'] and r['settling_time_s'] is None
  assert all(v is None for k,v in r.items() if k.startswith('last_2s_'))
 else:
  times=np.array([float(row['time_post']) for row in rows]);mask=times>6+1e-9
  np.testing.assert_allclose(r['last_2s_internal_velocity_error_rms_m_s'],np.sqrt(np.mean(np.sum((vel[mask]-des[mask])**2,axis=1))))
  assert len(rows)==800 and abs(times[-1]-8)<1e-12
 checks.append({'model':r['label'],'case':r['case'],'samples':len(rows),'completed':r['completed'],'maximum_recomputed_metric_difference':float(max(abs(r[k]-v) for k,v in metrics.items()))})
report={'source':'same-time physical CSV values, independently recomputed metrics', 'checks':checks, 'initial_snapshots_equal':True,'reference_sequences_equal':True}
(root/'evaluation_verification.json').write_text(json.dumps(report,indent=2)+'\n')
def fmt(x):return 'null' if x is None else f'{x:.6f}' if isinstance(x,float) else str(x)
def table(columns):
 lines=['| '+' | '.join(name for name,key in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
 for r in summary:lines.append('| '+' | '.join(fmt(r[key]) for _,key in columns)+' |')
 return '\n'.join(lines)
text='각 모델·케이스는 고정 초기 상태 1회 평가이며, partial은 실제 관측된 구간만의 지표다. 속도 오차는 v - norm_clip(-4 e_p,1.5)다.\n\n'
text+=table([('모델','label'),('케이스','case'),('완료','completed'),('구간','metric_scope'),('시간 s','actual_duration_sec'),('종료','end_reason'),('XY RMSE m','position_rmse_xy'),('Z RMSE m','position_rmse_z'),('3D RMSE m','position_rmse_total'),('정착 s','settling_time_s'),('속도 RMS m/s','actual_speed_rms_m_s'),('속도 오차 RMS m/s','internal_velocity_error_rmse_m_s')])
text+='\n\n자세 단위는 도(deg), 모터 지표는 샘플 비율(0–1)이다.\n\n'
text+=table([('모델','label'),('케이스','case'),('roll RMS deg','roll_rms_deg'),('pitch RMS deg','pitch_rms_deg'),('최대 tilt deg','tilt_max_deg'),('allocator clip','motor_allocation_clipped_fraction_any_motor'),('ESC lower','motor_command_at_lower_bound_fraction_any_motor'),('ESC upper','motor_command_at_upper_bound_fraction_any_motor')])
text+='\n\n마지막 2초 지표는 8초를 완료한 경우에만 계산한다.\n\n'
text+=table([('모델','label'),('케이스','case'),('위치 RMSE m','last_2s_position_rmse_total'),('실제 속도 RMS m/s','last_2s_speed_rms'),('속도 오차 RMS m/s','last_2s_internal_velocity_error_rms_m_s')])
(root/'evaluation_tables.md').write_text(text+'\n')
print(json.dumps(report,indent=2))
print(text)

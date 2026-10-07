"""Read-only CSV verification and derived reporting for this evaluation directory."""
import csv,json,hashlib,shutil
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text());summary=json.loads((root/'summary.json').read_text())
repo=Path(manifest['selection_record']).parents[3]
assert manifest['status']=='completed' and len(summary)==14
errors=[];identity=[];clips={};motor_rows=[];window_rows=[]
for r in summary:
 stem=r['condition']+'-'+r['label']
 with (root/(stem+'.csv')).open() as f:rows=list(csv.DictReader(f))
 def vec(key,n=3):return np.array([[float(row[f'{key}_{i}']) for i in range(n)] for row in rows])
 t=np.array([float(row['time_post']) for row in rows]);e=vec('position')-vec('reference_post');v=vec('velocity')
 des=-4*e;des*=np.minimum(1,1.5/np.maximum(np.linalg.norm(des,axis=1),1e-30))[:,None]
 np.testing.assert_array_equal(vec('position_error_world'),e)
 np.testing.assert_allclose(vec('desired_velocity'),des,atol=1e-12,rtol=1e-12)
 np.testing.assert_allclose(vec('internal_velocity_error'),v-des,atol=1e-12,rtol=1e-12)
 np.testing.assert_allclose(vec('policy_raw_velocity'),vec('velocity_before'),atol=1e-7,rtol=1e-7)
 for key,(a,b) in {'full_0_20':(0,20),'pre_3_5':(3,5),'post_5_20':(5,20),'tail_18_20':(18,20)}.items():
  mask=(t>a+1e-9)&(t<=b+1e-9);x=e[mask];s=r['windows'][key]
  if s is None:continue
  mean=x.mean(0);var=((x-mean)**2).mean(0);square=(x*x).mean(0)
  expected={'offset_xy_m':np.linalg.norm(mean[:2]),'rmse_xy_m':np.sqrt(square[:2].sum()),'sway_xy_rms_m':np.sqrt(var[:2].sum()),'mean_error_z_m':mean[2],'rmse_z_m':np.sqrt(square[2]),'sway_z_rms_m':np.sqrt(var[2])}
  errors.extend(abs(s[k]-value) for k,value in expected.items())
  np.testing.assert_allclose(list(expected.values()),[s[k] for k in expected],rtol=1e-12,atol=1e-12)
  identity.extend([abs(square[:2].sum()-mean[:2]@mean[:2]-var[:2].sum()),abs(square[2]-mean[2]**2-var[2])])
  window_rows.append({'condition':r['condition'],'model':r['label'],'window':key,**s})
 with (root/(stem+'-physics.csv')).open() as f:physical=list(csv.DictReader(f))
 assert len(physical)==5*len(rows)
 flags=np.array([[float(row[f'allocator_clipped_{i}'])>0 for i in range(4)] for row in physical])
 np.testing.assert_allclose(flags.sum(0)*.002,r['motor_diagnostics']['allocator_clipped']['duration_s_per_channel'])
 ids=np.flatnonzero(flags.any(1));groups=np.split(ids,np.where(np.diff(ids)>1)[0]+1)
 clips[stem]=[[float(physical[g[0]]['physics_time']),float(physical[g[-1]]['physics_time_post'])] for g in groups if len(g)]
 for i in range(4):
  md=r['motor_diagnostics'];item=dict(condition=r['condition'],model=r['label'],motor_number=i+1)
  for name,value in md.items():
   if isinstance(value,list):item[name]=value[i]
   elif isinstance(value,dict) and name not in ('policy_action_at_bound','policy_action_clipped'):
    item[name+'_duration_s']=value['duration_s_per_channel'][i]
    item[name+'_longest_s']=value['longest_continuous_s_per_channel'][i]
  motor_rows.append(item)
 # Nominal first 8s exactly reproduces the prior common A/B evaluation, ignoring new fields/horizon flags.
 if r['condition']=='nominal':
  prior=repo/'artifacts/runs/dr-transfer-a7_icqln'/('hover-'+r['label']+'.csv')
  with prior.open() as f:old=list(csv.DictReader(f))
  for oldrow,newrow in zip(old,rows):
   for key in ('position','velocity','quaternion','omega','action','motor_thrust'):
    n=4 if key in ('quaternion','action','motor_thrust') else 3
    for i in range(n):assert float(oldrow[f'{key}_{i}'])==float(newrow[f'{key}_{i}'])

for filename,records in [('window_metrics.csv',window_rows),('motor_summary.csv',motor_rows)]:
 with (root/filename).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
protected=json.loads((root/'protected_hashes_before.json').read_text())
older=json.loads((repo/'artifacts/runs/velocity-ab-training-comparison-isygddsb/preserved_hashes_before.json').read_text())
for p,h in {**protected,**older}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
proof=dict(verified_rollouts=14,control_samples=sum(r['sample_count'] for r in summary),physics_samples=sum(r['sample_count']*5 for r in summary),
 maximum_csv_metric_difference_m=float(max(errors)),maximum_bias_variance_identity_residual_m2=float(max(identity)),
 nominal_first8s_matches_prior_ab_evaluation=True,protected_files_checked=len({**protected,**older}),
 protected_files_unchanged=True,allocator_clipping_intervals=clips)
(root/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')

def f(v,scale=1,digits=3):return 'null' if v is None else f'{v*scale:.{digits}f}'
lines=['| 조건 | 모델 | 완료/종료 | 5–20s 최대 XY/absZ (mm) | tail XY offset / 흔들림 (mm) | tail Z 평균 / 흔들림 (mm) | 회복 XY/Z/3D (s) | allocator clip (s) |',
       '|---|---|---|---:|---:|---:|---|---:|']
for r in summary:
 tail=r['windows']['tail_18_20'];post=r['windows']['post_5_20'];md=r['motor_diagnostics'];fault=r['fault_scheduled']
 pair=lambda a,b:f'{f(a,1000)} / {f(b,1000)}'
 recovery=' / '.join(f(r['recovery_'+d+'_s'],digits=2) for d in ('xy','z','3d')) if fault else '해당 없음'
 lines.append('| '+' | '.join([r['condition'],r['label'],f"{r['actual_duration_sec']:.2f}s / {r['end_reason']}",pair(post['max_xy_error_m'],post['max_abs_z_error_m']),pair(tail['offset_xy_m'],tail['sway_xy_rms_m']),pair(tail['mean_error_z_m'],tail['sway_z_rms_m']),recovery,f(md['allocator_clipped']['duration_any_s'])])+' |')
(root/'comparison_table.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(proof,indent=2));print('\n'.join(lines))

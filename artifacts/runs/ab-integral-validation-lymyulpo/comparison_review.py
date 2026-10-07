"""Compact reporting from saved summaries; no rollouts or fitting."""
import csv,json
from pathlib import Path
p=Path(__file__).resolve().parent
results=json.loads((p/'summary.json').read_text())
flat=[]
for r in results:
 row={k:r[k] for k in ('condition','label','gain','completed','actual_duration_sec','end_reason')}
 for w in ('pre_3_5','tail_18_20','pre_second_28_30','tail_58_60'):
  metrics=r['windows'][w]
  for key in ('offset_xy_m','mean_error_z_m','sway_xy_rms_m','sway_z_rms_m','actual_velocity_xy_rms_m_s','actual_velocity_z_rms_m_s','yaw_error_mean_deg','yaw_error_max_abs_deg'):
   row[w+'__'+key]=None if metrics is None else metrics[key]
 for key in ('integral_frozen','integral_xy_at_limit','integral_z_at_limit','interval_allocator_clipping','interval_esc_boundary','interval_action_boundary'):
  row[key+'_duration_s']=r['integral_diagnostics'][key]['duration_any_s']
 row['longest_freeze_s']=max(r['integral_diagnostics']['integral_frozen']['longest_continuous_s_per_channel'])
 row['xi_final_m']=json.dumps(r['integral_diagnostics']['xi_final_m'])
 for i,e in enumerate(r['event_results']):
  for axis,value in e['recovery_s'].items():row[f'event_{i+1}_recovery_{axis}_s']=value
 flat.append(row)
fields=list(dict.fromkeys(k for row in flat for k in row))
with (p/'comparison_compact.csv').open('w',newline='') as file:
 writer=csv.DictWriter(file,fields);writer.writeheader();writer.writerows(flat)
print('FLIGHTS',sum(r['completed'] for r in results),'/',len(results))
for condition in dict.fromkeys(r['condition'] for r in results):
 cells=[]
 for r in results:
  if r['condition']!=condition:continue
  w=r['windows']['tail_58_60']
  cells.append(f"{w['offset_xy_m']*1000:.3f}/{w['mean_error_z_m']*1000:+.3f}" if w else f"FAIL {r['actual_duration_sec']:.2f}s")
 print(condition,' | '.join(cells))
for r in results:
 if len(r['event_results'])!=2:continue
 e=r['event_results'][1];w=r['windows'][f"{e['kind']}_30_60"]
 b=e['before_event']
 print('EVENT',r['key'],'REC',[x['recovery_s'] for x in r['event_results']],
  'MAX_XY_Z_mm',[w['max_xy_error_m']*1000,w['max_abs_z_error_m']*1000] if w else None,
  'TILT',w['max_tilt_deg'] if w else None,'BEFORE_ERR',b['e_true_before'],
  'BEFORE_SPEED',b['velocity'],'BEFORE_XI',b['xi_before'], 'CMD',e['motion_command_metrics'],
  'FINAL_XI',r['integral_diagnostics']['xi_final_m'])

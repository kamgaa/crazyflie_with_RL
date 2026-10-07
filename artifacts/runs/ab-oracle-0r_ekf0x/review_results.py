"""Compact physical comparison from saved metrics; no rollout or tuning."""
import csv,json
from pathlib import Path

p=Path(__file__).resolve().parent
results=json.loads((p/'summary.json').read_text());rows=[]
for r in results:
    row={k:r[k] for k in ('key','condition','label','configuration','allocator_mode','gain','completed','actual_duration_sec','end_reason','fault_to_termination_s')}
    tail=r['windows']['tail_58_60']
    for key in ('mean_error_xy_m','offset_xy_m','mean_error_z_m','sway_xy_rms_m','sway_z_rms_m','yaw_error_mean_deg','yaw_error_max_abs_deg'):
        row['tail_'+key]=None if tail is None else tail[key]
    for key in ('integral_frozen','interval_allocator_clipping','interval_esc_boundary','interval_action_boundary','integral_xy_at_limit','integral_z_at_limit'):
        row[key+'_time_s']=r['integral_diagnostics'][key]['duration_any_s']
        row[key+'_longest_s']=max(r['integral_diagnostics'][key]['longest_continuous_s_per_channel'])
    for event in r['event_results']:
        prefix=event['kind']+'_';metric=event['observed_metrics']
        row[prefix+'partial']=event['observed_metrics_partial']
        for key in ('max_xy_error_m','max_abs_z_error_m','max_tilt_deg','max_angular_speed_rad_s','yaw_error_max_abs_deg'):
            row[prefix+key]=None if metric is None else metric[key]
        for axis,value in event['recovery_s'].items():row[prefix+'recovery_'+axis+'_s']=value
        row[prefix+'before_event']=event['before_event']
        if metric:
            for term in ('allocation_residual_xml','actuator_response_residual_xml'):
                for measure,value in metric['wrench_diagnostics'][term].items():row[prefix+term+'_'+measure]=value
    rows.append(row)
fields=list(dict.fromkeys(k for row in rows for k in row))
with (p/'comparison_compact.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fields);writer.writeheader()
    writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
print('COMPLETED',sum(r['completed'] for r in results),'/',len(results))
for r in results:
    t=r['windows']['tail_58_60'];diag=r['integral_diagnostics']
    print(r['key'],f"duration={r['actual_duration_sec']:.2f}",
        'tail_XY_Z_mm='+('null' if t is None else f"{t['offset_xy_m']*1000:.3f}/{t['mean_error_z_m']*1000:+.3f}"),
        'tail_yaw_deg='+('null' if t is None else f"{t['yaw_error_mean_deg']:+.3f}"),
        'clip_s='+str(r['motor_diagnostics']['allocator_clipped']['duration_any_s']),
        'freeze_s='+str(diag['integral_frozen']['duration_any_s']),
        'event_rec='+str([(e['kind'],e['recovery_s']) for e in r['event_results']]))
for r in results:
    if not r['event_results']:continue
    print('TRANSIENT',r['key'])
    for e in r['event_results']:
        q=e['observed_metrics']
        print(e['kind'],'partial=',e['observed_metrics_partial'],'maxXY/Zmm=',None if q is None else [q['max_xy_error_m']*1000,q['max_abs_z_error_m']*1000],
              'tilt=',None if q is None else q['max_tilt_deg'],'recovery=',e['recovery_s'])

"""Post-evaluation numerical/terminal audit, CPU only, no optimizer."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from ppo_common import dump_json
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir)
models=['parent','control_stop_best','control_stop_final','track_return_best','track_return_final'];checks=[]
def rows(d,c):
 info=json.loads((d/'results.json').read_text());cp=info[0]['checkpoint_name']
 with (d/cp/(c+'.csv')).open() as f:return list(csv.DictReader(f))
for m in models:
 d8=r/'evaluation'/f'{m}_fixed_8s';d30=r/'evaluation'/f'{m}_fixed_30s'
 for case in ['nominal','x_plus','x_minus','y_plus','y_minus','z_plus','z_minus']:
  x=rows(d8,case);y=rows(d30,case);n=min(len(x),len(y));keys=['ex','ey','ez','vx','vy','vz','qw','qx','qy','qz','wx','wy','wz']+[f'raw_action_{i}' for i in range(4)]
  err={k:max(abs(float(x[i][k])-float(y[i][k])) for i in range(n)) for k in keys}
  checks.append(dict(model=m,condition=case,common_steps=n,max_position_difference_m=max(err[k] for k in ['ex','ey','ez']),max_state_action_difference=max(err.values()),field_errors=err))
# Not a bitwise PhysX replay assertion. Reload exactness is independently checked on 128 observations.
dump_json(r/'evaluation/horizon_prefix_comparison.json',dict(comparisons=checks,maximum_position_difference_m=max(x['max_position_difference_m'] for x in checks),maximum_state_action_difference=max(x['max_state_action_difference'] for x in checks),meaning='8s vs first 8s of 30s, same 7-env batch; horizon flag excluded. PhysX float variations are distinct from fixed-input network reload.'))
print('PREFIX_AUDIT',max(x['max_position_difference_m'] for x in checks))

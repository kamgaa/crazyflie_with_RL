import argparse,json,copy
from pathlib import Path
import torch
from ppo_common import dump_json
from phase6_common import differences
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir)
checks={}
for b in ['control_stop','track_return']:
 checks[b+'_preflight']=json.loads((r/b/'preflight.json').read_text())['success']
 checks[b+'_resume']=json.loads((r/b/'preflight_resume_verification.json').read_text())['success']
x=torch.load(r/'control_stop/preflight_initial_reset.pt',weights_only=True);y=torch.load(r/'track_return/preflight_initial_reset.pt',weights_only=True)
for k in x:checks['same_initial_'+k]=torch.equal(x[k],y[k])
c=json.loads((r/'control_stop/agent_config.json').read_text());t=json.loads((r/'track_return/agent_config.json').read_text());c.pop('run_name');t.pop('run_name');checks['same_agent_except_label']=c==t
rc=json.loads((r/'control_stop/reward_config.json').read_text());rt=json.loads((r/'track_return/reward_config.json').read_text());rd=differences(rc,rt);checks['only_velocity_reward_difference']=len(rd)==1 and rc['velocity_mode']=='stop' and rt['velocity_mode']=='return'
old=json.loads(Path('runs/phase8_exponential_20261012_164652/E_position/evaluation/reload_8s/results.json').read_text());new=json.loads((r/'evaluation/parent_preflight_8/results.json').read_text());details=[]
for o in old:
 if o['checkpoint_name']!='best':continue
 n=next(x for x in new if x['condition']==o['condition']);delta={k:abs(n[k]-o[k]) for k in ['position_rmse_m','roll_rms_rad','z_std_m']};details.append(dict(condition=o['condition'],absolute_difference=delta));checks['parent_reproduced_'+o['condition']]=all(v<1e-4 for v in delta.values()) and n['completed']==o['completed']
dump_json(r/'parent_reproduction.json',dict(comparison=details,note='Prior single environment vs new 7-environment batch; float32 origin offsets and PhysX numeric variation, not network reload error.'))
dump_json(r/'condition_diff.json',dict(agent=differences(c,t),reward=rd,environment='same .05 XYZ reset box; only intentional velocity reward reference differs',parent_reset_change=[0,.05]))
dump_json(r/'comparison_ready.json',dict(success=all(checks.values()),checks=checks));assert all(checks.values()),checks
print('COMPARISON_READY',checks)

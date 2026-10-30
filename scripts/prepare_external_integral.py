"""Create an isolated frozen-phase14 integral evaluation, never select another model."""
import datetime,json,shutil,subprocess
from pathlib import Path
from zoneinfo import ZoneInfo
from ppo_common import ROOT,sha256,dump_json
src=ROOT/'runs/phase14_vz_weight3_fresh_20261019_120809'
cp=src/'best.pt';sel=json.loads((src/'best_selection.json').read_text())
assert sel['selected']['additional_iteration']==448
assert sha256(cp)==sha256(src/'checkpoint_448.pt')==sel['selected']['sha256']
r=ROOT/'runs'/('phase14_external_integral_'+datetime.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S'))
r.mkdir(exist_ok=False)
for d in ['evidence','evaluation','plots','source_snapshot']: (r/d).mkdir()
for f in ['agent_config.json','reward_config.json','env_config.json','resolved_config.json','best_selection.json']:
 shutil.copy2(src/f,r/f)
original=Path('/home/mrl_6534/rl_crazyflie/migration_phase1_20261008/source/crazyflie_rl/integral_controller.py')
shutil.copy2(original,r/'source_snapshot/mujoco_integral_controller.py')
plan=dict(name='Frozen phase14 PPO, external position integral OFF/ON',created_at=datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
 hostname=subprocess.check_output(['hostname'],text=True).strip(),project=str(ROOT),source_run=str(src),checkpoint=str(cp),checkpoint_sha256=sha256(cp),selected_iteration=448,
 reference=str(src/'best_reference.pt'),reference_sha256=sha256(src/'best_reference.pt'),seed=42,device='cuda:0',physics='isaacsim_physx',optimizer_updates=0,
 modes=['off','on'],tasks=[dict(name='hover',horizon_s=30,period_s=None,initial_position=[0,0,1]),dict(name='T10',horizon_s=20,period_s=10,initial_position=[1,0,1]),dict(name='T5',horizon_s=10,period_s=5,initial_position=[1,0,1])],
 integral=dict(gain_s_inv=[.2,.2,.2],control_dt_s=.01,xy_norm_limit_m=.4,z_abs_limit_m=.15,default_enabled=False,error_sign='current minus actual target',policy_error='e_actual - xi',update='post-step with saved pre-step e_actual',freeze='any allocator command outside [0,0.289] N, all3 axes',reset='selected env xi=0; generation guards stale post-step updates',target_change='retain',future_fault_event='retain',pause='no update',zero_error='retain'),
 old_mujoco=dict(path=str(original),sha256=sha256(original),difference='Old code freezes on allocator OR ESC boundary OR action boundary. This specified experiment freezes only on actual allocator clipping; batched GPU state/reset generation added.'),
 timing='Action at t_k uses target(t_k), xi_k. Existing pre_physics hook advances true target to t_(k+1); poststep reward/failure/metrics use target(t_(k+1)). Integral uses saved e_actual(t_k).',
 comparison_tolerances=dict(fixed_action_max_abs=1e-6,off_historical_position_max_m=1e-4,off_historical_body_omega_max_rad_s=.02),no_new_best_selection=True)
dump_json(r/'experiment_plan.json',plan)
dump_json(r/'integral_config.json',plan['integral'])
files=set(p for d in ['src','scripts'] for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix not in ['.pyc'])
files.update(p for p in src.rglob('*') if p.is_file() and p.suffix!='.pyc')
files.update(p for p in (ROOT/'baselines/vz015_best').rglob('*') if p.is_file() and p.suffix!='.pyc')
dump_json(r/'evidence/protected_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in sorted(files)})
ps=subprocess.check_output(['ps','-eo','pid,etimes,args'],text=True)
(r/'evidence/processes_before.txt').write_text('\n'.join(l for l in ps.splitlines() if any(x in l for x in ['python','tensorboard','isaac','train_','evaluate_'])))
(r/'evidence/listening_ports_before.txt').write_text(subprocess.check_output(['ss','-ltn'],text=True))
for cmd,label in [(['git','status','--short'],'git_status'),(['git','rev-parse','HEAD'],'git_head')]:
 x=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True);(r/'evidence'/f'{label}.txt').write_text(x.stdout+x.stderr)
(ROOT/'runs/.external_integral_current').write_text(str(r)+'\n')
print(r,flush=True)

"""Focused PID identity/state/path validation; no simulation or optimization."""
import json,sys,copy,hashlib
from pathlib import Path
import numpy as np
from circle_eval_common import load_pid,wrench_to_action
r=Path(sys.argv[1]);plan=json.loads((r/'experiment_plan.json').read_text());old=Path(plan['historical_pid_run']);prior=json.loads((old/'experiment_plan.json').read_text());checks={}
checks['settings_exactly_historical']=plan['pid']==prior['pid'];checks['source_exactly_historical']=(r/'source_snapshot/original_controllers.py').read_bytes()==(old/'source_snapshot/original_controllers.py').read_bytes()
controllers=[load_pid(r,plan) for _ in plan['cases']];pid=controllers[0];zero=np.zeros(3);unit=[1.,0,0,0];target=[0,0,1]
w=pid(target,unit,zero,zero,target,0);checks['initial_wrench_nominal_hover']=bool(np.allclose(w,[0,0,0,.04338*9.81],atol=1e-14,rtol=0));checks['initial_normalized_command_zero']=bool(np.max(np.abs(wrench_to_action(w)))<1e-14)
controllers[0]([.05,0,1],unit,zero,zero,target,0);checks['independent_integrators']=all(np.all(p._i_vel==0) and np.all(p._i_rate==0) for p in controllers[1:]) and np.linalg.norm(controllers[0]._i_vel)>0
controllers[0].reset();checks['integrator_reset']=bool(np.all(controllers[0]._i_vel==0) and np.all(controllers[0]._i_rate==0));checks['nominal_model_no_payload_adaptation']=all(p.mass==.04338 and np.array_equal(p.inertia_diagonal,[2.3951e-5,2.3951e-5,3.2347e-5]) for p in controllers)
# Historical loader and a fresh copy receive the same state sequence, including saturated commands.
reference=load_pid(old,prior);reference.reset();controllers[0].reset();rng=np.random.default_rng(42);maxerr=0
for i in range(200):
 q=rng.normal(size=4);q/=np.linalg.norm(q);pos=rng.uniform(-1,1,3)+target;vel=rng.normal(size=3);omega=rng.normal(size=3)*3
 a=controllers[0](pos,q,vel,omega,target,0);b=reference(pos,q,vel,omega,target,0);maxerr=max(maxerr,float(np.max(np.abs(a-b))))
checks['historical_stateful_outputs_exact']=maxerr==0;assert all(checks.values()),checks
checks={k:bool(v) for k,v in checks.items()}
result=dict(success=True,checks=checks,historical_sequence_max_wrench_error=maxerr,initial_wrench=w.tolist(),scored_simulation_steps=0,optimizer_updates=0,PID_source_sha256=hashlib.sha256((r/'source_snapshot/original_controllers.py').read_bytes()).hexdigest())
(r/'pid_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

"""Focused mathematical and checkpoint-boundary checks, no simulator or optimizer."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
from ppo_common import dump_json,sha256
from circle_eval_common import reference,load_pid,wrench_to_action
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir);plan=json.loads((r/'experiment_plan.json').read_text())
checks={}
def ck(name,test):checks[name]=bool(test);assert test,name
for T in [10.,5.]:
 t=np.linspace(0,2*T,2001);p,v,acc=reference(t,T)
 ck(f'{T}_radius',np.allclose(np.linalg.norm(p[:,:2],axis=1),1,atol=1e-12))
 ck(f'{T}_z',np.all(p[:,2]==1))
 ck(f'{T}_two_laps',np.allclose(p[[0,1000,2000]],[[1,0,1]]*3,atol=1e-12))
 ck(f'{T}_speed',np.allclose(np.linalg.norm(v,axis=1),2*np.pi/T))
 ck(f'{T}_acceleration',np.allclose(np.linalg.norm(acc,axis=1),(2*np.pi/T)**2))
 eps=1e-5;vp=(reference(t+eps,T)[0]-reference(t-eps,T)[0])/(2*eps)
 ck(f'{T}_derivative',np.allclose(vp,v,atol=1e-8))
 ck(f'{T}_first_motion_ccw',reference(.01,T)[0][1]>0)
pid=load_pid(r,plan)
w=pid([1,0,1],[1,0,0,0],[0,0,0],[0,0,0],[1,0,1])
ck('PID_hover_wrench',np.allclose(w,[0,0,0,.04338*9.81],atol=1e-14))
ck('PID_hover_normalized_action',np.allclose(wrench_to_action(w),0,atol=1e-14))
pid.reset();w=pid([1,0,1],[1,0,0,0],[0,0,0],[0,0,0],[1,.05,1]);ck('positive_y_negative_roll',w[0]<0)
pid.reset();w=pid([1,0,1],[1,0,0,0],[0,0,0],[0,0,0],[1.05,0,1]);ck('positive_x_positive_pitch',w[1]>0)
rng=np.random.default_rng(42);wr=rng.uniform(-.001,.001,(50,4));wr[:,3]+=.04338*9.81
ck('wrench_action_roundtrip',np.allclose(wrench_to_action(wr)*[.0075,.0075,.001,.5]+[0,0,0,.04338*9.81],wr))
for name,record in plan['models'].items():ck(name+'_checkpoint_copy',sha256(record['copy'])==record['sha256']==sha256(record['source']))
ck('frozen_PID_source',sha256(r/'source_snapshot/original_controllers.py')==plan['pid']['sha256'])
dump_json(r/'preflight_verification.json',dict(success=all(checks.values()),checks=checks,optimizer_updates=0))
print('CIRCLE_PREFLIGHT',len(checks),'PASS')

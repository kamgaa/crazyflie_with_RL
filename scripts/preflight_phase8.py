"""Parent reproduction and all three restores precede any phase8 optimizer call."""
import argparse,json,csv,subprocess,sys,time,traceback,shutil
from pathlib import Path
import numpy as np
from ppo_common import ROOT,dump_json,sha256
from phase8_evalmetrics import augment_row,summarize
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
plan=json.loads((r/'experiment_plan.json').read_text());jobs=[]
try:
    assert json.loads((r/'reward_ready.json').read_text())['success']
    newroot=r/'common/evaluation_8s';new=json.loads((newroot/'results.json').read_text())
    parent=Path(plan['parent']);old=json.loads((parent/'evaluation/final_reload/results.json').read_text())
    comparisons=[]
    for x in new:
        prior=next(y for y in old if y['condition']==x['condition'])
        path=parent/'evaluation/final_reload/final'/(x['condition']+'.csv')
        with path.open() as f:rows=list(csv.DictReader(f))
        rows=[{k: (v=='True' if v in ('True','False') else float(v)) for k,v in row.items()} for row in rows]
        for row in rows:augment_row(row)
        metrics,_=summarize(rows,8)
        with (newroot/'initial'/(x['condition']+'.csv')).open() as f:nrows=list(csv.DictReader(f))
        trajectory_error=max(abs(float(nrows[j][k])-rows[j][k]) for j in range(min(len(rows),len(nrows))) for k in ['ex','ey','ez','vx','vy','vz','qw','qx','qy','qz','wx','wy','wz'])
        errors={k:abs(x[k]-metrics[k]) for k in ['position_rmse_m','roll_rms_rad','tail_z_std_m','tail_z_peak_to_peak_m','tail_body_omega_rms_rad_s']}
        ok=(x['completed']==prior['completed_8s'] and errors['position_rmse_m']<=.001+.05*metrics['position_rmse_m']
            and errors['roll_rms_rad']<=.001+.1*metrics['roll_rms_rad'] and errors['tail_z_std_m']<=.0001+.1*metrics['tail_z_std_m'])
        comparisons.append(dict(condition=x['condition'],success=ok,errors=errors,trajectory_max_abs_error_mixed_units=trajectory_error,
            old={k:metrics[k] for k in errors},new={k:x[k] for k in errors},
            tolerance='RMSE 1mm+5%; roll RMS .001rad+10%; tail Z std .1mm+10%; completion identical'))
    dump_json(r/'parent_reproduction.json',dict(success=all(x['success'] for x in comparisons),comparisons=comparisons,
        optimizer_updates=0,new_process_execution=json.loads((newroot/'execution.json').read_text()),reload=json.loads((newroot/'reload_checks.json').read_text())))
    assert all(x['success'] for x in comparisons)
    for name in plan['order']:
        out=r/name;cmd=[sys.executable,str(ROOT/'scripts/train_phase8.py'),'--run-dir',str(out),'--additional-iterations','256','--preflight','--device','cuda:0','--headless','--physics','isaacsim_physx']
        tick=time.time()
        with (out/'preflight.log').open('w') as f:res=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        jobs.append(dict(branch=name,command=cmd,returncode=res.returncode,wall_s=time.time()-tick));dump_json(r/'preflight_jobs.json',jobs)
        assert res.returncode==0,(name,res.returncode)
        shutil.copy2(r/'common/initial.pt',out/'initial.pt');shutil.copy2(r/'common/initial_reference.pt',out/'initial_reference.pt')
    proofs=[json.loads((r/n/'preflight_resume_verification.json').read_text()) for n in plan['order']]
    for key in ['actor_sha256','critic_sha256','mean_sha256','std_sha256','critic_output_sha256']:
        assert len({v[key] for v in proofs})==1,key
    files=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'src').rglob('*.py'))+list((ROOT/'src').rglob('*.usda'))
    for src in files:
        dst=r/'source_snapshot'/src.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    dump_json(r/'source_sha256.json',{str(p.relative_to(ROOT)):sha256(p) for p in files})
    dump_json(r/'comparison_ready.json',dict(success=True,optimizer_updates=0,time=time.time(),branches=plan['order'],initial_outputs_identical=True,parent_reproduction_success=True))
    print('COMPARISON_READY',flush=True)
except BaseException as e:
    dump_json(r/'preflight_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),jobs=jobs));raise


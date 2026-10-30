"""Snapshot and package completed phase6 without changing any training results."""
import argparse,json,hashlib,shutil,socket,subprocess,time,tarfile,urllib.request
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
from phase6_common import preserved
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
def read(p):return json.loads(Path(p).read_text())
assert read(r/'verification.json')['success']
plan=read(r/'experiment_plan.json');parent=Path(plan['parent'])
checks={'original_project':preserved(r/'evidence/project_before_sha256.json',ROOT),
        'parent_run':preserved(r/'evidence/parent_before_sha256.json',parent)}
assert all(x['success'] for x in checks.values())
initial=read(r/'source_sha256.json')
unchanged={k:sha256(ROOT/k)==v and sha256(r/'source_snapshot'/k)==v for k,v in initial.items()}
assert all(unchanged.values())
for src in (ROOT/'scripts').glob('*phase6*.py'):
    dest=r/'source_snapshot'/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
dump_json(r/'final_source_sha256.json',{str(p.relative_to(r/'source_snapshot')):sha256(p) for p in (r/'source_snapshot').rglob('*') if p.is_file()})
ps=subprocess.check_output(['ps','-eo','pid,args'],text=True)
related=[x.strip() for x in ps.splitlines() if ('scripts/train_phase' in x or 'scripts/evaluate_phase' in x) and ' - <<' not in x]
assert not related,related
tb=[x.strip() for x in ps.splitlines() if '-m tensorboard.main' in x]
state=dict(hostname=socket.gethostname(),project=str(ROOT),time=time.time(),active_training_evaluation=related,tensorboard=tb,
    tensorboard_http_status=urllib.request.urlopen('http://127.0.0.1:6006',timeout=5).status,
    nvidia_kernel=Path('/proc/driver/nvidia/version').read_text(),preservation=checks,
    executed_source_unchanged=unchanged,best_sha256=sha256(r/'best.pt'),final_sha256=sha256(r/'final.pt'),
    visual_review=['evaluation_vs_transitions','same_iteration_comparison','ppo_loss','best_trajectories','reward_terms','phase6_ppo_loss_detail','precision_reference'],
    visual_review_note='Inspected rendered PNGs; labels/curves/legends visible; raw data CSVs retained.')
dump_json(r/'completion_verification.json',state)
(ROOT/'REPORT_phase6_ko.txt').write_text((r/'REPORT_phase6_ko.txt').read_text())
files=[x for x in r.rglob('*') if x.is_file() and x.name!='manifest.json' and '__pycache__' not in x.parts]
manifest={str(x.relative_to(r)):sha256(x) for x in files};dump_json(r/'manifest.json',manifest)
archive=r.with_name(r.name+'_review.tar.gz');assert not archive.exists(),archive
start=time.perf_counter()
with tarfile.open(archive,'w:gz',compresslevel=6) as tf:
    for name in sorted([*manifest,'manifest.json']):tf.add(r/name,arcname=r.name+'/'+name,recursive=False)
with tarfile.open(archive,'r:gz') as tf:
    stored=json.load(tf.extractfile(r.name+'/manifest.json'));assert stored==manifest
    for name,expected in manifest.items():
        h=hashlib.sha256(tf.extractfile(r.name+'/'+name).read()).hexdigest()
        assert h==expected,name
digest=sha256(archive);archive.with_suffix(archive.suffix+'.sha256').write_text(digest+'  '+archive.name+'\n')
result=dict(archive=str(archive),sha256=digest,bytes=archive.stat().st_size,files=len(manifest)+1,
    verified_content_hashes=len(manifest),pack_and_verify_wall_s=time.perf_counter()-start,success=True)
dump_json(archive.with_name(r.name+'_archive_verification.json'),result);print(json.dumps(result,indent=2))


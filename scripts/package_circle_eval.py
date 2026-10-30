"""Immutable delivery archive with per-file and archive SHA256 verification."""
import argparse,hashlib,json,shutil,socket,subprocess,tarfile,time
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256,protected_hashes
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
read=lambda p:json.loads(Path(p).read_text())
assert read(r/'verification.json')['success']
assert not (r/'review.tar.gz').exists()
assert protected_hashes()==read(r/'evidence/protected_physics_sha256.json')
assert all(sha256(ROOT/f)==h for f,h in read(r/'evidence/existing_sources_sha256.json').items())
plan=read(r/'experiment_plan.json');assert all(sha256(x['source'])==x['sha256']==sha256(x['copy']) for x in plan['models'].values())
for p in (ROOT/'scripts').glob('*.py'):
 if 'circle_eval' in p.name or p.name=='evaluate_circle.py':shutil.copy2(p,r/'source_snapshot/scripts'/p.name)
dump_json(r/'final_source_sha256.json',{str(p.relative_to(r/'source_snapshot')):sha256(p) for p in (r/'source_snapshot').rglob('*') if p.is_file() and '__pycache__' not in p.parts})
dump_json(r/'completion.json',dict(hostname=socket.gethostname(),all_six_trials_executed=True,optimizer_updates=0,source_and_checkpoint_preserved=True,headless_tested=True,gui_executed=False,time=time.time()))
exclude={'manifest.json','files.sha256','review.tar.gz','review.tar.gz.sha256','archive_verification.json'}
files={str(p.relative_to(r)):sha256(p) for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in exclude}
dump_json(r/'manifest.json',files)
(r/'files.sha256').write_text('\n'.join(h+'  '+name for name,h in sorted({**files,'manifest.json':sha256(r/'manifest.json')}.items()))+'\n')
start=time.perf_counter()
with tarfile.open(r/'review.tar.gz','w:gz',compresslevel=6) as tf:
 for name in sorted([*files,'manifest.json','files.sha256']):tf.add(r/name,arcname=r.name+'/'+name,recursive=False)
count=0
with tarfile.open(r/'review.tar.gz','r|gz') as tf:
 for member in tf:
  name=str(Path(member.name).relative_to(r.name))
  if name in files:
   assert hashlib.sha256(tf.extractfile(member).read()).hexdigest()==files[name],name;count+=1
assert count==len(files)
digest=sha256(r/'review.tar.gz');(r/'review.tar.gz.sha256').write_text(digest+'  review.tar.gz\n')
receipt=dict(success=True,files=count+2,verified_hashes=count,sha256=digest,bytes=(r/'review.tar.gz').stat().st_size,wall_s=time.perf_counter()-start)
dump_json(r/'archive_verification.json',receipt);print(json.dumps(receipt),flush=True)

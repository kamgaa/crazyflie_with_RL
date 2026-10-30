import argparse,hashlib,json,shutil,tarfile,time
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();start=time.perf_counter();assert json.loads((r/'verification.json').read_text())['success']
for folder in ['scripts','src','reference']:
 if (ROOT/folder).exists():shutil.copytree(ROOT/folder,r/'source_snapshot'/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['pyproject.toml','README.rst']:
 if (ROOT/name).exists():shutil.copy2(ROOT/name,r/'source_snapshot'/name)
dump_json(r/'source_change_manifest.json',dict(existing_files_modified=[],new_files={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'scripts').glob('*phase13sq*.py')},training_entrypoint='scripts/train_phase13sq.py adapted from unchanged phase11; dedicated phase13sq reward/telemetry; exact fresh 512 iterations',circle_entrypoint='scripts/evaluate_circle_phase13sq.py retains original phase11 trajectory/control path; new reward monitor only'))
exclude={'review.tar.gz','review.tar.gz.sha256','MANIFEST.sha256','FILES.txt','archive_verification.json','package_phase13sq.log','delivery_sequence.log','delivery_sequence.json','postprocessing_sequence.log','postprocessing_sequence.json','delivery_refinement.log','package_phase13sq.py.refinement.log'}
files=sorted(p for p in r.rglob('*') if p.is_file() and p.name not in exclude and '__pycache__' not in p.parts)
(r/'FILES.txt').write_text('\n'.join(str(p.relative_to(r)) for p in files)+'\nFILES.txt\nMANIFEST.sha256\n');files.append(r/'FILES.txt')
hashes={str(p.relative_to(r)):sha256(p) for p in files};(r/'MANIFEST.sha256').write_text(''.join(v+'  '+k+'\n' for k,v in hashes.items()))
with tarfile.open(r/'review.tar.gz','w:gz',compresslevel=4) as t:
 for p in files+[r/'MANIFEST.sha256']:t.add(p,arcname=str(p.relative_to(r)),recursive=False)
count=0
with tarfile.open(r/'review.tar.gz','r:gz') as t:
 for x in t:
  if x.name not in hashes:continue
  h=hashlib.sha256();f=t.extractfile(x)
  while chunk:=f.read(1<<20):h.update(chunk)
  assert h.hexdigest()==hashes[x.name];count+=1
assert count==len(hashes) and all(sha256(r/k)==v for k,v in hashes.items())
h=sha256(r/'review.tar.gz');(r/'review.tar.gz.sha256').write_text(h+'  review.tar.gz\n');dump_json(r/'archive_verification.json',dict(success=True,files=count,sha256=h,bytes=(r/'review.tar.gz').stat().st_size,packaging_seconds=time.perf_counter()-start));print('ARCHIVE_VERIFIED',count,h,flush=True)

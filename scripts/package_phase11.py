import argparse,tarfile,hashlib,time,json,shutil
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();tick=time.perf_counter()
for directory in ['scripts','src','reference']:
 if (ROOT/directory).exists():
  shutil.copytree(ROOT/directory,r/'source_snapshot'/directory,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['pyproject.toml','README.rst']:
 if (ROOT/name).exists():shutil.copy2(ROOT/name,r/'source_snapshot'/name)
newfiles={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'scripts').glob('*phase11*.py')}
dump_json(r/'source_change_manifest.json',dict(new_files=newfiles,existing_files_modified=[],note='Dedicated entrypoints only. Full imported scripts/src snapshot included; original task, motor, PPO package unmodified.'))
excluded={'review.tar.gz','review.tar.gz.sha256','MANIFEST.sha256','FILES.txt','archive_verification.json','package_phase11.log'}
files=sorted(p for p in r.rglob('*') if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts)
(r/'FILES.txt').write_text('\n'.join(str(p.relative_to(r)) for p in files)+'\nFILES.txt\nMANIFEST.sha256\n');files.append(r/'FILES.txt');files.sort()
hashes={str(p.relative_to(r)):sha256(p) for p in files};(r/'MANIFEST.sha256').write_text(''.join(f'{h}  {f}\n' for f,h in hashes.items()))
archive=r/'review.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
 for p in files+[r/'MANIFEST.sha256']:tar.add(p,arcname=str(p.relative_to(r)),recursive=False)
verified=0
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  if member.name in hashes:
   h=hashlib.sha256();f=tar.extractfile(member)
   while chunk:=f.read(1024*1024):h.update(chunk)
   assert h.hexdigest()==hashes[member.name],member.name;verified+=1
assert verified==len(hashes) and all(sha256(r/f)==h for f,h in hashes.items())
h=sha256(archive);(r/'review.tar.gz.sha256').write_text(h+'  review.tar.gz\n');dump_json(r/'archive_verification.json',dict(success=True,files=verified,sha256=h,bytes=archive.stat().st_size,packaging_and_verification_s=time.perf_counter()-tick));print('ARCHIVE_VERIFIED',verified,h,flush=True)

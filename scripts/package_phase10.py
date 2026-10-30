import argparse,tarfile,hashlib,time,json
from pathlib import Path
from ppo_common import dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();tick=time.perf_counter()
excluded={'review.tar.gz','review.tar.gz.sha256','MANIFEST.sha256','archive_verification.json','package.log','package_phase10.log','delivery.log'}
files=sorted(p for p in r.rglob('*') if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts and p.name!='FILES.txt')
(r/'FILES.txt').write_text('\n'.join(str(p.relative_to(r)) for p in files)+'\nFILES.txt\nMANIFEST.sha256\n')
files.append(r/'FILES.txt');files.sort()
hashes={str(p.relative_to(r)):sha256(p) for p in files}
(r/'MANIFEST.sha256').write_text(''.join(f'{h}  {f}\n' for f,h in hashes.items()))
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
assert verified==len(hashes)
assert all(sha256(r/f)==h for f,h in hashes.items())
h=sha256(archive);(r/'review.tar.gz.sha256').write_text(h+'  review.tar.gz\n')
dump_json(r/'archive_verification.json',dict(success=True,files=verified,sha256=h,bytes=archive.stat().st_size,packaging_and_verification_s=time.perf_counter()-tick,manifest_scope='all archived files except manifest itself; archive checksum external'))
print('ARCHIVE_VERIFIED',verified,archive.stat().st_size,h,flush=True)

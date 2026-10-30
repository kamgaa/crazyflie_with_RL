import hashlib,json,tarfile,shutil,sys,time
from pathlib import Path
r=Path(sys.argv[1]).resolve();assert r.parent.name=='runs' and r.name.startswith('phase16_pid_rigid_payload_eval_')
shutil.copy2(Path(__file__),r/'source_snapshot/scripts'/Path(__file__).name)
excluded={'FILELIST.txt','SHA256SUMS.txt','review.tar.gz','review.tar.gz.sha256','archive_verification.json','package.log'}
files=sorted(p for p in r.rglob('*') if p.is_file() and not (p.parent==r and p.name in excluded))
names=[p.relative_to(r).as_posix() for p in files]+['FILELIST.txt','SHA256SUMS.txt'];(r/'FILELIST.txt').write_text('\n'.join(sorted(names))+'\n');files.append(r/'FILELIST.txt')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
checks={p.relative_to(r).as_posix():digest(p) for p in files};(r/'SHA256SUMS.txt').write_text(''.join(f'{v}  {k}\n' for k,v in sorted(checks.items())));files.append(r/'SHA256SUMS.txt')
with tarfile.open(r/'review.tar.gz','w:gz',compresslevel=6) as t:
 for p in sorted(files):t.add(p,arcname=r.name+'/'+p.relative_to(r).as_posix(),recursive=False)
checked=0
with tarfile.open(r/'review.tar.gz','r:gz') as t:
 members=t.getmembers();assert len(members)==len(names)
 for item in members:
  rel=item.name.split('/',1)[1]
  if rel in checks:
   h=hashlib.sha256();f=t.extractfile(item)
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   assert h.hexdigest()==checks[rel],rel;checked+=1
checksum=digest(r/'review.tar.gz');(r/'review.tar.gz.sha256').write_text(checksum+'  review.tar.gz\n');result=dict(success=True,archive_bytes=(r/'review.tar.gz').stat().st_size,archive_sha256=checksum,members=len(members),member_sha256_verified=checked,filelist_verified=set(names)=={m.name.split('/',1)[1] for m in members},baseline_checkpoint_included=False);assert result['filelist_verified'];(r/'archive_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

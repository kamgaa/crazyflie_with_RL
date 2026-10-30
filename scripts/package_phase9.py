"""Create snapshot and a self-verifiable review archive without touching prior outputs."""
import argparse,json,shutil,subprocess,tarfile,hashlib,time
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve();start=time.perf_counter()
assert json.loads((r/'verification.json').read_text())['success']
for dirname in ['src','scripts','reference']:
 shutil.copytree(ROOT/dirname,r/'source_snapshot'/dirname,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['pyproject.toml','setup.py','setup.cfg','README.rst']:
 if (ROOT/name).exists():shutil.copy2(ROOT/name,r/'source_snapshot'/name)
(r/'source_snapshot/git_status.txt').write_text(subprocess.run(['git','status','--short','--branch'],cwd=ROOT,capture_output=True,text=True).stdout)
(r/'source_snapshot/git_diff.patch').write_text(subprocess.run(['git','diff','--binary'],cwd=ROOT,capture_output=True,text=True).stdout)
(r/'source_snapshot/CHANGES_ko.txt').write_text('Phase9 전용 scripts/*phase9*.py 추가. 기존 src task, phase3~8 및 circle scripts는 변경하지 않았고 baseline SHA256 보존 검사를 통과했다. 세부 파일 설명은 REPORT_phase9_ko.txt 참조.\n')
(r/'evidence/processes_after.txt').write_text(subprocess.run(['ps','-eo','pid,etime,args'],capture_output=True,text=True).stdout)
(r/'evidence/listening_ports_after.txt').write_text(subprocess.run(['ss','-ltnp'],capture_output=True,text=True).stdout)
exclude={'finalization.log','finalize_package_phase9.py.log','review.tar.gz','review.tar.gz.sha256','MANIFEST.sha256','FILE_LIST.txt','archive_verification.json'}
files=sorted(p for p in r.rglob('*') if p.is_file() and p.relative_to(r).parts[0] not in exclude and '__pycache__' not in p.parts)
manifest={str(p.relative_to(r)):sha256(p) for p in files}
(r/'MANIFEST.sha256').write_text(''.join(f'{h}  {name}\n' for name,h in manifest.items()))
(r/'FILE_LIST.txt').write_text('\n'.join(manifest)+'\nMANIFEST.sha256\nFILE_LIST.txt\n')
archive=r/'review.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
 for pth in files+[r/'MANIFEST.sha256',r/'FILE_LIST.txt']:tar.add(pth,arcname=str(pth.relative_to(r)),recursive=False)
checked=0
with tarfile.open(archive,'r:gz') as tar:
 for name,expected in manifest.items():
  f=tar.extractfile(name);h=hashlib.sha256()
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  assert h.hexdigest()==expected,name;checked+=1
 h_manifest=tar.extractfile('MANIFEST.sha256').read();assert h_manifest==(r/'MANIFEST.sha256').read_bytes()
outer=sha256(archive);(r/'review.tar.gz.sha256').write_text(outer+'  review.tar.gz\n')
dump_json(r/'archive_verification.json',dict(success=True,files_verified=checked,manifest_verified=True,tar_read_all_members=True,archive_sha256=outer,bytes=archive.stat().st_size,packaging_wall_s=time.perf_counter()-start))
print('ARCHIVE_OK',checked,archive.stat().st_size,outer,flush=True)

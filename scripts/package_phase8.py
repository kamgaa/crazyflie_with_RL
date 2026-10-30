"""Phase8 archive, verified in one streaming pass; no prior artifacts modified."""
import argparse,json,shutil,time,subprocess,socket,hashlib,tarfile,urllib.request,importlib.metadata as md
from pathlib import Path
from ppo_common import ROOT,dump_json,sha256
from phase6_common import preserved
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();r=Path(a.run_dir).resolve()
def read(p):return json.loads(Path(p).read_text())
assert read(r/'verification.json')['success'] and read(r/'plot_generation.json')['success']
assert not (r/'review.tar.gz').exists()
checks={k:preserved(r/'evidence'/fn,ROOT) for k,fn in [('project','project_before_sha256.json'),('prior','prior_runs_sha256.json')]}
assert all(x['success'] for x in checks.values())
source=read(r/'source_sha256.json')
unchanged={k:sha256(ROOT/k)==h and sha256(r/'source_snapshot'/k)==h for k,h in source.items()}
assert all(unchanged.values()),[k for k,v in unchanged.items() if not v]
for src in (ROOT/'scripts').glob('*phase8*.py'):
    dest=r/'source_snapshot'/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
dump_json(r/'final_source_sha256.json',{str(p.relative_to(r/'source_snapshot')):sha256(p) for p in (r/'source_snapshot').rglob('*') if p.is_file()})
versions={}
for name in ['torch','isaaclab','isaacsim','rsl-rl-lib','tensorboard','newton']:
    try:versions[name]=md.version(name)
    except md.PackageNotFoundError:versions[name]='not found'
dump_json(r/'package_versions.json',versions)
ps=subprocess.check_output(['ps','-eo','pid,args'],text=True)
active=[x.strip() for x in ps.splitlines() if 'scripts/train_phase' in x or 'scripts/evaluate_phase' in x]
assert not active,active
try:tb_status=urllib.request.urlopen('http://127.0.0.1:6006',timeout=5).status
except Exception as e:tb_status=str(e)
dump_json(r/'completion_verification.json',dict(success=True,hostname=socket.gethostname(),time=time.time(),
    executed_source_unchanged=unchanged,preservation=checks,active_training_evaluation=active,
    tensorboard=[x.strip() for x in ps.splitlines() if '-m tensorboard.main' in x],tensorboard_http_status=tb_status,
    gui_executed=False,driver_repair_executed=False))
exclude={'manifest.json','files.sha256','review.tar.gz','review.tar.gz.sha256','archive_verification.json'}
files=[p for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in exclude]
manifest={str(p.relative_to(r)):sha256(p) for p in files};dump_json(r/'manifest.json',manifest)
hashlist=dict(manifest,**{'manifest.json':sha256(r/'manifest.json')})
(r/'files.sha256').write_text('\n'.join(h+'  '+name for name,h in sorted(hashlist.items()))+'\n')
start=time.perf_counter();archive=r/'review.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tf:
    for name in sorted([*manifest,'manifest.json','files.sha256']):tf.add(r/name,arcname=r.name+'/'+name,recursive=False)
verified=0
with tarfile.open(archive,'r|gz') as tf:
    for member in tf:
        name=str(Path(member.name).relative_to(r.name))
        if name in manifest:
            assert hashlib.sha256(tf.extractfile(member).read()).hexdigest()==manifest[name],name;verified+=1
        elif name=='manifest.json':assert json.load(tf.extractfile(member))==manifest
assert verified==len(manifest)
digest=sha256(archive);(r/'review.tar.gz.sha256').write_text(digest+'  review.tar.gz\n')
receipt=dict(success=True,path=str(archive),sha256=digest,bytes=archive.stat().st_size,files=len(manifest)+2,verified_hashes=verified,wall_s=time.perf_counter()-start)
dump_json(r/'archive_verification.json',receipt);print(json.dumps(receipt,indent=2),flush=True)


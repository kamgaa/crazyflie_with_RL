"""Run-local evidence helpers; no changes to task physics or reward."""
import hashlib,json,os,subprocess,threading,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASK='MRL-Crazyflie-Hover-E2E-v0'

def dump_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,default=str,allow_nan=False)+'\n')

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def protected_hashes():
    base=ROOT/'src/crazyflie_isaaclab'
    files=list((base/'tasks/crazyflie').glob('*.py'))+[base/'assets/cf21b.usda']
    return {str(p.relative_to(ROOT)):sha256(p) for p in files}

def model_hash(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def gpu_snapshot():
    def query(args):
        return subprocess.check_output(['nvidia-smi',*args,'--format=csv,noheader,nounits'],text=True,timeout=5).strip()
    try:
        rows=query(['--query-compute-apps=pid,used_gpu_memory']).splitlines()
        own=[float(r.split(',')[1].strip()) for r in rows if r.split(',')[0].strip()==str(os.getpid())]
        g=query(['--id=0','--query-gpu=memory.used,memory.total,utilization.gpu']).split(',')
        return dict(wall_time=time.time(),pid=os.getpid(),process_mib=sum(own),
                    device_used_mib=float(g[0]),device_total_mib=float(g[1]),gpu_util_percent=float(g[2]))
    except Exception as e:return dict(wall_time=time.time(),error=str(e))

class GpuSampler:
    def __init__(self):
        self.samples=[];self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        while not self.stop_event.is_set():
            self.samples.append(gpu_snapshot());self.stop_event.wait(1.0)
    def start(self):self.thread.start()
    def stop(self):self.stop_event.set();self.thread.join(timeout=6)

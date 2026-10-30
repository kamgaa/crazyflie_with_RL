"""Evidence recorder for offline verification (host transfers only in diagnostics)."""
import json,csv
from pathlib import Path
import numpy as np
import torch

def host(x):
    if isinstance(x,torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)

class Checks:
    def __init__(self): self.rows=[]
    def close(self,name,measured,expected,atol,unit='',note=''):
        m,e=host(measured),host(expected)
        err=float(np.max(np.abs(m.astype(float)-e.astype(float))))
        passed=bool(np.isfinite(m).all() and np.isfinite(e).all() and err<=atol)
        self.rows.append(dict(name=name,expected=e.tolist(),measured=m.tolist(),max_abs_error=err,
                              tolerance=atol,unit=unit,passed=passed,note=note))
        print(('PASS ' if passed else 'FAIL ')+name+f' error={err:.7g} tol={atol}',flush=True)
    def save(self,path,metadata=None):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        out=dict(metadata=metadata or {},passed=all(x['passed'] for x in self.rows),checks=self.rows)
        (path/'results.json').write_text(json.dumps(out,indent=2)+'\n')
        with (path/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['name','expected','measured','max_abs_error','tolerance','unit','passed','note'])
            writer.writeheader();writer.writerows(self.rows)
        return out['passed']

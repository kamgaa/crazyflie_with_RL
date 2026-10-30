"""Phase-5 additions to preserved phase-4 telemetry; no task/PPO arithmetic changes."""
import ast,inspect,textwrap,types,random,subprocess
from pathlib import Path
import numpy as np
import torch
from phase4_metrics import Telemetry,TERMS,REASONS
from ppo_common import dump_json,model_hash
BUDGET=8388608
INTERVAL=524288
COUNTS={64:4096,256:1024,1024:256}

def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())

def equal_state(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b

def memory():
    free,total=torch.cuda.mem_get_info()
    return dict(cuda_device_used_MiB=(total-free)/2**20,cuda_device_total_MiB=total/2**20,
        torch_allocated_MiB=torch.cuda.memory_allocated()/2**20,torch_reserved_MiB=torch.cuda.memory_reserved()/2**20,
        torch_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,torch_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)

def install_probe(alg,out):
    from rsl_rl.algorithms import PPO
    source=textwrap.dedent(inspect.getsource(PPO.update));new=source
    call='        _capture(self, loss, surrogate_loss, value_loss, entropy, ratio, batch, distribution_params, actions_log_prob)\n'
    assert source.count('        loss.backward()')==1
    new=new.replace('        loss.backward()',call+'        loss.backward()')
    replacements=[]
    for who in ['actor','critic']:
        orig=f'        nn.utils.clip_grad_norm_(self.{who}.parameters(), self.max_grad_norm)'
        repl=f'        _{who}_gn = nn.utils.clip_grad_norm_(self.{who}.parameters(), self.max_grad_norm)'
        assert new.count(orig)==1;new=new.replace(orig,repl);replacements.append((orig,repl))
    gc='        _capture_grad(self, _actor_gn, _critic_gn)\n'
    assert new.count('        self.optimizer.step()')==1
    new=new.replace('        self.optimizer.step()',gc+'        self.optimizer.step()')
    recovered=new.replace(call,'').replace(gc,'')
    for orig,repl in replacements:recovered=recovered.replace(repl,orig)
    assert ast.dump(ast.parse(recovered))==ast.dump(ast.parse(source))
    Path(out,'ppo_update_original.py').write_text(source);Path(out,'ppo_update_instrumented.py').write_text(new)
    def capture(self,loss,policy,value,entropy,ratio,batch,params,logp):
        with torch.no_grad():
            ent=entropy.mean();logratio=logp-torch.squeeze(batch.old_actions_log_prob)
            vals=(policy.detach(),value.detach(),self.value_loss_coef*value.detach(),-self.entropy_coef*ent,loss.detach(),ent,
                self.actor.get_kl_divergence(batch.old_distribution_params,params).mean(),
                ((ratio-1).abs()>self.clip_param).float().mean(),
                ((ratio-1)-logratio).mean(),ratio.mean(),ratio.min(),ratio.max(),
                (loss-(policy+self.value_loss_coef*value-self.entropy_coef*ent)).abs().detach())
            self._p5_losses.append(torch.stack(vals))
    def grad(self,a,c):self._p5_grads.append(torch.stack((a.detach(),c.detach(),(a>self.max_grad_norm).float(),(c>self.max_grad_norm).float())))
    ns=dict(PPO.update.__globals__);ns.update(_capture=capture,_capture_grad=grad)
    exec(compile(new,str(Path(out,'ppo_update_instrumented.py')),'exec'),ns)
    alg.update=types.MethodType(ns['update'],alg);alg._p5_losses=[];alg._p5_grads=[]

def loss_metrics(alg):
    v=torch.stack(alg._p5_losses);g=torch.stack(alg._p5_grads)
    values=torch.cat((v.mean(0),g.mean(0))).cpu().tolist()
    names=['Loss/policy','Loss/value','Loss/value_contribution','Loss/entropy_contribution','Loss/ppo_total','Policy/entropy',
        'PPO/kl','PPO/clip_fraction','PPO/approx_kl','PPO/ratio_mean','PPO/minibatch_ratio_min_mean','PPO/minibatch_ratio_max_mean','Validation/ppo_loss_residual',
        'Gradient/actor_preclip_norm','Gradient/critic_preclip_norm','Gradient/actor_clipped_fraction','Gradient/critic_clipped_fraction']
    alg._p5_losses.clear();alg._p5_grads.clear()
    return dict(zip(names,values))

def fixed_observations():
    # Independent CPU generator: does not consume the training generator state.
    gen=torch.Generator(device='cpu').manual_seed(20261010)
    x=torch.zeros((128,15));x[:,6]=1;x[:,14]=1
    x[1:,:3]=torch.randn((127,3),generator=gen)*.05
    x[1:,3:6]=torch.randn((127,3),generator=gen)*.1
    q=torch.randn((127,4),generator=gen);q[:,0]+=10;q/=q.norm(dim=-1,keepdim=True);x[1:,6:10]=q
    x[1:,10:13]=torch.randn((127,3),generator=gen)*.2
    w,a,b,c=x[:,6:10].unbind(-1);yaw=torch.atan2(2*(w*c+a*b),1-2*(b*b+c*c));x[:,13]=yaw.sin();x[:,14]=yaw.cos()
    return x

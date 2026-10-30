"""Compare independent single/vector PhysX processes after origin subtraction."""
from pathlib import Path
import numpy as np
from checks import Checks
root=Path(__file__).resolve().parents[1]
a=np.load(root/'results/physx_1/fixed_action_trace.npy')
b=np.load(root/'results/physx_16/fixed_action_trace.npy')
c=Checks()
c.close('single_vs_vector_env0',b[:,0],a[:,0],4e-6,'mixed SI')
c.close('single_vs_all_vector_envs',b,a,4e-6,'mixed SI',note='30 policy steps; state, motor, observation, reward')
raise SystemExit(0 if c.save(root/'results/independence',dict(single_envs=1,vector_envs=16,steps=30)) else 1)

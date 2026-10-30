"""Native motor order; user motor 1..4 maps to indices 3..0."""
import torch
from .constants import ALLOC_ARM, MOTOR_DIR, TORQUE_RATIO, ROTOR_POS, COM

def allocation_matrix(device, dtype=torch.float32):
    a = ALLOC_ARM
    return torch.tensor([[-a,-a,a,a],[-a,a,a,-a],
                         [TORQUE_RATIO*d for d in MOTOR_DIR],[1,1,1,1]], device=device, dtype=dtype)

def allocation_inverse(device, dtype=torch.float32):
    # Reference np.linalg.pinv is float64; calculate once, then store in task dtype.
    return torch.linalg.pinv(allocation_matrix(device, torch.float64)).to(dtype)

def rotor_wrench(force, reaction, arms):
    """Sum actual forces [N] and moments [Nm] about the composite COM."""
    f = torch.zeros((*force.shape, 3), device=force.device, dtype=force.dtype)
    f[..., 2] = force
    torque = torch.cross(arms.expand_as(f), f, dim=-1).sum(dim=-2)
    torque[..., 2] += reaction.sum(dim=-1)
    return f.sum(dim=-2).unsqueeze(1), torque.unsqueeze(1)

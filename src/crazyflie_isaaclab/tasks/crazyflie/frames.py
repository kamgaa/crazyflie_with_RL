"""Explicit Lab xyzw / policy wxyz conversions; vectors stay world Z-up / body FLU."""
import torch

def normalize_wxyz(q):
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    fallback = torch.cat((torch.ones_like(q[..., :1]), torch.zeros_like(q[..., 1:])), dim=-1)
    q = torch.where(norm < 1e-12, fallback, q / norm.clamp_min(1e-12))
    return torch.where(q[..., :1] < 0, -q, q)

def xyzw_to_wxyz(q):
    return normalize_wxyz(torch.cat((q[..., 3:], q[..., :3]), dim=-1))

def wxyz_to_xyzw(q):
    q = normalize_wxyz(q)
    return torch.cat((q[..., 1:], q[..., :1]), dim=-1)

def rotate(q, v):
    """Rotate body vectors to world with normalized wxyz quaternion."""
    xyz = q[..., 1:]
    uv = torch.cross(xyz, v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(xyz, uv, dim=-1))

def inverse_rotate(q, v):
    return rotate(torch.cat((q[..., :1], -q[..., 1:]), dim=-1), v)

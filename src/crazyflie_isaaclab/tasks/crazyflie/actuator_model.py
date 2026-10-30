"""Batched GPU motor dynamics. State omega [rad/s], not thrust lag."""
import math
import torch
from .constants import BRANCH_MIN, DT, GAIN, TAU, THRUST_MAX, TORQUE_RATIO, MOTOR_DIR

def polynomial(r):
    return -.23 * r**3 + .562 * r**2 - .043 * r

def thrust_from_omega(omega):
    ratio = (omega / GAIN).clamp(0, 1)
    value = polynomial(ratio).clamp(0, THRUST_MAX)
    return torch.where(ratio <= BRANCH_MIN, 0.0, value)

def inverse_thrust(command):
    command = command.clamp(0, THRUST_MAX)
    active = command > max(1e-12, polynomial(BRANCH_MIN))
    lower = torch.full_like(command, BRANCH_MIN)
    upper = torch.ones_like(command)
    # Same fixed 48 bisections as the reference; each iteration is batched.
    for _ in range(48):
        mid = .5 * (lower + upper)
        value = polynomial(mid)
        lower = torch.where((value < command) & active, mid, lower)
        upper = torch.where((value >= command) & active, mid, upper)
    return torch.where(active, .5 * (lower + upper) * GAIN, 0.0)

class MotorModel:
    def __init__(self, num_envs, device, dtype=torch.float32):
        self.omega = torch.zeros((num_envs, 4), device=device, dtype=dtype)
        self.command = torch.zeros_like(self.omega)
        self.esc = torch.zeros_like(self.omega)
        self.force = torch.zeros_like(self.omega)
        self.reaction = torch.zeros_like(self.omega)
        self.direction = torch.tensor(MOTOR_DIR, device=device, dtype=dtype)
        self.decay = math.exp(-DT / TAU)

    def set_command(self, force):
        self.command.copy_(force.clamp(0, THRUST_MAX))
        self.esc.copy_((inverse_thrust(self.command) / GAIN).clamp(0, 1))

    def advance(self):
        self.omega.mul_(self.decay).add_((1 - self.decay) * GAIN * self.esc)
        self.force.copy_(thrust_from_omega(self.omega))
        self.reaction.copy_(self.force * self.direction * TORQUE_RATIO)

    def reset(self, env_ids, force):
        self.command[env_ids] = force
        self.omega[env_ids] = inverse_thrust(force)
        self.esc[env_ids] = (self.omega[env_ids] / GAIN).clamp(0, 1)
        self.force[env_ids] = thrust_from_omega(self.omega[env_ids])
        self.reaction[env_ids] = self.force[env_ids] * self.direction * TORQUE_RATIO

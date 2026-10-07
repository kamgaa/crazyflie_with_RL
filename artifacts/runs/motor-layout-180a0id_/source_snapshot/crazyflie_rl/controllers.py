"""Crazyflie control math shared by the Gym environment and diagnostics.

The defaults in this module are the executable values from the pre-refactor
``crazyflie_residual_env.py``.  ``CascadePID`` can also consume the immutable
vehicle and PID sections of :class:`~crazyflie_rl.config.ExperimentConfig`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# Legacy public constants.  Keeping these aliases lets old scripts import the
# same values while config-driven runs use their resolved VehicleConfig.
MASS = 0.04338
GRAV = 9.81
ARM = 0.035355
K_TAU = 0.00594
MOTOR_DIR = np.array([1.0, -1.0, 1.0, -1.0])
THRUST_MIN = 0.0
THRUST_MAX = 0.20
J_DIAG = np.array([2.3951e-5, 2.3951e-5, 3.2347e-5])
PHYSICS_HZ = 500.0
TAU_MAX_RP = ARM * (2.0 * (2.0 * THRUST_MAX) - MASS * GRAV)


def quat_normalize_wxyz(q: Sequence[float]) -> np.ndarray:
    """Normalize a ``wxyz`` quaternion, preserving the legacy zero fallback."""

    value = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return value / norm


def rotmat_from_quat_wxyz(q: Sequence[float]) -> np.ndarray:
    """Return the body-to-world rotation matrix for a ``wxyz`` quaternion."""

    w, x, y, z = np.asarray(q, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def build_allocation_matrix(
    arm_length: float = ARM,
    motor_direction: Sequence[float] = MOTOR_DIR,
    torque_coefficient: float = K_TAU,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the preserved motor-order allocation matrix and pseudoinverse."""

    direction = np.asarray(motor_direction, dtype=float)
    if direction.shape != (4,) or not np.all(np.isfinite(direction)):
        raise ValueError("motor_direction must be a finite vector with shape (4,)")
    arm = float(arm_length)
    coefficient = float(torque_coefficient)
    if not np.isfinite(arm) or arm <= 0.0:
        raise ValueError("arm_length must be positive and finite")
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("torque_coefficient must be positive and finite")

    x = np.array([+arm, -arm, -arm, +arm])
    y = np.array([-arm, -arm, +arm, +arm])
    allocation = np.vstack(
        [
            y,
            -x,
            direction * coefficient,
            np.ones(4),
        ]
    ).astype(float)
    return allocation, np.linalg.pinv(allocation)


def _build_B() -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible name for the legacy default allocation matrix."""

    return build_allocation_matrix()


class CascadePID:
    """Stateful position/velocity/attitude/rate cascade from the master branch."""

    def __init__(
        self,
        dt: float,
        *,
        vehicle: Any = None,
        settings: Any = None,
        config: Any = None,
    ):
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("PID dt must be positive and finite")

        if config is not None:
            if vehicle is None:
                vehicle = config.vehicle
            if settings is None:
                settings = config.controller.pid

        # Position -> velocity setpoint.
        self.kp_pos = float(getattr(settings, "kp_position", 4.0))
        self.v_max = float(getattr(settings, "velocity_limit", 1.5))

        # Velocity PI -> acceleration setpoint.  The derivative gain remains a
        # stored compatibility value; master did not evaluate a D term.
        self.kp_vel = float(getattr(settings, "kp_velocity", 4.0))
        self.ki_vel = float(getattr(settings, "ki_velocity", 1.0))
        self.kd_vel = float(getattr(settings, "kd_velocity", 0.0))

        # Attitude P -> rate setpoint.
        self.kp_att = float(getattr(settings, "kp_attitude", 12.0))

        # Rate PI -> torque.  As on master, kd_rate is stored but not applied.
        self.kp_rate = np.asarray(
            getattr(settings, "kp_rate", (0.0008, 0.0008, 0.0006)), dtype=float
        ).copy()
        self.ki_rate = np.asarray(
            getattr(settings, "ki_rate", (0.0, 0.0, 0.0)), dtype=float
        ).copy()
        self.kd_rate = np.asarray(
            getattr(settings, "kd_rate", (0.0, 0.0, 0.0)), dtype=float
        ).copy()
        for name in ("kp_rate", "ki_rate", "kd_rate"):
            value = getattr(self, name)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite vector with shape (3,)")

        self.max_tilt = np.deg2rad(float(getattr(settings, "max_tilt_deg", 35.0)))
        self.max_tau = float(getattr(settings, "max_torque", 0.02))
        self.max_Fz = float(getattr(settings, "max_force", 1.0))
        self.integrator_limit = float(getattr(settings, "integrator_limit", 2.0))

        self.mass = float(getattr(vehicle, "mass", MASS))
        self.gravity = float(getattr(vehicle, "gravity", GRAV))
        self.inertia_diagonal = np.asarray(
            getattr(vehicle, "inertia_diagonal", tuple(J_DIAG)), dtype=float
        ).copy()
        if self.inertia_diagonal.shape != (3,) or not np.all(
            np.isfinite(self.inertia_diagonal)
        ):
            raise ValueError("inertia_diagonal must be a finite vector with shape (3,)")

        scalar_values = {
            "kp_position": self.kp_pos,
            "velocity_limit": self.v_max,
            "kp_velocity": self.kp_vel,
            "ki_velocity": self.ki_vel,
            "kp_attitude": self.kp_att,
            "max_tilt_deg": self.max_tilt,
            "max_torque": self.max_tau,
            "max_force": self.max_Fz,
            "integrator_limit": self.integrator_limit,
            "mass": self.mass,
            "gravity": self.gravity,
        }
        if any(not np.isfinite(value) for value in scalar_values.values()):
            raise ValueError("PID and vehicle values must all be finite")
        if self.v_max <= 0.0 or self.max_tau <= 0.0 or self.max_Fz <= 0.0:
            raise ValueError("PID limits must be positive")
        if self.integrator_limit <= 0.0 or self.mass <= 0.0 or self.gravity <= 0.0:
            raise ValueError("integrator limit, mass, and gravity must be positive")

        self.reset()
        # Kept for callers that inspected this legacy attribute.  External
        # disturbance injection itself belongs to the environment.
        self.dist_torque_body = np.zeros(3)

    def reset(self) -> None:
        self._i_vel = np.zeros(3)
        self._i_rate = np.zeros(3)

    def __call__(
        self,
        pos_W: Sequence[float],
        quat_wxyz: Sequence[float],
        vel_W: Sequence[float],
        omega_B: Sequence[float],
        pos_des: Sequence[float],
        yaw_des: float = 0.0,
    ) -> np.ndarray:
        pos = np.asarray(pos_W, dtype=float)
        quat = np.asarray(quat_wxyz, dtype=float)
        velocity = np.asarray(vel_W, dtype=float)
        omega = np.asarray(omega_B, dtype=float)
        target = np.asarray(pos_des, dtype=float)

        rotation = rotmat_from_quat_wxyz(quat)
        world_z = np.array([0.0, 0.0, 1.0])

        velocity_setpoint = self.kp_pos * (target - pos)
        setpoint_norm = np.linalg.norm(velocity_setpoint)
        if setpoint_norm > self.v_max:
            velocity_setpoint *= self.v_max / setpoint_norm

        velocity_error = velocity_setpoint - velocity
        self._i_vel += velocity_error * self.dt
        self._i_vel = np.clip(
            self._i_vel, -self.integrator_limit, self.integrator_limit
        )
        acceleration_setpoint = (
            self.kp_vel * velocity_error + self.ki_vel * self._i_vel
        )

        desired_force = self.mass * (
            acceleration_setpoint + self.gravity * world_z
        )
        collective_force = float(desired_force @ (rotation @ world_z))
        collective_force = np.clip(collective_force, 0.0, self.max_Fz)

        force_norm = np.linalg.norm(desired_force)
        body_z_desired = (
            desired_force / force_norm if force_norm > 1e-6 else world_z
        )
        cosine_tilt = np.clip(body_z_desired @ world_z, -1.0, 1.0)
        tilt = np.arccos(cosine_tilt)
        if tilt > self.max_tilt:
            axis = np.cross(world_z, body_z_desired)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis /= axis_norm
                body_z_desired = (
                    np.cos(self.max_tilt) * world_z
                    + np.sin(self.max_tilt) * np.cross(axis, world_z)
                )
                body_z_desired /= np.linalg.norm(body_z_desired)

        yaw_heading = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        body_y_desired = np.cross(body_z_desired, yaw_heading)
        body_y_desired /= max(np.linalg.norm(body_y_desired), 1e-6)
        body_x_desired = np.cross(body_y_desired, body_z_desired)
        desired_rotation = np.column_stack(
            [body_x_desired, body_y_desired, body_z_desired]
        )

        attitude_matrix = (
            desired_rotation.T @ rotation - rotation.T @ desired_rotation
        )
        attitude_error = 0.5 * np.array(
            [attitude_matrix[2, 1], attitude_matrix[0, 2], attitude_matrix[1, 0]]
        )
        rate_setpoint = -self.kp_att * attitude_error

        rate_error = rate_setpoint - omega
        # Deliberately no rate-integrator clipping: this is master behavior.
        self._i_rate += rate_error * self.dt
        torque = self.kp_rate * rate_error + self.ki_rate * self._i_rate
        torque += np.cross(omega, self.inertia_diagonal * omega)
        torque = np.clip(torque, -self.max_tau, self.max_tau)

        return np.array([torque[0], torque[1], torque[2], collective_force])


__all__ = [
    "ARM",
    "CascadePID",
    "GRAV",
    "J_DIAG",
    "K_TAU",
    "MASS",
    "MOTOR_DIR",
    "PHYSICS_HZ",
    "TAU_MAX_RP",
    "THRUST_MAX",
    "THRUST_MIN",
    "_build_B",
    "build_allocation_matrix",
    "quat_normalize_wxyz",
    "rotmat_from_quat_wxyz",
]

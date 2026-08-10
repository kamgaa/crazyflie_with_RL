"""Lazy plotting helpers for experiment-owned PNG artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np


def quaternion_to_euler_deg(quaternion: np.ndarray) -> np.ndarray:
    """Convert one ``wxyz`` quaternion to roll, pitch, yaw in degrees."""

    w, x, y, z = np.asarray(quaternion, dtype=float)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _new_path(path: str | Path) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite plot artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_hover_trace(
    path: str | Path,
    *,
    tag: str,
    time_sec: np.ndarray,
    position: np.ndarray,
    attitude_deg: np.ndarray,
    hover_altitude: float,
    line_width: float = 3.0,
) -> Path:
    """Save the legacy two-panel hover trajectory plot."""

    target = _new_path(path)
    plt = _pyplot()
    fig, (position_axis, attitude_axis) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True
    )

    for index, label in enumerate(("x", "y", "z")):
        position_axis.plot(time_sec, position[:, index], lw=line_width, label=label)
    position_axis.axhline(0.0, ls="--", color="gray", lw=1.5)
    position_axis.axhline(hover_altitude, ls=":", color="gray", lw=1.5)
    position_axis.set_ylabel("position [m]")
    position_axis.set_ylim(-1.0, 1.0)
    position_axis.set_title(f"{tag} — position (x,y,z)")
    position_axis.legend(loc="best")
    position_axis.grid(alpha=0.3)

    for index, label in enumerate(("roll", "pitch", "yaw")):
        attitude_axis.plot(
            time_sec, attitude_deg[:, index], lw=line_width, label=label
        )
    attitude_axis.axhline(0.0, ls="--", color="gray", lw=1.5)
    attitude_axis.set_ylabel("attitude [deg]")
    attitude_axis.set_xlabel("time [s]")
    attitude_axis.set_ylim(-20.0, 20.0)
    attitude_axis.set_title(f"{tag} — attitude (roll,pitch,yaw)")
    attitude_axis.legend(loc="best")
    attitude_axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def save_circle_trace(
    path: str | Path,
    *,
    tag: str,
    time_sec: np.ndarray,
    position: np.ndarray,
    attitude_deg: np.ndarray,
    reference_position: np.ndarray,
    position_error: np.ndarray,
    line_width: float = 3.0,
) -> Path:
    """Save the legacy position, attitude, OOD, and XY circle panels."""

    target = _new_path(path)
    plt = _pyplot()
    fig = plt.figure(figsize=(11, 13))
    position_axis = fig.add_subplot(4, 1, 1)
    attitude_axis = fig.add_subplot(4, 1, 2, sharex=position_axis)
    error_axis = fig.add_subplot(4, 1, 3, sharex=position_axis)
    xy_axis = fig.add_subplot(4, 1, 4)

    colors = ("tab:blue", "tab:orange", "tab:green")
    for index, (color, label) in enumerate(zip(colors, ("x", "y", "z"))):
        position_axis.plot(
            time_sec, position[:, index], lw=line_width, color=color, label=label
        )
        position_axis.plot(
            time_sec,
            reference_position[:, index],
            ls=":",
            lw=1.8,
            color=color,
            alpha=0.7,
        )
    position_axis.set_ylabel("position [m]")
    position_axis.set_title(f"{tag} — position (solid=actual, dotted=ref)")
    position_axis.legend(loc="best")
    position_axis.grid(alpha=0.3)

    for index, label in enumerate(("roll", "pitch", "yaw")):
        attitude_axis.plot(
            time_sec, attitude_deg[:, index], lw=line_width, label=label
        )
    attitude_axis.axhline(0.0, ls="--", color="gray", lw=1.5)
    attitude_axis.set_ylabel("attitude [deg]")
    attitude_axis.set_title(f"{tag} — attitude")
    attitude_axis.legend(loc="best")
    attitude_axis.grid(alpha=0.3)

    error_axis.plot(
        time_sec, position_error, lw=line_width, color="tab:red", label="|pos_err|"
    )
    error_axis.axhline(
        0.15, ls="--", color="black", lw=1.5, label="train dist. boundary (0.15m)"
    )
    error_axis.fill_between(
        time_sec,
        0.15,
        position_error,
        where=position_error > 0.15,
        color="tab:red",
        alpha=0.15,
    )
    error_axis.set_ylabel("|pos_err| [m]")
    error_axis.set_xlabel("time [s]")
    error_axis.set_title(f"{tag} — OOD indicator (red = outside training dist.)")
    error_axis.legend(loc="best")
    error_axis.grid(alpha=0.3)

    xy_axis.plot(
        position[:, 0], position[:, 1], lw=line_width, color="tab:blue", label="actual"
    )
    xy_axis.plot(
        reference_position[:, 0],
        reference_position[:, 1],
        ls=":",
        lw=1.8,
        color="black",
        label="ref",
    )
    xy_axis.set_xlabel("x [m]")
    xy_axis.set_ylabel("y [m]")
    xy_axis.set_title(f"{tag} — XY path")
    xy_axis.axis("equal")
    xy_axis.legend(loc="best")
    xy_axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def save_entropy_diagnostic(
    path: str | Path,
    series: Mapping[str, tuple[np.ndarray, np.ndarray] | None],
) -> Path:
    target = _new_path(path)
    plt = _pyplot()
    tags = tuple(series)
    fig, axes = plt.subplots(len(tags), 1, figsize=(10, 12), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, tag in zip(axes, tags):
        values = series[tag]
        if values is None:
            axis.set_title(f"{tag}  (tag unavailable)")
        else:
            steps, samples = values
            axis.plot(steps, samples, lw=2.0, label=tag)
            axis.set_title(tag)
            axis.legend(loc="best")
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("timesteps")
    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def save_iterm_diagnostic(
    path: str | Path,
    *,
    floor: np.ndarray,
    policy: np.ndarray,
    integrator_limit: float,
    force_scale: float,
    payload_mass: float,
    gravity: float,
    hover_altitude: float,
) -> Path:
    target = _new_path(path)
    plt = _pyplot()
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].axhline(
        hover_altitude,
        ls=":",
        color="black",
        lw=1.2,
        label=f"ref z={hover_altitude:g}",
    )
    axes[0].plot(floor[:, 0], floor[:, 1], lw=2.5, label="floor")
    axes[0].plot(policy[:, 0], policy[:, 1], lw=2.5, label="residual")
    axes[0].set_ylabel("z [m]")
    axes[0].set_title("z position (payload sag)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(floor[:, 0], floor[:, 3], lw=2.5, label="floor")
    axes[1].plot(policy[:, 0], policy[:, 3], lw=2.5, label="residual")
    axes[1].axhline(0.0, ls="--", color="gray", lw=1.0)
    axes[1].set_ylabel("pitch [deg]")
    axes[1].set_title("pitch (case-1 = pitch disturbance)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].axhline(
        integrator_limit, ls=":", color="red", lw=1.5, label="integrator clip"
    )
    axes[2].axhline(-integrator_limit, ls=":", color="red", lw=1.5)
    axes[2].plot(floor[:, 0], floor[:, 6], lw=2.5, label="floor i_vel_z")
    axes[2].plot(policy[:, 0], policy[:, 6], lw=2.5, label="residual i_vel_z")
    axes[2].set_ylabel("i_vel_z")
    axes[2].set_title("velocity I-term (z) vs anti-windup clip")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(
        policy[:, 0],
        force_scale * policy[:, 10],
        lw=2.5,
        color="tab:green",
        label="residual delta Fz [N]",
    )
    payload_weight = payload_mass * gravity
    axes[3].axhline(
        payload_weight,
        ls="--",
        color="black",
        lw=1.2,
        label=f"payload weight {payload_weight:.3f} N",
    )
    axes[3].set_ylabel("delta Fz [N]")
    axes[3].set_xlabel("time [s]")
    axes[3].set_title("residual delta Fz vs payload weight")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def save_learning_curve(
    path: str | Path,
    *,
    steps: np.ndarray,
    residual_error: np.ndarray,
    floor_error: float,
) -> Path:
    target = _new_path(path)
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(steps, residual_error, "o-", label="PID + residual")
    axis.axhline(
        floor_error,
        ls="--",
        color="gray",
        label=f"PID floor ({floor_error:.4f})",
    )
    axis.axvspan(40_000, 100_000, alpha=0.12, color="green")
    axis.annotate(
        "best ~80k",
        (80_000, 0.0267),
        textcoords="offset points",
        xytext=(0, -25),
        ha="center",
        arrowprops={"arrowstyle": "->"},
    )
    axis.set_xlabel("timesteps")
    axis.set_ylabel("steady-state pos_err [m] (lower is best)")
    axis.set_title("10g residual: WIN then collapse")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


__all__ = [
    "quaternion_to_euler_deg",
    "save_circle_trace",
    "save_entropy_diagnostic",
    "save_hover_trace",
    "save_iterm_diagnostic",
    "save_learning_curve",
]

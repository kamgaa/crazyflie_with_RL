"""Lazy plotting helpers for experiment-owned PNG artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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


@dataclass(frozen=True)
class _ComparisonRollout:
    """Validated view of one rollout used by the comparison plot."""

    key: str
    label: str
    time_sec: np.ndarray
    position: np.ndarray
    reference_position: np.ndarray
    attitude_deg: np.ndarray
    position_error: np.ndarray
    control_input: np.ndarray
    motor_thrust: np.ndarray
    phases: tuple[str, ...]


@dataclass(frozen=True)
class _PolicyRollout:
    """Validated view of one rollout used by the per-policy flight plot."""

    label: str
    time_sec: np.ndarray
    position: np.ndarray
    reference_position: np.ndarray
    linear_velocity: np.ndarray
    attitude_deg: np.ndarray
    angular_velocity: np.ndarray
    control_input: np.ndarray
    motor_thrust: np.ndarray
    motor_thrust_command: np.ndarray | None
    phases: tuple[str, ...]


_MISSING = object()


def _rollout_value(
    source: Mapping[str, Any] | Any,
    *names: str,
    default: Any = _MISSING,
) -> Any:
    """Read a rollout field from either a mapping or an attribute object."""

    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if not isinstance(source, Mapping) and hasattr(source, name):
            return getattr(source, name)
    if default is not _MISSING:
        return default
    choices = ", ".join(names)
    raise ValueError(f"comparison rollout is missing required field ({choices})")


def _comparison_array(
    value: Any,
    *,
    field: str,
    samples: int,
    columns: int | None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    expected = (samples,) if columns is None else (samples, columns)
    if array.shape != expected:
        raise ValueError(
            f"comparison rollout {field} must have shape {expected}, got {array.shape}"
        )
    return array


def _policy_array(
    value: Any,
    *,
    field: str,
    samples: int,
    columns: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    expected = (samples, columns)
    if array.shape != expected:
        raise ValueError(
            f"policy rollout {field} must have shape {expected}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"policy rollout {field} must contain finite values")
    return array


def _optional_policy_array(
    value: Any,
    *,
    field: str,
    samples: int,
    columns: int,
) -> np.ndarray | None:
    """Return an optional finite diagnostic array, ignoring legacy NaNs."""

    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    expected = (samples, columns)
    if array.shape != expected:
        raise ValueError(
            f"policy rollout {field} must have shape {expected}, got {array.shape}"
        )
    # Legacy fake/partial traces deliberately use NaN for unavailable BLDC
    # diagnostics.  They must remain plottable with the original panels.
    return array.copy() if np.all(np.isfinite(array)) else None


def _normalize_policy_rollout(
    rollout: Mapping[str, Any] | Any,
) -> _PolicyRollout:
    """Validate the signals needed by the seven-panel per-policy plot."""

    label = str(_rollout_value(rollout, "label", default="policy"))
    time_sec = np.asarray(
        _rollout_value(rollout, "time_sec", "time"), dtype=float
    )
    if time_sec.ndim != 1 or time_sec.size < 1:
        raise ValueError(
            "policy rollout time_sec must be a non-empty one-dimensional array"
        )
    if not np.all(np.isfinite(time_sec)):
        raise ValueError("policy rollout time_sec must contain finite values")
    if np.any(np.diff(time_sec) < 0.0):
        raise ValueError("policy rollout time_sec must be nondecreasing")
    samples = int(time_sec.size)

    phases = tuple(
        str(value) for value in _rollout_value(rollout, "phases", "phase")
    )
    if len(phases) != samples:
        raise ValueError("policy rollout phases must have one value per time sample")

    return _PolicyRollout(
        label=label,
        time_sec=time_sec,
        position=_policy_array(
            _rollout_value(rollout, "position", "actual_position"),
            field="position",
            samples=samples,
            columns=3,
        ),
        reference_position=_policy_array(
            _rollout_value(rollout, "reference_position", "reference"),
            field="reference_position",
            samples=samples,
            columns=3,
        ),
        linear_velocity=_policy_array(
            _rollout_value(
                rollout,
                "linear_velocity",
                "velocity",
                "velocity_world",
            ),
            field="linear_velocity",
            samples=samples,
            columns=3,
        ),
        attitude_deg=_policy_array(
            _rollout_value(rollout, "attitude_deg", "attitude"),
            field="attitude_deg",
            samples=samples,
            columns=3,
        ),
        angular_velocity=_policy_array(
            _rollout_value(
                rollout,
                "angular_velocity",
                "omega",
                "body_rate",
            ),
            field="angular_velocity",
            samples=samples,
            columns=3,
        ),
        control_input=_policy_array(
            _rollout_value(rollout, "control_input", "control_u", "action"),
            field="control_input",
            samples=samples,
            columns=4,
        ),
        motor_thrust=_policy_array(
            _rollout_value(
                rollout,
                "motor_thrust",
                "motor_output",
                "motor_force",
            ),
            field="motor_thrust",
            samples=samples,
            columns=4,
        ),
        motor_thrust_command=_optional_policy_array(
            _rollout_value(
                rollout,
                "motor_thrust_command",
                "motor_thrust_requested",
                "motor_thrust_desired",
                default=None,
            ),
            field="motor_thrust_command",
            samples=samples,
            columns=4,
        ),
        phases=phases,
    )


def _normalize_comparison_rollouts(
    rollouts: Mapping[str, Mapping[str, Any] | Any],
) -> tuple[_ComparisonRollout, ...]:
    """Validate and normalize the two policy traces accepted by the plot API."""

    if len(rollouts) != 2:
        raise ValueError(
            "policy comparison requires exactly two rollouts (floor and PPO)"
        )

    normalized: list[_ComparisonRollout] = []
    for raw_key, source in rollouts.items():
        key = str(raw_key)
        label = str(_rollout_value(source, "label", default=key))
        time_sec = np.asarray(
            _rollout_value(source, "time_sec", "time"), dtype=float
        )
        if time_sec.ndim != 1 or time_sec.size < 1:
            raise ValueError(
                "comparison rollout time_sec must be a non-empty one-dimensional array"
            )
        if not np.all(np.isfinite(time_sec)):
            raise ValueError("comparison rollout time_sec must contain finite values")
        if np.any(np.diff(time_sec) < 0.0):
            raise ValueError("comparison rollout time_sec must be nondecreasing")
        samples = int(time_sec.size)

        position = _comparison_array(
            _rollout_value(source, "position", "actual_position"),
            field="position",
            samples=samples,
            columns=3,
        )
        reference = _comparison_array(
            _rollout_value(source, "reference_position", "reference"),
            field="reference_position",
            samples=samples,
            columns=3,
        )
        attitude = _comparison_array(
            _rollout_value(source, "attitude_deg", "attitude"),
            field="attitude_deg",
            samples=samples,
            columns=3,
        )
        raw_error = np.asarray(
            _rollout_value(source, "position_error", "error"), dtype=float
        )
        if raw_error.shape == (samples, 3):
            position_error = np.linalg.norm(raw_error, axis=1)
        else:
            position_error = _comparison_array(
                raw_error,
                field="position_error",
                samples=samples,
                columns=None,
            )
        control_input = _comparison_array(
            _rollout_value(source, "control_input", "control_u", "action"),
            field="control_input",
            samples=samples,
            columns=4,
        )
        motor_thrust = _comparison_array(
            _rollout_value(
                source,
                "motor_thrust",
                "motor_output",
                "motor_force",
            ),
            field="motor_thrust",
            samples=samples,
            columns=4,
        )
        phases = tuple(
            str(value) for value in _rollout_value(source, "phases", "phase")
        )
        if len(phases) != samples:
            raise ValueError(
                "comparison rollout phases must have one value per time sample"
            )
        for field, array in (
            ("position", position),
            ("reference_position", reference),
            ("attitude_deg", attitude),
            ("position_error", position_error),
            ("control_input", control_input),
            ("motor_thrust", motor_thrust),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"comparison rollout {field} must contain finite values")

        normalized.append(
            _ComparisonRollout(
                key=key,
                label=label,
                time_sec=time_sec,
                position=position,
                reference_position=reference,
                attitude_deg=attitude,
                position_error=position_error,
                control_input=control_input,
                motor_thrust=motor_thrust,
                phases=phases,
            )
        )
    return tuple(normalized)


def _comparison_phase_boundaries(
    traces: Sequence[_ComparisonRollout],
) -> tuple[tuple[float, str], ...]:
    """Return the ordered union of phase transitions from both rollouts."""

    unique: dict[tuple[float, str], tuple[float, str]] = {}
    for trace in traces:
        previous = trace.phases[0]
        for index, phase in enumerate(trace.phases[1:], start=1):
            if phase == previous:
                continue
            boundary = float(trace.time_sec[index])
            unique.setdefault((round(boundary, 9), phase), (boundary, phase))
            previous = phase
    return tuple(sorted(unique.values(), key=lambda item: (item[0], item[1])))


def _comparison_parameter_text(
    mission_name: str,
    mission_parameters: Mapping[str, Any] | None,
) -> str:
    values = dict(mission_parameters or {})
    center = values.pop("center_xy", None)
    parts = [f"mission={mission_name}"]
    if center is not None:
        center_values = np.asarray(center, dtype=float)
        if center_values.shape == (2,):
            parts.append(f"center=({center_values[0]:g}, {center_values[1]:g}) m")
    for key, value in values.items():
        if isinstance(value, (str, int, float, bool, np.number)):
            parts.append(f"{key}={value}")
        if len(" | ".join(parts)) > 180:
            parts[-1] = "..."
            break
    return " | ".join(parts)


def _policy_phase_boundaries(
    trace: _PolicyRollout,
) -> tuple[tuple[float, str], ...]:
    """Return each phase transition in one rollout."""

    transitions: list[tuple[float, str]] = []
    previous = trace.phases[0]
    for index, phase in enumerate(trace.phases[1:], start=1):
        if phase == previous:
            continue
        transitions.append((float(trace.time_sec[index]), phase))
        previous = phase
    return tuple(transitions)


def save_policy_trace(
    path: str | Path,
    *,
    tag: str,
    rollout: Mapping[str, Any] | Any,
    mission_name: str = "trajectory",
    mission_parameters: Mapping[str, Any] | None = None,
    motor_unit: str = "N",
    line_width: float = 1.8,
) -> Path:
    """Save one policy's complete flight trace as a single 16:9 PNG.

    The seven panels are position, world-frame linear velocity, per-motor and
    total thrust, attitude, body angular velocity, an equal-scale XY path, and
    normalized control input ``u``. ``rollout`` may be a mapping or an
    attribute object and must provide one row per ``time_sec`` sample for all
    signals. The caller should invoke this once for the floor and once for PPO.
    """

    unit = str(motor_unit).strip()
    if not unit:
        raise ValueError("motor_unit must be a non-empty label")
    trace = _normalize_policy_rollout(rollout)
    target = _new_path(path)
    plt = _pyplot()
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 0.72),
        hspace=0.48,
        wspace=0.30,
    )
    position_axis = fig.add_subplot(grid[0, 0])
    linear_velocity_axis = fig.add_subplot(grid[0, 1], sharex=position_axis)
    thrust_axis = fig.add_subplot(grid[0, 2], sharex=position_axis)
    attitude_axis = fig.add_subplot(grid[1, 0], sharex=position_axis)
    angular_velocity_axis = fig.add_subplot(grid[1, 1], sharex=position_axis)
    path_axis = fig.add_subplot(grid[1, 2])
    # A wide final panel keeps all four control channels readable while
    # retaining the sketch's two rows of three primary flight-state panels.
    control_axis = fig.add_subplot(grid[2, :], sharex=position_axis)
    time_axes = (
        position_axis,
        linear_velocity_axis,
        thrust_axis,
        attitude_axis,
        angular_velocity_axis,
        control_axis,
    )

    xyz_colors = ("tab:blue", "tab:orange", "tab:green")
    for index, (name, color) in enumerate(zip(("x", "y", "z"), xyz_colors)):
        position_axis.plot(
            trace.time_sec,
            trace.position[:, index],
            color=color,
            lw=line_width,
            label=f"{name} actual",
        )
        position_axis.plot(
            trace.time_sec,
            trace.reference_position[:, index],
            color=color,
            ls=":",
            lw=max(1.0, line_width * 0.75),
            alpha=0.85,
            label=f"{name} ref",
        )
        linear_velocity_axis.plot(
            trace.time_sec,
            trace.linear_velocity[:, index],
            color=color,
            lw=line_width,
            label=f"v{name}",
        )
        desired = _rollout_value(rollout, 'desired_velocity', default=None)
        if desired is not None:
            linear_velocity_axis.plot(trace.time_sec, np.asarray(desired)[:, index],
                color=color, ls='--', lw=1., label=f'v{name} desired')

    angle_names = ("roll", "pitch", "yaw")
    omega_names = ("wx", "wy", "wz")
    for index, color in enumerate(xyz_colors):
        attitude_axis.plot(
            trace.time_sec,
            trace.attitude_deg[:, index],
            color=color,
            lw=line_width,
            label=angle_names[index],
        )
        angular_velocity_axis.plot(
            trace.time_sec,
            trace.angular_velocity[:, index],
            color=color,
            lw=line_width,
            label=omega_names[index],
        )

    motor_colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    show_requested_thrust = (
        trace.motor_thrust_command is not None
        and not np.allclose(
            trace.motor_thrust_command,
            trace.motor_thrust,
            rtol=1e-9,
            atol=1e-12,
        )
    )
    for index, color in enumerate(motor_colors):
        thrust_axis.plot(
            trace.time_sec,
            trace.motor_thrust[:, index],
            color=color,
            lw=line_width,
            label=f"M{index + 1}",
        )
        if show_requested_thrust:
            thrust_axis.plot(
                trace.time_sec,
                trace.motor_thrust_command[:, index],
                color=color,
                ls=":",
                lw=max(1.0, line_width * 0.8),
                alpha=0.85,
                label=f"M{index + 1} cmd",
            )
    thrust_axis.plot(
        trace.time_sec,
        np.sum(trace.motor_thrust, axis=1),
        color="black",
        ls="--",
        lw=max(1.2, line_width),
        label="total",
    )

    control_labels = ("u_tau_x", "u_tau_y", "u_tau_z", "u_Fz")
    control_styles = ("-", "--", "-.", ":")
    for index, (label, color, style) in enumerate(
        zip(control_labels, motor_colors, control_styles)
    ):
        control_axis.plot(
            trace.time_sec,
            trace.control_input[:, index],
            color=color,
            ls=style,
            lw=line_width,
            label=label,
        )

    path_axis.plot(
        trace.reference_position[:, 0],
        trace.reference_position[:, 1],
        color="black",
        ls=":",
        lw=max(1.0, line_width * 0.85),
        label="reference",
    )
    path_axis.plot(
        trace.position[:, 0],
        trace.position[:, 1],
        color="tab:blue",
        lw=line_width,
        label="actual",
    )
    path_axis.scatter(
        [trace.position[0, 0]],
        [trace.position[0, 1]],
        color="tab:green",
        marker="o",
        s=24,
        label="start",
        zorder=5,
    )
    path_axis.scatter(
        [trace.position[-1, 0]],
        [trace.position[-1, 1]],
        color="tab:red",
        marker="x",
        s=34,
        label="end",
        zorder=5,
    )
    center = dict(mission_parameters or {}).get("center_xy")
    if center is not None:
        center_xy = np.asarray(center, dtype=float)
        if center_xy.shape == (2,):
            path_axis.scatter(
                [center_xy[0]],
                [center_xy[1]],
                marker="+",
                s=70,
                color="tab:purple",
                label="center",
                zorder=5,
            )

    position_axis.set_title("Position (actual / reference)")
    position_axis.set_ylabel("position [m]")
    linear_velocity_axis.set_title("Linear velocity")
    linear_velocity_axis.set_ylabel("velocity [m/s]")
    thrust_axis.set_title(
        "Motor thrust (actual solid / requested dotted)"
        if show_requested_thrust
        else "Motor thrust"
    )
    thrust_axis.set_ylabel(f"thrust [{unit}]")
    attitude_axis.set_title("Attitude")
    attitude_axis.set_ylabel("angle [deg]")
    angular_velocity_axis.set_title("Angular velocity (body frame)")
    angular_velocity_axis.set_ylabel("angular velocity [rad/s]")
    path_axis.set_title(f"{mission_name} path overview")
    path_axis.set_xlabel("x [m]")
    path_axis.set_ylabel("y [m]")
    path_axis.axis("equal")
    control_axis.set_title("Normalized control input u")
    control_axis.set_ylabel("u [-1, 1]")
    control_axis.set_xlabel("time [s]")
    control_axis.set_ylim(-1.05, 1.05)

    for axis in time_axes[:-1]:
        axis.set_xlabel("time [s]")
    for axis in (attitude_axis, angular_velocity_axis, control_axis):
        axis.axhline(0.0, color="gray", ls=":", lw=0.8)
    thrust_axis.axhline(0.0, color="gray", ls=":", lw=0.8)

    for boundary, phase in _policy_phase_boundaries(trace):
        for axis in time_axes:
            axis.axvline(boundary, color="gray", lw=0.7, alpha=0.35)
        position_axis.annotate(
            phase,
            xy=(boundary, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -2),
            textcoords="offset points",
            rotation=90,
            va="top",
            fontsize=6.5,
            color="dimgray",
        )

    for axis in time_axes:
        axis.grid(alpha=0.25)
    path_axis.grid(alpha=0.25)
    position_axis.legend(loc="best", ncol=2, fontsize=6.5)
    linear_velocity_axis.legend(loc="best", ncol=3, fontsize=7)
    thrust_axis.legend(loc="best", ncol=3, fontsize=6.5)
    attitude_axis.legend(loc="best", ncol=3, fontsize=7)
    angular_velocity_axis.legend(loc="best", ncol=3, fontsize=7)
    path_axis.legend(loc="best", fontsize=6.5)
    control_axis.legend(loc="best", ncol=4, fontsize=7)

    fig.suptitle(f"{tag} - {trace.label}", fontsize=14)
    fig.text(
        0.01,
        0.008,
        _comparison_parameter_text(mission_name, mission_parameters),
        ha="left",
        va="bottom",
        fontsize=7,
        color="dimgray",
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.955))
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return target


def save_policy_comparison_trace(
    path: str | Path,
    *,
    tag: str,
    rollouts: Mapping[str, Mapping[str, Any] | Any],
    mission_name: str = "trajectory",
    mission_parameters: Mapping[str, Any] | None = None,
    motor_unit: str = "N",
    line_width: float = 1.8,
) -> Path:
    """Save one 16:9 figure comparing PID-floor and PPO rollouts.

    ``rollouts`` must contain exactly two entries. Each value can be a mapping or
    an attribute object and provides ``time_sec``, ``position``,
    ``reference_position``, ``attitude_deg``, ``position_error``,
    ``control_input`` (four channels), ``motor_thrust`` (four channels), and
    ``phases``. A human-readable ``label`` is optional. Common concise aliases
    such as ``time``, ``actual_position``, ``control_u``, and ``action`` are also
    accepted so the plotting layer remains independent of the runner dataclass.
    """

    unit = str(motor_unit).strip()
    if not unit:
        raise ValueError("motor_unit must be a non-empty label")
    traces = _normalize_comparison_rollouts(rollouts)
    target = _new_path(path)
    plt = _pyplot()
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        3,
        4,
        height_ratios=(1.15, 1.0, 1.1),
        hspace=0.42,
        wspace=0.34,
    )
    position_axis = fig.add_subplot(grid[0, :3])
    xy_axis = fig.add_subplot(grid[0, 3])
    error_axis = fig.add_subplot(grid[1, 0], sharex=position_axis)
    attitude_axis = fig.add_subplot(grid[1, 1:], sharex=position_axis)
    control_axis = fig.add_subplot(grid[2, :2], sharex=position_axis)
    motor_axis = fig.add_subplot(grid[2, 2:], sharex=position_axis)
    time_axes = (
        position_axis,
        error_axis,
        attitude_axis,
        control_axis,
        motor_axis,
    )

    policy_colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple")
    xyz_styles = ("-", "--", ":")
    channel_styles = ("-", "--", "-.", ":")
    for trace, color in zip(traces, policy_colors):
        for index, (coordinate, style) in enumerate(zip("xyz", xyz_styles)):
            position_axis.plot(
                trace.time_sec,
                trace.position[:, index],
                color=color,
                ls=style,
                lw=line_width,
                label=f"{trace.label} {coordinate}",
            )
        xy_axis.plot(
            trace.position[:, 0],
            trace.position[:, 1],
            color=color,
            lw=line_width,
            label=trace.label,
        )
        error_axis.plot(
            trace.time_sec,
            trace.position_error,
            color=color,
            lw=line_width,
            label=trace.label,
        )
        for index, (name, style) in enumerate(
            zip(("roll", "pitch", "yaw"), xyz_styles)
        ):
            attitude_axis.plot(
                trace.time_sec,
                trace.attitude_deg[:, index],
                color=color,
                ls=style,
                lw=line_width,
                label=f"{trace.label} {name}",
            )
        for index, style in enumerate(channel_styles):
            control_axis.plot(
                trace.time_sec,
                trace.control_input[:, index],
                color=color,
                ls=style,
                lw=line_width,
                label=f"{trace.label} u{index + 1}",
            )
            motor_axis.plot(
                trace.time_sec,
                trace.motor_thrust[:, index],
                color=color,
                ls=style,
                lw=line_width,
                label=f"{trace.label} M{index + 1}",
            )

    # The longest trace best represents the full common mission reference.
    reference_trace = max(traces, key=lambda trace: trace.time_sec.size)
    reference_colors = ("0.15", "0.4", "0.65")
    for index, (coordinate, color) in enumerate(zip("xyz", reference_colors)):
        position_axis.plot(
            reference_trace.time_sec,
            reference_trace.reference_position[:, index],
            color=color,
            ls=":",
            lw=1.3,
            alpha=0.85,
            label=f"ref {coordinate}",
        )
    xy_axis.plot(
        reference_trace.reference_position[:, 0],
        reference_trace.reference_position[:, 1],
        color="black",
        ls=":",
        lw=1.5,
        label="reference",
    )

    center = dict(mission_parameters or {}).get("center_xy")
    if center is not None:
        center_xy = np.asarray(center, dtype=float)
        if center_xy.shape == (2,):
            xy_axis.scatter(
                [center_xy[0]],
                [center_xy[1]],
                marker="+",
                s=90,
                color="tab:red",
                label="center",
                zorder=5,
            )

    position_axis.set_ylabel("position [m]")
    position_axis.set_title("XYZ tracking (policy color; coordinate line style)")
    error_axis.set_ylabel("|position error| [m]")
    error_axis.set_title("Position error")
    error_axis.set_xlabel("time [s]")
    attitude_axis.set_ylabel("attitude [deg]")
    attitude_axis.set_title("Attitude (roll / pitch / yaw)")
    attitude_axis.set_xlabel("time [s]")
    control_axis.set_ylabel("control input u")
    control_axis.set_title("Control input u (u1-u4)")
    control_axis.set_xlabel("time [s]")
    motor_axis.set_ylabel(f"motor thrust [{unit}]")
    motor_axis.set_title(f"Motor thrust (M1-M4) [{unit}]")
    motor_axis.set_xlabel("time [s]")
    xy_axis.set_xlabel("x [m]")
    xy_axis.set_ylabel("y [m]")
    xy_axis.set_title(f"{mission_name} XY path")
    xy_axis.axis("equal")

    error_axis.axhline(0.15, ls="--", color="black", lw=0.9, alpha=0.7)
    attitude_axis.axhline(0.0, ls=":", color="gray", lw=0.8)
    control_axis.axhline(0.0, ls=":", color="gray", lw=0.8)
    motor_axis.axhline(0.0, ls=":", color="gray", lw=0.8)

    for boundary, phase in _comparison_phase_boundaries(traces):
        for axis in time_axes:
            axis.axvline(boundary, color="gray", lw=0.7, alpha=0.35)
        position_axis.annotate(
            phase,
            xy=(boundary, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -2),
            textcoords="offset points",
            rotation=90,
            va="top",
            fontsize=6.5,
            color="dimgray",
        )

    for axis in time_axes:
        axis.grid(alpha=0.25)
    xy_axis.grid(alpha=0.25)
    position_axis.legend(loc="best", ncol=3, fontsize=6.5)
    xy_axis.legend(loc="best", fontsize=7)
    error_axis.legend(loc="best", fontsize=7)
    attitude_axis.legend(loc="best", ncol=3, fontsize=6.5)
    control_axis.legend(loc="best", ncol=4, fontsize=6)
    motor_axis.legend(loc="best", ncol=4, fontsize=6)

    fig.suptitle(f"{tag} - floor vs PPO", fontsize=14)
    fig.text(
        0.01,
        0.008,
        _comparison_parameter_text(mission_name, mission_parameters),
        ha="left",
        va="bottom",
        fontsize=7,
        color="dimgray",
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.955))
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return target


def save_hover_trace(
    path: str | Path,
    *,
    tag: str,
    time_sec: np.ndarray,
    position: np.ndarray,
    attitude_deg: np.ndarray,
    hover_altitude: float,
    reference_position: np.ndarray | None = None,
    position_error: np.ndarray | None = None,
    line_width: float = 3.0,
) -> Path:
    """Save hover position, attitude, and tracking-error panels."""

    target = _new_path(path)
    plt = _pyplot()
    fig, (position_axis, attitude_axis, error_axis) = plt.subplots(
        3, 1, figsize=(10, 10), sharex=True
    )

    for index, label in enumerate(("x", "y", "z")):
        position_axis.plot(time_sec, position[:, index], lw=line_width, label=label)
    position_axis.axhline(0.0, ls="--", color="gray", lw=1.5)
    position_axis.axhline(hover_altitude, ls=":", color="gray", lw=1.5)
    position_axis.set_ylabel("position [m]")
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
    attitude_axis.set_title(f"{tag} — attitude (roll,pitch,yaw)")
    attitude_axis.legend(loc="best")
    attitude_axis.grid(alpha=0.3)

    if reference_position is not None:
        reference = np.asarray(reference_position, dtype=float)
        for index, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
            position_axis.plot(
                time_sec,
                reference[:, index],
                ls=":",
                lw=1.5,
                color=color,
                alpha=0.7,
            )
    errors = (
        np.linalg.norm(
            np.asarray(position, dtype=float)
            - np.asarray(reference_position, dtype=float),
            axis=1,
        )
        if position_error is None and reference_position is not None
        else np.asarray(position_error, dtype=float)
        if position_error is not None
        else np.zeros_like(np.asarray(time_sec, dtype=float))
    )
    error_axis.plot(time_sec, errors, lw=line_width, color="tab:red")
    error_axis.axhline(0.15, ls="--", color="black", lw=1.2, label="0.15 m")
    error_axis.axhline(1.5, ls=":", color="gray", lw=1.2, label="1.5 m")
    error_axis.set_ylabel("|pos_err| [m]")
    error_axis.set_xlabel("time [s]")
    error_axis.set_title(f"{tag} — position error")
    error_axis.legend(loc="best")
    error_axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def save_tracking_trace(
    path: str | Path,
    *,
    tag: str,
    time_sec: np.ndarray,
    position: np.ndarray,
    attitude_deg: np.ndarray,
    reference_position: np.ndarray,
    position_error: np.ndarray,
    mission_name: str = "trajectory",
    mission_parameters: Mapping[str, Any] | None = None,
    phases: Sequence[str] = (),
    line_width: float = 3.0,
) -> Path:
    """Save common circle/Lissajous tracking and XY-path panels."""

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
    position_axis.set_title(
        f"{tag} — {mission_name} position (solid=actual, dotted=ref)"
    )
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
    xy_axis.set_title(f"{tag} — {mission_name} XY path")
    xy_axis.axis("equal")
    xy_axis.legend(loc="best")
    xy_axis.grid(alpha=0.3)

    parameters = dict(mission_parameters or {})
    center = parameters.get("center_xy")
    if center is not None:
        center_xy = np.asarray(center, dtype=float)
        if center_xy.shape == (2,):
            xy_axis.scatter(
                [center_xy[0]],
                [center_xy[1]],
                marker="+",
                s=100,
                color="tab:red",
                label="center",
                zorder=5,
            )
            xy_axis.legend(loc="best")

    if phases and len(phases) == len(time_sec):
        previous = phases[0] if phases else None
        for index, phase in enumerate(phases[1:], start=1):
            if phase == previous:
                continue
            boundary = float(time_sec[index])
            for axis in (position_axis, attitude_axis, error_axis):
                axis.axvline(boundary, color="gray", lw=0.8, alpha=0.35)
            error_axis.annotate(
                phase,
                xy=(boundary, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -2),
                textcoords="offset points",
                rotation=90,
                va="top",
                fontsize=7,
                color="dimgray",
            )
            previous = phase

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
    """Compatibility wrapper for callers using the legacy circle API."""

    return save_tracking_trace(
        path,
        tag=tag,
        time_sec=time_sec,
        position=position,
        attitude_deg=attitude_deg,
        reference_position=reference_position,
        position_error=position_error,
        mission_name="circle",
        line_width=line_width,
    )


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
    "save_policy_comparison_trace",
    "save_policy_trace",
    "save_tracking_trace",
]


def save_transfer_comparison_plot(path, rollouts, reference, case_name):
    """Compare any number of frozen policies at correctly aligned post times.

    Reuses the repository's headless backend, overwrite guard and quaternion
    conversion. Reference velocity is supplied by the mission, not synthesized here.
    """
    target = _new_path(path)
    plt = _pyplot()
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    model_keys = list(dict.fromkeys(rows[0].get('model_label', label)
                                   for label, rows in rollouts.items() if rows))
    colors = {key: f'C{i % 10}' for i, key in enumerate(model_keys)}

    def style(label, rows):
        return dict(color=colors[rows[0].get('model_label', label)],
                    linestyle='--' if rows[0].get('observation_velocity_mode') == 'error' else '-')
    try:
        for column, coordinate in enumerate('xyz'):
            axes[0, column].plot(reference['time'], reference['reference'][:, column],
                                 'k--', label=f'reference {coordinate}', lw=1.2)
            if 'reference_velocity' in reference:
                axes[1, column].plot(reference['time'], reference['reference_velocity'][:, column],
                                     'k:', label=f'reference velocity {coordinate}', lw=1.2)
        for label, rows in rollouts.items():
            if not rows:
                continue
            times = np.array([r['time_post'] for r in rows])
            values = [np.array([r[key] for r in rows]) for key in ('position', 'velocity')]
            values.append(np.array([quaternion_to_euler_deg(r['quaternion']) for r in rows]))
            values.append(np.array([r['omega'] for r in rows]))
            for row, array in enumerate(values):
                for column in range(3):
                    axes[row, column].plot(times, array[:, column], label=label, **style(label, rows))
            if rows[0].get('internal_velocity_reference_mode') == 'position_error':
                desired=np.array([r['desired_velocity'] for r in rows])
                for column in range(3):
                    axes[1,column].plot(times,desired[:,column],color=style(label,rows)['color'],
                                        ls='--',lw=1.,label=f'{label} internal desired')
        units = ('position [m]', 'absolute velocity [m/s]', 'attitude [deg]', 'body omega [rad/s]')
        for row, unit in enumerate(units):
            for column, axis in enumerate(axes[row]):
                axis.set_ylabel(f'{("roll", "pitch", "yaw")[column] if row == 2 else "xyz"[column]} {unit}')
                axis.grid(alpha=.25)
                axis.legend(fontsize=7)
                axis.set_xlim(0, reference['time'][-1])
                if row == 3:
                    axis.set_xlabel('post-state time [s]')
        phases = reference['phase']
        for index in range(1, len(phases)):
            if phases[index] != phases[index - 1]:
                for axis in axes.flat:
                    axis.axvline(reference['time'][index], color='gray', alpha=.2, lw=.7)
                axes[0, 0].text(reference['time'][index], 1.01, str(phases[index]),
                                transform=axes[0, 0].get_xaxis_transform(), fontsize=6)
        fig.suptitle(f'{case_name}: frozen policy comparison (partial traces end at termination)')
        fig.tight_layout()
        fig.savefig(target, dpi=140)
    finally:
        plt.close(fig)
    xy_target = _new_path(target.with_name(target.stem + '-xy.png'))
    fig, axis = plt.subplots(figsize=(8, 8))
    try:
        axis.plot(reference['reference'][:, 0], reference['reference'][:, 1], 'k:', label='reference', lw=2)
        for label, rows in rollouts.items():
            if rows:
                positions = np.vstack((rows[0]['position_before'], [r['position'] for r in rows]))
                axis.plot(positions[:, 0], positions[:, 1], label=label, **style(label, rows))
        axis.set(xlabel='x [m]', ylabel='y [m]', title=f'{case_name}: XY paths (partial traces end at termination)')
        axis.set_aspect('equal', adjustable='datalim')
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(xy_target, dpi=140)
    finally:
        plt.close(fig)
    return target


def save_interactive_motor_plot(path, rows, events):
    """Supplement the standard single-policy report with output-loss signals."""
    target = _new_path(path)
    plt = _pyplot()
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), sharex=True)
    try:
        t = np.array([r['time_post'] for r in rows])
        for i in range(4):
            axes[0,i].step([*[r['time_sec'] for r in rows], t[-1]],
                [*[r['motor_effectiveness'][i]*100 for r in rows], rows[-1]['motor_effectiveness'][i]*100],
                where='post',label='effectiveness')
            axes[0,i].set(title=f'Motor {i+1}',ylabel='effectiveness [%]',ylim=(-2,102))
            for key,label in (('motor_thrust_command','clipped command'),
                              ('motor_thrust_before_effectiveness','post-actuator nominal'),
                              ('motor_thrust_applied','applied force')):
                axes[1,i].plot(t,[r[key][i] for r in rows],label=label)
            axes[1,i].set(ylabel='thrust [N]',xlabel='post-state time [s]')
            for event in events:
                if event['event_type'] in ('degrade','restore'):
                    for ax in axes[:,i]:
                        ax.axvline(event['simulation_time'],color='gray',alpha=.3,linestyle=':')
            for ax in axes[:,i]:
                ax.grid(alpha=.25)
                ax.legend(fontsize=7)
        fig.suptitle('E2E interactive: rotor effectiveness and last-substep thrust; dotted lines = fault/restore events')
        fig.tight_layout()
        fig.savefig(target,dpi=140)
    finally:
        plt.close(fig)
    return target

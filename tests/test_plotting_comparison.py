from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from crazyflie_rl import plotting


def _rollout(*, label: str, offset: float = 0.0) -> dict[str, Any]:
    time_sec = np.array([0.0, 1.0, 2.0, 3.0])
    reference = np.column_stack(
        (time_sec / 3.0, np.sin(time_sec), np.ones(time_sec.size))
    )
    position = reference + np.array([offset, -offset, offset])
    return {
        "label": label,
        "time_sec": time_sec,
        "position": position,
        "reference_position": reference,
        "attitude_deg": np.column_stack(
            (time_sec, -2.0 * time_sec, 3.0 * time_sec)
        ),
        "linear_velocity": np.column_stack(
            (0.1 * time_sec, -0.2 * time_sec, 0.3 * time_sec)
        ),
        "angular_velocity": np.column_stack(
            (0.01 * time_sec, -0.02 * time_sec, 0.03 * time_sec)
        ),
        "position_error": np.linalg.norm(position - reference, axis=1),
        "control_input": np.column_stack(
            tuple((index + 1.0) * time_sec for index in range(4))
        ),
        "motor_thrust": np.column_stack(
            tuple(0.05 * (index + 1.0) + time_sec * 0.01 for index in range(4))
        ),
        "phases": ("GOTO", "CIRCLE", "CIRCLE", "HOLD"),
    }


class _FakeGrid:
    def __getitem__(self, item: Any) -> Any:
        return item


class _FakeAxis:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str):
        def record(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args, kwargs))

        return record


class _FakeFigure:
    def __init__(self, *, figsize: tuple[float, float]) -> None:
        self.figsize = figsize
        self.axes: list[_FakeAxis] = []
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def add_gridspec(self, *args: Any, **kwargs: Any) -> _FakeGrid:
        self.calls.append(("add_gridspec", args, kwargs))
        return _FakeGrid()

    def add_subplot(self, *args: Any, **kwargs: Any) -> _FakeAxis:
        self.calls.append(("add_subplot", args, kwargs))
        axis = _FakeAxis()
        self.axes.append(axis)
        return axis

    def suptitle(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("suptitle", args, kwargs))

    def text(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("text", args, kwargs))

    def tight_layout(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("tight_layout", args, kwargs))

    def savefig(self, path: Path, **kwargs: Any) -> None:
        self.calls.append(("savefig", (path,), kwargs))
        path.write_bytes(b"fake png")


class _FakePyplot:
    def __init__(self) -> None:
        self.figure_instance: _FakeFigure | None = None
        self.closed: list[_FakeFigure] = []

    def figure(self, *, figsize: tuple[float, float]) -> _FakeFigure:
        self.figure_instance = _FakeFigure(figsize=figsize)
        return self.figure_instance

    def close(self, figure: _FakeFigure) -> None:
        self.closed.append(figure)


def _call_names(axis: _FakeAxis) -> list[str]:
    return [name for name, _args, _kwargs in axis.calls]


def _plot_labels(axis: _FakeAxis) -> set[str]:
    return {
        kwargs["label"]
        for name, _args, kwargs in axis.calls
        if name == "plot" and "label" in kwargs
    }


def test_policy_plot_is_one_16_by_9_figure_with_seven_requested_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_plt = _FakePyplot()
    monkeypatch.setattr(plotting, "_pyplot", lambda: fake_plt)
    output = tmp_path / "c-small-floor.png"

    result = plotting.save_policy_trace(
        output,
        tag="c-small-T8-L2",
        rollout=_rollout(label="PID floor"),
        mission_name="circle",
        mission_parameters={"center_xy": (0.5, 0.0), "period_sec": 8.0, "laps": 2},
        motor_unit="N",
    )

    assert result == output
    assert output.read_bytes() == b"fake png"
    figure = fake_plt.figure_instance
    assert figure is not None
    assert figure.figsize == (16, 9)
    assert len(figure.axes) == 7
    assert fake_plt.closed == [figure]

    position, linear_velocity, thrust, attitude, angular_velocity, path, control = (
        figure.axes
    )
    assert {"x actual", "x ref", "y actual", "y ref", "z actual", "z ref"} <= (
        _plot_labels(position)
    )
    assert _plot_labels(linear_velocity) == {"vx", "vy", "vz"}
    assert {"M1", "M2", "M3", "M4", "total"} == _plot_labels(thrust)
    assert _plot_labels(attitude) == {"roll", "pitch", "yaw"}
    assert _plot_labels(angular_velocity) == {"wx", "wy", "wz"}
    assert {"reference", "actual"} <= _plot_labels(path)
    assert _plot_labels(control) == {"u_tau_x", "u_tau_y", "u_tau_z", "u_Fz"}
    assert any(name == "axis" and args == ("equal",) for name, args, _ in path.calls)
    assert any(
        name == "set_ylabel" and args == ("thrust [N]",)
        for name, args, _kwargs in thrust.calls
    )
    assert "annotate" in _call_names(position)
    for axis in (
        position,
        linear_velocity,
        thrust,
        attitude,
        angular_velocity,
        control,
    ):
        assert "axvline" in _call_names(axis)


def test_policy_plot_accepts_attribute_aliases() -> None:
    source = _rollout(label="PPO residual", offset=0.02)
    rollout = SimpleNamespace(
        label=source["label"],
        time=source["time_sec"],
        actual_position=source["position"],
        reference=source["reference_position"],
        velocity=source["linear_velocity"],
        attitude=source["attitude_deg"],
        omega=source["angular_velocity"],
        action=source["control_input"],
        motor_force=source["motor_thrust"],
        phase=source["phases"],
    )

    normalized = plotting._normalize_policy_rollout(rollout)

    assert normalized.label == "PPO residual"
    np.testing.assert_allclose(normalized.linear_velocity, source["linear_velocity"])
    np.testing.assert_allclose(normalized.angular_velocity, source["angular_velocity"])


def test_policy_plot_overlays_requested_thrust_only_when_bldc_signal_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _rollout(label="BLDC PPO")
    source["motor_thrust_command"] = source["motor_thrust"] + 0.02
    fake_plt = _FakePyplot()
    monkeypatch.setattr(plotting, "_pyplot", lambda: fake_plt)

    plotting.save_policy_trace(
        tmp_path / "bldc.png",
        tag="bldc",
        rollout=source,
    )

    figure = fake_plt.figure_instance
    assert figure is not None
    thrust_axis = figure.axes[2]
    assert {"M1 cmd", "M2 cmd", "M3 cmd", "M4 cmd"} <= _plot_labels(
        thrust_axis
    )
    assert any(
        name == "set_title" and "requested dotted" in args[0]
        for name, args, _kwargs in thrust_axis.calls
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("linear_velocity", np.zeros((4, 2)), "linear_velocity must have shape"),
        ("angular_velocity", np.zeros((4, 4)), "angular_velocity must have shape"),
        ("motor_thrust", np.full((4, 4), np.nan), "motor_thrust must contain finite"),
        ("phases", ("GOTO",), "phases must have one value per time sample"),
    ],
)
def test_policy_rollout_validation(
    field: str, value: Any, message: str
) -> None:
    rollout = {**_rollout(label="floor"), field: value}

    with pytest.raises(ValueError, match=message):
        plotting._normalize_policy_rollout(rollout)


def test_policy_plot_refuses_overwrite_before_importing_matplotlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "floor.png"
    output.write_bytes(b"keep")
    imported = False

    def fail_if_called() -> None:
        nonlocal imported
        imported = True
        raise AssertionError("matplotlib should not be loaded")

    monkeypatch.setattr(plotting, "_pyplot", fail_if_called)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        plotting.save_policy_trace(
            output,
            tag="hover",
            rollout=_rollout(label="floor"),
        )
    assert imported is False
    assert output.read_bytes() == b"keep"


def test_comparison_plot_is_one_16_by_9_figure_with_all_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_plt = _FakePyplot()
    monkeypatch.setattr(plotting, "_pyplot", lambda: fake_plt)
    output = tmp_path / "comparison.png"

    result = plotting.save_policy_comparison_trace(
        output,
        tag="Circle preset A",
        rollouts={
            "floor": _rollout(label="PID floor"),
            "ppo": _rollout(label="PPO residual", offset=0.02),
        },
        mission_name="circle",
        mission_parameters={"center_xy": (0.5, 0.0), "period_sec": 8.0, "laps": 2},
        motor_unit="N",
    )

    assert result == output
    assert output.read_bytes() == b"fake png"
    figure = fake_plt.figure_instance
    assert figure is not None
    assert figure.figsize == (16, 9)
    assert len(figure.axes) == 6
    assert fake_plt.closed == [figure]

    position, xy, error, attitude, control, motor = figure.axes
    assert any(
        call[0] == "axis" and call[1] == ("equal",) for call in xy.calls
    )
    assert "annotate" in _call_names(position)
    assert "axvline" in _call_names(error)
    assert "axvline" in _call_names(attitude)
    assert "axvline" in _call_names(control)
    assert "axvline" in _call_names(motor)

    control_labels = {
        kwargs["label"]
        for name, _args, kwargs in control.calls
        if name == "plot" and "label" in kwargs
    }
    motor_labels = {
        kwargs["label"]
        for name, _args, kwargs in motor.calls
        if name == "plot" and "label" in kwargs
    }
    assert {"PID floor u1", "PID floor u4", "PPO residual u1", "PPO residual u4"} <= control_labels
    assert {"PID floor M1", "PID floor M4", "PPO residual M1", "PPO residual M4"} <= motor_labels
    assert any(
        name == "set_ylabel" and args == ("motor thrust [N]",)
        for name, args, _kwargs in motor.calls
    )


def test_comparison_plot_accepts_attribute_objects_aliases_and_vector_error() -> None:
    floor = _rollout(label="floor")
    ppo = _rollout(label="ppo", offset=0.1)
    ppo_object = SimpleNamespace(
        label=ppo["label"],
        time=ppo["time_sec"],
        actual_position=ppo["position"],
        reference=ppo["reference_position"],
        attitude=ppo["attitude_deg"],
        error=ppo["position"] - ppo["reference_position"],
        action=ppo["control_input"],
        motor_force=ppo["motor_thrust"],
        phase=ppo["phases"],
    )

    normalized = plotting._normalize_comparison_rollouts(
        {"floor": floor, "ppo": ppo_object}
    )

    assert tuple(trace.label for trace in normalized) == ("floor", "ppo")
    np.testing.assert_allclose(
        normalized[1].position_error,
        np.linalg.norm(ppo_object.error, axis=1),
    )
    assert plotting._comparison_phase_boundaries(normalized) == (
        (1.0, "CIRCLE"),
        (3.0, "HOLD"),
    )


@pytest.mark.parametrize(
    ("rollouts", "message"),
    [
        ({"floor": _rollout(label="floor")}, "exactly two rollouts"),
        (
            {
                "floor": _rollout(label="floor"),
                "ppo": {**_rollout(label="ppo"), "control_input": np.zeros((4, 3))},
            },
            "control_input must have shape",
        ),
        (
            {
                "floor": _rollout(label="floor"),
                "ppo": {**_rollout(label="ppo"), "phases": ("GOTO",)},
            },
            "phases must have one value per time sample",
        ),
    ],
)
def test_comparison_rollout_validation(
    rollouts: dict[str, dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plotting._normalize_comparison_rollouts(rollouts)


def test_comparison_plot_refuses_overwrite_before_importing_matplotlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "already-there.png"
    output.write_bytes(b"keep")
    imported = False

    def fail_if_called() -> None:
        nonlocal imported
        imported = True
        raise AssertionError("matplotlib should not be loaded")

    monkeypatch.setattr(plotting, "_pyplot", fail_if_called)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        plotting.save_policy_comparison_trace(
            output,
            tag="circle",
            rollouts={
                "floor": _rollout(label="floor"),
                "ppo": _rollout(label="ppo"),
            },
        )
    assert imported is False
    assert output.read_bytes() == b"keep"

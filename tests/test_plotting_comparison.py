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

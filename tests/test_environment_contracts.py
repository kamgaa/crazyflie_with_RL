from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest


np = pytest.importorskip("numpy")
REAL_MUJOCO_RUNTIME = all(
    importlib.util.find_spec(name) is not None for name in ("gymnasium", "mujoco")
)

# The pure observation/control/reward regression tests do not need a simulator.
# Minimal import stubs let those tests run on review machines without MuJoCo;
# the real XML integration test below still requires and identifies the real stack.
if not REAL_MUJOCO_RUNTIME:
    fake_mujoco = types.ModuleType("mujoco")
    fake_mujoco.mj_step = lambda *_args: None
    fake_mujoco.mj_resetData = lambda *_args: None
    fake_mujoco.mj_forward = lambda *_args: None
    sys.modules.setdefault("mujoco", fake_mujoco)

    fake_gymnasium = types.ModuleType("gymnasium")
    fake_gymnasium.Env = object
    fake_spaces = types.ModuleType("gymnasium.spaces")
    fake_spaces.Box = object
    fake_gymnasium.spaces = fake_spaces
    sys.modules.setdefault("gymnasium", fake_gymnasium)
    sys.modules.setdefault("gymnasium.spaces", fake_spaces)

import crazyflie_residual_env as env_module
from crazyflie_residual_env import (
    CrazyflieResidualEnv,
    DEFAULT_RESIDUAL_SCALE,
    MASS,
    GRAV,
    TAU_MAX_RP,
)
from crazyflie_rl.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _observation_only_env(mode: str) -> CrazyflieResidualEnv:
    env = CrazyflieResidualEnv.__new__(CrazyflieResidualEnv)
    env.mode = mode
    env.pos_des = np.array([0.0, 0.0, 1.0])
    env.yaw_des = 0.0
    return env


def test_default_residual_scale_is_the_confirmed_four_vector() -> None:
    default = inspect.signature(CrazyflieResidualEnv.__init__).parameters[
        "residual_scale"
    ].default
    assert tuple(default) == (0.022, 0.022, 0.0001, 0.3)
    assert tuple(DEFAULT_RESIDUAL_SCALE) == tuple(default)


def test_observation_builders_share_the_first_thirteen_values() -> None:
    pos = np.array([0.2, -0.1, 1.4])
    vel = np.array([1.0, 2.0, 3.0])
    quat = np.array([np.cos(0.2), 0.0, 0.0, np.sin(0.2)])
    omega = np.array([0.4, 0.5, 0.6])

    residual_obs = _observation_only_env("residual")._obs(pos, quat, vel, omega)
    e2e_obs = _observation_only_env("e2e")._obs(pos, quat, vel, omega)

    assert residual_obs.shape == (13,)
    assert e2e_obs.shape == (15,)
    np.testing.assert_allclose(e2e_obs[:13], residual_obs)
    np.testing.assert_allclose(
        residual_obs,
        np.concatenate([pos - np.array([0.0, 0.0, 1.0]), vel, quat, omega]),
    )


@pytest.mark.parametrize(
    ("profile", "expected_shape"),
    [("residual_train.yaml", (13,)), ("e2e_train.yaml", (15,))],
)
def test_real_mujoco_reset_shape_when_resources_exist(
    profile: str,
    expected_shape: tuple[int],
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / profile)
    xml_path = config.resolve_path("mujoco_xml")
    if not REAL_MUJOCO_RUNTIME:
        pytest.skip("gymnasium and MuJoCo are not installed in this test runtime")
    if not xml_path.is_file():
        pytest.skip(f"real MuJoCo resource is not available: {xml_path}")

    environment = config.data["environment"]
    env = CrazyflieResidualEnv(
        str(xml_path),
        mode=config.control_mode,
        residual_scale=config.residual_scale,
        policy_hz=environment["policy_hz"],
        episode_sec=environment["episode_sec"],
        com_bias_randomize=environment["com_bias_randomize"],
        com_bias_mass=environment["com_bias_mass"],
        com_bias_offset=environment["com_bias_offset"],
        att_perturb_deg=environment["att_perturb_deg"],
        pos_perturb=environment["pos_perturb"],
    )
    try:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.shape == expected_shape
        assert observation.shape == expected_shape
    finally:
        env.close()


def _step_only_env(mode: str, state, captured_controls: list[np.ndarray]):
    env = _observation_only_env(mode)
    env.residual_scale = np.asarray(DEFAULT_RESIDUAL_SCALE)
    env.substeps = 1
    env._read_state = lambda: tuple(np.asarray(value).copy() for value in state)
    env.pid = lambda *_args: np.array([0.1, -0.2, 0.3, 0.4])
    env._apply_control = lambda value: captured_controls.append(np.asarray(value).copy())
    env._com_off3 = np.zeros(3)
    env._com_mw = 0.0
    env.dist_torque_body = np.zeros(3)
    env.drone_bid = 0
    env.data = SimpleNamespace(xfrc_applied=np.zeros((1, 6)))
    env.model = SimpleNamespace()
    env._prev_action = np.zeros(4)
    env.w_dact = 0.0
    env._step = 0
    env.max_steps = 100
    return env


@pytest.mark.parametrize("mode", ["residual", "e2e"])
def test_control_composition_reward_and_termination_regression(
    mode: str,
    monkeypatch,
) -> None:
    monkeypatch.setattr(env_module.mujoco, "mj_step", lambda *_args: None)
    state = (
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
    )
    captured: list[np.ndarray] = []
    env = _step_only_env(mode, state, captured)
    action = np.array([0.5, -0.25, 1.0, 0.1])

    observation, reward, terminated, truncated, _ = env.step(action)

    scaled = np.asarray(DEFAULT_RESIDUAL_SCALE) * action
    if mode == "residual":
        expected_control = np.array([0.1, -0.2, 0.3, 0.4]) + scaled
        expected_reward = -0.001 * float(action @ action)
        assert observation.shape == (13,)
    else:
        expected_control = scaled + np.array([0.0, 0.0, 0.0, MASS * GRAV])
        expected_reward = 0.0
        assert observation.shape == (15,)
    np.testing.assert_allclose(captured[0], expected_control)
    assert reward == pytest.approx(expected_reward)
    assert terminated is False
    assert truncated is False


def test_crash_penalty_and_altitude_termination_regression(monkeypatch) -> None:
    monkeypatch.setattr(env_module.mujoco, "mj_step", lambda *_args: None)
    state = (
        np.array([0.0, 0.0, 0.1]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
    )
    env = _step_only_env("residual", state, [])

    _observation, reward, terminated, truncated, _ = env.step(np.zeros(4))

    # Position cost is 3 * (0.1 - 1.0)^2, followed by the preserved -10 crash penalty.
    assert reward == pytest.approx(-(3.0 * 0.9**2) - 10.0)
    assert terminated is True
    assert truncated is False


def test_domain_randomization_sampling_regression(monkeypatch) -> None:
    monkeypatch.setattr(env_module.mujoco, "mj_resetData", lambda *_args: None)
    monkeypatch.setattr(env_module.mujoco, "mj_forward", lambda *_args: None)

    env = _observation_only_env("residual")
    env.model = SimpleNamespace()
    env.data = SimpleNamespace(
        xfrc_applied=np.zeros((1, 6)),
        qpos=np.zeros(7),
        qvel=np.zeros(6),
    )
    env.com_bias_randomize = True
    env.com_bias_mass = 0.0
    env.com_bias_offset = np.zeros(2)
    env.pos_perturb = 0.0
    env.att_perturb_deg = 0.0
    env.pid = SimpleNamespace(reset=lambda: None)
    env._prev_action = np.zeros(4)
    env._step = 99
    captured: dict[str, object] = {}

    def capture_bias(mass, offset) -> None:
        captured["mass"] = mass
        captured["offset"] = np.asarray(offset).copy()

    env._set_com_bias = capture_bias
    env._read_state = lambda: (
        env.data.qpos[0:3].copy(),
        env.data.qpos[3:7].copy(),
        np.zeros(3),
        np.zeros(3),
    )

    env.reset(seed=123)

    expected_rng = np.random.default_rng(123)
    theta = expected_rng.uniform(0.0, 2.0 * np.pi)
    radius = expected_rng.uniform(0.02, 0.10)
    torque = expected_rng.uniform(0.0, 0.5 * TAU_MAX_RP)
    expected_mass = min(torque / (radius * GRAV), 0.015)
    expected_offset = np.array(
        [radius * np.cos(theta), radius * np.sin(theta)]
    )

    assert captured["mass"] == pytest.approx(expected_mass)
    np.testing.assert_allclose(captured["offset"], expected_offset)
    assert np.linalg.norm(captured["offset"]) == pytest.approx(radius)
    assert 0.0 <= captured["mass"] <= 0.015
    assert torque <= 0.5 * TAU_MAX_RP

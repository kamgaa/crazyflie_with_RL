"""Actual MuJoCo tests, distinct from prescribed-force vs motor-lag rollouts."""

from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from crazyflie_rl.config import load_config
from crazyflie_rl.environment import CrazyflieResidualEnv
from crazyflie_rl.payload_physics import inertia_body, static_hover, subtree_properties
from crazyflie_rl.controllers import rotmat_from_quat_wxyz

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (0.0, (0.0, 0.0)),
    (0.01, (0.03, 0)),
    (0.01, (-0.03, 0)),
    (0.01, (0, 0.03)),
    (0.01, (0, -0.03)),
    (0.01, (0.03, 0.04)),
]


def make(mass=0.0, offset=(0.0, 0.0), mode="e2e", xml=None):
    return CrazyflieResidualEnv(
        config=load_config(ROOT / "configs/e2e_train.yaml"),
        mode=mode,
        xml_path=xml,
        com_bias_mass=mass,
        com_bias_offset=offset,
        pos_perturb=0,
        att_perturb_deg=0,
    )


def prescribed(env, total_thrust, tilted=False, external=None):
    env.data.qpos[:3] = [0, 0, 1]
    env.data.qpos[3:7] = (
        [np.cos(0.3), np.sin(0.3) / np.sqrt(2), np.sin(0.3) / np.sqrt(2), 0]
        if tilted
        else [1, 0, 0, 0]
    )
    env.data.qvel[:] = 0
    env.data.ctrl[:] = 0
    env.data.ctrl[env.act_force] = total_thrust / 4
    env.data.xfrc_applied[:] = 0
    if external is not None:
        env.data.xfrc_applied[env.drone_bid] = external
    mujoco.mj_forward(env.model, env.data)


@pytest.mark.parametrize("mode", ["residual", "e2e"])
@pytest.mark.parametrize("mass,offset", CASES)
def test_com_inertia_and_freefall(mode, mass, offset):
    env = make(mass, offset, mode)
    obs, info = env.reset(seed=42)
    r = np.r_[offset, 0.0]
    m0, m = env._m0, env._m0 + mass
    center = (m0 * env._ipos0 + mass * r) / m
    d = r - env._ipos0
    expected = env._inertia_body0 + m0 * mass / m * (
        (d @ d) * np.eye(3) - d[:, None] * d[None, :]
    )
    np.testing.assert_allclose(
        inertia_body(env.model, env.drone_bid), expected, atol=1e-18
    )
    np.testing.assert_allclose(
        env.data.xipos[env.drone_bid] - env.data.xpos[env.drone_bid], center, atol=1e-15
    )
    eig = np.linalg.eigvalsh(expected)
    assert eig[0] > 0 and eig[2] <= eig[0] + eig[1] + 1e-15
    assert obs.shape == (15,) and env.action_space.shape == (4,)
    assert info["payload"]["mass_kg"] == mass
    assert info["physics_model_version"] == "rigid_point_payload_v2"
    for tilt in (False, True):
        prescribed(env, 0, tilt)
        np.testing.assert_allclose(env.data.qacc[:3], env.model.opt.gravity, atol=1e-11)
        np.testing.assert_allclose(env.data.qacc[3:], 0, atol=1e-10)


@pytest.mark.parametrize("mass,offset", CASES)
@pytest.mark.parametrize("tilted", [False, True])
def test_prescribed_actual_thrust_bias_scales_with_thrust(mass, offset, tilted):
    env = make(mass, offset)
    env.reset(seed=42)
    total, center, _, _ = subtree_properties(env)
    for fraction in (0, 0.5, 1, 1.2):
        thrust = fraction * total * 9.81
        prescribed(env, thrust, tilted)
        snapshot = env.physics_wrench_snapshot()
        expected = np.cross(-center, [0, 0, thrust])
        np.testing.assert_allclose(
            snapshot["motor_wrench_vehicle_com_body"][:3], expected, atol=1e-15
        )
        np.testing.assert_allclose(
            snapshot["external_applied_wrench_vehicle_com_body"], 0, atol=1e-15
        )
        # Remove translation/child freedoms from the actual articulated mass matrix.
        full = np.zeros((env.model.nv, env.model.nv))
        mujoco.mj_fullM(env.model, env.data, full)
        keep = [3, 4, 5]
        other = [0, 1, 2, *range(6, env.model.nv)]
        effective = full[np.ix_(keep, keep)] - full[
            np.ix_(keep, other)
        ] @ np.linalg.solve(full[np.ix_(other, other)], full[np.ix_(other, keep)])
        np.testing.assert_allclose(effective @ env.data.qacc[3:6], expected, atol=2e-14)


def test_reset_restores_original_and_preserves_children_rng_and_allocator():
    env = make()
    env.reset(seed=42)
    original = [
        env.model.body_mass.copy(),
        env.model.body_ipos.copy(),
        env.model.body_inertia.copy(),
        env.model.body_iquat.copy(),
    ]
    B = env.B.copy()
    sites = env.model.site_pos.copy()
    for mass, offset in [(0.01, (0.03, 0.04)), (0.02, (-0.02, 0.01)), (0, (0, 0))]:
        env.com_bias_mass, env.com_bias_offset = mass, np.array(offset)
        env.reset(seed=42)
        for old, new in zip(
            original,
            [
                env.model.body_mass,
                env.model.body_ipos,
                env.model.body_inertia,
                env.model.body_iquat,
            ],
        ):
            np.testing.assert_array_equal(old[2:], new[2:])
        np.testing.assert_array_equal(env.B, B)
        np.testing.assert_array_equal(env.model.site_pos, sites)
    for old, new in zip(
        original,
        [
            env.model.body_mass,
            env.model.body_ipos,
            env.model.body_inertia,
            env.model.body_iquat,
        ],
    ):
        np.testing.assert_array_equal(old, new)
    env.com_bias_randomize = True
    a, ia = env.reset(seed=42)
    env.step(np.zeros(4))
    b, ib = env.reset(seed=42)
    np.testing.assert_array_equal(a, b)
    assert ia == ib
    assert ia["payload"]["mass_kg"] == pytest.approx(0.010512206395535115)
    expected = env.model.body_mass.copy()
    for _ in range(10):
        env.step(np.zeros(4))
    np.testing.assert_array_equal(expected, env.model.body_mass)


def custom_xml(tmp_path, *, full=None):
    tree = ET.parse(ROOT / "resources/mujoco/cf21B_500.xml")
    tree.getroot().find("compiler").set(
        "meshdir", str(ROOT / "resources/mujoco/assets")
    )
    inertial = tree.find(".//body[@name='drone']/inertial")
    inertial.set("pos", ".005 -.003 .002")
    inertial.set("quat", ".9238795325112867 0 0 .3826834323650898")
    if full is not None:
        mass, center, tensor = full
        inertial.attrib.clear()
        inertial.set("mass", str(mass))
        inertial.set("pos", " ".join(map(str, center)))
        inertial.set(
            "fullinertia",
            " ".join(
                map(
                    str,
                    [
                        tensor[0, 0],
                        tensor[1, 1],
                        tensor[2, 2],
                        tensor[0, 1],
                        tensor[0, 2],
                        tensor[1, 2],
                    ],
                )
            ),
        )
    else:
        inertial.set("diaginertia", "2e-5 3e-5 4e-5")
    path = tmp_path / ("reference.xml" if full is not None else "original.xml")
    tree.write(path)
    return str(path)


def test_rotated_offset_original_against_independent_explicit_reference(tmp_path):
    env = make(0.01, (0.03, 0.04), xml=custom_xml(tmp_path))
    env.reset(seed=3)
    # Independent body-axis tensor and two shifted components, no production compose helper.
    angle = np.pi / 4
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    c0 = np.array([0.005, -0.003, 0.002])
    rp = np.array([0.03, 0.04, 0])
    mass = 0.04338 + 0.01
    center = (0.04338 * c0 + 0.01 * rp) / mass
    tensor = R @ np.diag([2e-5, 3e-5, 4e-5]) @ R.T
    for m, point in [(0.04338, c0), (0.01, rp)]:
        d = point - center
        tensor += m * ((d @ d) * np.eye(3) - np.outer(d, d))
    np.testing.assert_allclose(
        inertia_body(env.model, env.drone_bid), tensor, atol=1e-18
    )
    reference = make(xml=custom_xml(tmp_path, full=(mass, center, tensor)))
    reference.reset(seed=3)
    for tilted in (False, True):
        for thrust in (0, 0.3, 0.6, 0.7):
            for e in (env, reference):
                prescribed(e, thrust, tilted)
            # MuJoCo's XML fullinertia diagonalization has finite eigensolver
            # tolerance; compare against that independently compiled model.
            np.testing.assert_allclose(
                env.data.qacc, reference.data.qacc, rtol=2e-6, atol=5e-5
            )
            matrices = []
            for e in (env, reference):
                matrix = np.zeros((e.model.nv, e.model.nv))
                mujoco.mj_fullM(e.model, e.data, matrix)
                matrices.append(matrix)
            np.testing.assert_allclose(*matrices, atol=1e-11)
    env.com_bias_mass = 0
    env.reset(seed=3)
    np.testing.assert_array_equal(env.model.body_iquat[env.drone_bid], env._iquat0)
    np.testing.assert_array_equal(env.model.body_inertia[env.drone_bid], env._J0)


def test_tilted_external_wrench_is_separate():
    env = make(0.01, (0.03, 0.04))
    env.reset(seed=42)
    external = np.array([0.01, -0.02, 0.03, 0.001, 0.002, -0.003])
    prescribed(env, 0.5, True, external)
    s = env.physics_wrench_snapshot()
    assert s["nominal_allocator_wrench_order"] == "torque_xyz_collective_thrust"
    assert len(s["nominal_allocator_actual_wrench"]) == 4
    R = env.data.xmat[env.drone_bid].reshape(3, 3)
    center = np.asarray(s["vehicle_com_body_m"])
    c = env.model.body_ipos[env.drone_bid]
    expected = np.r_[
        R.T @ external[3:] + np.cross(c - center, R.T @ external[:3]),
        R.T @ external[:3],
    ]
    np.testing.assert_allclose(
        s["external_applied_wrench_vehicle_com_body"], expected, atol=1e-16
    )
    np.testing.assert_allclose(
        s["motor_wrench_vehicle_com_body"][:3],
        np.cross(-center, [0, 0, 0.5]),
        atol=1e-16,
    )
    np.testing.assert_allclose(
        s["motor_wrench_vehicle_com_world"][:3],
        R @ np.cross(-center, [0, 0, 0.5]),
        atol=1e-16,
    )


def test_hover_physical_and_policy_reachability_are_distinct():
    env = make(0.03, (0.03, 0))
    _, info = env.reset(seed=1)
    h = info["static_hover"]
    assert h["physical_status"] == "infeasible"
    assert h["policy_allocator_reachable"] is False
    assert max(h["unbounded_equilibrium_motor_thrust_n"]) > 0.2
    env = make(0.01, (0.03, 0))
    env.reset(seed=1)
    h = static_hover(env)
    assert h["physically_feasible"] and h["policy_allocator_reachable"]
    env.residual_scale[:] = 1e-8
    h = static_hover(env)
    assert h["physically_feasible"] and not h["policy_allocator_reachable"]


@pytest.mark.parametrize("mode", ["residual", "e2e"])
def test_real_rollout_preserves_motor_lag_and_nominal_controller(mode):
    env = make(0.01, (0.03, 0), mode)
    _, info = env.reset(seed=42)
    assert sum(info["actuator_initial_state"]["actual_thrust_n"]) == pytest.approx(
        (0.04338 + 0.01) * 9.81
    )
    assert env.pid.mass == env.mass == 0.04338
    env.substeps = 1
    env.step(np.zeros(4))
    assert sum(env._last_f_cmd) == pytest.approx(0.04338 * 9.81)
    assert sum(env._last_f) == pytest.approx(0.5197109959871)
    np.testing.assert_array_equal(env.data.xfrc_applied, 0)
    assert env.data.qacc[4] > 0


@pytest.mark.parametrize("mode", ["residual", "e2e"])
def test_payload_free_rollout_against_untouched_xml(mode):
    env = make(mode=mode)
    env.reset(seed=42)
    model = mujoco.MjModel.from_xml_path(env.xml_path)
    data = mujoco.MjData(model)
    data.qpos[:] = env.data.qpos
    data.qvel[:] = env.data.qvel
    mujoco.mj_forward(model, data)

    def compare(**kwargs):
        data.ctrl[:] = env.data.ctrl
        mujoco.mj_step(model, data)
        np.testing.assert_allclose(env.data.qpos, data.qpos, atol=2e-14)
        np.testing.assert_allclose(env.data.qvel, data.qvel, atol=2e-13)

    env.set_physics_substep_observer(compare)
    for _ in range(10):
        env.step(np.array([0.005, -0.003, 0.01, 0.01]))


def test_fixed_payload_evaluation_and_viewer_record_actual_reset():
    from crazyflie_rl.evaluation import PolicyEvaluator
    from crazyflie_rl.eval_cli import EvaluationRunner, trace_metrics

    config = load_config(ROOT / "configs/residual_hover_eval.yaml")
    config = replace(
        config,
        environment=replace(config.environment, episode_sec=0.02),
        evaluation=replace(config.evaluation, episode_count=2),
        mission=replace(
            config.mission,
            force_floor_start=False,
            hover=replace(config.mission.hover, duration=0.02),
        ),
    )
    result = PolicyEvaluator(config).evaluate(None).as_metrics()
    assert len(result["episodes"]) == 2
    assert all(e["payload"]["mass_kg"] == 0.03 for e in result["episodes"])
    assert all(not e["static_hover"]["physically_feasible"] for e in result["episodes"])
    runner = EvaluationRunner(
        config, None, headless=True, realtime=False, camera_tracking=False
    )
    a = runner.run(None, "floor", "audit")
    b = runner.run(None, "floor", "audit")
    np.testing.assert_array_equal(a.position, b.position)
    metrics = trace_metrics(a, 0.3)
    assert metrics["reset_info"]["payload"]["attachment_body_m"] == [0.03, 0.0, 0.0]
    assert len(metrics["physics_wrenches"]) == 2
    assert "rollout_initial_actuator_state" in metrics["reset_info"]


def test_version_config_manifest_and_cross_physics_provenance(tmp_path, monkeypatch):
    import json
    import yaml
    from crazyflie_rl import artifacts
    from crazyflie_rl.eval_cli import _model_provenance
    from crazyflie_rl.evaluation import PolicyEvaluator
    from crazyflie_rl.physics_version import PHYSICS_MODEL_VERSION, physics_comparison

    monkeypatch.setattr(artifacts, "_git_metadata", lambda _: {})
    config = load_config(ROOT / "configs/e2e_train.yaml")
    config = replace(config, paths=replace(config.paths, artifact_root=tmp_path))
    run = artifacts.ArtifactManager.create(config, command=["payload-unit-test"])
    assert run.manifest["physics_model_version"] == PHYSICS_MODEL_VERSION
    resolved = yaml.safe_load(
        (run.run_dir / run.manifest["resolved_config_path"]).read_text()
    )
    assert resolved["physics_model_version"] == PHYSICS_MODEL_VERSION
    # A synthetic old manifest, no real checkpoint is loaded or overwritten.
    path = run.run_dir / "models" / "legacy-test.zip"
    path.write_bytes(b"test fixture, not PPO")
    old = {
        "models": {"final": {"path": "models/legacy-test.zip"}},
        "resolved_config": {},
    }
    run.manifest_path.write_text(json.dumps(old))
    original = run.manifest_path.read_bytes()
    provenance = _model_provenance(path)
    assert (
        physics_comparison(provenance["physics_model_version"])[
            "cross_physics_evaluation"
        ]
        is True
    )
    assert run.manifest_path.read_bytes() == original
    assert physics_comparison(None)["cross_physics_evaluation"] is None

    # Avoid a full default-duration rollout while checking no-policy provenance.
    short = replace(
        config,
        environment=replace(config.environment, episode_sec=0.01),
        evaluation=replace(config.evaluation, episode_count=1),
    )
    assert (
        PolicyEvaluator(short)
        .evaluate(None)
        .as_metrics()["physics_provenance"]["physics_comparison_status"]
        == "same_physics"
    )


def test_polynomial_hover_search_does_not_claim_false_certificate():
    config = load_config(ROOT / "configs/cf21b_actuator_torque_poly_eval.yaml")
    env = CrazyflieResidualEnv(
        config=config,
        com_bias_mass=0,
        com_bias_offset=(0, 0),
        pos_perturb=0,
        att_perturb_deg=0,
    )
    _, info = env.reset(seed=1)
    assert info["static_hover"]["reaction_torque_model"] == "paper_polynomial"
    assert info["static_hover"]["physical_status"] == "feasible"
    np.testing.assert_allclose(
        info["static_hover"]["equilibrium_residual_torque_force"], 0, atol=1e-9
    )

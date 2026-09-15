from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.factories import EnvironmentFactory
from crazyflie_rl.environment import legacy_reward_breakdown
from crazyflie_rl.observation import (
    AuxiliaryObservationState,
    LEGACY_STATE_READER,
    observation_dimension,
)
from crazyflie_rl.recovery import RecoveryCase, set_recovery_case


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PROFILE = ROOT / "configs" / "e2e_train_payload_dr_pos6_vel05.yaml"
PROFILE = ROOT / "configs" / "e2e_train_payload_dr_history_integral_v1.yaml"


def _mask(observation: np.ndarray, slot: int) -> float:
    return float(observation[15 + slot * 20 + 19])


def test_configured_dimensions_and_schema() -> None:
    config = load_config(PROFILE)
    settings = config.environment.observation
    assert settings is not None
    assert config.observation_shape == (418,)
    assert observation_dimension(settings) == 418
    assert (
        observation_dimension(
            replace(
                settings,
                position_error_integral=replace(
                    settings.position_error_integral, enabled=False
                ),
            )
        )
        == 415
    )
    assert (
        observation_dimension(
            replace(settings, history=replace(settings.history, enabled=False))
        )
        == 18
    )
    assert (
        observation_dimension(
            replace(
                settings,
                history=replace(settings.history, enabled=False),
                position_error_integral=replace(
                    settings.position_error_integral, enabled=False
                ),
            )
        )
        == 15
    )
    assert config.observation_schema["history"]["order"] == "newest_to_oldest"
    assert config.observation_schema["dimension"] == 418


def test_new_viewer_profiles_keep_fixed_nominal_and_exact_30mm_radius() -> None:
    nominal = load_config(
        ROOT / "configs" / "view_payload_history_integral_v1_nominal.yaml"
    )
    assert nominal.observation_shape == (418,)
    assert nominal.environment.payload.mass == 0.0
    assert nominal.environment.payload.offset == (0.0, 0.0)
    assert nominal.environment.payload.curriculum.enabled is False
    assert nominal.environment.initial_state_randomization.enabled is False
    for azimuth in range(0, 360, 45):
        profile = ROOT / "configs" / (
            f"view_payload_history_integral_v1_az{azimuth:03d}.yaml"
        )
        config = load_config(profile)
        assert config.observation_shape == (418,)
        assert config.environment.payload.mass == 0.010
        assert np.linalg.norm(config.environment.payload.offset) == pytest.approx(0.030)
        assert config.environment.payload.curriculum.enabled is False
        assert config.environment.initial_state_randomization.enabled is False


def test_legacy_reward_change_affects_only_tilt_and_rate_contributions() -> None:
    donor = load_config(LEGACY_PROFILE).environment.reward
    improved = load_config(PROFILE).environment.reward
    assert (donor.position_weight, improved.position_weight) == (6.0, 6.0)
    assert (donor.velocity_weight, improved.velocity_weight) == (0.5, 0.5)
    assert (donor.tilt_weight, improved.tilt_weight) == (1.0, 2.0)
    assert (donor.angular_velocity_weight, improved.angular_velocity_weight) == (
        0.01,
        0.05,
    )

    state = {
        "position_error": [0.10, -0.20, 0.05],
        "velocity": [0.30, 0.20, -0.10],
        "tilt_error": 0.12,
        "omega_body": [1.0, -2.0, 0.5],
        "yaw_error": 0.3,
        "action": [0.2, -0.1, 0.4, 0.0],
        "action_delta": [0.1, 0.2, -0.2, 0.3],
    }

    def weights(reward):
        return {
            "position": reward.position_weight,
            "linear_velocity": reward.velocity_weight,
            "tilt": reward.tilt_weight,
            "angular_velocity": reward.angular_velocity_weight,
            "yaw": reward.yaw_weight,
            # E2E legacy reward disables action/action-rate contributions.
            "action": 0.0,
            "action_rate": 0.0,
        }

    before = legacy_reward_breakdown(**state, weights=weights(donor))
    after = legacy_reward_breakdown(**state, weights=weights(improved))
    assert before["raw_costs"] == after["raw_costs"]
    for name in ("position", "linear_velocity", "yaw", "action", "action_rate"):
        assert before["weighted_costs"][name] == after["weighted_costs"][name]
    assert after["weighted_costs"]["tilt"] == pytest.approx(
        2.0 * before["weighted_costs"]["tilt"]
    )
    assert after["weighted_costs"]["angular_velocity"] == pytest.approx(
        5.0 * before["weighted_costs"]["angular_velocity"]
    )


def test_history_padding_order_action_alignment_and_pure_reads() -> None:
    settings = load_config(PROFILE).environment.observation
    assert settings is not None
    state = AuxiliaryObservationState(settings, policy_dt=0.01)
    initial = np.arange(15, dtype=float)
    initial[6:10] = (1.0, 0.0, 0.0, 0.0)
    state.reset(initial)

    reset_observation = state.compose(initial)
    assert reset_observation.shape == (418,)
    assert all(_mask(reset_observation, slot) == 0.0 for slot in range(20))
    np.testing.assert_array_equal(reset_observation[15:30], initial)

    action0 = np.array([1.0, -1.0, 0.25, 0.5])
    state.advance(
        base_observation_before=initial,
        applied_action=action0,
        reference_minus_position_before=np.array([0.1, -0.2, 0.3]),
    )
    current = initial + 100.0
    composed = state.compose(current)
    np.testing.assert_array_equal(composed[15:30], initial)
    np.testing.assert_array_equal(composed[30:34], action0)
    assert _mask(composed, 0) == 1.0
    assert _mask(composed, 1) == 0.0
    before = state.state_snapshot()
    np.testing.assert_array_equal(state.compose(current), state.compose(current))
    after = state.state_snapshot()
    assert before["transition_count"] == after["transition_count"]
    np.testing.assert_array_equal(before["integral"], after["integral"])
    np.testing.assert_array_equal(
        before["history"][0]["observation"],
        after["history"][0]["observation"],
    )

    action1 = np.array([-0.5, 0.0, 0.75, -1.0])
    state.advance(
        base_observation_before=current,
        applied_action=action1,
        reference_minus_position_before=np.zeros(3),
    )
    newest_first = state.compose(current + 1.0)
    np.testing.assert_array_equal(newest_first[15:30], current)
    np.testing.assert_array_equal(newest_first[30:34], action1)
    np.testing.assert_array_equal(newest_first[35:50], initial)
    np.testing.assert_array_equal(newest_first[50:54], action0)


def test_integral_sign_dt_clamp_and_reset() -> None:
    settings = load_config(PROFILE).environment.observation
    assert settings is not None
    settings = replace(
        settings,
        position_error_integral=replace(
            settings.position_error_integral, clamp_m_s=(0.01, 0.01, 0.01)
        ),
    )
    state = AuxiliaryObservationState(settings, policy_dt=0.01)
    base = np.zeros(15)
    base[6] = 1.0
    state.reset(base)
    state.advance(
        base_observation_before=base,
        applied_action=np.zeros(4),
        reference_minus_position_before=np.array([1.0, -2.0, 0.5]),
    )
    snapshot = state.state_snapshot()
    np.testing.assert_allclose(snapshot["integral"], [0.01, -0.01, 0.005])
    assert state.diagnostics()["integral_clamp_transition_count_xyz"] == [0, 1, 0]
    np.testing.assert_allclose(state.compose(base)[-3:], [1.0, -1.0, 0.5])
    state.reset(base)
    assert state.diagnostics()["transition_count"] == 0
    np.testing.assert_array_equal(state.state_snapshot()["integral"], np.zeros(3))


def test_current_body_rate_frame_and_read_side_effects() -> None:
    config = load_config(PROFILE)
    environment = EnvironmentFactory(config).make(
        seed=1000,
        payload_curriculum_enabled=False,
        initial_state_randomization_enabled=False,
        com_bias_mass=0.010,
        com_bias_offset=(0.030, 0.020),
        pos_perturb=0.0,
        att_perturb_deg=0.0,
    )
    try:
        import mujoco

        environment.reset(seed=1000)
        qpos = environment._freejoint_qpos_address
        dof = environment._freejoint_dof_address
        angle = np.deg2rad(31.0)
        environment.data.qpos[qpos + 3 : qpos + 7] = (
            np.cos(angle / 2.0),
            np.sin(angle / 2.0),
            0.0,
            0.0,
        )
        expected_rate = np.array([0.7, -1.1, 0.35])
        environment.data.qvel[dof + 3 : dof + 6] = expected_rate
        mujoco.mj_forward(environment.model, environment.data)
        assert not np.allclose(
            environment.model.body_iquat[environment.drone_bid],
            [1.0, 0.0, 0.0, 0.0],
        )
        sensor_address = int(environment.model.sensor_adr[environment.gyro_sid])
        np.testing.assert_allclose(
            environment.data.sensordata[sensor_address : sensor_address + 3],
            expected_rate,
            rtol=0.0,
            atol=1e-12,
        )
        snapshot = {
            "time": float(environment.data.time),
            "qpos": environment.data.qpos.copy(),
            "qvel": environment.data.qvel.copy(),
            "ctrl": environment.data.ctrl.copy(),
            "omega": environment._last_omega.copy(),
        }
        np.testing.assert_allclose(environment._read_state()[3], expected_rate)
        environment.reset_auxiliary_observation_state()
        first = environment.current_observation()
        second = environment.current_observation()
        np.testing.assert_array_equal(first, second)
        assert float(environment.data.time) == snapshot["time"]
        for field in ("qpos", "qvel", "ctrl"):
            np.testing.assert_array_equal(
                getattr(environment.data, field), snapshot[field]
            )
        np.testing.assert_array_equal(environment._last_omega, snapshot["omega"])
        assert (
            environment._auxiliary_observation.state_snapshot()["transition_count"] == 0
        )
    finally:
        environment.close()


def test_environment_advances_auxiliary_once_with_clipped_applied_action() -> None:
    config = load_config(PROFILE)
    environment = EnvironmentFactory(config).make(
        seed=1000,
        payload_curriculum_enabled=False,
        initial_state_randomization_enabled=False,
        pos_perturb=0.0,
        att_perturb_deg=0.0,
    )
    try:
        import mujoco

        environment.reset(seed=1000)
        qpos = environment._freejoint_qpos_address
        environment.data.qpos[qpos : qpos + 3] = environment.pos_des + np.array(
            [0.1, -0.2, 0.3]
        )
        environment.data.qpos[qpos + 3 : qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        environment.data.qvel[:] = 0.0
        mujoco.mj_forward(environment.model, environment.data)
        before = environment.reset_auxiliary_observation_state()[:15].copy()
        _obs, _reward, _terminated, _truncated, info = environment.step(
            np.array([2.0, -2.0, 0.5, 0.0], dtype=np.float32)
        )
        state = environment._auxiliary_observation.state_snapshot()
        assert state["transition_count"] == 1
        assert len(state["history"]) == 1
        np.testing.assert_array_equal(state["history"][0]["observation"], before)
        np.testing.assert_array_equal(
            state["history"][0]["action"], [1.0, -1.0, 0.5, 0.0]
        )
        np.testing.assert_allclose(state["integral"], [-0.001, 0.002, -0.003])
        assert info["observation_diagnostics"]["history_valid_count"] == 1
        assert info["transition_timing"]["physics_substeps"] == 5
        assert info["transition_timing"]["policy_dt_s"] == pytest.approx(0.01)
        assert abs(
            info["legacy_reward_terms"]["weighted_cost_sum_minus_legacy_cost"]
        ) < 1e-7
    finally:
        environment.close()


def test_recovery_install_resets_episode_memory_after_final_pose() -> None:
    environment = EnvironmentFactory(load_config(PROFILE)).make(
        seed=1000,
        payload_curriculum_enabled=False,
        initial_state_randomization_enabled=False,
    )
    try:
        environment.reset(seed=1000)
        environment.step(np.zeros(4))
        case = RecoveryCase(
            category="test",
            name="roll_plus_20",
            tilt_deg=20.0,
            tilt_axis_xyz=(1.0, 0.0, 0.0),
            position_offset_xyz_m=(0.0, 0.0, 0.0),
        )
        observation = set_recovery_case(environment, case, seed=1000)
        diagnostics = environment._auxiliary_observation.diagnostics()
        assert diagnostics["history_valid_count"] == 0
        assert diagnostics["transition_count"] == 0
        assert all(_mask(observation, slot) == 0.0 for slot in range(20))
        np.testing.assert_allclose(observation[15:30], observation[:15])
        np.testing.assert_array_equal(observation[-3:], np.zeros(3))
    finally:
        environment.close()


def test_feature_off_legacy_reader_preserves_reset_rng_and_observation() -> None:
    legacy = load_config(LEGACY_PROFILE)
    enhanced = load_config(PROFILE)
    settings = enhanced.environment.observation
    assert settings is not None
    disabled = replace(
        settings,
        state_reader=LEGACY_STATE_READER,
        history=replace(settings.history, enabled=False, length_steps=0),
        position_error_integral=replace(
            settings.position_error_integral, enabled=False
        ),
    )
    compatible = replace(
        enhanced,
        environment=replace(
            legacy.environment,
            observation=disabled,
        ),
    )
    left = EnvironmentFactory(legacy).make(seed=123)
    right = EnvironmentFactory(compatible).make(seed=123)
    try:
        for _ in range(3):
            left_observation, _ = left.reset()
            right_observation, _ = right.reset()
            np.testing.assert_array_equal(left_observation, right_observation)
            left_payload = left.payload_snapshot()
            right_payload = right.payload_snapshot()
            assert left_payload["mass_kg"] == right_payload["mass_kg"]
            assert (
                left_payload["attachment_body_m"] == right_payload["attachment_body_m"]
            )
    finally:
        left.close()
        right.close()


def test_dummy_vec_auto_reset_keeps_terminal_history_out_of_next_episode() -> None:
    sb3 = pytest.importorskip("stable_baselines3.common.vec_env")
    config = load_config(PROFILE)
    short = replace(
        config,
        environment=replace(config.environment, episode_sec=0.01),
    )
    vector = sb3.DummyVecEnv(
        [
            lambda: EnvironmentFactory(short).make(
                seed=1000,
                payload_curriculum_enabled=False,
                initial_state_randomization_enabled=False,
                pos_perturb=0.0,
                att_perturb_deg=0.0,
            )
        ]
    )
    try:
        reset_observation = vector.reset()[0]
        assert all(_mask(reset_observation, slot) == 0.0 for slot in range(20))
        next_observation, _reward, done, infos = vector.step(
            np.zeros((1, 4), dtype=np.float32)
        )
        assert bool(done[0])
        terminal = infos[0]["terminal_observation"]
        assert _mask(terminal, 0) == 1.0
        assert all(_mask(next_observation[0], slot) == 0.0 for slot in range(20))
        np.testing.assert_array_equal(next_observation[0, -3:], np.zeros(3))
    finally:
        vector.close()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import crazyflie_rl.environment as environment_module
from crazyflie_rl.config import load_config
from crazyflie_rl.factories import EnvironmentFactory
from crazyflie_rl.payload_evaluation import (
    BEST_PAYLOAD_RANKING,
    best_payload_ranking_key,
    fixed_payload_evaluation_cases,
    summarize_fixed_payload_suite,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "e2e_train_payload_dr_v1.yaml"
DONOR_MANIFEST = ROOT / (
    "artifacts/runs/ppo_e2e_hover_legacy-recovery-finetune-v2_seed42_20260907-145105/"
    "manifests/ppo_e2e_hover_legacy-recovery-finetune-v2_seed42_manifest_"
    "20260907-145105.json"
)


def test_payload_profile_preserves_donor_training_and_control_contracts() -> None:
    config = load_config(PROFILE)
    donor = json.loads(DONOR_MANIFEST.read_text(encoding="utf-8"))["resolved_config"]
    resolved = config.resolved_dict()

    assert config.physics_model_version == "rigid_point_payload_v2"
    assert resolved["vehicle"] == donor["vehicle"]
    assert resolved["actuator"] == donor["actuator"]
    assert resolved["controller"] == donor["controller"]
    assert resolved["training"] == donor["training"]
    assert resolved["environment"]["reward"] == donor["environment"]["reward"]
    assert resolved["environment"]["initial_state_randomization"] == donor["environment"]["initial_state_randomization"]
    for key in (
        "control_mode", "policy_hz", "episode_sec", "residual_scale",
        "position_target", "yaw_target", "position_perturbation",
        "attitude_perturbation_deg", "termination",
    ):
        assert resolved["environment"][key] == donor["environment"][key]
    assert config.observation_shape == (15,)
    assert config.action_shape == (4,)
    assert config.training.policy_initialization_required


def test_payload_profile_requires_explicit_donor_without_discovery() -> None:
    import train_ppo_02

    with pytest.raises(SystemExit, match="checkpoint reselection are intentionally disabled"):
        train_ppo_02.main(["--config", str(PROFILE), "--total-timesteps", "2048"])


def test_payload_curriculum_stage_boundaries_and_si_contract() -> None:
    from crazyflie_rl.environment import payload_curriculum_stage

    settings = load_config(PROFILE).environment.payload.curriculum
    expected = {
        0: "centered_0_to_5g",
        199_999: "centered_0_to_5g",
        200_000: "center20_or_radius_0_to_10mm",
        499_999: "center20_or_radius_0_to_10mm",
        500_000: "center20_or_radius_0_to_30mm",
        1_000_000: "center20_or_radius_0_to_30mm",
    }
    assert {
        step: payload_curriculum_stage(settings, step).name for step in expected
    } == expected
    assert settings.payload_free_probability == pytest.approx(0.30)
    assert settings.attachment_z_m == 0.0
    assert settings.mass_radius_independent
    assert settings.sample_at_reset and settings.fixed_within_episode
    assert settings.feasibility_filter.minimum_lower_thrust_margin_n == pytest.approx(0.02)
    assert settings.feasibility_filter.minimum_upper_thrust_margin_n == pytest.approx(0.02)
    assert settings.feasibility_filter.max_resample_attempts == 32
    assert settings.feasibility_filter.exhausted_action == "raise"


@pytest.mark.parametrize("step", [0, 199_999, 200_000, 499_999, 500_000])
def test_payload_rng_does_not_change_initial_pose_stream(step: int) -> None:
    config = load_config(PROFILE)
    factory = EnvironmentFactory(config)
    for seed in (7, 42, 1000):
        enabled = factory.make(
            seed=seed, payload_curriculum_enabled=True, track_curriculum_steps=True
        )
        disabled = factory.make(
            seed=seed, payload_curriculum_enabled=False, track_curriculum_steps=True
        )
        try:
            enabled.set_curriculum_global_step(step)
            disabled.set_curriculum_global_step(step)
            _, enabled_info = enabled.reset(seed=seed)
            _, disabled_info = disabled.reset(seed=seed)
            np.testing.assert_array_equal(enabled.data.qpos[:7], disabled.data.qpos[:7])
            np.testing.assert_array_equal(enabled.data.qvel, np.zeros_like(enabled.data.qvel))
            assert enabled_info["initial_position_offset_xyz_m"] == disabled_info["initial_position_offset_xyz_m"]
            assert enabled_info["initial_tilt_deg"] == disabled_info["initial_tilt_deg"]
        finally:
            enabled.close()
            disabled.close()


def test_payload_mixture_ranges_filter_and_episode_fixity() -> None:
    config = load_config(PROFILE)
    environment = EnvironmentFactory(config).make(
        seed=1000, track_curriculum_steps=True
    )
    environment.set_curriculum_global_step(500_000)
    try:
        payloads = []
        for index in range(1000):
            _, info = environment.reset(seed=1000 if index == 0 else None)
            curriculum = info["payload_curriculum"]
            payload = info["payload"]
            payloads.append((curriculum, payload))
            assert curriculum["stage"]["name"] == "center20_or_radius_0_to_30mm"
            assert 0.0 <= payload["mass_kg"] <= 0.010
            assert payload["attachment_body_m"][2] == 0.0
            assert np.linalg.norm(payload["attachment_body_m"][:2]) <= 0.030 + 1e-15
            assert curriculum["acceptance"]["physical_status"] == "feasible"
            assert curriculum["acceptance"]["e2e_action_reachable"] is True
            assert curriculum["acceptance"]["minimum_lower_thrust_margin_n"] >= 0.020 - 1e-12
            assert curriculum["acceptance"]["minimum_upper_thrust_margin_n"] >= 0.020 - 1e-12
        stats = environment.payload_curriculum_statistics()
        assert stats["accepted_payload_free_ratio"] == pytest.approx(0.30, abs=0.04)
        assert stats["accepted_center_ratio_within_payload"] == pytest.approx(0.20, abs=0.04)

        before = environment.payload_snapshot()
        for _ in range(5):
            environment.step(np.zeros(4, dtype=np.float32))
        after = environment.payload_snapshot()
        assert after["mass_kg"] == before["mass_kg"]
        assert after["attachment_body_m"] == before["attachment_body_m"]
    finally:
        environment.close()


def test_payload_stage_change_waits_for_next_reset() -> None:
    config = load_config(PROFILE)
    environment = EnvironmentFactory(config).make(seed=31, track_curriculum_steps=True)
    try:
        environment.set_curriculum_global_step(199_999)
        _, first_info = environment.reset(seed=31)
        before = environment.payload_snapshot()
        environment.set_curriculum_global_step(200_000)
        assert environment.payload_snapshot() == before
        _, second_info = environment.reset()
        assert first_info["payload_curriculum"]["stage"]["name"] == "centered_0_to_5g"
        assert second_info["payload_curriculum"]["stage"]["name"] == "center20_or_radius_0_to_10mm"
    finally:
        environment.close()


def test_payload_filter_exhaustion_raises_without_nominal_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(PROFILE)
    environment = EnvironmentFactory(config).make(seed=5, track_curriculum_steps=True)
    monkeypatch.setattr(
        environment_module,
        "static_hover",
        lambda _environment: {
            "physical_status": "infeasible",
            "physically_feasible": False,
            "policy_allocator_reachable": False,
            "equilibrium_motor_thrust_n": None,
            "actual_thrust_limits_n": [[0.0, 0.2]] * 4,
        },
    )
    try:
        with pytest.raises(RuntimeError, match="no nominal fallback"):
            environment.reset(seed=5)
        stats = environment.payload_curriculum_statistics()
        assert stats["candidate_attempt_count"] == 32
        assert stats["rejected_candidate_count"] == 32
        assert stats["exhausted_reset_count"] == 1
        assert stats["rejection_reason_counts"] == {"physical_infeasible": 32}
    finally:
        environment.close()


def _result(
    category: str,
    *,
    success: bool,
    completed: bool,
    tail: float | None,
    name: str,
) -> dict[str, object]:
    return {
        "category": category,
        "name": name,
        "execution_status": "executed",
        "success": success,
        "completed_full_duration": completed,
        "position_rmse_last_1s_m": tail,
        "termination_reason": ["time_limit"] if completed else ["excessive_tilt"],
    }


def test_fixed_suite_and_best_payload_ranking_contract() -> None:
    cases = fixed_payload_evaluation_cases()
    assert len(cases) == 14
    diagonals = [case for case in cases if case.azimuth_deg in {45.0, 135.0, 225.0, 315.0}]
    assert all(np.linalg.norm(case.attachment_xy_m) == pytest.approx(0.030) for case in diagonals)

    rows = [_result("nominal", success=True, completed=True, tail=0.01, name="nominal")]
    rows += [
        _result("center_payload", success=True, completed=True, tail=0.02, name=f"center{i}")
        for i in range(2)
    ]
    rows += [
        _result("offset_payload", success=i < 3, completed=i < 6, tail=0.03 + i * 0.01 if i < 6 else None, name=f"offset{i}")
        for i in range(8)
    ]
    rows += [
        _result("recovery", success=False, completed=False, tail=None, name=f"recovery{i}")
        for i in range(3)
    ]
    summary = summarize_fixed_payload_suite(rows)
    assert summary["payload_success_count"] == 5
    assert summary["payload_completed_full_duration_count"] == 8
    assert summary["payload_completed_worst_tail_position_rmse_m"] == pytest.approx(0.08)
    assert summary["roll_minus_20deg_success_is_required"] is False
    assert summary["ranking"] == list(BEST_PAYLOAD_RANKING)
    assert summary["criteria"]["continuous_hold_s"] == pytest.approx(1.0)
    assert summary["execution_contract"]["fixed_case_feasibility_filter_applied"] is False
    assert best_payload_ranking_key(summary) == pytest.approx((-5.0, -8.0, 0.08))
    with pytest.raises(ValueError, match="nominal success"):
        best_payload_ranking_key({**summary, "nominal_success": False})

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

import crazyflie_rl.environment as env_module
from crazyflie_rl.config import ConfigError, load_config
from crazyflie_rl.eval_cli import RolloutTrace, trace_metrics
from crazyflie_rl.factories import EnvironmentFactory
from crazyflie_rl.plotting import save_lyapunov_trace
from crazyflie_rl.rewards import (
    TrackingError,
    compute_lyapunov_reward,
    desired_rotation_from_yaw,
    geometric_attitude_error,
    lyapunov_candidate,
    tracking_error_from_state,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _error(
    *,
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    attitude=(0.0, 0.0, 0.0),
    angular_rate=(0.0, 0.0, 0.0),
) -> TrackingError:
    return TrackingError(
        position=np.asarray(position, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
        attitude=np.asarray(attitude, dtype=float),
        angular_rate=np.asarray(angular_rate, dtype=float),
    )


def _reward_config(
    mode: str = "lyapunov",
    *,
    potential: bool = True,
    decay: bool = True,
):
    reward = load_config(CONFIGS / "base.yaml").environment.reward
    return replace(
        reward,
        mode=mode,
        lyapunov=replace(
            reward.lyapunov,
            potential_shaping_enabled=potential,
            decay_penalty_enabled=decay,
        ),
    )


def _terms(
    before: TrackingError,
    after: TrackingError,
    *,
    mode: str = "lyapunov",
    gamma: float = 0.9,
    dt: float = 0.1,
    terminated: bool = False,
    truncated: bool = False,
    potential: bool = True,
    decay: bool = True,
):
    return compute_lyapunov_reward(
        error_before=before,
        error_after=after,
        dt=dt,
        gamma=gamma,
        config=_reward_config(mode, potential=potential, decay=decay),
        terminated=terminated,
        truncated=truncated,
    )


def _runtime_config(profile: str, *, mode: str = "lyapunov"):
    config = load_config(CONFIGS / profile)
    reward = replace(config.environment.reward, mode=mode)
    return replace(
        config,
        environment=replace(config.environment, reward=reward),
    )


def _require_runtime() -> None:
    if env_module.gym is None or env_module.mujoco is None:
        pytest.skip("Gymnasium and MuJoCo are not installed")


def test_zero_tracking_error_has_zero_candidate() -> None:
    settings = _reward_config().lyapunov
    assert lyapunov_candidate(_error(), settings) == 0.0


@pytest.mark.parametrize(
    "error",
    [
        _error(position=(1.0, 0.0, 0.0)),
        _error(velocity=(0.0, -1.0, 0.0)),
        _error(attitude=(0.0, 0.0, 0.2)),
        _error(angular_rate=(0.0, 0.0, 2.0)),
    ],
)
def test_nonzero_tracking_error_has_positive_candidate(error: TrackingError) -> None:
    assert lyapunov_candidate(error, _reward_config().lyapunov) > 0.0


def test_each_normalization_scale_and_matrix_block_weight_is_applied() -> None:
    base = _reward_config().lyapunov
    normalization = replace(
        base.normalization,
        position=(1.0, 2.0, 4.0),
        velocity=(2.0, 4.0, 8.0),
        attitude=(0.5, 1.0, 2.0),
        angular_rate=(4.0, 2.0, 1.0),
    )
    weights = replace(
        base.matrix_weights,
        position=2.0,
        velocity=3.0,
        attitude=5.0,
        angular_rate=7.0,
    )
    settings = replace(base, normalization=normalization, matrix_weights=weights)
    error = _error(
        position=(1.0, 2.0, 4.0),
        velocity=(2.0, 4.0, 8.0),
        attitude=(0.5, 1.0, 2.0),
        angular_rate=(4.0, 2.0, 1.0),
    )

    expected = 3.0 * (2.0 + 3.0 + 5.0 + 7.0)
    assert lyapunov_candidate(error, settings) == pytest.approx(expected)


def test_quaternion_sign_does_not_change_geometric_attitude_error() -> None:
    quaternion = np.array([0.8, -0.2, 0.3, 0.4], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    desired = desired_rotation_from_yaw(0.7)

    positive = geometric_attitude_error(quaternion, desired)
    negative = geometric_attitude_error(-quaternion, desired)

    np.testing.assert_array_equal(positive, negative)


def test_geometric_attitude_error_uses_short_rotation_across_yaw_wrap() -> None:
    actual_yaw = np.deg2rad(179.0)
    desired_yaw = np.deg2rad(-179.0)
    actual = np.array(
        [np.cos(actual_yaw / 2.0), 0.0, 0.0, np.sin(actual_yaw / 2.0)]
    )

    error = geometric_attitude_error(
        actual, desired_rotation_from_yaw(desired_yaw)
    )

    assert abs(error[2]) == pytest.approx(np.sin(np.deg2rad(2.0)), abs=1e-12)
    assert np.linalg.norm(error[:2]) == pytest.approx(0.0, abs=1e-12)


def test_tracking_error_frames_match_world_translation_and_body_rate_formula() -> None:
    yaw = np.pi / 2.0
    quaternion = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    error = tracking_error_from_state(
        position=(2.0, 3.0, 4.0),
        quaternion_wxyz=quaternion,
        velocity_world=(4.0, 5.0, 6.0),
        angular_rate_body=(0.5, 0.5, 0.0),
        position_reference=(1.0, 1.0, 1.0),
        yaw_reference=0.0,
        velocity_reference_world=(1.0, 2.0, 3.0),
        angular_rate_reference_body=(1.0, 0.0, 0.0),
    )

    np.testing.assert_array_equal(error.position, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(error.velocity, [3.0, 3.0, 3.0])
    # R.T @ R_d maps desired-body x into actual-body -y at +90 deg yaw.
    np.testing.assert_allclose(error.angular_rate, [0.5, 1.5, 0.0], atol=1e-12)


def test_potential_difference_is_exactly_v_before_minus_gamma_v_after() -> None:
    terms = _terms(
        _error(position=(1.0, 0.0, 0.0)),
        _error(position=(0.4, 0.0, 0.0)),
        mode="legacy_plus_lyapunov",
        gamma=0.83,
        decay=False,
    )

    expected = terms.v_before - 0.83 * terms.v_after
    assert terms.potential_difference == pytest.approx(expected)
    assert terms.potential_shaping == pytest.approx(
        _reward_config().lyapunov.potential_weight * expected
    )
    assert terms.state_cost == 0.0


def test_potential_shaping_is_positive_for_a_sufficient_candidate_decrease() -> None:
    terms = _terms(
        _error(position=(1.0, 0.0, 0.0)),
        _error(position=(0.5, 0.0, 0.0)),
        decay=False,
    )
    assert terms.v_after < terms.v_before
    assert terms.potential_shaping > 0.0


def test_decay_violation_is_zero_when_exponential_condition_is_met() -> None:
    terms = _terms(
        _error(position=(1.0, 0.0, 0.0)),
        _error(position=(0.8, 0.0, 0.0)),
        potential=False,
    )
    assert terms.v_after <= terms.decay_target
    assert terms.decay_violation == 0.0
    assert terms.decay_penalty == 0.0
    assert terms.decay_condition_satisfied


def test_decay_condition_violation_produces_only_a_negative_penalty() -> None:
    terms = _terms(
        _error(position=(1.0, 0.0, 0.0)),
        _error(position=(1.1, 0.0, 0.0)),
        potential=False,
    )
    assert terms.v_after > terms.decay_target
    assert terms.decay_violation > 0.0
    assert terms.decay_penalty < 0.0
    assert not terms.decay_condition_satisfied


def test_absorbing_termination_and_time_limit_truncation_use_different_potentials() -> None:
    before = _error(position=(1.0, 0.0, 0.0))
    after = _error(position=(0.7, 0.0, 0.0))
    terminated = _terms(before, after, terminated=True, truncated=False)
    truncated = _terms(before, after, terminated=False, truncated=True)

    assert terminated.terminal_potential_zeroed
    assert terminated.potential_difference == pytest.approx(terminated.v_before)
    assert not truncated.terminal_potential_zeroed
    assert truncated.potential_difference == pytest.approx(
        truncated.v_before - 0.9 * truncated.v_after
    )


def test_terminal_potential_shaping_telescopes_and_does_not_reward_earlier_crash() -> None:
    gamma = 0.9
    first = _terms(
        _error(position=(1.0, 0.0, 0.0)),
        _error(position=(0.8, 0.0, 0.0)),
        mode="legacy_plus_lyapunov",
        gamma=gamma,
        decay=False,
    )
    terminal = _terms(
        _error(position=(0.8, 0.0, 0.0)),
        _error(position=(2.0, 0.0, 0.0)),
        mode="legacy_plus_lyapunov",
        gamma=gamma,
        terminated=True,
        decay=False,
    )

    discounted_shaping = first.potential_difference + gamma * terminal.potential_difference
    assert discounted_shaping == pytest.approx(first.v_before)


def test_reset_clears_transition_candidate_diagnostics() -> None:
    _require_runtime()
    config = _runtime_config("residual_train_lyapunov.yaml")
    env = EnvironmentFactory(config).make(seed=13)
    try:
        env.reset(seed=13)
        _observation, _reward, _terminated, _truncated, info = env.step(
            np.zeros(4, dtype=np.float32)
        )
        assert "reward_terms" in info
        assert env._last_lyapunov_terms is not None
        env.reset(seed=14)
        assert env._last_lyapunov_terms is None
    finally:
        env.close()


def test_environment_candidate_uses_action_before_and_after_states_without_offset() -> None:
    _require_runtime()
    config = _runtime_config("e2e_train_lyapunov.yaml")
    env = EnvironmentFactory(config).make(seed=17)
    try:
        env.reset(seed=17)
        expected_before = lyapunov_candidate(
            env._tracking_error(*env._read_state()),
            config.environment.reward.lyapunov,
        )
        _observation, _reward, _terminated, _truncated, info = env.step(
            np.zeros(4, dtype=np.float32)
        )
        expected_after = lyapunov_candidate(
            env._tracking_error(*env._read_state()),
            config.environment.reward.lyapunov,
        )

        assert info["reward_terms"]["v_before"] == pytest.approx(expected_before)
        assert info["reward_terms"]["v_after"] == pytest.approx(expected_after)
    finally:
        env.close()


def test_independent_environment_instances_do_not_share_candidate_state() -> None:
    _require_runtime()
    config = _runtime_config("residual_train_lyapunov.yaml")
    first = EnvironmentFactory(config).make(seed=18)
    second = EnvironmentFactory(config).make(seed=19)
    try:
        first.reset(seed=18)
        second.reset(seed=19)
        first.step(np.zeros(4, dtype=np.float32))
        assert first._last_lyapunov_terms is not None
        assert second._last_lyapunov_terms is None
        first_snapshot = dict(first._last_lyapunov_terms)
        second.step(np.zeros(4, dtype=np.float32))
        assert first._last_lyapunov_terms == first_snapshot
        assert second._last_lyapunov_terms is not first._last_lyapunov_terms
    finally:
        first.close()
        second.close()


def test_pre_feature_reward_config_defaults_to_legacy(tmp_path: Path) -> None:
    data = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))
    data["environment"]["reward"].pop("mode")
    data["environment"]["reward"].pop("lyapunov")
    data["environment"]["reward"].pop("e2e_torque_xy_weight")
    data["environment"]["reward"].pop("e2e_torque_yaw_weight")
    profile = tmp_path / "pre_feature.yaml"
    profile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_config(profile)

    assert config.environment.reward.mode == "legacy"
    assert config.environment.reward.e2e_torque_xy_weight == 0.0
    assert config.environment.reward.e2e_torque_yaw_weight == 0.0


def test_all_preexisting_profiles_remain_default_off_legacy() -> None:
    opt_in = {
        "residual_train_lyapunov.yaml",
        "e2e_train_lyapunov.yaml",
        "e2e_train_lyapunov_rate01.yaml",
        "e2e_train_lyapunov_rate01_initial_perturb.yaml",
    }
    for profile in CONFIGS.glob("*.yaml"):
        if profile.name in opt_in:
            continue
        assert load_config(profile).environment.reward.mode == "legacy", profile.name


def test_disabled_candidate_terms_are_exactly_legacy_on_same_seed_rollout() -> None:
    _require_runtime()
    legacy = _runtime_config("residual_train.yaml", mode="legacy")
    disabled_settings = replace(
        legacy.environment.reward.lyapunov,
        potential_shaping_enabled=False,
        decay_penalty_enabled=False,
    )
    disabled_reward = replace(
        legacy.environment.reward,
        mode="legacy_plus_lyapunov",
        lyapunov=disabled_settings,
    )
    ablation = replace(
        legacy,
        environment=replace(legacy.environment, reward=disabled_reward),
    )
    legacy_env = EnvironmentFactory(legacy).make(seed=21)
    ablation_env = EnvironmentFactory(ablation).make(seed=21)
    actions = (
        np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        np.array([-0.5, 0.4, -0.3, 0.2], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    try:
        legacy_observation, _ = legacy_env.reset(seed=21)
        ablation_observation, _ = ablation_env.reset(seed=21)
        np.testing.assert_array_equal(legacy_observation, ablation_observation)
        for action in actions:
            legacy_step = legacy_env.step(action)
            ablation_step = ablation_env.step(action)
            np.testing.assert_array_equal(legacy_step[0], ablation_step[0])
            assert legacy_step[1] == ablation_step[1]
            assert legacy_step[2:4] == ablation_step[2:4]
            assert legacy_step[4] == {}
            assert ablation_step[4]["reward_terms"]["reward_total"] == 0.0
    finally:
        legacy_env.close()
        ablation_env.close()


def test_lyapunov_mode_does_not_double_count_legacy_tracking_weights() -> None:
    _require_runtime()
    baseline = _runtime_config("e2e_train_lyapunov.yaml")
    changed_legacy_weights = replace(
        baseline.environment.reward,
        position_weight=999.0,
        velocity_weight=777.0,
        tilt_weight=555.0,
        angular_velocity_weight=333.0,
        yaw_weight=111.0,
    )
    changed = replace(
        baseline,
        environment=replace(
            baseline.environment,
            reward=changed_legacy_weights,
        ),
    )
    first = EnvironmentFactory(baseline).make(seed=29)
    second = EnvironmentFactory(changed).make(seed=29)
    try:
        first.reset(seed=29)
        second.reset(seed=29)
        first_step = first.step(np.zeros(4, dtype=np.float32))
        second_step = second.step(np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(first_step[0], second_step[0])
        assert first_step[1] == second_step[1]
        assert first_step[4]["reward_terms"] == second_step[4]["reward_terms"]
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize(
    ("profile", "expected_weight"),
    [
        ("residual_train_lyapunov.yaml", 0.001),
        ("e2e_train_lyapunov.yaml", 0.0),
    ],
)
def test_nontracking_effort_uses_only_existing_normalized_action_penalty(
    profile: str, expected_weight: float
) -> None:
    _require_runtime()
    config = load_config(CONFIGS / profile)
    env = EnvironmentFactory(config).make(seed=31)
    action = np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float32)
    try:
        env.reset(seed=31)
        _observation, _reward, _terminated, _truncated, info = env.step(action)
        expected = -expected_weight * float(action @ action)
        assert info["reward_terms"]["nontracking_reward"] == pytest.approx(expected)
    finally:
        env.close()


def test_torque_penalty_defaults_and_e2e_profile_override() -> None:
    base = load_config(CONFIGS / "base.yaml")
    residual = load_config(CONFIGS / "residual_train_lyapunov.yaml")
    e2e = load_config(CONFIGS / "e2e_train_lyapunov.yaml")

    assert base.environment.reward.e2e_torque_xy_weight == 0.0
    assert base.environment.reward.e2e_torque_yaw_weight == 0.0
    assert residual.environment.reward.e2e_torque_xy_weight == 0.0
    assert residual.environment.reward.e2e_torque_yaw_weight == 0.0
    assert e2e.environment.reward.e2e_torque_xy_weight == 0.1
    assert e2e.environment.reward.e2e_torque_yaw_weight == 0.02


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("e2e_torque_xy_weight", -1.0, "e2e_torque_xy_weight"),
        ("e2e_torque_yaw_weight", float("nan"), "e2e_torque_yaw_weight"),
        ("e2e_torque_yaw_weight", float("inf"), "e2e_torque_yaw_weight"),
    ],
)
def test_invalid_torque_penalty_weights_fail_during_config_load(
    tmp_path: Path, field: str, value: float, message: str
) -> None:
    data = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))
    data["environment"]["reward"][field] = value
    profile = tmp_path / "invalid_torque_weight.yaml"
    profile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(profile)


def test_torque_penalty_logging_and_reward_use_only_normalized_torque() -> None:
    _require_runtime()
    config = load_config(CONFIGS / "e2e_train_lyapunov.yaml")
    zero_reward = replace(
        config.environment.reward,
        e2e_torque_xy_weight=0.0,
        e2e_torque_yaw_weight=0.0,
    )
    zero_config = replace(
        config,
        environment=replace(config.environment, reward=zero_reward),
    )
    penalized_env = EnvironmentFactory(config).make(seed=71)
    zero_env = EnvironmentFactory(zero_config).make(seed=71)
    action = np.array([0.3, -0.4, 0.5, 0.9], dtype=np.float32)
    try:
        penalized_env.reset(seed=71)
        zero_env.reset(seed=71)
        penalized = penalized_env.step(action)
        unpenalized = zero_env.step(action)
        terms = penalized[4]["reward_terms"]
        expected_xy = 0.1 * (0.3**2 + 0.4**2)
        expected_yaw = 0.02 * 0.5**2
        expected_total = expected_xy + expected_yaw
        assert terms["normalized_torque"] == pytest.approx([0.3, -0.4, 0.5])
        assert terms["e2e_torque_xy_cost"] == pytest.approx(expected_xy)
        assert terms["e2e_torque_yaw_cost"] == pytest.approx(expected_yaw)
        assert terms["e2e_torque_cost"] == pytest.approx(expected_total)
        assert terms["e2e_torque_reward"] == pytest.approx(-expected_total)
        assert penalized[1] == pytest.approx(unpenalized[1] - expected_total)
        assert terms["e2e_torque_reward"] == pytest.approx(
            terms["e2e_torque_cost"] * -1.0
        )
    finally:
        penalized_env.close()
        zero_env.close()


def test_torque_penalty_is_independent_of_thrust_action() -> None:
    _require_runtime()
    config = load_config(CONFIGS / "e2e_train_lyapunov.yaml")
    first = EnvironmentFactory(config).make(seed=72)
    second = EnvironmentFactory(config).make(seed=72)
    try:
        first.reset(seed=72)
        second.reset(seed=72)
        first_terms = first.step(np.array([0.3, -0.4, 0.5, -1.0], dtype=np.float32))[4][
            "reward_terms"
        ]
        second_terms = second.step(np.array([0.3, -0.4, 0.5, 1.0], dtype=np.float32))[4][
            "reward_terms"
        ]
        for key in (
            "e2e_torque_xy_cost",
            "e2e_torque_yaw_cost",
            "e2e_torque_cost",
            "e2e_torque_reward",
        ):
            assert first_terms[key] == pytest.approx(second_terms[key])
    finally:
        first.close()
        second.close()


def test_torque_penalty_is_zero_for_residual_even_when_configured() -> None:
    _require_runtime()
    config = load_config(CONFIGS / "residual_train_lyapunov.yaml")
    configured_reward = replace(
        config.environment.reward,
        e2e_torque_xy_weight=0.9,
        e2e_torque_yaw_weight=0.8,
    )
    configured = replace(
        config,
        environment=replace(config.environment, reward=configured_reward),
    )
    baseline = EnvironmentFactory(config).make(seed=73)
    changed = EnvironmentFactory(configured).make(seed=73)
    action = np.array([0.3, -0.4, 0.5, 0.9], dtype=np.float32)
    try:
        baseline.reset(seed=73)
        changed.reset(seed=73)
        expected = baseline.step(action)
        actual = changed.step(action)
        np.testing.assert_array_equal(expected[0], actual[0])
        assert expected[1] == actual[1]
        terms = actual[4]["reward_terms"]
        assert terms["e2e_torque_xy_cost"] == 0.0
        assert terms["e2e_torque_yaw_cost"] == 0.0
        assert terms["e2e_torque_cost"] == 0.0
        assert terms["e2e_torque_reward"] == 0.0
    finally:
        baseline.close()
        changed.close()


def test_torque_penalty_does_not_change_legacy_reward() -> None:
    _require_runtime()
    config = load_config(CONFIGS / "e2e_train_lyapunov.yaml")
    legacy_reward = replace(config.environment.reward, mode="legacy")
    zero_reward = replace(
        legacy_reward,
        e2e_torque_xy_weight=0.0,
        e2e_torque_yaw_weight=0.0,
    )
    with_weights = replace(
        config,
        environment=replace(config.environment, reward=legacy_reward),
    )
    without_weights = replace(
        config,
        environment=replace(config.environment, reward=zero_reward),
    )
    first = EnvironmentFactory(with_weights).make(seed=74)
    second = EnvironmentFactory(without_weights).make(seed=74)
    action = np.array([0.3, -0.4, 0.5, 0.9], dtype=np.float32)
    try:
        first.reset(seed=74)
        second.reset(seed=74)
        first_step = first.step(action)
        second_step = second.step(action)
        np.testing.assert_array_equal(first_step[0], second_step[0])
        assert first_step[1] == second_step[1]
        assert first_step[4] == {}
        assert second_step[4] == {}
    finally:
        first.close()
        second.close()


def test_torque_penalty_is_subtracted_once_in_legacy_plus_lyapunov() -> None:
    _require_runtime()
    config = load_config(CONFIGS / "e2e_train_lyapunov.yaml")
    weighted_reward = replace(config.environment.reward, mode="legacy_plus_lyapunov")
    zero_reward = replace(
        weighted_reward,
        e2e_torque_xy_weight=0.0,
        e2e_torque_yaw_weight=0.0,
    )
    weighted = EnvironmentFactory(
        replace(config, environment=replace(config.environment, reward=weighted_reward))
    ).make(seed=75)
    unweighted = EnvironmentFactory(
        replace(config, environment=replace(config.environment, reward=zero_reward))
    ).make(seed=75)
    action = np.array([0.3, -0.4, 0.5, 0.9], dtype=np.float32)
    try:
        weighted.reset(seed=75)
        unweighted.reset(seed=75)
        weighted_step = weighted.step(action)
        unweighted_step = unweighted.step(action)
        assert weighted_step[1] == pytest.approx(unweighted_step[1] - 0.03)
        assert weighted_step[4]["reward_terms"]["e2e_torque_cost"] == pytest.approx(0.03)
        assert weighted_step[4]["reward_terms"]["reward_components_consistent"]
        assert weighted_step[4]["reward_terms"]["reward_components"][
            "e2e_torque_reward"
        ] == pytest.approx(-0.03)
    finally:
        weighted.close()
        unweighted.close()


@pytest.mark.parametrize(
    ("profile", "expected_mode"),
    [
        ("residual_train_lyapunov.yaml", "residual"),
        ("e2e_train_lyapunov.yaml", "e2e"),
    ],
)
def test_candidate_profiles_preserve_observation_and_action_shapes(
    profile: str, expected_mode: str
) -> None:
    config = load_config(CONFIGS / profile)
    assert config.control_mode == expected_mode
    assert config.environment.reward.mode == "lyapunov"
    assert config.observation_shape == (15,)
    assert config.action_shape == (4,)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["environment"]["reward"]["lyapunov"][
                "matrix_weights"
            ].__setitem__("position", 0.0),
            "matrix_weights.position.*positive",
        ),
        (
            lambda data: data["environment"]["reward"]["lyapunov"][
                "matrix_weights"
            ].__setitem__("velocity", float("inf")),
            "matrix_weights.velocity.*finite",
        ),
        (
            lambda data: data["environment"]["reward"]["lyapunov"][
                "normalization"
            ].__setitem__("attitude", [1.0, 0.0, 1.0]),
            r"normalization.attitude\[1\].*positive",
        ),
        (
            lambda data: data["environment"]["reward"]["lyapunov"][
                "normalization"
            ].__setitem__("velocity", [1.0, float("nan"), 1.0]),
            r"normalization.velocity\[1\].*finite",
        ),
        (
            lambda data: data["environment"]["reward"]["lyapunov"].__setitem__(
                "decay_rate", 100.0
            ),
            "0 < decay_rate",
        ),
    ],
)
def test_invalid_candidate_parameters_fail_during_config_load(
    tmp_path: Path, mutate, message: str
) -> None:
    data = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))
    mutate(data)
    profile = tmp_path / "invalid.yaml"
    profile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(profile)


@pytest.mark.parametrize(
    "profile", ["residual_train_lyapunov.yaml", "e2e_train_lyapunov.yaml"]
)
def test_same_seed_short_candidate_rollout_has_no_nan_or_inf(profile: str) -> None:
    _require_runtime()
    config = load_config(CONFIGS / profile)
    env = EnvironmentFactory(config).make(seed=37)
    try:
        observation, _ = env.reset(seed=37)
        assert observation.shape == (15,)
        for _ in range(20):
            observation, reward, terminated, truncated, info = env.step(
                np.zeros(4, dtype=np.float32)
            )
            assert observation.shape == (15,)
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            terms = info["reward_terms"]
            assert terms["gamma"] == config.training.ppo.gamma
            for key in (
                "reward_total",
                "v_before",
                "v_after",
                "delta_v",
                "potential_shaping",
                "decay_target",
                "decay_violation",
                "state_cost",
            ):
                assert np.isfinite(terms[key]), key
            if terminated or truncated:
                break
    finally:
        env.close()


def test_candidate_mode_retains_full_crash_penalty() -> None:
    _require_runtime()
    config = _runtime_config("e2e_train_lyapunov.yaml")
    forced_termination = replace(
        config.environment.termination,
        min_altitude=2.0,
        max_altitude=2.5,
    )
    with_penalty = replace(
        config,
        environment=replace(
            config.environment,
            termination=forced_termination,
        ),
    )
    zero_penalty_reward = replace(
        config.environment.reward,
        crash_penalty=0.0,
    )
    without_penalty = replace(
        config,
        environment=replace(
            config.environment,
            termination=forced_termination,
            reward=zero_penalty_reward,
        ),
    )
    penalized_env = EnvironmentFactory(with_penalty).make(seed=44)
    unpenalized_env = EnvironmentFactory(without_penalty).make(seed=44)
    try:
        penalized_env.reset(seed=44)
        unpenalized_env.reset(seed=44)
        penalized = penalized_env.step(np.zeros(4, dtype=np.float32))
        unpenalized = unpenalized_env.step(np.zeros(4, dtype=np.float32))
        assert penalized[2] is True
        assert unpenalized[2] is True
        assert penalized[1] == pytest.approx(
            unpenalized[1] - config.environment.reward.crash_penalty,
            rel=0.0,
            abs=1e-12,
        )
        assert penalized[4]["reward_terms"]["terminal_potential_zeroed"]
        assert penalized[4]["reward_terms"]["crash_or_ood_reward"] == pytest.approx(
            -config.environment.reward.crash_penalty
        )
        assert penalized[4]["reward_terms"]["reward_components_consistent"]
        assert "below_minimum_altitude" in penalized[4]["termination_reasons"]
    finally:
        penalized_env.close()
        unpenalized_env.close()


def test_trace_metrics_record_candidate_tracking_action_and_safety_contract() -> None:
    trace = RolloutTrace(
        policy="residual",
        label="candidate",
        time_sec=np.array([0.0, 0.1, 0.2]),
        position=np.zeros((3, 3)),
        attitude_deg=np.zeros((3, 3)),
        reference_position=np.zeros((3, 3)),
        position_error=np.array([1.0, 2.0, 3.0]),
        phases=("HOVER", "HOVER", "HOVER"),
        training_boundary_crossed_at=0.0,
        guard_boundary_crossed_at=0.1,
        terminated_at=0.2,
        truncated_at=None,
        diverged_at=None,
        control_input=np.array(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        motor_thrust=np.zeros((3, 4)),
        linear_velocity=np.zeros((3, 3)),
        angular_velocity=np.zeros((3, 3)),
        control_mode="residual",
        lyapunov_v_before=np.array([4.0, 3.0, 2.0]),
        lyapunov_v=np.array([3.0, 2.0, 2.5]),
        lyapunov_delta_v=np.array([-1.0, -1.0, 0.5]),
        lyapunov_decay_target=np.array([3.6, 2.7, 1.8]),
        normalized_tracking_error=np.array([2.0, 1.0, 3.0]),
        geometric_attitude_error=np.array(
            [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]]
        ),
        angular_rate_error=np.array(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        ),
        actuator_saturation_fraction=np.array([0.0, 0.25, 0.5]),
        terminated_transition=np.array([False, False, True]),
        truncated_transition=np.array([False, False, False]),
    )

    metrics = trace_metrics(trace, 0.3)

    assert metrics["mean_v"] == pytest.approx(2.5)
    assert metrics["max_v"] == 3.0
    assert metrics["terminal_v"] == 2.5
    assert metrics["mean_delta_v"] == pytest.approx(-0.5)
    assert metrics["v_decrease_transition_ratio"] == pytest.approx(2.0 / 3.0)
    assert metrics["decay_condition_satisfied_ratio"] == pytest.approx(2.0 / 3.0)
    assert metrics["integrated_tracking_error"] == pytest.approx(0.6)
    assert metrics["integrated_normalized_tracking_error"] == pytest.approx(0.6)
    assert metrics["residual_action_rms"] == metrics["action_rms"]
    assert metrics["action_total_variation"] == pytest.approx(2.0)
    assert metrics["actuator_saturation_ratio"] == pytest.approx(0.25)
    assert metrics["crash_or_guard_termination"] is True


def test_candidate_plot_records_v_delta_and_decay_bound(tmp_path: Path) -> None:
    output = save_lyapunov_trace(
        tmp_path / "candidate.png",
        tag="unit test",
        time_sec=np.array([0.0, 0.1, 0.2]),
        v_before=np.array([3.0, 2.0, 1.0]),
        v_after=np.array([2.0, 1.0, 0.5]),
        delta_v=np.array([-1.0, -1.0, -0.5]),
        decay_target=np.array([2.7, 1.8, 0.9]),
    )
    assert output.is_file()
    assert output.stat().st_size > 0

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.reward_alignment import (
    discounted_reward_totals,
    reward_alignment_cases,
    reward_alignment_warnings,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_combined_profile_exactly_composes_reward_and_initial_curriculum() -> None:
    combined = load_config(
        CONFIGS / "e2e_train_lyapunov_rate01_initial_perturb.yaml"
    )
    reward_parent = load_config(CONFIGS / "e2e_train_lyapunov_rate01.yaml")
    reset_parent = load_config(CONFIGS / "e2e_train_initial_perturb.yaml")

    assert combined.control_mode == "e2e"
    assert combined.environment.reward == reward_parent.environment.reward
    assert combined.training == reward_parent.training
    assert combined.environment.initial_state_randomization == (
        reset_parent.environment.initial_state_randomization
    )
    assert combined.environment.position_perturbation == 0.0
    assert combined.environment.attitude_perturbation_deg == 0.0
    assert combined.environment.reward.mode == "lyapunov"
    assert combined.environment.reward.angular_velocity_weight == 0.01
    assert combined.environment.reward.e2e_torque_xy_weight == 0.1
    assert combined.environment.reward.e2e_torque_yaw_weight == 0.02
    assert combined.environment.initial_state_randomization.enabled is True
    assert combined.actuator.model == "cf21b_first_order"
    assert combined.observation_shape == (15,)
    assert combined.action_shape == (4,)
    assert combined.training.seed == 42


def test_reward_alignment_case_grid_is_nominal_and_bidirectional_roll_pitch() -> None:
    cases = reward_alignment_cases()
    assert len(cases) == 13
    assert cases[0].name == "nominal"
    assert cases[0].tilt_deg == 0.0
    assert {case.tilt_deg for case in cases[1:]} == {5.0, 15.0, 30.0}
    for tilt in (5.0, 15.0, 30.0):
        assert {
            case.name for case in cases if case.tilt_deg == tilt
        } == {
            f"roll_plus_{tilt:g}deg",
            f"roll_minus_{tilt:g}deg",
            f"pitch_plus_{tilt:g}deg",
            f"pitch_minus_{tilt:g}deg",
        }


def test_discounted_reward_totals_use_ppo_gamma_for_every_component() -> None:
    first = {
        "state_reward": -1.0,
        "potential_reward": 2.0,
        "decay_reward": -3.0,
        "e2e_torque_reward": -4.0,
        "crash_or_ood_reward": 0.0,
        "nontracking_reward": -0.5,
    }
    second = {
        "state_reward": -2.0,
        "potential_reward": 4.0,
        "decay_reward": -6.0,
        "e2e_torque_reward": -8.0,
        "crash_or_ood_reward": -10.0,
        "nontracking_reward": -1.0,
    }
    totals = discounted_reward_totals(
        [(-6.5, first), (-23.0, second)], gamma=0.5
    )

    assert totals["discounted_state_reward"] == pytest.approx(-2.0)
    assert totals["discounted_potential_reward"] == pytest.approx(4.0)
    assert totals["discounted_decay_reward"] == pytest.approx(-6.0)
    assert totals["discounted_torque_penalty"] == pytest.approx(-8.0)
    assert totals["discounted_crash_penalty"] == pytest.approx(-5.0)
    assert totals["discounted_nontracking_reward"] == pytest.approx(-1.0)
    assert totals["total_discounted_return"] == pytest.approx(-18.0)
    assert totals["discounted_logged_component_sum"] == pytest.approx(-18.0)


def test_alignment_warning_detects_success_return_inversion() -> None:
    rows = [
        {
            "name": "pitch_plus_30deg",
            "controller": "pid_floor",
            "success": True,
            "total_discounted_return": -20.0,
        },
        {
            "name": "pitch_plus_30deg",
            "controller": "e2e_ppo",
            "success": False,
            "total_discounted_return": -10.0,
        },
    ]
    warnings = reward_alignment_warnings(rows)
    assert len(warnings) == 1
    assert "PID succeeds but PPO fails" in warnings[0]
    assert "pitch_plus_30deg" in warnings[0]


@pytest.mark.parametrize(
    "pid_success,ppo_success,pid_return,ppo_return",
    [
        (False, False, -20.0, -10.0),
        (True, True, -20.0, -10.0),
        (True, False, -5.0, -10.0),
    ],
)
def test_alignment_warning_requires_exact_inversion_condition(
    pid_success: bool,
    ppo_success: bool,
    pid_return: float,
    ppo_return: float,
) -> None:
    rows = [
        {
            "name": "roll_plus_15deg",
            "controller": "pid_floor",
            "success": pid_success,
            "total_discounted_return": pid_return,
        },
        {
            "name": "roll_plus_15deg",
            "controller": "e2e_ppo",
            "success": ppo_success,
            "total_discounted_return": ppo_return,
        },
    ]
    assert reward_alignment_warnings(rows) == []


def test_reward_alignment_cli_help_is_import_safe() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "evaluate_reward_alignment.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert "--model" in completed.stdout
    assert "--seed" in completed.stdout

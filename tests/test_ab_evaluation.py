from pathlib import Path
import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.dr_transfer import actuation_metrics, validate_common_config
from crazyflie_rl.velocity_reference import velocity_semantics, desired_velocity_semantics


def test_motor_clipping_and_bounds_are_separate_from_tracking_lag():
    rows=[]
    for k in range(2):
        rows.append(dict(quaternion=np.array([1.,0,0,0]),
            motor_allocation_clipped=np.array([k==0,k==1,False,False]),
            motor_command_at_lower_bound=np.array([k==0,False,False,False]),
            motor_command_at_upper_bound=np.array([False,False,k==1,False]),
            motor_actual_at_thrust_lower_bound=np.zeros(4,bool),
            motor_actual_at_thrust_upper_bound=np.zeros(4,bool),
            motor_thrust=np.full(4,.04+.01*k)))
    result=actuation_metrics(rows)
    assert result['motor_allocation_clipped_fraction_any_motor']==1
    assert result['motor_allocation_clipped_fraction_per_motor']==[.5,.5,0,0]
    assert result['motor_command_at_lower_bound_fraction_any_motor']==.5
    assert result['motor_command_at_upper_bound_fraction_any_motor']==.5
    assert result['motor_actual_at_thrust_upper_bound_fraction_any_motor']==0
    assert result['motor_thrust_min_n_per_motor']==[.04]*4
    assert result['tilt_max_deg']==0


def test_empty_or_missing_actuation_is_unmeasured_not_zero():
    assert actuation_metrics([])['motor_allocation_clipped_fraction_any_motor'] is None
    row={'quaternion':np.array([1.,0,0,0])}
    assert actuation_metrics([row])['motor_allocation_clipped_fraction_any_motor'] is None


def test_common_evaluation_contract():
    config=load_config(Path(__file__).resolve().parents[1]/'configs/eval_velocity_ab.yaml')
    validate_common_config(config)
    assert velocity_semantics(config)=={'mode':'absolute'}
    assert desired_velocity_semantics(config)=={'mode':'position_error','position_gain':4.,'max_speed':1.5}
    assert config.environment.position_perturbation==0


def test_tail_velocity_error_only_for_completed_evaluation():
    from crazyflie_rl.dr_transfer import summarize, Thresholds, Case
    goal=np.array([0.,0.,1.])
    rows=[]
    for k in range(800):
        rows.append(dict(time=k*.01,time_post=(k+1)*.01,position=goal.copy(),
            reference=goal.copy(),reference_post=goal.copy(),position_before=goal.copy(),
            velocity=np.zeros(3),velocity_before=np.zeros(3),quaternion=np.array([1.,0,0,0]),
            internal_velocity_error=np.array([.3,.4,0]),terminated=False,truncated=k==799))
    result=summarize(rows,Case('hover',8.,goal=tuple(goal)),Thresholds())
    assert result['last_2s_internal_velocity_error_rms_m_s']==pytest.approx(.5)
    assert result['completed_count']==result['trial_count']==1
    result=summarize(rows[:100],Case('hover',8.,goal=tuple(goal)),Thresholds())
    assert result['partial'] and result['last_2s_internal_velocity_error_rms_m_s'] is None

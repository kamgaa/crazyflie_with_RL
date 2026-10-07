import numpy as np
import pytest

from audit_coordinate_contracts import (new_env,engine_geometry,establish_frames,matrix_audit,
    increments,efficiency_loss_checks,state_frame_checks,policy_audit)
from crazyflie_rl.config import load_config
from crazyflie_rl.dr_transfer import ROOT


@pytest.fixture(scope='module')
def audit():
    config=load_config(ROOT/'configs/eval_velocity_ab.yaml');env,adapter=new_env(config)
    geometry=engine_geometry(env);frames=establish_frames(env,geometry)
    yield config,env,geometry,frames
    env.close()


def test_model_front_world_and_proper_transform(audit):
    _,env,g,f=audit
    np.testing.assert_allclose(f['camera_forward_native'],[1,0,0],atol=1e-14)
    np.testing.assert_allclose(f['camera_right_native'],[0,-1,0],atol=1e-14)
    np.testing.assert_allclose(f['gravity_world_m_s2'],[0,0,-9.81],atol=1e-14)
    S=f['S_native_to_frd'];assert np.linalg.det(S)==1
    np.testing.assert_array_equal(S@S.T,np.eye(3))
    assert f['z_only_flip_determinant']==-1


def test_motor_permutation_and_spin_are_distinct(audit):
    _,env,g,f=audit
    assert [r['user_motor_number'] for r in f['motors']]==[4,3,2,1]
    np.testing.assert_array_equal(f['P_native_from_user'],np.eye(4)[::-1])
    assert [r['normal_body_reaction_top_view'] for r in f['motors']]==['CCW','CW','CCW','CW']
    assert all(not r['matches_user_reaction'] for r in f['motors'])
    assert all(r['site_parent']=='drone' for r in f['motors'])
    assert env.model.nu==8 and np.all(env.data.qvel[6:]==0) and np.all(env.actuator_model.omega>0)


def test_three_matrices_and_command_coordinate_equivalence(audit):
    _,env,g,f=audit;r=matrix_audit(env,g,f)
    assert r['internal_sign_and_order_match']
    assert r['axis_arm_difference_m']==pytest.approx(.000005,abs=1e-15)
    assert r['target_roll_pitch_T_match'] and r['target_yaw_reversed']
    assert r['motor_command_equivalence_max_n']<1e-12 and r['oracle_equivalence_max_n']<1e-12
    assert r['naive_target_allocator_only']['actual_existing_plant_frd_wrench'][2]==pytest.approx(-.0001)
    # A matching geometric sign in the old native index does NOT establish the
    # target's front-left rotor number or physical propeller spin.
    np.testing.assert_allclose(np.array(r['physical_user_frd'])[2],-.00594*np.array([-1,1,-1,1]))


@pytest.fixture(scope='module')
def probes(audit):
    config,env,g,f=audit
    return increments(config,f['S_native_to_frd'])


@pytest.mark.parametrize('condition',['nominal','payload_5g_pos_x'])
@pytest.mark.parametrize('pose',['level','rotated'])
def test_engine_force_com_moment_and_acceleration(audit,probes,condition,pose):
    records,masses=probes
    rs=[r for r in records if r['condition']==condition and r['pose']==pose]
    assert len(rs)==8
    for r in rs:
        assert r['contacts_before']==r['contacts_after']==0
        assert max(r['errors'].values())<1e-8
        delta=np.array(r['delta_torque_origin_native_nm'])-np.cross(r['whole_com_native_m'],r['delta_force_native_n'])
        np.testing.assert_allclose(delta,r['delta_torque_com_native_nm'],atol=1e-14)
        np.testing.assert_allclose(r['delta_angular_acceleration_native_rad_s2'],r['delta_angular_acceleration_predicted_from_com_rad_s2'],atol=1e-8)
    entry=next(r for r in masses if r['condition']==condition and r['pose']==pose)
    assert entry['total_mass_kg']==pytest.approx(.043384+(.005 if condition.startswith('payload') else 0))
    if condition.startswith('payload'):
        assert not np.array_equal(entry['whole_com_native_m'],entry['drone_body_ipos_m'])
        assert any(np.linalg.norm(np.array(r['delta_origin_acceleration_world_m_s2'])-r['delta_whole_com_acceleration_world_m_s2'])>.001 for r in rs)


def test_efficiency_loss_is_increment_not_normal_reaction(audit):
    config,env,g,f=audit;records=efficiency_loss_checks(config,f)
    user=next(r for r in records if r['test']=='user_FRD_motor_1')
    assert user['current_motor_number']==4 and user['user_motor_number']==1
    d=user['loss_magnitude_n'];expected=np.array([-.03536*d,-.03536*d,-.00594*d])
    np.testing.assert_allclose(user['delta_torque_origin_frd_nm'],expected,atol=1e-14)
    assert user['user_motor1_expected_fault_torque_frd_nm'][2]>0 and user['delta_torque_origin_frd_nm'][2]<0
    for r in records:
        assert r['motor_internal_state_unchanged'] and r['physical_state_and_reference_unchanged']
        np.testing.assert_array_equal(np.array(r['nominal_thrust_before'])*r['effectiveness'],r['actual_thrust_after'])


def test_mixed_state_frames_nonzero_orientation_and_payload(audit):
    config,env,g,f=audit;r=state_frame_checks(config,f['S_native_to_frd'])
    assert r['gyro_matches_body_qvel'] and r['quaternion_matches_engine']
    assert r['origin_velocity_differs_from_whole_com']
    assert not np.allclose(r['gyro_world'],r['gyro_native_body'])


def test_frozen_policy_roundtrip_without_connecting_adapter(audit,monkeypatch):
    from stable_baselines3 import PPO
    import torch
    def forbidden(*args,**kwargs):raise AssertionError('training forbidden')
    monkeypatch.setattr(PPO,'learn',forbidden);monkeypatch.setattr(PPO,'train',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    config,env,g,f=audit
    records,samples=policy_audit(config,f['S_native_to_frd'],f['P_native_from_user'])
    assert len(samples)==8 and len(records)==2
    assert all(not r['adapter_connected'] and not r['evaluation_rollout_executed'] and r['parameters_unchanged'] for r in records)
    assert max(r['action_max_error'] for r in samples)<1e-6

"""Nonlearning checks for physical layout selection and public motor boundaries."""
from dataclasses import replace
import numpy as np
import pytest

from crazyflie_rl.config import load_config, ConfigError
from crazyflie_rl.dr_transfer import ROOT, EvaluationAdapter
from crazyflie_rl.motor_layout import (S,P,user_from_native,native_from_user,exposed_motor_index,
    reaction_directions,world_from_frd)
from crazyflie_rl.layout_evaluation import configuration_pair, static_validation, LayoutObserver, NOMINAL, FAULTS
from crazyflie_rl.interactive_eval import Controls
from crazyflie_rl.integral_controller import IntegralController
from crazyflie_rl.integral_validation import Scenario, GAINS, make_case
from crazyflie_rl.oracle_allocation import OracleAllocationEnv, efficiency_matrix
from crazyflie_rl.oracle_recovery import scenario_at
from crazyflie_rl.payload_motor_eval import condition_config
from crazyflie_rl.dr_transfer import run_case
from crazyflie_rl.oracle_eval import verify_oracle


@pytest.fixture(scope='module')
def configs():
    return configuration_pair(ROOT/'configs/eval_velocity_ab.yaml',ROOT/'configs/eval_velocity_ab_user_frd.yaml')


def test_profiles_default_and_shared_native_contract(configs):
    assert len(NOMINAL)*2*2+len(FAULTS)*2==30
    assert load_config(ROOT/'configs/e2e_train.yaml').vehicle.reaction_torque_layout=='legacy'
    assert configs['legacy'].vehicle.motor_direction==(1.,-1.,1.,-1.)
    assert configs['user_frd'].vehicle.motor_direction==(-1.,1.,-1.,1.)
    for c in configs.values():
        assert c.observation_shape==(15,) and c.action_shape==(4,)
        assert c.environment.residual_scale==(.0075,.0075,.001,.5)
    with pytest.raises(ValueError):reaction_directions('typo',[1,-1,1,-1])


def test_legacy_serialized_config_keeps_archived_comparison_cli_contract(configs):
    import json,yaml
    old=json.loads((ROOT/'artifacts/runs/oracle-recovery-m3dh3vij/manifest.json').read_text())
    assert configs['legacy'].resolved_dict()==old['common_resolved_config']
    for c in configs.values():
        saved=yaml.safe_load(yaml.safe_dump(c.resolved_dict()))
        assert saved['vehicle']['motor_direction']==list(c.vehicle.motor_direction)
    assert configs['user_frd'].resolved_dict()['vehicle']['reaction_torque_layout']=='user_frd'


def test_proper_frame_permutation_and_level_thrust():
    np.testing.assert_array_equal(S@S.T,np.eye(3));assert np.linalg.det(S)==1
    rng=np.random.default_rng(42);v=rng.normal(size=(5,4))
    np.testing.assert_array_equal(native_from_user(user_from_native(v)),v)
    np.testing.assert_array_equal(world_from_frd(np.eye(3)),S)
    np.testing.assert_array_equal(world_from_frd(np.eye(3))@[0,0,-1],[0,0,1])
    assert [exposed_motor_index(i,'user_frd') for i in range(1,5)]==[3,2,1,0]


def test_engine_static_matrices_com_increment_and_loss(configs):
    result=static_validation(configs)
    for layout,r in result.items():
        assert r['rotor_increment_count']==32
        np.testing.assert_allclose(r['physical_user_frd'],r['expected_user_frd'],atol=1e-14)
        assert max(max(v['errors'].values()) for v in r['increments'])<1e-8
    user=next(v for v in result['user_frd']['efficiency_losses'] if v['test']=='user_FRD_motor_1')
    np.testing.assert_allclose(user['delta_torque_origin_frd_nm'],np.array([-.03536,-.03536,.00594])*user['loss_magnitude_n'],atol=1e-14)


@pytest.mark.parametrize('layout',['legacy','user_frd'])
@pytest.mark.parametrize('motor',[1,2,3,4])
def test_keyboard_numbering_independence_and_restore(configs,layout,motor):
    env=OracleAllocationEnv(config=configs[layout])
    try:
        adapter=EvaluationAdapter(env);adapter.reset_to_case_initial_state(make_case(Scenario('keys')),42)
        before=adapter.snapshot();controls=Controls();events=[]
        controls.process([str(motor)]*15,env,0,events.append)
        index=exposed_motor_index(motor,layout)
        eta=np.ones(4);eta[index]=.7
        np.testing.assert_array_equal(env.motor_effectiveness,eta)
        assert events[-1]['native_motor_index']==index and events[-1]['user_motor_id']==4-index
        assert before==adapter.snapshot()
        controls.process(['0'],env,0,events.append)
        np.testing.assert_array_equal(env.motor_effectiveness,np.ones(4));assert before==adapter.snapshot()
    finally:env.close()


@pytest.mark.parametrize('layout',['legacy','user_frd'])
def test_oracle_column_event_and_current_efficiency_no_reset(configs,layout):
    native=scenario_at('combined_motor2_70',71)
    scenario=native if layout=='legacy' else replace(native,motor_number=3)
    env=OracleAllocationEnv(config=condition_config(configs[layout],scenario),allocator_mode='oracle')
    try:
        adapter=EvaluationAdapter(env);adapter.reset_to_case_initial_state(make_case(scenario),42)
        ctrl=IntegralController(None,GAINS[1]);obs=LayoutObserver(scenario,ctrl);obs.on_reset(adapter)
        ctrl.xi[:]=[.01,-.02,.03]
        env.data.time=4.99;obs.before_step(adapter,499,4.99)
        np.testing.assert_array_equal(env.allocator_efficiency,np.ones(4))
        env.data.time=5.;before=adapter.snapshot();obs.before_step(adapter,500,5.)
        assert before==adapter.snapshot();np.testing.assert_array_equal(ctrl.xi,[.01,-.02,.03])
        np.testing.assert_array_equal(env.allocator_efficiency,[1,.71,1,1])
        B=efficiency_matrix(env.B,env.motor_effectiveness)
        w=B@np.array([.1,.12,.08,.11]);f=env.B_pinv@w
        np.testing.assert_allclose(B@f,w,atol=1e-14)
        before_omega=env.actuator_model.omega.copy();env._apply_control(w)
        np.testing.assert_array_equal(env._last_f,env.motor_effectiveness*env.nominal_thrust)
        np.testing.assert_array_equal(env._last_q_actual,env.motor_effectiveness*env.nominal_reaction_torque)
        # Motor lag stays enabled, so nominal output has not jumped to command.
        assert not np.allclose(env.nominal_thrust,env._last_f_cmd,atol=1e-5)
        np.testing.assert_allclose(env.B_pinv,np.linalg.pinv(B),atol=1e-14)
    finally:env.close()


@pytest.mark.parametrize('layout',['legacy','user_frd'])
def test_displaced_reference_uses_actual_case_in_signal_verification(configs,layout):
    class UnitPolicy:
        def bind(self,env):pass
        def predict(self,observation):return np.zeros(4)
    scenario=Scenario('unit_reference');case=replace(make_case(scenario,.03),goal=(0.,.05,1.))
    controller=IntegralController(UnitPolicy(),GAINS[0]);observer=LayoutObserver(scenario,controller,-5.)
    rows,initial,error,reasons=run_case(configs[layout],case,controller,42,env_factory=OracleAllocationEnv,
        observer=observer,observation_transform=controller.prepare_observation)
    assert error is None
    verify_oracle(rows,observer.physics_rows,scenario,'existing',case=case)
    np.testing.assert_allclose(initial['observation'][:3],[0,-.05,0],atol=1e-8)
    assert initial['quaternion'][3]<0


def test_user_interactive_real_checkpoint_logging(configs,tmp_path):
    from crazyflie_rl.payload_motor_eval import select_models,DEFAULT_RECORD
    from crazyflie_rl.interactive_eval import run,KeyEvents
    from crazyflie_rl.dr_policy import sha256
    import csv,json
    c=configs['user_frd'];c=replace(c,paths=replace(c.paths,artifact_root=tmp_path/'runs'))
    policy=select_models(DEFAULT_RECORD,c)[0];before=sha256(policy.provenance['path'])
    keys=KeyEvents();keys(ord('1'))
    root,trace,outcome=run(c,policy,duration=.03,headless=True,realtime=False,keys=keys)
    assert outcome['error'] is None and before==sha256(policy.provenance['path'])
    with (root/'rollout.csv').open() as f:rows=list(csv.DictReader(f))
    assert rows[0]['reaction_torque_layout']=='user_frd'
    assert float(rows[0]['motor_effectiveness_4'])==.98 and float(rows[0]['user_motor_effectiveness_1'])==.98
    with (root/'events.csv').open() as f:event=next(csv.DictReader(f))
    assert event['native_motor_index']=='3' and event['user_motor_id']=='1'
    assert (root/'plots/motor_effectiveness.png').exists()

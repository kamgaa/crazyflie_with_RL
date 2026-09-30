"""No PPO learning: state/clock contracts, metric oracles and frozen inference."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.dr_policy import load_frozen_policy, validate_metadata, labeled, sha256
from crazyflie_rl.dr_transfer import (
    ROOT, Case, EvaluationAdapter, Thresholds, make_cases, reference_sequence,
    rmse, run_case, summarize, validate_common_config, main,
)
from crazyflie_rl.environment import CrazyflieResidualEnv

CONFIG = ROOT / 'configs/eval_dr_transfer.yaml'
CHECKPOINT = ROOT / 'artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip'


@pytest.fixture
def config():
    return load_config(CONFIG)


@pytest.mark.parametrize('name,expected', [('step-005', -.05), ('step-050', -.5)])
def test_first_observation_state_and_reference_purity(config, name, expected):
    case = make_cases(config, [name])[0]
    env = CrazyflieResidualEnv(config=config)
    try:
        a = EvaluationAdapter(env)
        obs = a.reset_to_case_initial_state(case, 42)
        np.testing.assert_array_equal(obs[:3], np.array([expected, 0, 0], np.float32))
        np.testing.assert_array_equal(env.data.qpos[:7], [0, 0, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(env.data.qvel, 0)
        assert env._step == 0 and env.data.time == 0
        np.testing.assert_array_equal(env._prev_action, 0)
        np.testing.assert_array_equal(env.pid._i_vel, 0)
        assert env._last_f.sum() == pytest.approx(config.vehicle.mass * config.vehicle.gravity, rel=1e-5)
        assert np.all(env._last_omega > 0)
        before = a.snapshot()
        a.set_reference([.7, -.3, 1.1])
        a.current_observation()
        a.current_observation()
        after = a.snapshot()
        for k in before.keys() - {'reference', 'observation'}:
            assert before[k] == after[k], k
        # Absolute world velocity is preserved, not replaced by a reference derivative.
        env.data.qvel[:3] = [1, 2, 3]
        np.testing.assert_array_equal(a.current_observation()[3:6], [1, 2, 3])
    finally:
        env.close()


def test_config_disables_both_dr_paths_and_retains_plant(config):
    validate_common_config(config)
    assert not config.environment.initial_pose_randomization.enabled
    assert config.environment.position_perturbation == 0
    assert config.environment.attitude_perturbation_deg == 0
    base = load_config(ROOT / 'configs/e2e_train.yaml')
    assert config.controller == base.controller
    assert config.vehicle == base.vehicle
    assert config.actuator == base.actuator
    assert config.environment.termination == base.environment.termination
    for seed in (1, 42):
        env = CrazyflieResidualEnv(config=config)
        try:
            obs, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(obs[:3], 0)
            np.testing.assert_array_equal(obs[6:10], [1, 0, 0, 0])
        finally:
            env.close()
    with pytest.raises(ValueError, match='perturbations'):
        validate_common_config(replace(config, environment=replace(config.environment, position_perturbation=.1)))


def test_circle_reuses_mission_and_sets_26_second_horizon(config):
    case = make_cases(config, ['circle'])[0]
    assert case.horizon == 26
    sequence = reference_sequence(case, .01)
    assert len(sequence['time']) == 2601
    old_config = load_config(ROOT / 'configs/view_live_circle_eval.yaml')
    from crazyflie_rl.missions import mission_from_experiment
    old = mission_from_experiment(old_config)
    for t, ref, phase in zip(sequence['time'], sequence['reference'], sequence['phase']):
        old_ref, old_phase = old.reference(t)
        np.testing.assert_array_equal(ref, old_ref)
        assert phase == old_phase
    env = CrazyflieResidualEnv(config=config, episode_sec=case.horizon)
    try:
        assert env.max_steps == 2600
        adapter = EvaluationAdapter(env)
        adapter.reset_to_case_initial_state(case, 42)
        np.testing.assert_array_equal(env.data.qpos[:3], [0, 0, .02])
        np.testing.assert_array_equal(env._last_omega, 0)
    finally:
        env.close()


def rows_at(times, errors=None, velocities=None, *, terminated=False):
    times = np.asarray(times, float)
    errors = np.zeros((len(times), 3)) if errors is None else np.asarray(errors)
    velocities = np.zeros_like(errors) if velocities is None else np.asarray(velocities)
    goal = np.array([.05, 0, 1.])
    return [dict(time=float(t-.01), time_post=float(t), reference=goal.copy(), reference_post=goal.copy(),
                 phase='STEP', phase_post='STEP', position=goal + e, velocity=v,
                 quaternion=np.array([1.,0.,0.,0.]),
                 position_before=goal + e, velocity_before=v, terminated=terminated and i==len(times)-1,
                 truncated=not terminated and t==8.) for i, (t,e,v) in enumerate(zip(times, errors, velocities))]


def test_rmse_oracle_tail_std_and_partial_null():
    errors = [[3,4,2], [0,0,-4], [6,8,0]]
    assert rmse(errors) == pytest.approx(dict(position_rmse_xy=np.sqrt(125/3),
                                             position_rmse_z=np.sqrt(20/3), position_rmse_total=np.sqrt(145/3)))
    case = Case('step-005', 8., (.05,0,1))
    times = np.arange(1, 801) * .01
    errors = np.tile([.001, .002, -.003], (800,1))
    velocities = np.tile([.003, .004, 0], (800,1))
    rows = rows_at(times, errors, velocities)
    r = summarize(rows, case, Thresholds())
    assert r['completed'] and not r['partial']
    assert r['last_2s_speed_rms'] == pytest.approx(.005)
    assert r['last_2s_position_rmse_total'] == pytest.approx(np.sqrt(14)*.001)
    for axis in 'xyz':
        assert r['last_2s_position_std_'+axis] == pytest.approx(0, abs=1e-14)
    assert r['overshoot_m'] == pytest.approx(.001)
    assert r['settled'] and r['settling_time_s'] == 0
    partial = summarize(rows_at(times[:50], errors[:50], terminated=True), case, Thresholds(), reasons=['min_altitude'])
    assert partial['partial'] and not partial['completed']
    assert partial['position_rmse_total'] > 0
    assert all(v is None for k,v in partial.items() if k.startswith('last_2s_'))
    assert partial['settling_time_s'] is None and not partial['settled']
    assert partial['termination_reasons'] == ['min_altitude']


def test_settling_requires_good_suffix_minimum_hold_and_position_only_entry():
    times = np.arange(1, 801) * .01
    errors = np.zeros((800,3))
    velocities = np.zeros((800,3))
    velocities[:599,0] = .1
    case = Case('step-005',8.,(.05,0,1))
    result = summarize(rows_at(times,errors,velocities),case,Thresholds())
    assert result['first_position_entry_s'] == 0
    assert result['settling_time_s'] == 6.
    velocities[750,0] = .1  # Less than a second remains after this departure.
    result = summarize(rows_at(times,errors,velocities),case,Thresholds())
    assert result['settling_time_s'] is None and not result['settled']
    errors[-1,0] = .006
    assert not summarize(rows_at(times,errors),case,Thresholds())['settled']
    empty = summarize([],case,Thresholds(),error='no prediction')
    assert empty['position_rmse_xy'] is None and empty['partial']


def test_circle_metrics_use_post_reference_and_do_not_add_step_metrics(config):
    case = make_cases(config, ['circle'])[0]
    rows = rows_at([12.01,12.02])
    for row in rows:
        row['reference'] = case.reference(row['time'])[0]
        row['reference_post'], row['phase_post'] = case.reference(row['time_post'])
        row['position'] = row['reference_post'].copy()
    result = summarize(rows,case,Thresholds())
    assert result['circle_phase']['sample_count']==2
    assert result['position_rmse_total']==0
    assert result['circle_phase']['partial']
    assert 'settling_time_s' not in result and 'overshoot_m' not in result


@pytest.mark.parametrize('metadata', [
    {'residual_scale':[.1,.1,.1,.1]}, {'control_mode':'residual'}, {'action_shape':[3]},
    {'observation_shape':[16]}, {'velocity_input':'error'}, {'wrappers':['FrameStack']},
    {'resolved_config': {'environment': {'residual_scale':[1,2,3,4]}}},
])
def test_contract_mismatch_is_error(config, metadata):
    with pytest.raises(ValueError):
        validate_metadata(metadata,config)


def fake_zip(tmp_path, saved):
    path=tmp_path/'fake.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('data',json.dumps(saved))
    return path


def test_unknown_preprocessing_and_missing_statistics_fail(config,tmp_path):
    path=fake_zip(tmp_path, {'_last_original_obs':[1,2,3]})
    with pytest.raises(ValueError,match='statistics required'):
        load_frozen_policy('norm',str(path),config)
    with pytest.raises(ValueError,match='none is incompatible'):
        load_frozen_policy('norm',str(path),config,normalization='none')
    path=fake_zip(tmp_path, {})
    with pytest.raises(ValueError,match='preprocessing is unknown'):
        load_frozen_policy('unknown',str(path),config)
    with pytest.raises(ValueError,match='duplicate label'):
        labeled(['x=a.zip','x=b.zip'])


def test_unknown_scale_is_explicit_and_normalizer_is_frozen(config,tmp_path,monkeypatch):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from crazyflie_rl import dr_policy
    class Stub:
        action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32)
        observation_space=gym.spaces.Box(-np.inf,np.inf,(15,),dtype=np.float32)
        policy=SimpleNamespace(set_training_mode=lambda _:None)
        def predict(self, obs, deterministic):
            assert deterministic
            self.last_obs=obs.copy()
            return np.zeros(4),None
    model=Stub()
    monkeypatch.setattr(dr_policy,'_load_policy',lambda *args:model)
    path=fake_zip(tmp_path, {'_last_original_obs':[1]})
    env=CrazyflieResidualEnv(config=config)
    normalizer=VecNormalize(DummyVecEnv([lambda:env]))
    normalizer.obs_rms.mean[:]=1
    normalizer.obs_rms.var[:]=4
    statistics=tmp_path/'vecnormalize.pkl'
    normalizer.save(statistics)
    digest=sha256(statistics)
    policy=load_frozen_policy('norm',str(path),config,normalization=str(statistics))
    assert policy.provenance['action_scale_verification'].startswith('unknown')
    try:
        policy.bind(env)
        assert not policy.normalizer.training and not policy.normalizer.norm_reward
        before=policy.normalizer.obs_rms.mean.copy(),policy.normalizer.obs_rms.var.copy(),policy.normalizer.obs_rms.count
        policy.predict(np.ones(15)*3)
        np.testing.assert_allclose(model.last_obs, np.ones(15),rtol=1e-7)
        np.testing.assert_array_equal(before[0],policy.normalizer.obs_rms.mean)
        np.testing.assert_array_equal(before[1],policy.normalizer.obs_rms.var)
        assert before[2]==policy.normalizer.obs_rms.count
    finally:
        env.close()
    assert sha256(statistics)==digest


def test_existing_checkpoint_same_model_all_cases(tmp_path,config,monkeypatch):
    if not CHECKPOINT.is_file():
        pytest.skip('explicit local smoke checkpoint unavailable')
    from stable_baselines3 import PPO
    def forbidden(*args,**kwargs):
        raise AssertionError('PPO training is forbidden')
    monkeypatch.setattr(PPO,'learn',forbidden)
    monkeypatch.setattr(PPO,'train',forbidden)
    before=sha256(CHECKPOINT)
    # Full 8s step runs, strict early-termination circle; no invented success.
    code=main(['--config',str(CONFIG),'--model',f'a={CHECKPOINT}', '--model',f'b={CHECKPOINT}',
               '--output-dir',str(tmp_path)])
    assert code==0
    directory=next(tmp_path.glob('dr-transfer-*'))
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['initial_snapshots_equal'] and manifest['reference_sequences_equal']
    assert manifest['cases']['circle']['policy_steps']==2600
    for name in ('step-005','step-050','circle'):
        assert (directory/f'{name}-a.csv').read_text().replace(',a,',',LABEL,') == (directory/f'{name}-b.csv').read_text().replace(',b,',',LABEL,')
        assert (directory/f'{name}-comparison.png').is_file()
        assert manifest['initial_snapshots'][name]['a']==manifest['initial_snapshots'][name]['b']
    results=json.loads((directory/'summary.json').read_text())
    for a,b in zip(results[::2],results[1::2]):
        assert {k:v for k,v in a.items() if k not in ('label','display_name')}=={k:v for k,v in b.items() if k not in ('label','display_name')}
    circle=results[-1]
    assert circle['terminated'] and circle['partial']
    assert circle['termination_reasons']==['min_altitude']
    assert circle['circle_phase']['position_rmse_total'] is None
    assert sha256(CHECKPOINT)==before
    marker=tmp_path/'untouched.txt'
    marker.write_text('keep')
    # Dry-run must not create a run directory or overwrite any output.
    assert main(['--config',str(CONFIG),'--model',f'a={CHECKPOINT}','--model',f'b={CHECKPOINT}',
                 '--dry-run','--output-dir',str(tmp_path)])==0
    assert len(list(tmp_path.glob('dr-transfer-*')))==1
    assert marker.read_text()=='keep'


def test_first_predict_receives_new_observation_and_clock_alignment(config):
    class Recorder:
        def bind(self, env):
            self.env=env
            self.first=None
        def predict(self, obs):
            if self.first is None:
                self.first=obs.copy()
            # Deliberately end the test at the first transition. No PPO results
            # are inferred from this unit-test double.
            self.env.max_steps=1
            return np.zeros(4)
    for case in make_cases(config,['step-005','step-050','circle','circle-air']):
        policy=Recorder()
        rows,snapshot,error,_=run_case(config,case,policy,42)
        assert error is None
        np.testing.assert_array_equal(policy.first,np.array(snapshot['observation'],np.float32))
        np.testing.assert_array_equal(policy.first[:3],
                                      (np.array(snapshot['position'])-case.reference(0)[0]).astype(np.float32))
        assert rows[0]['time']==0 and rows[0]['time_post']==.01
        np.testing.assert_array_equal(rows[0]['reference'],case.reference(0)[0])
        np.testing.assert_array_equal(rows[0]['reference_post'],case.reference(.01)[0])
        if case.name=='circle':
            assert not np.array_equal(rows[0]['reference'],rows[0]['reference_post'])


def test_checkpoint_input_bounds_and_shape_rejected(config,tmp_path,monkeypatch):
    import gymnasium as gym
    from crazyflie_rl import dr_policy
    path=fake_zip(tmp_path,{})
    stub=SimpleNamespace(action_space=gym.spaces.Box(-2,2,(4,),dtype=np.float32))
    monkeypatch.setattr(dr_policy,'_load_policy',lambda *args:stub)
    with pytest.raises(ValueError,match='action bounds'):
        load_frozen_policy('wrong',str(path),config,normalization='none')


def test_new_entrypoint_import_has_no_runtime_imports():
    import subprocess,sys
    code="import compare_dr_policies,sys; assert 'mujoco' not in sys.modules; assert 'stable_baselines3' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_circle_air_extracts_original_ramp_and_hold(config):
    original, air = make_cases(config, ['circle', 'circle-air'])
    offset = original.mission.boundaries.settle2_end
    assert air.reference_time_offset_sec == offset == 12
    assert air.horizon == original.horizon - offset == 14
    sequence = reference_sequence(air, .01)
    assert len(sequence['time']) == 1401
    assert set(sequence['phase']) == {'CIRCLE', 'HOLD'}
    for t, ref, phase in zip(sequence['time'], sequence['reference'], sequence['phase']):
        expected, expected_phase = original.reference(t + offset)
        np.testing.assert_array_equal(ref, expected)
        assert phase == expected_phase
    np.testing.assert_array_equal(air.initial_position(), [1, 0, 1])
    assert air.description(.01)['phase_boundaries'] == {'circle_end': 12, 'total': 14}
    # The reused ramp starts from rest; normal circular speed is ~0.628 m/s.
    assert np.linalg.norm(air.reference(.001)[0] - air.reference(0)[0]) / .001 < .001
    snapshots = []
    for _ in range(2):
        env = CrazyflieResidualEnv(config=config, episode_sec=air.horizon)
        try:
            adapter = EvaluationAdapter(env)
            obs = adapter.reset_to_case_initial_state(air, 42)
            np.testing.assert_array_equal(obs[:3], 0)
            np.testing.assert_array_equal(env.data.qpos[:3], air.reference(0)[0])
            np.testing.assert_array_equal(env.data.qpos[3:7], [1, 0, 0, 0])
            np.testing.assert_array_equal(env.data.qvel, 0)
            assert env.min_altitude < env.data.qpos[2] < env.max_altitude
            assert env.max_steps == 1400
            assert env._last_f.sum() == pytest.approx(config.vehicle.mass * config.vehicle.gravity, rel=1e-5)
            snapshots.append(adapter.snapshot())
            adapter.set_reference(air.reference(1)[0])
            adapter.current_observation()
            after = adapter.snapshot()
            for key in snapshots[-1].keys() - {'reference', 'observation'}:
                assert after[key] == snapshots[-1][key]
        finally:
            env.close()
    assert snapshots[0] == snapshots[1]
    empty = summarize([], air, Thresholds())
    assert empty['partial'] and empty['circle_phase']['position_rmse_total'] is None
    assert empty['circle_phase']['altitude_error_max_abs_m'] is None
    assert 'settling_time_s' not in empty and 'overshoot_m' not in empty


def test_circle_air_same_checkpoint_reproducible(tmp_path, monkeypatch):
    if not CHECKPOINT.is_file():
        pytest.skip('explicit local smoke checkpoint unavailable')
    from stable_baselines3 import PPO
    def forbidden(*args, **kwargs):
        raise AssertionError('PPO training is forbidden')
    monkeypatch.setattr(PPO, 'learn', forbidden)
    monkeypatch.setattr(PPO, 'train', forbidden)
    before = sha256(CHECKPOINT)
    assert main(['--config',str(CONFIG),'--model',f'a={CHECKPOINT}', '--model',f'b={CHECKPOINT}',
                 '--cases','circle-air','--output-dir',str(tmp_path)]) == 0
    directory = next(tmp_path.glob('dr-transfer-*'))
    assert (directory/'circle-air-a.csv').read_text().replace(',a,',',LABEL,') == (directory/'circle-air-b.csv').read_text().replace(',b,',',LABEL,')
    manifest = json.loads((directory/'manifest.json').read_text())
    assert manifest['initial_snapshots_equal'] and manifest['reference_sequences_equal']
    assert manifest['cases']['circle-air']['policy_steps'] == 1400
    assert sha256(CHECKPOINT) == before


def test_altitude_metrics_use_signed_same_time_errors():
    from crazyflie_rl.dr_transfer import altitude_metrics
    positions = np.array([[0, 0, .9], [0, 0, 1.2]])
    errors = positions - [0, 0, 1]
    metrics = altitude_metrics(positions, errors)
    assert metrics['altitude_min_m'] == .9
    assert metrics['altitude_max_m'] == 1.2
    assert metrics['altitude_error_mean_m'] == pytest.approx(.05)
    assert metrics['altitude_error_min_m'] == pytest.approx(-.1)
    assert metrics['altitude_error_max_abs_m'] == pytest.approx(.2)

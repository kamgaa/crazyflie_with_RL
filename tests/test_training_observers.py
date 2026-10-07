from pathlib import Path
from types import SimpleNamespace
import csv

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.environment import CrazyflieResidualEnv
from crazyflie_rl.training_observers import EpisodeCSVRecorder, make_periodic_checkpoint_callback

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT/'configs/e2e_train_velocity_ab_a.yaml')


def test_episode_recorder_preserves_sequence_and_time_limit(tmp_path):
    bare = CrazyflieResidualEnv(config=config(), episode_sec=.03)
    wrapped = EpisodeCSVRecorder(CrazyflieResidualEnv(config=config(), episode_sec=.03), tmp_path/'episodes.csv')
    try:
        np.testing.assert_array_equal(bare.reset(seed=42)[0], wrapped.reset(seed=42)[0])
        for _ in range(3):
            x, y = bare.step(np.zeros(4)), wrapped.step(np.zeros(4))
            for a,b in zip(x[:4],y[:4]): np.testing.assert_array_equal(a,b)
        assert y[4]['episode']['l'] == 3 and y[4]['episode_end_reason'] == 'time_limit'
        wrapped.reset(seed=43)
        wrapped.step(np.zeros(4))
    finally:
        bare.close(); wrapped.close()
    with (tmp_path/'episodes.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==2
    assert rows[0]['terminated']=='False' and rows[0]['truncated']=='True'
    assert rows[0]['episode_ended']=='True' and float(rows[0]['duration_sec'])==pytest.approx(.03)
    assert rows[1]['episode_ended']=='False' and rows[1]['end_reason']=='collector_closed'
    assert sum(int(r['length_steps']) for r in rows)==4
    assert sum(float(v) for k,v in rows[0].items() if k.startswith('reward_sum_'))==pytest.approx(float(rows[0]['return']))


def test_recorder_reads_terminal_state_before_vec_auto_reset(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv
    base=CrazyflieResidualEnv(config=config())
    vec=DummyVecEnv([lambda: EpisodeCSVRecorder(base,tmp_path/'episodes.csv')])
    try:
        vec.reset()
        base.data.qpos[2]=.1
        _,_,done,infos=vec.step(np.zeros((1,4)))
        assert done[0]
        assert 'min_altitude' in infos[0]['episode_end_reason']
        assert base.data.qpos[2]>.9  # already reset; CSV must contain the old terminal state.
    finally: vec.close()
    with (tmp_path/'episodes.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['physical_termination']=='True'
    assert rows[0]['time_limit_reached']=='False'
    assert float(rows[0]['final_position_error_m'])>.8


def test_periodic_checkpoints_independent_of_best_and_after_updates():
    class Artifacts:
        def __init__(self): self.saved=[]; self.metrics=[]
        def save_model(self, model, kind, **kw): self.saved.append((kind,kw))
        def write_metrics(self, kind, value): self.metrics.append((kind,value))
    artifacts=Artifacts()
    callback=make_periodic_checkpoint_callback(artifacts,100000)
    callback.model=SimpleNamespace(policy=SimpleNamespace(state_dict=lambda:{}),seed=42,get_vec_normalize_env=lambda:None)
    callback._on_training_start()
    for step in [0,98304,100352,102400,198656,200704]:
        callback.num_timesteps=step; callback._on_rollout_start()
    callback.num_timesteps=1001472; callback._on_training_end()
    assert [x[1]['timestep'] for x in artifacts.saved]==[100352,200704,1001472]
    assert all(kind=='intermediate' and kw['metadata']['optimizer_update_completed'] for kind,kw in artifacts.saved)
    assert artifacts.metrics[0][1]['normalization_enabled'] is False


def test_intermediate_save_keeps_best_final_and_normalization(tmp_path):
    from dataclasses import replace
    from crazyflie_rl.artifacts import ArtifactManager
    from crazyflie_rl.dr_policy import training_manifest
    c=config();c=replace(c,paths=replace(c.paths,artifact_root=tmp_path))
    manager=ArtifactManager.create(c)
    class Model:
        def save(self,path): Path(path+'.zip').write_bytes(b'test archive')
        def get_vec_normalize_env(self): return None
    path=manager.save_model(Model(),'intermediate',timestep=100352)
    assert 'step0100352' in path.name
    assert manager.manifest['models']=={'best':None,'final':None}
    _,metadata=training_manifest(path)
    assert metadata['normalization']=={'enabled':False}
    assert metadata['checkpoint_sha256']==manager.manifest['model_history'][0]['sha256']

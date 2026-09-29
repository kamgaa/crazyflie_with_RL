"""4D post-processing, episode-mass provenance and action-bias semantics."""
from dataclasses import fields, replace
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from crazyflie_rl.config import load_config
from crazyflie_rl.eval_cli import RolloutTrace, _cached_episode_mass
from crazyflie_rl.wrench_authority import (
    AXES, analyze_wrench_authority, build_wrench_authority_report,
    format_wrench_authority, save_wrench_authority_plot,
)
from crazyflie_rl.yaw_authority import analyze_yaw_authority

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT/'configs/view_live_circle_eval.yaml')


def trace(policy='residual', mode='e2e', mass=.05):
    c = config()
    action = np.array([[.25, -.5, .75, -.9], [.5, -.75, .9, -.95],
                       [.75, -.9, .95, -.99], [1., -1., 1., -1.]])
    command = action * np.asarray(c.environment.residual_scale)
    command[:,3] += c.vehicle.mass * c.vehicle.gravity
    actual = command * .9
    return RolloutTrace(
        policy=policy,label=policy,control_mode=mode,time_sec=np.arange(4)*.01,
        position=np.zeros((4,3)),reference_position=np.zeros((4,3)),
        position_error=np.zeros(4),attitude_deg=np.zeros((4,3)),
        phases=('GOTO','CIRCLE','CIRCLE','HOLD'),
        training_boundary_crossed_at=None,guard_boundary_crossed_at=None,
        terminated_at=None,truncated_at=None,diverged_at=None,
        control_input=action,wrench_command=command,wrench_actual=actual,
        allocation_error=command-actual,episode_mass_kg=mass,
    )


def test_extraction_hover_centering_physical_conversion_and_tracking():
    c,t=config(),trace()
    m=analyze_wrench_authority(t,c)
    assert m['hover_force_n']==.05*c.vehicle.gravity
    assert m['e2e_command_bias_force_n']==c.vehicle.mass*c.vehicle.gravity
    for i,axis in enumerate(AXES):
        a=m['overall']['axes'][axis]
        effort=t.wrench_command[:,i]-(.05*c.vehicle.gravity if i==3 else 0)
        assert a['command_effort']['abs_p99']==pytest.approx(np.percentile(abs(effort),99))
        actual=t.wrench_actual[:,i]-(.05*c.vehicle.gravity if i==3 else 0)
        assert a['actual_effort']['abs_p95']==pytest.approx(np.percentile(abs(actual),95))
        assert a['normalized_action']['u_abs_p99']==np.percentile(abs(t.control_input[:,i]),99)
        physical=t.control_input[:,i]*c.environment.residual_scale[i]
        assert a['physical_action_contribution']['physical_abs_p99']==np.percentile(abs(physical),99)
        tracking=a['absolute_wrench_tracking']
        assert tracking['command_rms']==pytest.approx(np.sqrt(np.mean(t.wrench_command[:,i]**2)))
        assert tracking['tracking_error_rms']==pytest.approx(np.sqrt(np.mean(t.allocation_error[:,i]**2)))
        assert a['e2e_command_mapping_error_abs_max']==0
    # A loaded episode need NOT satisfy command = episode_hover + scale*action.
    assert m['hover_force_n'] != m['e2e_command_bias_force_n']


def test_cached_mass_missing_metadata_and_no_model_access():
    class CachedEnvironment:
        _m0=.04
        _com_mw=.0123
        @property
        def model(self):
            raise AssertionError('must not access simulator')
    assert _cached_episode_mass(CachedEnvironment())==.0523
    assert _cached_episode_mass(SimpleNamespace()) is None
    m=analyze_wrench_authority(trace(mass=None),config())
    assert m['hover_force_n'] is None
    assert m['overall']['axes']['delta_fz']['command_effort']['abs_p99'] is None
    assert m['overall']['axes']['delta_fz']['absolute_wrench_tracking']['command_abs_p99'] is not None
    assert m['overall']['axes']['delta_fz']['physical_action_contribution']['physical_abs_p99'] is not None
    json.dumps(m,allow_nan=False)


def test_ratios_candidates_exploration_and_pid_action_exclusion():
    c=config()
    c=replace(c,environment=replace(c.environment,residual_scale=(.1,.2,.003,.4)),
              training=replace(c.training,ppo=replace(c.training.ppo,log_std_init=-2.)))
    pid,ppo=trace('floor','residual'),trace()
    report=build_wrench_authority_report([pid,ppo],c,model='checkpoint.zip')
    assert list(report['configured_action_scale'].values())==list(c.environment.residual_scale)
    sigma=np.exp(-2.)
    assert report['initial_exploration_sigma_normalized']==sigma
    for i,axis in enumerate(AXES):
        p99=report['floor']['overall']['axes'][axis]['command_effort']['abs_p99']
        p95=report['floor']['overall']['axes'][axis]['command_effort']['abs_p95']
        scale=c.environment.residual_scale[i]
        r=report['scale_analysis']['overall'][axis]
        assert r['pid_p99_over_action_scale']==pytest.approx(p99/scale)
        assert r['pid_p95_over_action_scale']==pytest.approx(p95/scale)
        for usage,key in [(.1,'0p1'),(.2,'0p2'),(.3,'0p3')]:
            assert r['scale_if_pid_p99_maps_to_target_usage']['target_usage_'+key]==pytest.approx(p99/usage)
        assert report['initial_exploration_sigma_physical'][axis]==pytest.approx(scale*sigma)
        assert r['initial_sigma_over_pid_p99']==pytest.approx(scale*sigma/p99)
        physical=np.percentile(abs(ppo.control_input[:,i]*scale),99)
        assert r['ppo_physical_p99_over_pid_p99']==pytest.approx(physical/p99)
        assert report['floor']['overall']['axes'][axis]['normalized_action']['u_rms'] is None
    assert 'CIRCLE' in format_wrench_authority(report)
    assert format_wrench_authority(json.loads(json.dumps(report, sort_keys=True))) == format_wrench_authority(report)
    json.dumps(report,allow_nan=False)


def test_boundaries_phase_segmentation_and_tracking_error_reuse():
    t=trace()
    t.control_input[:]=np.array([.25,.5,.75,1.])[:,None]
    m=analyze_wrench_authority(t,config())
    assert set(m['phases'])=={'GOTO','CIRCLE','HOLD'}
    assert m['phases']['CIRCLE']['sample_count']==2
    for a in m['overall']['axes'].values():
        for suffix,expected in [('0p25',.75),('0p5',.5),('0p75',.25),('0p9',.25),('0p95',.25),('0p99',.25)]:
            assert a['normalized_action']['fraction_abs_gt_'+suffix]==expected
    t.allocation_error[:]=.123
    m=analyze_wrench_authority(t,config())
    assert all(a['absolute_wrench_tracking']['tracking_error_rms']==pytest.approx(.123) for a in m['overall']['axes'].values())
    m=analyze_wrench_authority(replace(t,allocation_error=None),config())
    assert m['overall']['axes']['tau_x']['absolute_wrench_tracking']['tracking_error_rms']!=pytest.approx(.123)


def test_guards_and_json_safe():
    c=config()
    pid=trace('floor','residual',mass=c.vehicle.mass)
    pid.wrench_command[:]=0
    pid.wrench_command[:,3]=c.vehicle.mass*c.vehicle.gravity
    pid.wrench_actual[:]=pid.wrench_command
    pid.allocation_error[:]=0
    pid.control_input[:]=0
    report=build_wrench_authority_report([pid],c)
    for r in report['scale_analysis']['overall'].values():
        assert r['initial_sigma_over_pid_p99'] is None
        assert r['ppo_physical_p99_over_pid_p99'] is None
    assert all(m['absolute_wrench_tracking']['command_actual_correlation'] is None
               for m in report['floor']['overall']['axes'].values())
    c=replace(c,environment=replace(c.environment,residual_scale=(0,0,0,0)))
    r=build_wrench_authority_report([pid],c)
    assert all(a['pid_p99_over_action_scale'] is None for a in r['scale_analysis']['overall'].values())
    pid=replace(pid,wrench_actual=None,allocation_error=None)
    json.dumps(build_wrench_authority_report([pid],c),allow_nan=False)
    empty=replace(pid,time_sec=np.array([]),phases=(),control_input=np.empty((0,4)),
                  wrench_command=np.empty((0,4)),wrench_actual=np.empty((0,4)))
    json.dumps(analyze_wrench_authority(empty,c),allow_nan=False)


@pytest.mark.parametrize('policy,mode',[('floor','residual'),('residual','e2e')])
def test_yaw_consistency_and_readonly_rng_preservation(policy,mode):
    c,t=config(),trace(policy,mode)
    saved={f.name:getattr(t,f.name).copy() for f in fields(t) if isinstance(getattr(t,f.name),np.ndarray)}
    for name in saved:
        getattr(t,name).setflags(write=False)
    py,npstate=random.getstate(),np.random.get_state()
    yaw=analyze_yaw_authority(t,c)
    wrench=analyze_wrench_authority(t,c)
    for scope in ['overall',*t.phases]:
        y=yaw['overall'] if scope=='overall' else yaw['phases'][scope]
        w=(wrench['overall'] if scope=='overall' else wrench['phases'][scope])['axes']['tau_z']
        for p in [95,99]:
            assert w['command_effort'][f'abs_p{p}']==y[f'tau_z_cmd_abs_p{p}_nm']
            assert w['normalized_action'][f'u_abs_p{p}']==y[f'u_tau_z_abs_p{p}']
        assert w['absolute_wrench_tracking']['command_actual_correlation']==y['tau_z_actual_vs_cmd_correlation']
        assert w['normalized_action']['fraction_abs_gt_0p9']==y['u_tau_z_fraction_gt_0p9']
    assert random.getstate()==py
    assert all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),npstate))
    for name,value in saved.items():
        assert np.array_equal(getattr(t,name),value)


@pytest.mark.parametrize('payload,randomized',[(0.,False),(.01,False),(.01,True)])
def test_actual_episode_mass_and_e2e_bias_consistency(payload,randomized):
    pytest.importorskip('mujoco')
    from crazyflie_rl.environment import CrazyflieResidualEnv
    from crazyflie_rl.eval_cli import EvaluationRunner
    from crazyflie_rl.missions import mission_from_experiment
    c=config()
    if not Path(c.paths.mujoco_xml).is_file():
        pytest.skip('MuJoCo XML unavailable')
    c=replace(c,environment=replace(c.environment,payload=replace(c.environment.payload,mass=payload,randomize=randomized)))
    c=replace(c,mission=replace(c.mission,takeoff_sec=.1,settle_sec=.1,goto_sec=.1,post_hold_sec=.1,
                               circle=replace(c.mission.circle,period=.2,laps=1,ramp_sec=0)))
    mission=mission_from_experiment(c,legacy_circle_preset=False)
    captured=[]
    class Factory:
        def make(self,**kwargs):
            env=CrazyflieResidualEnv(config=c,seed=1000,**kwargs)
            captured.append(env)
            return env
    class Policy:
        def predict(self,observation,deterministic):
            return np.array([.1,-.2,.3,-.4],dtype=np.float32),None
    t=EvaluationRunner(c,None,headless=True,realtime=False,camera_tracking=False,
                       mission=mission,environment_factory=Factory()).run(Policy(),'residual','PPO')
    env=captured[0]
    # Simulator read exists only in this test oracle.
    assert t.episode_mass_kg==float(env.model.body_mass[env.drone_bid])
    assert t.episode_mass_kg==env._m0+env._com_mw
    expected=t.control_input*np.array(c.environment.residual_scale)
    expected[:,3]+=c.vehicle.mass*c.vehicle.gravity
    assert np.array_equal(t.wrench_command,expected)
    result=analyze_wrench_authority(t,c)
    assert result['hover_force_n']==t.episode_mass_kg*c.vehicle.gravity
    if payload==0:
        assert result['hover_force_n']==result['e2e_command_bias_force_n']
    else:
        assert result['hover_force_n']!=result['e2e_command_bias_force_n']


def test_plot_separates_force_and_torque_and_refuses_overwrite(tmp_path):
    path=save_wrench_authority_plot(tmp_path/'wrench.png',[trace('floor','residual'),trace()],config())
    assert path.read_bytes().startswith(b'\x89PNG')
    with pytest.raises(FileExistsError):
        save_wrench_authority_plot(path,[trace()],config())

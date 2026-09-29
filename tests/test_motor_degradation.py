from dataclasses import replace
import json
import numpy as np
import pytest
from crazyflie_rl.config import load_config
from crazyflie_rl.environment import CrazyflieResidualEnv
from crazyflie_rl.motor_degradation import Settings, DiagnosticEnv, experiment_config, rollout, summarize, recovery_time


def setup():
    s=Settings(activation_sec=.1,disturbance_start=.3,disturbance_duration=.1,total_sec=.8,hold_sec=.1,pre_window_sec=.1)
    return experiment_config(load_config('configs/base.yaml'),s),s


@pytest.mark.parametrize('condition,effectiveness',[('BASE',.7),('B',1.)])
def test_existing_pid_exact(condition,effectiveness):
    c,s=setup(); s=replace(s,effectiveness=effectiveness)
    a=CrazyflieResidualEnv(config=c); b=DiagnosticEnv(c,s,condition)
    try:
        np.testing.assert_array_equal(a.reset(seed=42)[0],b.reset(seed=42)[0])
        for _ in range(80):
            ra=a.step(np.zeros(4)); rb=b.step(np.zeros(4))
            np.testing.assert_array_equal(ra[0],rb[0]); assert ra[1:4]==rb[1:4]
            for field in ('qpos','qvel','ctrl'): np.testing.assert_array_equal(getattr(a.data,field),getattr(b.data,field))
            for field in ('_last_f','_last_omega','_last_f_cmd','_last_motor_cmd','_last_q_actual','_last_wrench_actual','B','B_pinv'):
                np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    finally: a.close(); b.close()


def test_effectiveness_reaction_blindness():
    c,s=setup(); env=DiagnosticEnv(c,replace(s,activation_sec=0.),'B'); normal=CrazyflieResidualEnv(config=c)
    try:
        env.reset(seed=2); normal.reset(seed=2)
        before={k:np.asarray(v).copy() for k,v in vars(env.pid).items()}
        wrench=np.array([.0001,0.,0.,c.vehicle.mass*c.vehicle.gravity])
        env._apply_control(wrench); normal._apply_control(wrench)
        np.testing.assert_array_equal(env.B,normal.B)
        np.testing.assert_array_equal(env._last_f[1:],normal._last_f[1:])
        assert env._last_f[0]==normal._last_f[0]*.7
        np.testing.assert_allclose(env._last_q_actual,env.motor_direction*c.actuator.reaction_torque.legacy_ratio_m*env._last_f,rtol=1e-15)
        np.testing.assert_array_equal(env._actuator.last_f,normal._actuator.last_f)
        for k,v in before.items(): np.testing.assert_array_equal(v,getattr(env.pid,k))
    finally: env.close(); normal.close()


@pytest.mark.parametrize('axis',['roll','-roll','pitch','-pitch','yaw','-yaw'])
def test_timing_determinism_metrics(axis):
    c,s=setup(); s=replace(s,disturbance_axis=axis)
    x,end=rollout(c,s,'C'); y,other=rollout(c,s,'C')
    for k in x: np.testing.assert_array_equal(x[k],y[k])
    mask=(x['time']>=s.disturbance_start)&(x['time']<s.disturbance_start+s.disturbance_duration)
    assert np.count_nonzero(mask)==50
    assert np.all(np.linalg.norm(x['disturbance_body_nm'][mask],axis=1)==s.disturbance_torque)
    assert not np.any(x['disturbance_body_nm'][~mask])
    result=summarize(x,c,s,'C',end); json.dumps(result,allow_nan=False)
    np.testing.assert_allclose(x['allocation_error'],x['wrench_command']-x['wrench_actual'])
    np.testing.assert_allclose(x['effective_upper_margin_n'],x['effectiveness']*c.vehicle.thrust_max-x['actual_thrust'])
    assert end==other


def test_recovery_hold_and_failure():
    t=np.arange(10)*.1
    assert recovery_time(t,np.zeros(10),.05,.2,.3,.1)==0.
    assert recovery_time(t,np.ones(10),.05,.2,.3,.1) is None


@pytest.mark.parametrize('kwargs',[{'effectiveness':0},{'effectiveness':1.1},{'motor_index':4},{'disturbance_torque':float('nan')},{'hold_sec':0}])
def test_validation(kwargs):
    with pytest.raises(ValueError): Settings(**kwargs)


def test_standalone_config_and_cli_summary(tmp_path, monkeypatch, capsys):
    import sys
    import yaml
    from pathlib import Path
    import diag_motor_degradation as cli
    path=Path('configs/diagnostics/pid_motor_degradation_diagnostic.yaml')
    values=yaml.safe_load(path.read_text())
    base=load_config(path.parent/values.pop('experiment_config'))
    s=Settings(**values)
    assert experiment_config(base,s).environment.control_mode=='residual'
    # Short non-learning integration of the actual CLI/artifact path.
    monkeypatch.setattr(sys,'argv',['diag_motor_degradation.py','--output-root',str(tmp_path),
        '--total-sec','.8','--activation-sec','.1','--disturbance-start','.3',
        '--disturbance-duration','.1','--hold-sec','.1','--pre-window-sec','.1'])
    monkeypatch.setattr(cli,'plot_trace',lambda *args: [])
    cli.main()
    root=next(tmp_path.iterdir())
    result=json.loads((root/'metrics/comparison.json').read_text())
    assert set(result)=={'BASE','A','B','C'}
    assert all(r['termination']['initial']==result['BASE']['termination']['initial'] for r in result.values())
    assert len(list((root/'traces').glob('*.npz')))==4
    assert 'Pre effective headroom' in capsys.readouterr().out

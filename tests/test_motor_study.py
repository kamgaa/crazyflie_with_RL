from dataclasses import replace
import numpy as np
import pytest
from crazyflie_rl import motor_study as m
from crazyflie_rl.config import load_config


def synthetic(n=100,dt=.1):
    x={key:np.zeros((n,3)) for key in ('position_error','velocity','angular_velocity','attitude_deg','i_velocity','i_rate')}
    x.update(time=np.arange(n)*dt,attitude_error_rad=np.zeros(n))
    for key in ('raw_upper_margin_normalized','raw_thrust','raw_upper_margin_n','effective_upper_margin_n','motor_command','clipped_thrust','actual_thrust'):
        x[key]=np.full((n,4),.1)
    return x


def test_settling_window_completion_and_nonsettled():
    s=m.Study(); x=synthetic()
    assert m.settling_time(x,s,.1)==pytest.approx(4.)
    x['position_error'][:,0]=1
    assert m.settling_time(x,s,.1) is None
    x['position_error'][:]=0; x['velocity'][:,0]=1
    assert m.settling_time(x,s,.1) is None
    assert m.settling_time(synthetic(10),s,.1) is None


def test_joint_recovery_requires_velocity_and_contiguous_hold():
    s=m.Study(); x=synthetic()
    x['velocity'][:25,0]=1.
    assert m.joint_recovery_time(x,s,2.,.1)==pytest.approx(.5)
    x['attitude_error_rad'][:]=.5
    assert m.joint_recovery_time(x,s,2.,.1) is None


def test_integrals_and_pre_metrics():
    c=load_config('configs/base.yaml'); x=synthetic(100,1/c.vehicle.physics_hz)
    x['position_error'][:,0]=.02; x['attitude_error_rad'][:]=.01; x['angular_velocity'][:,0]=.3
    s=m.Study(); end=dict(time=.2,terminated=False)
    r=m.response(x,c,s,.1,end)
    assert r['integrated_position_error_m2_s']==pytest.approx(.02**2*.1)
    assert r['integrated_attitude_error_rad2_s']==pytest.approx(.01**2*.1)
    assert r['integrated_angular_rate_rad2_s']==pytest.approx(.3**2*.1)
    assert r['pre']['position_error_rms_m']==pytest.approx(.02)


def row(peak=0.,rec=0.,sat=0.,terminated=False):
    return dict(peak_position_error_m=peak,peak_attitude_error_deg=0.,recovery_time_sec=rec,
        pulse_start_sec=6.,termination=dict(time=14.1,terminated=terminated),post=dict(any_motor_saturation_fraction=sat))


def test_critical_per_criterion_and_censoring():
    s=m.Study()
    blocks=[dict(amplitude_nm=a,conditions={k:row() for k in ('N+','N-','D+','D-')}) for a in (.0001,.0002)]
    blocks[0]['conditions']['D+']=row(peak=.11)
    blocks[1]['conditions']['N+']=row(sat=.01)
    levels,ratios=m.critical_levels(blocks,s)
    assert levels['D+']['per_criterion']['position']==.0001
    assert levels['N+']['per_criterion']['saturation']==.0002
    assert ratios['+']['value']==.5
    assert ratios['-']==dict(value=None,status='lower_bound_only')
    assert m.critical_events(row(rec=None),s)['recovery']
    assert m.critical_events(row(terminated=True),s)['termination']
    early=row(rec=None); early['termination']['time']=6.2
    assert not m.critical_events(early,s)['recovery']


def test_common_timing_selection_and_asymmetry():
    sweep=dict(candidate_lambda_star=.7,rows=[dict(effectiveness=1.,eligible=True,settling_time_sec=4.),dict(effectiveness=.7,eligible=True,settling_time_sec=9.)])
    s=m.Study()
    assert m.select_lambda(sweep)==.7
    assert m.pulse_times(sweep,.7,s,.002)==dict(N=6.,D=11.)
    assert m.pulse_times(sweep,.7,replace(s,timing_mode='common'),.002)==dict(N=11.,D=11.)
    assert m.asymmetry(0.,0.)['normalized_difference'] is None
    assert m.asymmetry(1.,3.)['normalized_difference']==1.
    with pytest.raises(ValueError): m.select_lambda(sweep,.6)


def test_each_pulse_has_new_env_seed_and_sign(tmp_path,monkeypatch):
    from crazyflie_rl import motor_degradation as legacy
    original=legacy.DiagnosticEnv
    created=[]
    class CountEnv(original):
        def __init__(self,*args):
            super().__init__(*args); created.append(self)
    monkeypatch.setattr(legacy,'DiagnosticEnv',CountEnv)
    s=m.Study(activation_sec=.1,duration=2.,steady_state_window_sec=.5,settle_hold_sec=.1,
        settle_buffer_sec=.1,pre_window_sec=.1,post_pulse_sec=.4,critical_recovery_sec=.2,recovery_hold_sec=.1)
    (tmp_path/'traces').mkdir()
    c=load_config('configs/base.yaml')
    r=m.four_pulses(c,s,tmp_path,.7,dict(N=.3,D=.3),.0001,'test')
    assert len(created)==4 and len({id(v) for v in created})==4
    assert all(v['termination']['initial']==r['N+']['termination']['initial'] for v in r.values())
    for label in r:
        with np.load(r[label]['trace']) as x:
            applied=x['disturbance_body_nm'][:,0]
            assert np.count_nonzero(applied)==50
            np.testing.assert_array_equal(applied[applied!=0],np.full(50,.0001 if label[1]=='+' else -.0001))
            assert x['time'][-1]==pytest.approx(.798)


def test_staircase_order_and_no_reuse(tmp_path,monkeypatch):
    s=m.Study(); seen=[]
    def fake(base,settings,root,lam,times,amplitude,prefix):
        seen.append((amplitude,prefix,lam))
        return {k:row() for k in ('N+','N-','D+','D-')}
    monkeypatch.setattr(m,'four_pulses',fake)
    (tmp_path/'metrics').mkdir()
    r=m.staircase(None,s,tmp_path,.6,dict(N=6.,D=12.))
    assert [x[0] for x in seen]==list(s.pulse_amplitudes)
    assert len({x[1] for x in seen})==len(seen)
    assert all(x[2]==.6 for x in seen)
    assert r['lambda_star']==.6


@pytest.mark.parametrize('kwargs',[dict(pulse_amplitudes=(.2,.1)),dict(pulse_duration=0),dict(effectiveness_values=(.7,)),dict(settle_hold_sec=float('nan')),dict(critical_motor_saturation=1)])
def test_validation(kwargs):
    with pytest.raises(ValueError): m.Study(**kwargs)

"""Fixed circle-fault experiment, using the existing evaluation/control pipeline.

The new cubic angular-speed ramp is an explicit experiment protocol; legacy
missions are unmodified. Analytic velocity is logged for evaluation only.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import json
from pathlib import Path
import shlex
import shutil
import tempfile
import numpy as np

from .artifacts import _git_metadata
from .config import load_config
from .dr_policy import sha256
from .dr_transfer import ROOT, Case, EvaluationAdapter, run_case, write_rollout, validate_common_config
from .estimated_allocation import EstimatedAllocationEnv
from .estimated_allocation_eval import AllocationObserver, TRACE_KEYS, EXTRA_KEYS, capture_protected_files
from .fault_estimation_eval import nominal_model, offline_verify, write_json, blocks
from .fault_estimator import EstimatorSettings
from .integral_controller import IntegralController
from .integral_eval import parameter_digest, verify_integral_rows, read_columns
from .integral_validation import Event, GAINS
from .interactive_eval import evaluation_config
from .motor_layout import exposed_motor_index
from .oracle_allocation import efficiency_matrix, plant_geometry
from .oracle_eval import flat_csv
from .oracle_recovery import tilt_deg, weighted_dwell
from .payload_motor_eval import DEFAULT_RECORD, select_models

PREVIOUS = ROOT/'artifacts/runs/estimated-allocation-trv1ikys'
SETTINGS = ROOT/'artifacts/runs/motor-estimation-g43e4hbx/estimator_settings.json'
CONFIG = ROOT/'configs/eval_circle_fault_0289.yaml'


@dataclass(frozen=True)
class CircleProtocol:
    period: float

    def __post_init__(self):
        if self.period not in (5., 10.):
            raise ValueError('This fixed experiment supports only periods 5 and 10 seconds')

    @property
    def fault_time(self): return 6.+2*self.period
    @property
    def horizon(self): return 6.+5*self.period

    def phase(self, t):
        omega = 2*np.pi/self.period
        if t <= 5.: return 0., 0.
        if t < 7.:
            s = (t-5.)/2.
            return 2*omega*(s**3-.5*s**4), omega*(3*s*s-2*s**3)
        return omega*(t-6.), omega

    def reference(self, t):
        theta, _ = self.phase(t)
        stage = ('HOVER' if t < 5 else 'RAMP' if t < 7 else
                 'WARMUP' if t < 6+self.period else 'PRE_FAULT_LAP' if t < self.fault_time else
                 f'POST_LAP_{min(3,1+int((t-self.fault_time)/self.period))}')
        return np.array([np.cos(theta),np.sin(theta),1.]), stage

    def reference_velocity(self, t):
        theta, rate = self.phase(t)
        return rate*np.array([-np.sin(theta),np.cos(theta),0.])

    def windows(self):
        f,T = self.fault_time,self.period
        return dict(hover=(0.,5.),entry_warmup=(5.,6+T),prefault_lap=(6+T,f),
                    post_first_2s=(f,f+2),post_lap_1=(f,f+T),post_lap_2=(f+T,f+2*T),
                    post_lap_3=(f+2*T,f+3*T),post_all=(f,self.horizon),last_lap=(f+2*T,f+3*T))

    def parameters(self):
        return dict(period_s=self.period,center_world_m=[0,0,1],radius_m=1.,initial_position=[1,0,1],
            yaw_native_rad=0.,direction='CCW viewed from world +Z',hover_s=5.,ramp_s=2.,
            theta_ramp='2*omega*(s^3 - s^4/2), s=(t-5)/2',
            theta_dot_ramp='omega*(3*s^2-2*s^3)',theta_after_ramp='omega*(t-6)',
            fault_time_s=self.fault_time,horizon_s=self.horizon,windows=self.windows(),
            reference_velocity_use='same-time evaluation only; not fed to PID or PPO',
            radial_error='(p-p_ref) dot [cos(theta),sin(theta),0], positive outward from reference point',
            tangential_error='(p-p_ref) dot [-sin(theta),cos(theta),0], positive along CCW tangent')


@dataclass(frozen=True)
class CircleCase(Case):
    protocol: CircleProtocol | None = None
    def initial_position(self): return self.protocol.reference(0)[0].copy()
    def reference(self,t): return self.protocol.reference(t)
    def reference_velocity(self,t): return self.protocol.reference_velocity(t)
    def description(self,dt):
        return dict(name=self.name,horizon_sec=self.horizon,control_dt=dt,
                    reference=self.protocol.parameters(),initial_velocity=[0,0,0],
                    initial_quaternion=[1,0,0,0],initial_omega=[0,0,0],
                    initialization='existing nominal airborne reset, no per-controller trim')


@dataclass(frozen=True)
class CircleScenario:
    name: str
    fault_time: float
    has_fault: bool
    mass: float = 0.
    offset: tuple = (0.,0.)
    def events(self):
        return (Event(self.fault_time,'fault',1,.7),) if self.has_fault else ()


class PIDOnly:
    """Existing residual-env baseline: zero residual, PID runs each physics step."""
    def bind(self,env):
        if env.mode != 'residual': raise ValueError('PID baseline requires residual environment')
    def predict(self,observation): return np.zeros(4,dtype=np.float32)


DIAGNOSTIC_KEYS = ('desired_wrench','baseline_unclipped_motor_command',
    'allocator_unclipped_motor_command','allocator_clipped_motor_command',
    'allocator_predicted_wrench','static_rotor_wrench_xml','static_rotor_wrench_b0',
    'allocation_residual_b0','geometry_wrench_difference','total_rotor_residual_xml')
CIRCLE_KEYS = ('phase','phase_post','reference_velocity','reference_velocity_post','velocity_error',
    'theta','theta_post','theta_dot','theta_dot_post','radial_error_m','tangential_error_m',
    'policy_raw_observation','observation','policy_input_time','pid_i_velocity','pid_i_rate',
    'physics_wrench_command','physics_static_rotor_wrench_xml','physics_actual_rotor_wrench_xml',
    'physics_allocation_residual_xml','physics_actuator_response_residual_xml',
    'physics_unclipped_motor_command','physics_clipped_motor_command',
    'counterfactual_oracle_unclipped','counterfactual_inverse_component_error_n',
    'reward','motor_sample_time','motor_sample_time_post','integral_enabled') + DIAGNOSTIC_KEYS


class CircleObserver(AllocationObserver):
    fixed_reference = False

    def __init__(self,scenario,controller,model,settings,mode,protocol,is_pid=False):
        super().__init__(scenario,controller,model,settings,mode)
        self.protocol=protocol;self.is_pid=is_pid;self.comparison_marker=None

    def on_reset(self,adapter):
        super().on_reset(adapter)
        self.metadata.update(protocol=self.protocol.parameters(),external_integral_enabled=not self.is_pid,
            controller='existing CascadePID, zero residual' if self.is_pid else 'frozen PPO + I',
            control_rate_hz=1/adapter.control_dt,physics_rate_hz=1/adapter.env.dt_phys,
            pid_call_rate_hz=1/adapter.env.dt_phys if self.is_pid else None,
            nominal_thrust_limit_n=adapter.env.thrust_max,
            B0_allocator=adapter.env.B0, B0_physical_xml=plant_geometry(adapter.env)['matrix'],
            wrench_frame='native FLU body origin; [tau_x,tau_y,tau_z,T], Nm,Nm,Nm,N; positive thrust magnitude')

    def before_step(self,adapter,step,t):
        # Pure reference update before the event snapshot; common loop repeats it
        # without advancing state/history. Events see p_ref(t), not p_ref(t-dt).
        adapter.set_reference(self.protocol.reference(t)[0])
        before_count=len(self.events)
        pid=(adapter.env.pid._i_vel.copy(),adapter.env.pid._i_rate.copy())
        super().before_step(adapter,step,t)
        if step == round(self.protocol.fault_time/adapter.control_dt):
            self.comparison_marker=dict(time=t,theta=self.protocol.phase(t)[0],
                target=adapter.env.pos_des.copy(),snapshot=adapter.snapshot(),xi=self.controller.xi.copy(),
                estimator_updates=self.estimator.update_count,fault_injected=self.condition.has_fault)
        if len(self.events)>before_count:
            np.testing.assert_array_equal(pid[0],adapter.env.pid._i_vel)
            np.testing.assert_array_equal(pid[1],adapter.env.pid._i_rate)
            self.events[-1].update(theta=self.protocol.phase(t)[0],theta_dot=self.protocol.phase(t)[1],
                pid_i_velocity=pid[0],pid_i_rate=pid[1],pid_state_preserved=True)

    def after_step(self,env,row):
        physics=self.physics_rows[self.physics_start:]
        for key in DIAGNOSTIC_KEYS: row[key]=physics[-1][key].copy()
        for source,dest in (
            ('desired_wrench','physics_wrench_command'),
            ('static_rotor_wrench_xml','physics_static_rotor_wrench_xml'),
            ('actual_rotor_wrench_xml','physics_actual_rotor_wrench_xml'),
            ('allocation_residual_xml','physics_allocation_residual_xml'),
            ('actuator_response_residual_xml','physics_actuator_response_residual_xml'),
            ('allocator_unclipped_motor_command','physics_unclipped_motor_command'),
            ('allocator_clipped_motor_command','physics_clipped_motor_command')):
            row[dest]=np.array([p[source] for p in physics])
        row.update(motor_sample_time=physics[-1]['physics_time'],motor_sample_time_post=physics[-1]['physics_time_post'])
        # Same-command diagnostic only, never feedback to any controller.
        truth=env.motor_effectiveness.copy()
        cf=np.linalg.pinv(efficiency_matrix(env.B0,truth))@row['wrench_command']
        expected=(env.B0_pinv@row['wrench_command'])/truth
        row.update(counterfactual_oracle_unclipped=cf,
            counterfactual_inverse_component_error_n=float(np.max(np.abs(cf-expected))))
        super().after_step(env,row)
        th,rate=self.protocol.phase(row['time']);th1,rate1=self.protocol.phase(row['time_post'])
        e=row['position_error_world']
        row.update(theta=th,theta_post=th1,theta_dot=rate,theta_dot_post=rate1,
            radial_error_m=float(e@np.array([np.cos(th1),np.sin(th1),0.])),
            tangential_error_m=float(e@np.array([-np.sin(th1),np.cos(th1),0.])),
            pid_i_velocity=env.pid._i_vel.copy(),pid_i_rate=env.pid._i_rate.copy())
        if self.is_pid:
            assert not row['integral_enabled'] and not np.any(row['xi_next'])
        np.testing.assert_array_equal(row['policy_raw_observation'][3:6],row['observation'][3:6])


def execute(config,policy,protocol,model,settings,mode,fault,*,horizon=None):
    is_pid=isinstance(policy,PIDOnly)
    gain=GAINS[0] if is_pid else GAINS[1]
    cfg=replace(config,environment=replace(config.environment,control_mode='residual')) if is_pid else config
    controller=IntegralController(policy,gain)  # disabled transparent diagnostics for PID
    scenario=CircleScenario('circle_fault' if fault else 'circle_healthy',protocol.fault_time,fault)
    observer=CircleObserver(scenario,controller,model,settings,mode,protocol,is_pid)
    case=CircleCase(f'circle-T{protocol.period:g}',horizon or protocol.horizon,protocol=protocol)
    rows,initial,error,reasons=run_case(cfg,case,controller,42,
        env_factory=partial(EstimatedAllocationEnv,allocation_source=mode),observer=observer,
        observation_transform=controller.prepare_observation)
    if error: raise RuntimeError(error)
    verify_integral_rows(rows,gain)
    return rows,initial,observer,case,reasons


def segment_metrics(rows):
    if not rows:return None
    e=np.array([r['position_error_world'] for r in rows]);ve=np.array([r['velocity_error'] for r in rows])
    att=np.array([r['attitude_deg'] for r in rows]);yaw=np.array([r['yaw_error_rad'] for r in rows])
    dt=np.array([r['time_post']-r['time'] for r in rows]);tilt=tilt_deg([r['quaternion'] for r in rows])
    bits=np.array([r['physics_allocator_clipped'] for r in rows]);subdt=np.repeat(dt/bits.shape[1],bits.shape[1])
    stat=dict(samples=len(rows),observed_start_time=rows[0]['time'],observed_end_time=rows[-1]['time_post'],
        position_rmse_xy=float(np.sqrt(np.mean(np.sum(e[:,:2]**2,axis=1)))),
        position_rmse_z=float(np.sqrt(np.mean(e[:,2]**2))),position_rmse_3d=float(np.sqrt(np.mean(np.sum(e**2,axis=1)))),
        max_xy_error_m=float(np.linalg.norm(e[:,:2],axis=1).max()),max_abs_z_error_m=float(np.abs(e[:,2]).max()),
        velocity_reference_error_rms_xy=float(np.sqrt(np.mean(np.sum(ve[:,:2]**2,axis=1)))),
        velocity_reference_error_rms_z=float(np.sqrt(np.mean(ve[:,2]**2))),
        mean_error_xyz_m=np.mean(e,axis=0),mean_xy_offset_m=float(np.linalg.norm(np.mean(e[:,:2],axis=0))),
        mean_z_error_m=float(e[:,2].mean()),
        roll_rms_deg=float(np.sqrt(np.mean(att[:,0]**2))),pitch_rms_deg=float(np.sqrt(np.mean(att[:,1]**2))),
        tilt_max_deg=float(max(tilt)),yaw_max_abs_deg=float(np.degrees(np.abs(yaw)).max()),
        omega_max_rad_s=float(np.linalg.norm([r['omega'] for r in rows],axis=1).max()),
        radial_error_mean_m=float(np.mean([r['radial_error_m'] for r in rows])),
        radial_error_rms_m=float(np.sqrt(np.mean(np.square([r['radial_error_m'] for r in rows])))),
        tangential_error_mean_m=float(np.mean([r['tangential_error_m'] for r in rows])),
        tangential_error_rms_m=float(np.sqrt(np.mean(np.square([r['tangential_error_m'] for r in rows])))),
        clipping=weighted_dwell(bits.reshape(-1,4),subdt),
        esc_boundary=weighted_dwell(np.array([r['physics_esc_boundary'] for r in rows]).reshape(-1,4),subdt),
        action_boundary=weighted_dwell([r['policy_action_at_bound'] for r in rows],dt),
        integral_freeze=weighted_dwell([r['integral_frozen'] for r in rows],dt))
    for key in ('physics_allocation_residual_xml','physics_actuator_response_residual_xml'):
        residual=np.array([r[key] for r in rows]).reshape(-1,4)
        stat[key+'_rms']=np.sqrt(np.average(residual**2,axis=0,weights=subdt))
    return stat


def estimation_metrics(rows,protocol,fault,mode):
    t=np.array([r['time_post'] for r in rows]);start=np.array([r['time'] for r in rows]);dt=t-start
    post=start>=protocol.fault_time-1e-9
    state=np.array([r['estimator_state'] for r in rows]);motor=np.array([r['estimated_motor'] for r in rows])
    truth=np.array([r['truth_efficiency_user'] for r in rows]);eta=np.array([r['estimated_efficiency_user'] for r in rows])
    healthy=~post if fault else np.ones(len(rows),bool)
    fp=healthy&(state=='fault');mis=post&(state=='fault')&(motor!=1) if fault else np.zeros(len(rows),bool)
    def first(mask,axis=t):return float(axis[np.flatnonzero(mask)[0]]) if mask.any() else None
    detection=first(post&(state=='fault')&(motor==1)) if fault else None
    changed=np.any(np.array([r['allocator_efficiency_user'] for r in rows])!=1,axis=1)
    applied=first(changed,start)
    out=dict(confirmed_detection_time=detection,detection_delay_s=None if detection is None else detection-protocol.fault_time,
        first_compensation_time=applied,first_compensation_delay_s=None if applied is None or not fault else applied-protocol.fault_time,
        false_positive_episodes=blocks(fp),false_positive_seconds=float(np.sum(dt[fp])),
        misidentification_seconds=float(np.sum(dt[mis])),uncertain_seconds=float(np.sum(dt[state=='uncertain'])),
        insufficient_data_seconds=float(np.sum(dt[state=='insufficient_data'])),efficiency_windows={},
        detection_null_reason='not_applicable' if not fault else 'not_confirmed_in_observed_interval' if detection is None else None,
        compensation_null_reason='blind_by_design' if mode=='blind' else 'none_applied' if applied is None else None)
    runtime=np.array([r['estimator_runtime_ms'] for r in rows])
    out['runtime_ms']=dict(median=float(np.median(runtime)),p95=float(np.percentile(runtime,95)),max=float(runtime.max()),over_10ms_count=int(sum(runtime>10)))
    if fault:
        for name,a,b in [('post_first_05s',protocol.fault_time,protocol.fault_time+.5),
                         ('post_first_1s',protocol.fault_time,protocol.fault_time+1),
                         ('post_all',protocol.fault_time,protocol.horizon),('last_lap',protocol.horizon-protocol.period,protocol.horizon)]:
            mask=(t>a+1e-9)&(t<=b+1e-9);er=eta[mask,0]-truth[mask,0]
            value=dict(mae=float(np.mean(np.abs(er))),rmse=float(np.sqrt(np.mean(er**2))),samples=int(sum(mask))) if mask.any() else None
            full=t[-1]>=b-1e-9
            out['efficiency_windows'][name]=dict(full=value if full else None,partial=value if not full else None)
    return out


def metrics(rows,observer,case,reasons):
    duration=rows[-1]['time_post'] if rows else 0.
    completed=bool(rows and not rows[-1]['terminated'] and abs(duration-case.horizon)<1e-8)
    windows={};partials={}
    for name,(a,b) in observer.protocol.windows().items():
        selected=[r for r in rows if a+1e-9<r['time_post']<=b+1e-9]
        stat=segment_metrics(selected);full=duration>=b-1e-9
        windows[name]=stat if full else None;partials[name]=stat if not full else None
    return dict(completed=completed,terminated=bool(rows and rows[-1]['terminated']),
        truncated=bool(rows and rows[-1]['truncated']),duration_s=duration,expected_duration_s=case.horizon,
        termination_reasons=reasons,windows=windows,partial_observed_windows=partials,
        observed_all=segment_metrics(rows),fault_applied=bool(observer.events),
        estimation=estimation_metrics(rows,observer.protocol,observer.condition.has_fault,observer.mode),
        counterfactual_inverse_max_error_n=max(r['counterfactual_inverse_component_error_n'] for r in rows))


def focused_motor_checks(config,model):
    from .actuators import Cf21bFirstOrderActuatorModel
    a=Cf21bFirstOrderActuatorModel(**model.actuator_kwargs)
    env=EstimatedAllocationEnv(config=config,allocation_source='oracle')
    out={};curve=[]
    try:
        EvaluationAdapter(env).reset_to_case_initial_state(Case('hover',1.,(0,0,1)),42)
        assert env.thrust_max==model.actuator_kwargs['thrust_max']==.289
        np.testing.assert_array_equal(env.actuator_model.time_constant_s,a.time_constant_s)
        np.testing.assert_array_equal(a.time_constant_s,np.full(4,.05))
        errors=[]
        for command in np.linspace(0,.289,101):
            omega=a.inverse_thrust(command);u=np.clip(omega/a.steady_state_gain_rad_s,0,1)
            thrust=a.thrust_from_omega(u*a.steady_state_gain_rad_s)
            errors.append(float(max(abs(thrust-command))))
            curve.append(dict(command_n=command,esc=u[0],omega_rad_s=omega[0],steady_nominal_n=thrust[0]))
        assert max(errors)<1e-12
        old=env.B_pinv.copy();env.sync_allocator();np.testing.assert_array_equal(old,env.B_pinv)
        assert exposed_motor_index(1,'user_frd')==3
        eta=np.ones(4);eta[3]=.7;env.motor_effectiveness=eta;env.sync_allocator()
        raw=np.array([.09,.10,.11,.12]);w=env.B0@raw
        expected=raw/eta;actual=env.B_pinv@w
        np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-14)
        residual=env.allocator_matrix@actual-w
        assert max(abs(residual))<1e-14 and max(actual)<.289
        for _ in range(1000):env._apply_control(w)
        np.testing.assert_allclose(env._last_f,eta*env.nominal_thrust,rtol=0,atol=0)
        np.testing.assert_allclose(env._last_q_actual,eta*env.nominal_reaction_torque,rtol=0,atol=0)
        np.testing.assert_allclose(env._last_f,raw,atol=1e-12)
        a.reset(airborne=False)
        for _ in range(1000):signal=a.apply(np.full(4,.289))
        np.testing.assert_allclose(signal.f_actual,.289,atol=1e-12)
        out=dict(nominal_limit_n=.289,allocator_limit_n=env.thrust_max,
            plant_forward_inverse_cap_n=env.actuator_model._mapping_thrust_max,
            estimator_forward_inverse_cap_n=a._mapping_thrust_max,plant_tau_s=env.actuator_model.time_constant_s,
            estimator_tau_s=a.time_constant_s,max_forward_inverse_error_n=max(errors),
            oracle_inverse_component_error_n=float(max(abs(actual-expected))),
            oracle_wrench_error=list(residual),unity_inverse_bitwise_equal=True,
            nominal_max_probe_n=signal.f_actual,xml_geometry_matrix=plant_geometry(env)['matrix'],B0=env.B0,
            mapping='user motor 1 -> native index 3',plant_single_application=True,
            old_020_configs_unchanged=True,manufacturer_curve_refit=False,
            cap_semantics='shared operational cap and forward/inverse map bound; cubic and ESC gain unchanged')
    finally:env.close()
    return out,curve


def comparisons(results):
    output=[]
    for T in (10.,5.):
        for label in ('PID','A_best','B_best'):
            group={r['variant']:r for r in results if r['period_s']==T and r['label']==label}
            for left,right in [('healthy','blind_fault'),('blind_fault','oracle_fault'),('oracle_fault','estimated_fault'),('healthy','estimated_fault')]:
                if left not in group or right not in group:continue
                for window in CircleProtocol(T).windows():
                    a,b=group[left]['windows'][window],group[right]['windows'][window]
                    row=dict(period_s=T,label=label,from_variant=left,to_variant=right,window=window,
                        full_windows_available=a is not None and b is not None)
                    for key in ('position_rmse_xy','position_rmse_z','max_xy_error_m','max_abs_z_error_m','velocity_reference_error_rms_xy'):
                        row[key+'_from']=a[key] if a else None;row[key+'_to']=b[key] if b else None
                        row[key+'_increase']=b[key]-a[key] if a and b else None
                    output.append(row)
    return output


def trace_comparison(left,right,end):
    a,b=read_columns(left),read_columns(right)
    ma=a.time_post<=end+1e-9;mb=b.time_post<=end+1e-9
    keys=[k for k in a if k in b and k.startswith(('position','quaternion','velocity','omega','action_',
          'wrench_command','motor_thrust','motor_reaction','delivered_esc','xi_t','xi_next','pid_i_'))]
    maximum=0.;worst=None
    if int(ma.sum())!=int(mb.sum()):return dict(equal=False,reason='different observed duration',samples=[int(ma.sum()),int(mb.sum())])
    for k in keys:
        value=float(np.max(np.abs(a[k][ma]-b[k][mb])))
        if value>maximum:maximum=value;worst=k
    return dict(equal=maximum==0.,max_abs_error=maximum,worst_column=worst,columns=len(keys),samples=int(ma.sum()),
                interval='commands [0,boundary), resulting states through boundary before event')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--completion',type=Path,default=DEFAULT_RECORD)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Three independent 8-second load/ramp/save checks; not the main 20 runs')
    args=p.parse_args(argv)
    config=evaluation_config(load_config(args.config));validate_common_config(config)
    if config.vehicle.thrust_max!=.289 or config.vehicle.reaction_torque_layout!='user_frd':raise ValueError('requires user_frd and 0.289 N')
    settings=EstimatorSettings(**json.loads(SETTINGS.read_text()));previous=json.loads((PREVIOUS/'manifest.json').read_text())
    assert sha256(SETTINGS)==previous['estimator_settings_sha256']
    model,meta=nominal_model(config);policies=select_models(args.completion,config)
    for policy in policies:
        old=next(x for x in previous['models'] if x['label']==policy.provenance['label'])
        assert all(old[k]==policy.provenance[k] for k in ('path','sha256'))
    jobs=[]
    for T in (10.,5.):
        for label,policy in [('PID',PIDOnly())]+[(p.provenance['label'],p) for p in policies]:
            variants=[('healthy','blind',False),('blind_fault','blind',True)]
            if label!='PID':variants += [('oracle_fault','oracle',True),('estimated_fault','estimated',True)]
            for variant,mode,fault in variants: jobs.append((T,label,policy,variant,mode,fault))
    if args.smoke:jobs=[j for j in jobs if j[0]==5 and j[3]=='healthy']
    plan=[dict(period_s=T,label=l,variant=v,allocator=m,fault=f) for T,l,_,v,m,f in jobs]
    if args.dry_run:
        print(json.dumps(dict(jobs=plan,config=config.resolved_dict(),settings=asdict(settings),
            protocols=[CircleProtocol(T).parameters() for T in (10.,5.)],models=[p.provenance for p in policies]),indent=2));return 0
    directory=args.output_dir or Path(tempfile.mkdtemp(prefix='circle-fault-smoke-' if args.smoke else 'circle-fault-',dir=ROOT/'artifacts/runs'))
    directory.mkdir(parents=True,exist_ok=True)
    # Allow only preservation files prepared before the one common-source edit.
    if any(x.name not in ('protected_hashes_before.json','git_status_before.txt','integral_validation_before.py.txt') for x in directory.iterdir()):
        raise FileExistsError('output must be new; existing results are never overwritten')
    if not (directory/'protected_hashes_before.json').exists():capture_protected_files(directory)
    from stable_baselines3 import PPO
    import torch
    def forbidden(*a,**k):raise RuntimeError('training/optimizer updates prohibited')
    PPO.learn=PPO.train=forbidden;torch.optim.Adam.step=forbidden
    digests={p.provenance['label']:parameter_digest(p) for p in policies}
    shutil.copyfile(SETTINGS,directory/'estimator_settings.json')
    motor_checks,curve=focused_motor_checks(config,model)
    write_json(directory/'motor_path_verification.json',motor_checks);flat_csv(directory/'motor_command_curve.csv',curve)
    manifest=dict(status='running',smoke=args.smoke,jobs=plan,config=config.resolved_dict(),seed=42,deterministic=True,
        settings_source=str(SETTINGS),estimator_settings_sha256=sha256(SETTINGS),estimator_settings=asdict(settings),
        previous_comparison=str(PREVIOUS),previous_manifest_sha256=sha256(PREVIOUS/'manifest.json'),
        models=[p.provenance for p in policies],nominal_model=meta,
        protocols=[CircleProtocol(T).parameters() for T in (10.,5.)],integral=asdict(GAINS[1]),
        resolved_config_by_controller=dict(PPO=config.resolved_dict(),PID=replace(config,environment=replace(config.environment,control_mode='residual')).resolved_dict()),
        maximum_thrust_override='0.289N explicit latest instruction overrides stale 0.20N paragraph; old YAML preserved',
        pid='existing CascadePID at 500Hz, zero residual; position->limited desired velocity->velocity PI->attitude/rate control; no new feedforward/filter/external integral',
        ppo='frozen PPO + external I at 100Hz; raw absolute world velocity, native body omega, frozen normalization, no analytic v_ref input',
        timing='control/reference/xi_t at t, held reference over [t,t+dt]; physical state/error/analytic reference at t+dt; estimator available at t+dt for next command; gyro timestamp t+dt-0.002',
        metric_samples='post-states in (start,end]; event at start before following command. No hover settling criterion used.',
        source_sha256={str(x.relative_to(ROOT)):sha256(x) for x in list((ROOT/'crazyflie_rl').glob('*.py'))+[args.config,ROOT/'compare_circle_faults.py']},
        git=_git_metadata(ROOT),runs={})
    results=[];verification=dict(offline_replay={},prefault={},single_efficiency_application_every_substep=True)
    def save():
        write_json(directory/'manifest.json',manifest);write_json(directory/'summary.json',results)
        flat_csv(directory/'summary.csv',results);flat_csv(directory/'comparisons.csv',comparisons(results))
        flat_csv(directory/'segments.csv',[dict(key=r['key'],period_s=r['period_s'],label=r['label'],variant=r['variant'],
            window=w,start_s=CircleProtocol(r['period_s']).windows()[w][0],end_s=CircleProtocol(r['period_s']).windows()[w][1],
            complete_window=s is not None,metrics=s,partial_metrics=r['partial_observed_windows'][w])
            for r in results for w,s in r['windows'].items()])
        write_json(directory/'verification.json',verification)
    save();print('RESULT_DIR',directory,flush=True)
    try:
        for T,label,policy,variant,mode,fault in jobs:
            key=f'T{T:g}-{label}-{variant}';print('RUN',key,flush=True)
            rows,initial,observer,case,reasons=execute(config,policy,CircleProtocol(T),model,settings,mode,fault,
                horizon=8. if args.smoke else None)
            r=metrics(rows,observer,case,reasons);r.update(key=key,period_s=T,label=label,variant=variant,
                controller='PID' if label=='PID' else 'PPO + I',allocation_mode=mode)
            csv=directory/(key+'.csv');keys=tuple(dict.fromkeys(TRACE_KEYS+EXTRA_KEYS+CIRCLE_KEYS))
            write_rollout(csv,[{k:row[k] for k in keys} for row in rows])
            flat_csv(directory/(key+'-events.csv'),observer.events);write_json(directory/(key+'-events.json'),observer.events)
            verification['offline_replay'][key]=offline_verify(csv,model,settings)
            reference=directory/f'T{T:g}-{label}-healthy.csv'
            if variant!='healthy':verification['prefault'][key]=trace_comparison(csv,reference,min(case.horizon,case.protocol.fault_time,r['duration_s']))
            if label!='PID':assert parameter_digest(policy)==digests[label]
            manifest['runs'][key]=dict(initial=initial,metadata=observer.metadata,comparison_boundary=observer.comparison_marker,
                events=observer.events,csv_sha256=sha256(csv),checkpoint_unchanged=True)
            results.append(r);save();print('DONE',len(results),key,r['duration_s'],r['termination_reasons'],flush=True)
        if not args.smoke:
            from .circle_fault_plots import plots
            plots(directory,results)
        script='#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'
        script+='run_dir=$(mktemp -d artifacts/runs/circle-fault-XXXXXX)\n'
        script+='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_circle_faults.py --config '+shlex.quote(str(args.config.resolve()))+' --output-dir "$run_dir"'+(' --smoke' if args.smoke else '')+'\n'
        script+='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_circle_faults.py --run-dir "$run_dir"\n'
        (directory/'rerun.sh').write_text(script)
        manifest.update(status='complete',completed_rollouts=len(results),no_training=True)
        save();print('COMPLETE',directory,flush=True)
    except BaseException as exc:
        manifest.update(status='error',error=repr(exc));save();raise
    return 0

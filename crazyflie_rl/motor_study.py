"""Non-learning PID experiments with explicit empirical settling/limit criteria.

Settling is the completion time of the first full hold-duration RMS window
(after degradation activation) passing all three thresholds. It is an empirical
screen, NOT a proof of stability. Recovery instead requires pointwise thresholds
for the whole hold and reports the start of that hold relative to pulse end.
"""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import numpy as np
from .motor_degradation import Settings, experiment_config, rollout


@dataclass(frozen=True)
class Study:
    motor_index: int = 0
    seed: int = 42
    effectiveness_values: tuple = (1., .9, .8, .7, .6, .55, .5)
    duration: float = 30.
    activation_sec: float = 2.
    steady_state_window_sec: float = 5.
    settle_position_m: float = .03
    settle_attitude_deg: float = 3.
    settle_velocity_m_s: float = .03
    settle_hold_sec: float = 2.
    settle_buffer_sec: float = 2.
    max_steady_saturation_fraction: float = .01
    timing_mode: str = 'individual'
    pre_window_sec: float = 1.
    pulse_amplitude: float = .0001
    pulse_duration: float = .1
    post_pulse_sec: float = 8.
    pulse_amplitudes: tuple = (.00005, .0001, .00015, .0002, .0003, .0004, .0005)
    recovery_position_m: float = .05
    recovery_attitude_deg: float = 5.
    recovery_velocity_m_s: float = .05
    recovery_hold_sec: float = .5
    critical_position_m: float = .1
    critical_attitude_deg: float = 10.
    critical_recovery_sec: float = 5.
    critical_motor_saturation: bool = True
    critical_termination: bool = True

    def __post_init__(self):
        if type(self.motor_index) is not int or self.motor_index not in range(4):
            raise ValueError('motor index must be 0..3')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if self.timing_mode not in ('individual', 'common'):
            raise ValueError('timing mode must be individual or common')
        for key, value in asdict(self).items():
            if key.startswith('critical_') and key in ('critical_motor_saturation', 'critical_termination'):
                if type(value) is not bool:
                    raise ValueError(f'{key} must be bool')
            elif key not in ('motor_index', 'seed', 'timing_mode', 'effectiveness_values', 'pulse_amplitudes'):
                if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                    raise ValueError(f'{key} must be finite and nonnegative')
                if value == 0 and key not in ('activation_sec', 'settle_buffer_sec', 'max_steady_saturation_fraction'):
                    raise ValueError(f'{key} must be positive')
        for seq in (self.effectiveness_values, self.pulse_amplitudes):
            if not len(seq) or any(isinstance(x, bool) or not np.isfinite(x) or x <= 0 for x in seq):
                raise ValueError('sweeps must contain positive finite values')
            if len(set(seq)) != len(seq):
                raise ValueError('duplicate sweep values')
        if max(self.effectiveness_values) > 1 or 1. not in self.effectiveness_values:
            raise ValueError('effectiveness sweep must include nominal 1.0, and stay <= 1')
        if any(a >= b for a,b in zip(self.pulse_amplitudes, self.pulse_amplitudes[1:])):
            raise ValueError('pulse amplitudes must be strictly increasing; input order is preserved')
        if self.duration <= self.activation_sec + max(self.settle_hold_sec, self.steady_state_window_sec):
            raise ValueError('duration must leave complete post-activation analysis windows')
        if self.settle_buffer_sec < self.pre_window_sec:
            raise ValueError('settle buffer must contain the whole pre window')
        if self.post_pulse_sec < self.critical_recovery_sec + self.recovery_hold_sec:
            raise ValueError('post window must cover recovery critical deadline plus hold')
        if self.max_steady_saturation_fraction > 1:
            raise ValueError('saturation fraction must be <= 1')


def magnitudes(x):
    return (np.linalg.norm(x['position_error'], axis=1),
            np.rad2deg(x['attitude_error_rad']),
            np.linalg.norm(x['velocity'], axis=1),
            np.linalg.norm(x['angular_velocity'], axis=1))


def settling_time(x, s, dt):
    p,a,v,_ = magnitudes(x)
    mask = x['time'] >= s.activation_sec
    times = x['time'][mask]
    n = int(np.ceil(s.settle_hold_sec / dt))
    if len(times) < n:
        return None
    good = np.ones(len(times)-n+1, dtype=bool)
    for values, limit in ((p,s.settle_position_m),(a,s.settle_attitude_deg),(v,s.settle_velocity_m_s)):
        squares = values[mask]**2
        cumulative = np.r_[0., np.cumsum(squares)]
        mean = (cumulative[n:] - cumulative[:-n]) / n
        good &= mean < limit**2
    indices = np.flatnonzero(good)
    return float(times[indices[0]+n-1]+dt) if indices.size else None


def joint_recovery_time(x, s, pulse_end, dt):
    p,a,v,_ = magnitudes(x)
    good = ((p<s.recovery_position_m) & (a<s.recovery_attitude_deg) &
            (v<s.recovery_velocity_m_s) & (x['time']>=pulse_end-1e-12))
    n = int(np.ceil(s.recovery_hold_sec/dt))
    run = 0
    for i, ok in enumerate(good):
        run = run+1 if ok else 0
        if run >= n:
            return max(0., float(x['time'][i-n+1]-pulse_end))
    return None


def window_metrics(x, mask, config):
    if not np.any(mask):
        return None
    p,a,v,w = magnitudes(x)
    rms = lambda arr: float(np.sqrt(np.mean(arr[mask]**2)))
    margin = x['raw_upper_margin_normalized'][mask]
    raw = x['raw_thrust'][mask]
    upper, lower = raw >= config.vehicle.thrust_max, raw <= config.vehicle.thrust_min
    result = dict(position_error_rms_m=rms(p), position_error_max_m=float(np.max(p[mask])),
        attitude_error_rms_deg=rms(a), attitude_error_max_deg=float(np.max(a[mask])),
        velocity_rms_m_s=rms(v), angular_velocity_rms_rad_s=rms(w),
        euler_rms_deg=np.sqrt(np.mean(x['attitude_deg'][mask]**2,axis=0)).tolist(),
        minimum_margin_n=float(np.min(x['raw_upper_margin_n'][mask])),
        minimum_effective_margin_n=float(np.min(x['effective_upper_margin_n'][mask])),
        per_motor_minimum_upper_margin_n=np.min(x['raw_upper_margin_n'][mask],axis=0).tolist(),
        per_motor_p01_margin_n=np.percentile(x['raw_upper_margin_n'][mask],1,axis=0).tolist(),
        per_motor_p01_normalized_margin=np.percentile(margin,1,axis=0).tolist(),
        fraction_margin_lt_10pct=np.mean(margin<.1,axis=0).tolist(),
        fraction_margin_lt_5pct=np.mean(margin<.05,axis=0).tolist(),
        upper_saturation_fraction=np.mean(upper,axis=0).tolist(),
        lower_saturation_fraction=np.mean(lower,axis=0).tolist(),
        any_motor_saturation_fraction=float(np.mean(np.any(upper|lower,axis=1))))
    for key in ('motor_command', 'clipped_thrust', 'actual_thrust', 'i_velocity', 'i_rate'):
        y=x[key][mask]
        result[key] = dict(mean=np.mean(y,axis=0).tolist(), p95=np.percentile(y,95,axis=0).tolist(),
            p99=np.percentile(y,99,axis=0).tolist(), abs_max=np.max(np.abs(y),axis=0).tolist())
    return result


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def run_trace(base, s, lam, start, amplitude, sign, total, path, pulse):
    # Every invocation constructs a NEW environment, resets with the SAME seed,
    # and retains the full response; no recovery-based early stopping.
    settings = Settings(motor_index=s.motor_index, effectiveness=lam, seed=s.seed,
        activation_sec=s.activation_sec, disturbance_axis='roll' if sign>0 else '-roll',
        disturbance_torque=amplitude, disturbance_start=start,
        disturbance_duration=s.pulse_duration, total_sec=total)
    config = experiment_config(base, settings)
    x,end = rollout(config, settings, 'C' if pulse else 'B')
    np.savez_compressed(path, **x)
    return x,end,config


def long_hover(base, s, root):
    results=[]
    for i,lam in enumerate(s.effectiveness_values):
        x,end,config=run_trace(base,s,lam,s.duration-s.pulse_duration-.1,0.,1,s.duration,
                             root/'traces'/f'hover_{i}.npz',False)
        dt=1/base.vehicle.physics_hz
        settle=settling_time(x,s,dt)
        steady=window_metrics(x,x['time']>=s.duration-s.steady_state_window_sec,config)
        complete=not end['terminated'] and end['time'] >= s.duration-1e-8
        eligible=bool(complete and settle is not None and steady and
            steady['position_error_rms_m']<s.settle_position_m and
            steady['attitude_error_rms_deg']<s.settle_attitude_deg and
            steady['velocity_rms_m_s']<s.settle_velocity_m_s and
            steady['any_motor_saturation_fraction']<=s.max_steady_saturation_fraction)
        row=dict(effectiveness=lam, settled=settle is not None, settling_time_sec=settle,
                 eligible=eligible, steady_state=steady, termination=end)
        results.append(row)
        print('hover',lam,'settle',settle,'eligible',eligible, 'position RMS', steady['position_error_rms_m'] if steady else None, flush=True)
    candidates=[r['effectiveness'] for r in results if r['eligible'] and r['effectiveness']<1]
    candidate=min(candidates) if candidates else None
    warnings=[]
    if candidate is not None:
        chosen=next(r for r in results if r['effectiveness']==candidate)
        if chosen['steady_state']['per_motor_p01_normalized_margin'][s.motor_index]<.05:
            warnings.append('candidate degraded-motor upper-margin P01 < 5%')
    result=dict(rows=results,candidate_lambda_star=candidate,warnings=warnings,
        settling_definition='completion of first full hold-duration RMS window after activation; empirical criterion, not stability proof')
    dump(root/'metrics/long_hover_sweep.json',result)
    return result


def select_lambda(sweep, requested=None):
    lam=sweep['candidate_lambda_star'] if requested is None else requested
    if lam is None or not 0<lam<1:
        raise ValueError('No eligible degraded candidate. Inspect long-hover results; no pulse experiments executed.')
    row=next((r for r in sweep['rows'] if r['effectiveness']==lam),None)
    nominal=next(r for r in sweep['rows'] if r['effectiveness']==1.)
    if row is None or not row['eligible'] or not nominal['eligible']:
        raise ValueError('Selected effectiveness and nominal must pass this run\'s long-hover screen')
    return lam


def pulse_times(sweep, lam, s, dt):
    times={name: next(r['settling_time_sec'] for r in sweep['rows'] if r['effectiveness']==level)+s.settle_buffer_sec
           for name,level in (('N',1.),('D',lam))}
    if s.timing_mode=='common':
        times={k:max(times.values()) for k in times}
    return {k:float(np.ceil((v-1e-10)/dt)*dt) for k,v in times.items()}


def response(x, config, s, pulse_start, end):
    dt=1/config.vehicle.physics_hz
    pre=window_metrics(x,(x['time']>=pulse_start-s.pre_window_sec)&(x['time']<pulse_start),config)
    mask=x['time']>=pulse_start-1e-12
    post=window_metrics(x,mask,config)
    p,a,v,w=magnitudes(x)
    recovery=joint_recovery_time(x,s,pulse_start+s.pulse_duration,dt)
    complete=end['time']>=pulse_start+s.pulse_duration+s.post_pulse_sec-1e-8 and not end['terminated']
    return dict(pre=pre, post=post, pulse_start_sec=pulse_start, termination=end,
        full_post_window=complete, recovery_time_sec=recovery,
        peak_position_error_m=float(np.max(p[mask])) if np.any(mask) else None,
        peak_attitude_error_deg=float(np.max(a[mask])) if np.any(mask) else None,
        peak_roll_abs_deg=float(np.max(np.abs(x['attitude_deg'][mask,0]))) if np.any(mask) else None,
        peak_angular_rate_rad_s=float(np.max(w[mask])) if np.any(mask) else None,
        integrated_position_error_m2_s=float(np.sum(p[mask]**2)*dt) if np.any(mask) else None,
        integrated_attitude_error_rad2_s=float(np.sum(np.deg2rad(a[mask])**2)*dt) if np.any(mask) else None,
        integrated_angular_rate_rad2_s=float(np.sum(w[mask]**2)*dt) if np.any(mask) else None)


def asymmetry(plus, minus):
    if plus is None or minus is None:
        return dict(absolute_difference=None, normalized_difference=None)
    mean=.5*(plus+minus)
    return dict(absolute_difference=abs(plus-minus), normalized_difference=abs(plus-minus)/mean if mean>1e-12 else None)


def four_pulses(base,s,root,lam,times,amplitude,prefix):
    rows={}
    for label in ('N+','N-','D+','D-'):
        level=1. if label[0]=='N' else lam
        start=times[label[0]]
        total=start+s.pulse_duration+s.post_pulse_sec
        # Round horizon upwards to policy step boundary to retain the complete window.
        total=np.ceil((total-1e-10)*base.environment.policy_hz)/base.environment.policy_hz
        path=root/'traces'/f'{prefix}_{label}.npz'
        x,end,config=run_trace(base,s,level,start,amplitude,1 if label[1]=='+' else -1,total,path,True)
        rows[label]=response(x,config,s,start,end)
        rows[label].update(effectiveness=level, amplitude_nm=amplitude, trace=str(path))
        print(prefix,label,'peak pos',rows[label]['peak_position_error_m'],'recovery',rows[label]['recovery_time_sec'],flush=True)
    return rows


def paired(base,s,root,lam,times):
    rows=four_pulses(base,s,root,lam,times,s.pulse_amplitude,'paired')
    differences={sign:dict(position_rms_difference_m=abs(rows['N'+sign]['pre']['position_error_rms_m']-rows['D'+sign]['pre']['position_error_rms_m']),
        attitude_rms_difference_deg=abs(rows['N'+sign]['pre']['attitude_error_rms_deg']-rows['D'+sign]['pre']['attitude_error_rms_deg'])) for sign in ('+','-')}
    result=dict(lambda_star=lam, rows=rows, pre_differences=differences,
        directional_asymmetry={k:{field:asymmetry(rows[k+'+'][field],rows[k+'-'][field])
            for field in ('recovery_time_sec','peak_position_error_m','peak_attitude_error_deg')} for k in ('N','D')})
    dump(root/'metrics/paired_roll_pulse.json',result)
    return result


def critical_events(r,s):
    # None recovery is a failure only with enough observed follow-up or termination.
    horizon=r['termination']['time']-(r['pulse_start_sec']+s.pulse_duration)
    return dict(position=r['peak_position_error_m'] is not None and r['peak_position_error_m']>s.critical_position_m,
        attitude=r['peak_attitude_error_deg'] is not None and r['peak_attitude_error_deg']>s.critical_attitude_deg,
        recovery=(r['recovery_time_sec']>s.critical_recovery_sec if r['recovery_time_sec'] is not None
                  else horizon>=s.critical_recovery_sec+s.recovery_hold_sec),
        saturation=bool(s.critical_motor_saturation and r['post'] and r['post']['any_motor_saturation_fraction']>0),
        termination=bool(s.critical_termination and r['termination']['terminated']))


def critical_levels(rows,s):
    results={}
    for label in ('N+','N-','D+','D-'):
        per={key:None for key in ('position','attitude','recovery','saturation','termination')}
        for block in rows:
            events=critical_events(block['conditions'][label],s)
            block['conditions'][label]['critical_events']=events
            for key,hit in events.items():
                if hit and per[key] is None:
                    per[key]=block['amplitude_nm']
        observed=[v for v in per.values() if v is not None]
        results[label]=dict(per_criterion=per,overall=min(observed) if observed else None,
            status='observed_on_grid' if observed else 'lower_bound_only',sweep_max_nm=rows[-1]['amplitude_nm'])
    ratios={}
    for sign in ('+','-'):
        n,d=results['N'+sign]['overall'],results['D'+sign]['overall']
        ratios[sign]=dict(value=d/n if n is not None and d is not None else None,
                          status='observed_on_grid' if n is not None and d is not None else 'lower_bound_only')
    return results,ratios


def staircase(base,s,root,lam,times):
    rows=[]
    for i,amplitude in enumerate(s.pulse_amplitudes):
        rows.append(dict(amplitude_nm=amplitude,impulse_nm_s=amplitude*s.pulse_duration,
            conditions=four_pulses(base,s,root,lam,times,amplitude,f'staircase_{i}')))
    levels,ratios=critical_levels(rows,s)
    result=dict(lambda_star=lam,rows=rows,critical_levels=levels,robustness_ratio=ratios,
        interpretation='Empirical grid thresholds; no interpolation or stability guarantee. Missing thresholds are right-censored.')
    dump(root/'metrics/pulse_staircase.json',result)
    return result


def plots(root,sweep,pair=None,stairs=None,s=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    for ax,key,label in zip(axes.flat,
        ('position_error_rms_m','attitude_error_rms_deg','settling_time_sec','minimum_margin_n','motor_p99','any_motor_saturation_fraction'),
        ('Position RMS [m]','Attitude RMS [deg]','Settling completion [s]','Minimum raw margin [N]','Worst normalized motor P99','Saturation fraction')):
        values=[]
        for r in sweep['rows']:
            m=r['steady_state']
            val=r['settling_time_sec'] if key=='settling_time_sec' else (max(m['motor_command']['p99']) if key=='motor_p99' else m[key]) if m else None
            values.append(np.nan if val is None else val)
        ax.plot([r['effectiveness'] for r in sweep['rows']],values,'o-'); ax.set_xlabel('Effectiveness'); ax.set_ylabel(label); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(root/'plots/long_hover_sweep.png'); plt.close(fig)
    if pair:
        fig,axes=plt.subplots(9,2,figsize=(16,25),sharex='col')
        for col,sign in enumerate(('+','-')):
            for system,style in (('N','-'),('D','--')):
                r=pair['rows'][system+sign]
                with np.load(r['trace']) as x:
                    t=x['time']-r['pulse_start_sec']
                    data=[x['attitude_deg'][:,0],x['angular_velocity'][:,0],np.linalg.norm(x['position_error'],axis=1),
                        x['wrench_command'][:,0],x['wrench_actual'][:,0],x['clipped_thrust'],x['actual_thrust'],
                        np.min(x['raw_upper_margin_n'],axis=1),np.column_stack((x['i_velocity'],x['i_rate']))]
                    for j,y in enumerate(data):
                        if y.ndim==1: axes[j,col].plot(t,y,style,label=system)
                        else:
                            for k in range(y.shape[1]):
                                label=f'{system} M{k}' if j in (5,6) else f'{system} '+('Ivel' if k<3 else 'Irate')+str(k%3)
                                axes[j,col].plot(t,y[:,k],style,label=label,color=f'C{k}')
            for j,label in enumerate(('Roll [deg]','Roll rate [rad/s]','Position error [m]','Requested roll [Nm]',
                'Actual roll [Nm]','Motor commands [N]','Actual thrust [N]','Min raw margin [N]','PID integrals')):
                ax=axes[j,col]; ax.axvspan(0,s.pulse_duration,color='orange',alpha=.2); ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend(fontsize=6,ncol=2)
            axes[0,col].set_title(sign+'roll'); axes[-1,col].set_xlabel('Time relative to pulse [s]')
        fig.tight_layout(); fig.savefig(root/'plots/paired_roll_pulse.png'); plt.close(fig)
    if stairs:
        fig,axes=plt.subplots(3,2,figsize=(12,11))
        for label in ('N+','N-','D+','D-'):
            for ax,key,title in zip(axes.flat,('peak_position_error_m','peak_attitude_error_deg','recovery_time_sec','minimum_margin_n','any_motor_saturation_fraction','minimum_effective_margin_n'),
                ('Peak position [m]','Peak attitude [deg]','Recovery [s]','Min raw margin [N]','Saturation fraction','Min effective margin [N]')):
                ys=[]
                for block in stairs['rows']:
                    r=block['conditions'][label]; v=r.get(key) if key in r else r['post'][key] if r['post'] else None
                    ys.append(v if v is not None else np.nan)
                ax.plot([r['amplitude_nm'] for r in stairs['rows']],ys,'o-' if label[1]=='+' else 's--',label=label)
                ax.set_xlabel('Pulse magnitude [Nm]'); ax.set_ylabel(title); ax.grid(alpha=.2); ax.legend()
        fig.tight_layout(); fig.savefig(root/'plots/pulse_staircase.png'); plt.close(fig)


def print_tables(sweep,pair,stairs):
    def f(v): return 'null' if v is None else f'{v:.6g}'
    print('\nExperiment 1: lambda settled t_settle pos_RMS att_RMS min_margin saturation')
    for r in sweep['rows']:
        m=r['steady_state']
        print(r['effectiveness'],r['settled'],f(r['settling_time_sec']),*(f(m[k]) if m else 'null' for k in ('position_error_rms_m','attitude_error_rms_deg','minimum_margin_n','any_motor_saturation_fraction')))
    if pair:
        print('\nExperiment 2: condition pre_pos pre_att min_margin peak_pos peak_roll recovery')
        for label,r in pair['rows'].items():
            print(label,*(f(r['pre'][k]) for k in ('position_error_rms_m','attitude_error_rms_deg','minimum_margin_n')),
                  *(f(r[k]) for k in ('peak_position_error_m','peak_roll_abs_deg','recovery_time_sec')))
    if stairs:
        for metric in ('peak_position_error_m','peak_attitude_error_deg','recovery_time_sec','minimum_margin_n','any_motor_saturation_fraction'):
            print('\nExperiment 3:',metric,'amplitude N+ N- D+ D-')
            for row in stairs['rows']:
                print(f(row['amplitude_nm']),*(f(r[metric] if metric in r else r['post'][metric] if r['post'] else None) for r in row['conditions'].values()))


def main(argv=None):
    import argparse
    import tempfile
    import yaml
    from .config import load_config
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=('long-hover','paired-pulse','staircase','study'))
    parser.add_argument('--config',default='configs/diagnostics/pid_motor_study.yaml')
    parser.add_argument('--output-root',default='artifacts/runs')
    parser.add_argument('--effectiveness',type=float,help='Override candidate, only if eligible in this run sweep')
    for key,value in asdict(Study()).items():
        kw=dict(default=None)
        if key in ('effectiveness_values','pulse_amplitudes'): kw.update(type=float,nargs='+')
        elif type(value) is bool: kw.update(action=argparse.BooleanOptionalAction)
        else: kw['type']=type(value)
        parser.add_argument('--'+key.replace('_','-'),**kw)
    args=parser.parse_args(argv)
    source=Path(args.config).resolve(); values=yaml.safe_load(source.read_text())
    base=load_config(source.parent/values.pop('experiment_config'))
    for key in asdict(Study()):
        if getattr(args,key) is not None: values[key]=getattr(args,key)
    s=Study(**values)
    # Pulse boundaries are exact physics-grid intervals; avoid silent rounding.
    for value in (s.pulse_duration,s.activation_sec):
        if not np.isclose(value*base.vehicle.physics_hz,round(value*base.vehicle.physics_hz),rtol=0,atol=1e-8):
            raise ValueError('activation and pulse duration must align to physics substeps')
    directory=Path(args.output_root); directory.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='pid_motor_study_',dir=directory)).resolve()
    for name in ('metrics','plots','traces','config'): (root/name).mkdir()
    dump(root/'config/study.json',asdict(s))
    (root/'config/source_experiment.json').write_text(json.dumps(asdict(base),indent=2,default=str))
    print('Artifacts:',root,flush=True)
    sweep=long_hover(base,s,root)
    plots(root,sweep)
    print('Candidate lambda*:',sweep['candidate_lambda_star'],sweep['warnings'],flush=True)
    pair=stairs=None; selected=None; times=None
    if args.experiment!='long-hover':
        selected=select_lambda(sweep,args.effectiveness)
        times=pulse_times(sweep,selected,s,1/base.vehicle.physics_hz)
        if args.experiment in ('paired-pulse','study'): pair=paired(base,s,root,selected,times)
        if args.experiment in ('staircase','study'): stairs=staircase(base,s,root,selected,times)
        plots(root,sweep,pair,stairs,s)
    dump(root/'metrics/summary.json',dict(candidate_lambda_star=sweep['candidate_lambda_star'],selected_lambda_star=selected,
        pulse_times_sec=times,experiment=args.experiment,settings=asdict(s),
        long_hover=sweep,paired=pair,staircase=stairs))
    print_tables(sweep,pair,stairs)
    return root

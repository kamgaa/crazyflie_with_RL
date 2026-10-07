"""PID-only fault experiment. No production environment/controller modifications.

Samples are taken at physics-substep start, after PID/actuator advancement and
before plant integration. Integrators therefore include this substep's update.
Recovery starts after pulse end; reported latency is measured from pulse end.
"""
from dataclasses import dataclass, replace, asdict
import numpy as np
from .environment import CrazyflieResidualEnv
from .plotting import quaternion_to_euler_deg


def apply_motor_effectiveness(env, effectiveness):
    """Scale fresh post-actuator outputs once, leaving allocator/RPM state intact.

    Call only immediately after the production _apply_control. This is a rotor
    output-loss model: both generated force and signed reaction torque scale.
    """
    lam = np.asarray(effectiveness, dtype=float)
    if lam.shape != (4,) or not np.all(np.isfinite(lam)) or np.any((lam < 0) | (lam > 1)):
        raise ValueError('motor effectiveness must be four finite values in [0,1]')
    nominal = env._last_f.copy()
    nominal_torque = env._last_q_actual.copy()
    if np.any(lam != 1):
        env._last_f *= lam
        env._last_q_actual *= lam
        env._last_wrench_actual = env.B @ env._last_f
        env._last_wrench_actual[2] = np.sum(env._last_q_actual)
        env._last_allocation_error = env._last_wrench_cmd - env._last_wrench_actual
        env._write_applied_motor_controls()
    return nominal, nominal_torque


@dataclass(frozen=True)
class Settings:
    motor_index: int = 0
    effectiveness: float = .7
    activation_sec: float = 2.
    disturbance_axis: str = 'roll'
    disturbance_torque: float = .0001
    disturbance_start: float = 6.
    disturbance_duration: float = .5
    total_sec: float = 12.
    pre_window_sec: float = 1.
    position_threshold_m: float = .05
    attitude_threshold_deg: float = 5.
    hold_sec: float = .5
    seed: int = 42

    def __post_init__(self):
        if type(self.motor_index) is not int or self.motor_index not in range(4):
            raise ValueError('motor_index must be 0..3')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        for k, v in asdict(self).items():
            if k not in ('motor_index', 'seed', 'disturbance_axis'):
                if isinstance(v, bool) or not np.isfinite(v) or v < 0:
                    raise ValueError(f'{k} must be finite and nonnegative')
        if not 0 < self.effectiveness <= 1:
            raise ValueError('effectiveness must be in (0,1]')
        if self.disturbance_axis.lstrip('+-') not in ('roll','pitch','yaw'):
            raise ValueError('axis must be +/-roll, +/-pitch, or +/-yaw')
        if not (self.activation_sec < self.disturbance_start < self.disturbance_start + self.disturbance_duration < self.total_sec):
            raise ValueError('require activation < pulse start < pulse end < experiment end')
        if min(self.hold_sec, self.pre_window_sec, self.position_threshold_m, self.attitude_threshold_deg) <= 0:
            raise ValueError('windows and thresholds must be positive')


def experiment_config(config, settings):
    """Explicit nominal initial hover; preserve all plant/PID/reward parameters."""
    e = replace(config.environment, control_mode='residual', episode_sec=settings.total_sec,
                position_target=(0.,0.,1.), yaw_target=0., position_perturbation=0.,
                attitude_perturbation_deg=0., initial_pose_randomization=None)
    if e.payload.randomize or e.payload.mass != 0 or config.actuator.randomization.enabled:
        raise ValueError('Use a nominal no-payload, nonrandomized actuator profile')
    return replace(config, environment=e)


class DiagnosticEnv(CrazyflieResidualEnv):
    def __init__(self, config, settings, condition):
        if condition not in ('BASE','A','B','C'):
            raise ValueError(condition)
        if config.actuator.reaction_torque.model != 'legacy_ratio':
            raise ValueError('This experiment requires legacy_ratio reaction torque')
        self.settings, self.condition, self.rows = settings, condition, []
        super().__init__(config=config)

    def _apply_control(self, wrench):
        # Preserve the production allocator, clipping, and complete actuator state.
        super()._apply_control(wrench)
        s = self.settings
        # Integer substep clock prevents floating accumulation shifting pulse edges.
        t = round(float(self.data.time) / self.dt_phys) * self.dt_phys
        lam = np.ones(4)
        if self.condition in ('B','C') and t >= s.activation_sec:
            lam[s.motor_index] = s.effectiveness
        nominal, _ = apply_motor_effectiveness(self, lam)
        self.dist_torque_body[:] = 0.
        if self.condition in ('A','C') and s.disturbance_start <= t < s.disturbance_start + s.disturbance_duration:
            axis = ('roll','pitch','yaw').index(s.disturbance_axis.lstrip('+-'))
            self.dist_torque_body[axis] = s.disturbance_torque * (-1 if s.disturbance_axis.startswith('-') else 1)
        p,q,v,w = self._read_state()
        raw = self.B_pinv @ np.asarray(wrench)
        width = self.thrust_max - self.thrust_min
        self.rows.append(dict(time=t, position=p, position_error=p-self.pos_des,
            velocity=v, quaternion=q, attitude_deg=quaternion_to_euler_deg(q), angular_velocity=w,
            attitude_error_rad=2*np.arccos(np.clip(abs(q[0]),0,1)),
            tilt_error_rad=np.arccos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)), yaw_error_rad=self._yaw_err(q),
            wrench_command=self._last_wrench_cmd.copy(), wrench_actual=self._last_wrench_actual.copy(),
            allocation_error=self._last_allocation_error.copy(), raw_thrust=raw,
            clipped_thrust=self._last_f_cmd.copy(), motor_command=self._last_motor_cmd.copy(),
            motor_omega=self._last_omega.copy(), nominal_thrust=nominal, effectiveness=lam,
            actual_thrust=self._last_f.copy(), reaction_torque=self._last_q_actual.copy(),
            i_velocity=self.pid._i_vel.copy(), i_rate=self.pid._i_rate.copy(),
            raw_upper_margin_n=self.thrust_max-raw,
            raw_upper_margin_normalized=(self.thrust_max-raw)/width,
            clipped_upper_margin_n=self.thrust_max-self._last_f_cmd,
            effective_upper_margin_n=lam*self.thrust_max-self._last_f,
            disturbance_body_nm=self.dist_torque_body.copy()))


def rollout(config, settings, condition):
    env = DiagnosticEnv(config, settings, condition)
    try:
        obs, _ = env.reset(seed=settings.seed)
        initial = dict(qpos=env.data.qpos.tolist(), qvel=env.data.qvel.tolist())
        while True:
            obs, reward, terminated, truncated, _ = env.step(np.zeros(4))
            if terminated or truncated:
                break
        trace = {k: np.asarray([r[k] for r in env.rows]) for k in env.rows[0]}
        end = dict(time=float(env.data.time), terminated=bool(terminated), truncated=bool(truncated), initial=initial)
        return trace, end
    finally:
        env.close()


def recovery_time(t, error, threshold, pulse_end, hold, dt):
    run = 0
    count = int(np.ceil(hold/dt))
    for i in range(len(t)):
        run = run+1 if t[i] >= pulse_end and error[i] < threshold else 0
        if run >= count:
            return float(t[i-count+1]-pulse_end)
    return None


def summarize(x, config, s, condition, end):
    t=x['time']; dt=1/config.vehicle.physics_hz
    p=np.linalg.norm(x['position_error'],axis=1); a=x['attitude_error_rad']
    pre=(t>=s.disturbance_start-s.pre_window_sec)&(t<s.disturbance_start)
    post=t>=s.disturbance_start
    def rms(v): return float(np.sqrt(np.mean(v*v))) if v.size else None
    def stats(mask):
        if not np.any(mask): return None
        raw=x['raw_thrust'][mask]; margin=x['raw_upper_margin_normalized'][mask]
        upper=raw>=config.vehicle.thrust_max; lower=raw<=config.vehicle.thrust_min
        return dict(raw_upper_margin_min_n=np.min(x['raw_upper_margin_n'][mask],axis=0).tolist(),
            normalized_margin_min=np.min(margin,axis=0).tolist(),
            normalized_margin_p01=np.percentile(margin,1,axis=0).tolist(),
            fraction_margin_lt_10pct=np.mean(margin<.1,axis=0).tolist(),
            fraction_margin_lt_5pct=np.mean(margin<.05,axis=0).tolist(),
            upper_saturation_fraction=np.mean(upper,axis=0).tolist(),
            lower_saturation_fraction=np.mean(lower,axis=0).tolist(),
            any_motor_saturation_fraction=float(np.mean(np.any(upper|lower,axis=1))),
            clipped_headroom_min_n=np.min(x['clipped_upper_margin_n'][mask],axis=0).tolist(),
            effective_headroom_min_n=np.min(x['effective_upper_margin_n'][mask],axis=0).tolist(),
            motor_command_mean=np.mean(x['motor_command'][mask],axis=0).tolist(),
            motor_command_p95=np.percentile(x['motor_command'][mask],95,axis=0).tolist(),
            motor_command_p99=np.percentile(x['motor_command'][mask],99,axis=0).tolist())
    return dict(condition=condition, settings=asdict(s), termination=end,
        definitions=dict(
            raw_upper_margin_n='f_max - raw allocator thrust request; may be negative',
            normalized_margin='(f_max - raw allocator thrust request) / (f_max - f_min)',
            clipped_headroom_n='f_max - clipped nominal thrust command',
            effective_headroom_n='lambda * f_max - actual degraded thrust; ground truth only',
            saturation='raw request >= upper limit or <= lower limit',
            attitude_error='principal SO(3) angle relative to identity, not tilt alone',
            recovery_time='first below-threshold hold start after pulse end minus pulse end; null if not observed',
            integrated_error='left-rectangle sum at physics substep starts from pulse start to observed termination',
            motor_command='normalized actuator command, distinct from thrust in N',
            intermediate_pid_outputs='position/velocity/attitude loop setpoints are local variables, not recorded; rate output is requested torque'),
        degradation_enabled=condition in ('B','C'), disturbance_enabled=condition in ('A','C'),
        sample_semantics='physics substep start; controller/actuator already advanced; plant not yet integrated',
        pre_disturbance=dict(position_rms_m=rms(p[pre]), attitude_rms_deg=rms(np.rad2deg(a[pre])),
            angular_velocity_rms_radps=rms(np.linalg.norm(x['angular_velocity'][pre],axis=1)),
            motors=stats(pre), integrator_velocity_last=x['i_velocity'][pre][-1].tolist() if np.any(pre) else None,
            integrator_rate_last=x['i_rate'][pre][-1].tolist() if np.any(pre) else None),
        recovery=dict(peak_position_error_m=float(np.max(p[post])) if np.any(post) else None,
            peak_attitude_error_deg=float(np.rad2deg(np.max(a[post]))) if np.any(post) else None,
            position_recovery_time_s=recovery_time(t,p,s.position_threshold_m,s.disturbance_start+s.disturbance_duration,s.hold_sec,dt),
            attitude_recovery_time_s=recovery_time(t,np.rad2deg(a),s.attitude_threshold_deg,s.disturbance_start+s.disturbance_duration,s.hold_sec,dt),
            integrated_position_error_m2_s=float(np.sum(p[post]**2)*dt),
            integrated_attitude_error_rad2_s=float(np.sum(a[post]**2)*dt)),
        motor_metrics=stats(np.ones(len(t),bool)), post_disturbance_motor_metrics=stats(post))


def plot_trace(x, s, config, condition, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=x['time']; paths=[]
    for kind in ('tracking','wrench','motors','margin'):
        fig, axes=plt.subplots(4,1,figsize=(11,11),sharex=True)
        if kind=='tracking':
            for ax,key,unit in zip(axes,('position','position_error','attitude_deg','angular_velocity'),('m','m','deg','rad/s')):
                values=np.linalg.norm(x[key],axis=1) if key=='position_error' else x[key]
                ax.plot(t,values); ax.set_ylabel(key+' ['+unit+']')
        elif kind=='wrench':
            for i,ax in enumerate(axes):
                ax.plot(t,x['wrench_command'][:,i],label='requested'); ax.plot(t,x['wrench_actual'][:,i],label='actual')
                ax.set_ylabel(('tau_x [Nm]','tau_y [Nm]','tau_z [Nm]','Fz [N]')[i])
        else:
            for i,ax in enumerate(axes):
                keys=('raw_thrust','clipped_thrust','actual_thrust') if kind=='motors' else ('raw_upper_margin_n','clipped_upper_margin_n','effective_upper_margin_n')
                for key in keys: ax.plot(t,x[key][:,i],label=key)
                if kind=='motors':
                    ax.axhline(config.vehicle.thrust_max,color='k',ls=':'); ax.axhline(config.vehicle.thrust_min,color='k',ls=':')
                else: ax.axhline(0,color='k',ls=':')
                ax.set_ylabel(f'Motor {i}'+(' (degraded)' if i==s.motor_index and condition in ('B','C') else '')+' [N]')
        for ax in axes:
            ax.axvline(s.activation_sec,color='gray',ls='--')
            ax.axvspan(s.disturbance_start,s.disturbance_start+s.disturbance_duration,color='orange',alpha=.15)
            ax.grid(alpha=.2)
            if kind!='tracking': ax.legend(fontsize=7)
        axes[-1].set_xlabel('Time [s]'); fig.suptitle(f'{condition}: {kind} (shading: scheduled pulse window)')
        fig.tight_layout(); path=directory/f'{condition}_{kind}.png'; fig.savefig(path); plt.close(fig); paths.append(str(path))
    return paths

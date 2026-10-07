"""Single frozen E2E policy with queued setpoints and post-actuator rotor losses."""
from __future__ import annotations

import argparse
from collections import deque
import copy
import csv
from dataclasses import replace, fields
import math
from pathlib import Path
import queue
import sys
import time

import numpy as np

from .artifacts import ArtifactManager
from .config import load_config
from .dr_policy import load_frozen_policy, sha256
from .dr_transfer import Case, EvaluationAdapter, termination_reasons
from .environment import CrazyflieResidualEnv
from .eval_cli import RolloutTrace, trace_metrics, _save_policy_report
from .missions import HoverMission
from .motor_degradation import apply_motor_effectiveness
from .motor_layout import (exposed_motor_index, exposed_values, user_from_native,
                           body_frd_from_native, layout_metadata, user_motor_signals)
from .plotting import quaternion_to_euler_deg

ROOT = Path(__file__).resolve().parents[1]
HELP = 'W/S: X +/- | D/A: Y +/- | R/F: Z +/- | 1-4: motor -2pp | 0: restore | Space: pause | Q: save & quit'


def evaluation_config(config):
    if config.control_mode != 'e2e':
        raise ValueError('interactive evaluation requires control_mode=e2e')
    e = config.environment
    e = replace(e, position_target=(0., 0., 1.), yaw_target=0., position_perturbation=0.,
                attitude_perturbation_deg=0., initial_pose_randomization=None,
                payload=replace(e.payload, randomize=False))
    if not e.termination.min_altitude < 1 < e.termination.max_altitude:
        raise ValueError('initial altitude 1 m must be inside the configured altitude limits')
    return replace(config, environment=e,
                   actuator=replace(config.actuator, reset_rpm_mode='auto',
                                    randomization=replace(config.actuator.randomization, enabled=False)),
                   evaluation=replace(config.evaluation, deterministic=True),
                   mission=replace(config.mission, type='hover', force_floor_start=False,
                                   hover=replace(config.mission.hover, target=(0.,0.,1.), yaw_deg=0.)))


class InteractiveEnv(CrazyflieResidualEnv):
    """Only the existing, post-actuator output-loss model changes the plant."""
    def __init__(self, *args, **kwargs):
        self.motor_effectiveness = np.ones(4)
        super().__init__(*args, **kwargs)

    def _apply_control(self, wrench):
        super()._apply_control(wrench)
        self.nominal_thrust, self.nominal_reaction_torque = apply_motor_effectiveness(self, self.motor_effectiveness)


class KeyEvents:
    """Viewer callback writes only this thread-safe queue; no simulator access.

    MuJoCo 3.12.0 UIAdapterWithPyCallback calls Python only for GLFW_PRESS,
    not REPEAT or RELEASE. No timer debounce that would lose rapid presses.
    """
    def __init__(self):
        self.queue = queue.SimpleQueue()

    def __call__(self, keycode):
        key = chr(keycode).upper() if isinstance(keycode, int) and 0 <= keycode < 128 else ''
        if key in ('W','S','D','A','R','F','1','2','3','4','0',' ','Q'):
            self.queue.put(key)

    def drain(self):
        while True:
            try:
                yield self.queue.get_nowait()
            except queue.Empty:
                return


class Controls:
    def __init__(self, position_step=.02):
        if not math.isfinite(position_step) or position_step <= 0:
            raise ValueError('position-step must be finite and positive')
        self.position_step = position_step
        self.counts = np.zeros(4, dtype=int)
        self.paused = False
        self.quit = False
        self.pending = deque()
        self.sequence = 0

    @property
    def effectiveness(self):
        return (50 - self.counts) / 50.

    def process(self, keys, env, step, sink):
        """Called only at control boundaries. Deferred edits retain FIFO order."""
        def apply(key, sequence, mutate=True):
            before_target, before_eff = env.pos_des.copy(), self.effectiveness.copy()
            motor = None
            kind = 'pending_unapplied'
            if mutate:
                if key in 'WSDARF':
                    axis, sign = {'W':(0,1),'S':(0,-1),'D':(1,1),'A':(1,-1),'R':(2,1),'F':(2,-1)}[key]
                    env.pos_des = env.pos_des.copy()
                    env.pos_des[axis] += sign * self.position_step
                    kind = 'target'
                elif key in '1234':
                    motor = int(key)
                    index = exposed_motor_index(motor, getattr(env, 'reaction_torque_layout', 'legacy'))
                    self.counts[index] = min(50, self.counts[index]+1)
                    env.motor_effectiveness = self.effectiveness.copy()
                    kind = 'degrade'
                elif key == '0':
                    self.counts[:] = 0
                    env.motor_effectiveness = self.effectiveness.copy()
                    kind = 'restore'
                elif key == ' ':
                    self.paused = not self.paused
                    kind = 'pause' if self.paused else 'resume'
                elif key == 'Q':
                    self.quit = True
                    kind = 'quit'
            index = (exposed_motor_index(motor, getattr(env, 'reaction_torque_layout', 'legacy'))
                     if motor is not None else None)
            sink(dict(sequence=sequence, simulation_time=float(env.data.time), control_step=step,
                      event_type=kind, key='Space' if key == ' ' else key, motor=motor, applied=mutate,
                      native_motor_index=index, user_motor_id=None if index is None else 4-index,
                      reaction_torque_layout=getattr(env, 'reaction_torque_layout', 'legacy'),
                      user_effectiveness_before=user_from_native(before_eff),
                      user_effectiveness_after=user_from_native(self.effectiveness),
                      effectiveness_before=before_eff, effectiveness_after=self.effectiveness.copy(),
                      target_before=before_target, target_after=env.pos_des.copy()))
        for key in keys:
            sequence = self.sequence
            self.sequence += 1
            if self.quit:
                apply(key, sequence, False)
            elif key in (' ', 'Q'):
                apply(key, sequence)
                if not self.paused and not self.quit:
                    while self.pending:
                        saved, number = self.pending.popleft()
                        apply(saved, number)
            elif self.paused:
                self.pending.append((key, sequence))
            else:
                apply(key, sequence)

    def cancel_pending(self, env, step, sink):
        for key, sequence in self.pending:
            sink(dict(sequence=sequence, simulation_time=float(env.data.time), control_step=step,
                      event_type='pending_unapplied', key=key, motor=None, applied=False,
                      native_motor_index=None, user_motor_id=None,
                      reaction_torque_layout=getattr(env, 'reaction_torque_layout', 'legacy'),
                      user_effectiveness_before=user_from_native(self.effectiveness),
                      user_effectiveness_after=user_from_native(self.effectiveness),
                      effectiveness_before=self.effectiveness.copy(), effectiveness_after=self.effectiveness.copy(),
                      target_before=env.pos_des.copy(), target_after=env.pos_des.copy()))
        self.pending.clear()


def rotor_mapping(env):
    import mujoco
    result = []
    rotation = env.data.xmat[env.drone_bid].reshape(3,3)
    for i, aid in enumerate(env.act_force):
        sid = int(env.model.actuator_trnid[aid,0])
        if env.model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_SITE:
            raise ValueError('interactive rotor reporting requires the existing site transmission')
        layout = getattr(env, 'reaction_torque_layout', 'legacy')
        position = rotation.T @ (env.data.site_xpos[sid]-env.data.xpos[env.drone_bid])
        result.append(dict(number=i+1 if layout == 'legacy' else 4-i, allocator_index=i,
                           native_motor_index=i, user_motor_id=4-i,
                           reaction_torque_layout=layout,
                           position_user_frd_m=body_frd_from_native(position).tolist(),
                           force_actuator=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_ACTUATOR,aid),
                           torque_actuator=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_ACTUATOR,env.act_torque[i]),
                           site=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_SITE,sid),
                           position_body_m=(rotation.T @ (env.data.site_xpos[sid]-env.data.xpos[env.drone_bid])).tolist(),
                           motor_direction=float(env.motor_direction[i]), force_gear=env.model.actuator_gear[aid].tolist(),
                           torque_gear=env.model.actuator_gear[env.act_torque[i]].tolist()))
    return sorted(result, key=lambda r:r['number'])


class CsvStream:
    """CSV serialization of existing trace field names plus explicitly named extras."""
    def __init__(self, path, empty_fields):
        self.file = path.open('x', newline='')
        self.writer = None
        self.empty_fields = empty_fields

    def append(self, row):
        flat = {}
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                flat.update({f'{key}_{i+1}': float(v) for i,v in enumerate(value.flat)})
            else:
                flat[key] = value
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(flat))
            self.writer.writeheader()
        self.writer.writerow(flat)
        self.file.flush()

    def close(self):
        if self.writer is None:
            csv.writer(self.file).writerow(self.empty_fields)
        self.file.close()


def make_trace(rows, env, outcome):
    widths = dict(position=3, attitude_deg=3, reference_position=3, control_input=4, motor_thrust=4,
                  linear_velocity=3, angular_velocity=3, motor_thrust_command=4, motor_command=4,
                  motor_omega_rad_s=4, reaction_torque_nm=4, wrench_command=4, wrench_actual=4, allocation_error=4,
                  desired_velocity=3,velocity_error=3)
    arrays = {key: np.asarray([r[key] for r in rows]).reshape((-1,n)) for key,n in widths.items()}
    return RolloutTrace(policy='e2e', label='E2E interactive', control_mode='e2e',
        time_sec=np.array([r['time_sec'] for r in rows]), position_error=np.array([r['position_error'] for r in rows]),
        time_post=np.array([r['time_post'] for r in rows]),
        phases=tuple(r['phase'] for r in rows), training_boundary_crossed_at=next((r['time_sec'] for r in rows if r['position_error']>.15),None),
        guard_boundary_crossed_at=next((r['time_sec'] for r in rows if r['position_error']>1.5),None),
        terminated_at=rows[-1]['time_sec'] if rows and outcome['terminated'] else None,
        truncated_at=rows[-1]['time_sec'] if rows and outcome['truncated'] else None,
        diverged_at=None, error=outcome.get('error'), actuator=env.actuator_snapshot(),
        episode_mass_kg=float(env._m0+env._com_mw), **arrays)


class LiveViewer:
    """Public passive viewer on a render-only copy; GUI edits cannot reset the plant."""
    def __init__(self, env, keys):
        import mujoco
        import mujoco.viewer
        self.mj = mujoco
        self.model = copy.copy(env.model)
        self.data = mujoco.MjData(self.model)
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(env.model,self.spec))
        self.copy_state(env)
        self.handle = mujoco.viewer.launch_passive(self.model,self.data,key_callback=keys,
                                                   show_left_ui=False,show_right_ui=False)
        with self.handle.lock():
            self.handle.cam.distance = 3.7
            self.handle.cam.lookat[:] = [0,0,1]
            self.geomgroup = self.handle.opt.geomgroup.copy()

    def copy_state(self, env):
        self.mj.mj_getState(env.model,env.data,self.state,self.spec)
        self.mj.mj_setState(self.model,self.data,self.state,self.spec)
        self.mj.mj_forward(self.model,self.data)

    def is_running(self):
        return self.handle.is_running()

    def update(self, env, controls):
        with self.handle.lock():
            # Native 0..4 shortcuts also toggle visual geometry groups. Reserve
            # these keys for effectiveness so the aircraft does not disappear.
            self.handle.opt.geomgroup[:5] = self.geomgroup[:5]
            self.copy_state(env)
            scene = self.handle.user_scn
            if scene is not None:
                scene.ngeom = 1
                self.mj.mjv_initGeom(scene.geoms[0],self.mj.mjtGeom.mjGEOM_SPHERE,
                    np.array([.025]*3),env.pos_des,np.eye(3).ravel(),np.array([.1,1,.2,.8],np.float32))
        self.handle.set_texts((None,None,HELP,
            f't={env.data.time:.2f}s  {"PAUSED" if controls.paused else "RUNNING"}\n'
            f'Target {env.pos_des.round(3)}\nMotor efficiency % {(exposed_values(controls.effectiveness, env.reaction_torque_layout)*100).round(0)}'))
        self.handle.sync()

    def close(self):
        self.handle.close()


def save_reports(artifacts, config, trace, rows, events, outcome, policy):
    metrics = trace_metrics(trace,config.evaluation.tail_fraction)
    metrics.update(outcome)
    np.savez_compressed(artifacts.run_dir/'trace.npz',
        **{f.name:getattr(trace,f.name) for f in fields(trace) if isinstance(getattr(trace,f.name),np.ndarray)},
        phases=np.asarray(trace.phases))
    # Same view_live report/metrics functions with a single real trace.
    from .reward_balance import build_reward_balance_report
    from .yaw_authority import build_yaw_authority_report, save_yaw_authority_plot
    from .wrench_authority import build_wrench_authority_report, save_wrench_authority_plot
    metrics['plot'] = None
    if rows:
        plot_trace, plot_rows = trace, rows
        if config.vehicle.reaction_torque_layout == 'user_frd':
            # Display only: native trace/CSV/authority calculations remain intact.
            motor_fields = ('motor_thrust','motor_thrust_command','motor_command',
                            'motor_omega_rad_s','reaction_torque_nm')
            plot_trace = replace(trace, **{k:user_from_native(getattr(trace,k)) for k in motor_fields})
            plot_rows = [dict(r, **{k:user_from_native(r[k]) for k in
                ('motor_effectiveness','motor_thrust_command','motor_thrust_before_effectiveness','motor_thrust_applied')}) for r in rows]
        path = _save_policy_report(artifacts,plot_trace,HoverMission.from_experiment(config),title_condition='interactive / '+config.vehicle.reaction_torque_layout+' motor numbering')
        metrics['plot'] = str(path.relative_to(artifacts.run_dir))
        from .plotting import save_interactive_motor_plot
        save_interactive_motor_plot(artifacts.run_dir/'plots/motor_effectiveness.png',plot_rows,events)
    from .velocity_reference import observation_contract, velocity_semantics, velocity_reward_semantics
    artifacts.write_metrics('evaluation',dict(status='failed' if outcome.get('error') else 'completed',
        observation_contract=observation_contract(config),velocity_semantics=velocity_semantics(config),
        velocity_reward_semantics=velocity_reward_semantics(config),
        control_mode='e2e', selected_policy='e2e', policies={'e2e':metrics}))
    for name, builder, plotter in (
        ('reward_balance',build_reward_balance_report,None),
        ('yaw_authority',build_yaw_authority_report,save_yaw_authority_plot),
        ('wrench_authority',build_wrench_authority_report,save_wrench_authority_plot)):
        report = builder([trace],config,model=policy.provenance['path'])
        if plotter and rows:
            plot = artifacts.path('plots',name,'.png')
            plotter(plot,[trace],config)
            report['plot'] = str(plot.relative_to(artifacts.run_dir))
        artifacts.write_metrics(name,report)
    return metrics


def run(config, policy, *, duration=None, seed=42, position_step=.02, headless=False,
        realtime=True, keys=None, boundary_hook=None, viewer_factory=LiveViewer, command=None):
    """Hooks inject events for tests only; each iteration performs at most one env.step."""
    config = evaluation_config(config)
    controls, keys = Controls(position_step), keys or KeyEvents()
    dt = 1/config.environment.policy_hz
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError('duration must be positive or None (unlimited)')
    limit = max(1,math.ceil(duration/dt-1e-10)) if duration is not None else math.inf
    artifacts = ArtifactManager.create(config,command=command,condition='interactive-motor-effectiveness',mission='hover',seed=seed)
    try:
        env = InteractiveEnv(config=config)
    except Exception as exc:
        artifacts.finalize('failed',end_reason='environment_initialization',error=f'{type(exc).__name__}: {exc}')
        print(f'results: {artifacts.run_dir}',flush=True)
        raise
    env.max_steps = limit  # Evaluation-only horizon; physical guards remain in env.step.
    rows, events = [], []
    stream = CsvStream(artifacts.run_dir/'rollout.csv',['time_sec','time_post','control_step'])
    event_stream = CsvStream(artifacts.run_dir/'events.csv',
        ['sequence','simulation_time','control_step','event_type','key','motor','applied']+
        [f'{key}_{i+1}' for key,n in (('effectiveness_before',4),('effectiveness_after',4),('target_before',3),('target_after',3)) for i in range(n)])
    outcome = dict(terminated=False,truncated=False,end_reason=None,error=None)
    viewer = None
    step = 0
    def event_sink(event):
        events.append(event)
        event_stream.append(event)
        print(f"[event t={env.data.time:.2f} step={step}] {event['event_type']} {event['key']}: target={env.pos_des.round(3)} effectiveness={exposed_values(controls.effectiveness, env.reaction_torque_layout).round(2)}",flush=True)
    try:
        adapter = EvaluationAdapter(env)
        adapter.reset_to_case_initial_state(Case('interactive',8.,goal=(0.,0.,1.)),seed)
        policy.bind(env)
        initial = adapter.snapshot()
        motors = rotor_mapping(env)
        payload = dict(randomize=False, mass_kg=config.environment.payload.mass,
            offset_body_xy_m=list(config.environment.payload.offset),
            offset_reference='drone body origin; z=0',
            total_mass_kg=float(env.model.body_mass[env.drone_bid]),
            center_of_mass_body_m=env.model.body_ipos[env.drone_bid].tolist(),
            inertia_diagonal_kg_m2=env.model.body_inertia[env.drone_bid].tolist(),
            center_of_mass_world_m=env.data.xipos[env.drone_bid].tolist(),
            inertia_frame_wxyz=env.model.body_iquat[env.drone_bid].tolist(),
            model='full compound inertia; refreshed constants; no explicit payload gravity torque',
            policy_hover_mass_kg=env.mass)
        print(f"Payload: {payload['mass_kg']*1000:g} g at body xy={payload['offset_body_xy_m']} m; total mass={payload['total_mass_kg']:g} kg",flush=True)
        for motor in motors:
            print(f"Motor {motor['number']} = {motor['force_actuator']} / {motor['site']}: body_xyz={motor['position_body_m']} motor_direction={motor['motor_direction']:+g}")
        print(HELP,flush=True)
        artifacts.write_runtime_config(dict(model=policy.provenance,initial_snapshot=initial,motors=motors,payload=payload,
            motor_layout=layout_metadata(env),
            duration_requested_sec=duration,max_policy_steps=None if math.isinf(limit) else limit,
            unlimited=duration is None,seed=seed,position_step_m=position_step,headless=headless,realtime=realtime,
            initial_effectiveness=[1.,1.,1.,1.], v_ref_world=[0.,0.,0.],normalization_frozen=True,
            input_config=str(config.source_path), overrides=dict(position_perturbation=0,attitude_perturbation_deg=0,
                initial_pose_randomization=None,payload_randomize=False,payload_mass_offset='preserved from input config',actuator_parameter_randomization=False,
                reset_rpm_mode='auto',evaluation_horizon='env.max_steps replaced only in evaluation instance'),
            timing=dict(time_sec='legacy view_live control-interval start; row state is post-step',
                time_post='actual post-state time; reference constant over interval',
                actuator='last physics substep output in [time_sec,time_post]',events='control boundary before predict'),
            effectiveness_model='fresh post-actuator f and signed reaction torque multiplied by lambda once; RPM state unchanged',
            viewer='MuJoCo 3.12.0 public PRESS-only callback; render-only model/data copies; refresh capped at 30 Hz',
            effectiveness_counter='integer count 0..50; lambda=(50-count)/50',
            source_sha256={name:sha256(ROOT/name) for name in ('view_e2e_interactive.py','crazyflie_rl/interactive_eval.py',
                'crazyflie_rl/motor_degradation.py','crazyflie_rl/plotting.py','crazyflie_rl/environment.py',
                'crazyflie_rl/velocity_reference.py','crazyflie_rl/config.py','crazyflie_rl/dr_policy.py')}))
        if not headless:
            viewer = viewer_factory(env,keys)
        deadline = time.monotonic()
        next_render = deadline
        while True:
            if boundary_hook:
                boundary_hook(step,keys,env,controls)
            if viewer and not viewer.is_running():
                outcome['end_reason'] = 'window_closed'
                break
            controls.process(keys.drain(),env,step,event_sink)
            if controls.quit:
                outcome['end_reason'] = 'quit'
                break
            if controls.paused:
                if viewer and time.monotonic() >= next_render:
                    viewer.update(env,controls)
                    next_render = time.monotonic() + 1/30
                time.sleep(.01)
                deadline = time.monotonic()
                continue
            t = step*dt
            obs = adapter.current_observation()
            before = adapter.read_state()
            v_des_before = env.desired_velocity(before['position'])
            action = np.asarray(policy.predict(obs))
            if action.shape != (4,) or not np.all(np.isfinite(action)):
                raise ValueError('policy returned invalid action')
            post_obs,reward,terminated,truncated,info = env.step(action)
            if not np.all(np.isfinite(post_obs)):
                raise ValueError('non-finite observation after physics step')
            # Preserve the view_live float32-derived trace fields and clock meaning.
            error_vector = post_obs[:3].astype(float)
            row = dict(time_sec=t,time_post=(step+1)*dt,control_step=step,phase='HOVER',
                position=error_vector+env.pos_des,reference_position=env.pos_des.copy(),
                attitude_deg=quaternion_to_euler_deg(post_obs[6:10]),position_error=float(np.linalg.norm(error_vector)),
                linear_velocity=(env._read_state()[2] if env.e2e_velocity is not None else post_obs[3:6].astype(float)),
                desired_velocity=info['desired_velocity'],velocity_error=info['velocity_error'],
                desired_velocity_before=v_des_before,velocity_error_before=before['velocity']-v_des_before,
                policy_input_time=t,angular_velocity=post_obs[10:13].astype(float),
                control_input=np.clip(action.astype(float),-1,1),policy_action=action.copy(),
                motor_thrust=env._last_f.copy(),motor_thrust_command=env._last_f_cmd.copy(),
                motor_command=env._last_motor_cmd.copy(),motor_omega_rad_s=env._last_omega.copy(),
                reaction_torque_nm=env._last_q_actual.copy(),wrench_command=env._last_wrench_cmd.copy(),
                wrench_actual=env._last_wrench_actual.copy(),allocation_error=env._last_allocation_error.copy(),
                motor_effectiveness=env.motor_effectiveness.copy(),motor_thrust_before_effectiveness=env.nominal_thrust.copy(),
                motor_thrust_applied=env._last_f.copy(),reaction_torque_before_effectiveness_nm=env.nominal_reaction_torque.copy(),
                position_before=before['position'],velocity_before=before['velocity'],raw_observation=obs.copy(),
                quaternion=env._read_state()[1],reward=float(reward),terminated=bool(terminated),truncated=bool(truncated))
            row.update(reaction_torque_layout=env.reaction_torque_layout, **user_motor_signals(row))
            rows.append(row)
            stream.append(row)
            step += 1
            outcome.update(terminated=bool(terminated),truncated=bool(truncated))
            if terminated or truncated:
                outcome['end_reason'] = ';'.join(termination_reasons(env,adapter.read_state(),env.pos_des)) if terminated else 'duration'
                break
            if viewer and time.monotonic() >= next_render:
                viewer.update(env,controls)
                next_render = time.monotonic() + 1/30
            if realtime:
                deadline += dt
                time.sleep(max(0.,deadline-time.monotonic()))
                if time.monotonic()-deadline > dt:
                    deadline = time.monotonic()  # Slow renderer: no skipped physics or catch-up burst.
    except KeyboardInterrupt:
        outcome['end_reason'] = 'keyboard_interrupt'
    except Exception as exc:
        outcome.update(end_reason='exception',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        controls.cancel_pending(env,step,event_sink)
        stream.close()
        event_stream.close()
        if viewer:
            try:
                viewer.close()
            except Exception as close_error:
                outcome['viewer_close_error'] = f'{type(close_error).__name__}: {close_error}'
                outcome['error'] = outcome.get('error') or outcome['viewer_close_error']
        outcome.update(actual_duration_sec=step*dt,requested_duration_sec=duration,control_steps=step,
                       checkpoint_hash_unchanged=sha256(policy.provenance['path'])==policy.provenance['sha256'])
        try:
            trace = make_trace(rows,env,outcome)
            save_reports(artifacts,config,trace,rows,events,outcome,policy)
        except Exception as save_error:
            # Streaming CSVs already exist. Never mask an active simulation exception.
            outcome['save_error'] = f'{type(save_error).__name__}: {save_error}'
            print(f'Report save failed: {outcome["save_error"]}',file=sys.stderr)
            if outcome.get('error') is None:
                outcome['error'] = outcome['save_error']
        finally:
            artifacts.finalize('failed' if outcome.get('error') else 'completed',**outcome)
            env.close()
            print(f'results: {artifacts.run_dir}',flush=True)
    if outcome.get('error'):
        raise RuntimeError(outcome['error'])
    return artifacts.run_dir,trace,outcome


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/eval_dr_transfer.yaml')
    p.add_argument('--model',required=True,help='one explicit E2E checkpoint zip')
    p.add_argument('--manifest',type=Path,help='optional archived training manifest')
    p.add_argument('--normalization',help='saved VecNormalize statistics or explicit none')
    p.add_argument('--position-step',type=float,default=.02,help='setpoint increment per key press, metres (default .02)')
    p.add_argument('--duration',type=float,default=0,help='simulation seconds; 0 (default) runs until user/physical termination')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--headless',action='store_true',help='noninteractive smoke check; requires finite --duration')
    p.add_argument('--no-realtime',action='store_true',help='headless checks only; GUI always uses wall-clock pacing')
    args = p.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 0 or args.seed < 0:
        p.error('duration must be finite and nonnegative; seed nonnegative')
    if args.headless and args.duration == 0:
        p.error('--headless requires positive --duration')
    if args.no_realtime and not args.headless:
        p.error('--no-realtime is only available with --headless')
    try:
        config = evaluation_config(load_config(args.config))
        Controls(args.position_step)
        policy = load_frozen_policy('e2e',args.model,config,manifest=args.manifest,normalization=args.normalization)
        metadata = policy.provenance['training_manifest'] or {}
        mode = metadata.get('control_mode',metadata.get('resolved_config',{}).get('environment',{}).get('control_mode'))
        if mode != 'e2e':
            raise ValueError('checkpoint must have verifiable E2E training metadata; use --manifest')
    except (ValueError,FileNotFoundError) as exc:
        p.error(str(exc))
    run(config,policy,duration=args.duration or None,seed=args.seed,position_step=args.position_step,
        headless=args.headless,realtime=not args.no_realtime,
        command=['view_e2e_interactive.py',*(sys.argv[1:] if argv is None else argv)])
    return 0

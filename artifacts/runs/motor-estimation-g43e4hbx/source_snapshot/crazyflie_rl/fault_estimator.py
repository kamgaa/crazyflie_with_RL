"""Causal five-hypothesis scalar window least squares, with no plant access.

Inputs are observable states and delivered ESC commands only. The nominal
motor replica has its own state; no effectiveness, actual thrust, RPM, qacc,
scenario, env or MuJoCo data is accepted by this module. This is not an MHE.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
import numpy as np

from .actuators import Cf21bFirstOrderActuatorModel
from .controllers import rotmat_from_quat_wxyz
from .motor_layout import exposed_motor_index, user_from_native


def frozen_array(value, shape):
    out = np.array(value, dtype=float, copy=True)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f'expected finite array {shape}')
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class ObservedState:
    time: float
    position_world: np.ndarray
    quaternion_wxyz: np.ndarray
    velocity_origin_world: np.ndarray
    omega_native: np.ndarray
    omega_time: float

    def __post_init__(self):
        if not np.isfinite([self.time, self.omega_time]).all() or self.omega_time > self.time+1e-12:
            raise ValueError('invalid observable timestamps')
        for key, shape in [('position_world',(3,)),('quaternion_wxyz',(4,)),
                           ('velocity_origin_world',(3,)),('omega_native',(3,))]:
            object.__setattr__(self, key, frozen_array(getattr(self,key), shape))
        if abs(np.linalg.norm(self.quaternion_wxyz)-1) > 1e-6:
            raise ValueError('quaternion must be normalized')


@dataclass(frozen=True)
class DeliveredCommands:
    start: float
    substep_dt: float
    esc_native: np.ndarray

    def __post_init__(self):
        if not np.isfinite([self.start,self.substep_dt]).all() or self.substep_dt <= 0:
            raise ValueError('invalid command interval')
        u=np.asarray(self.esc_native)
        if u.ndim != 2 or u.shape[1] != 4 or len(u)==0:
            raise ValueError('expected substeps x four delivered ESC inputs')
        object.__setattr__(self,'esc_native',frozen_array(u,u.shape))
        if np.any((u<0)|(u>1)): raise ValueError('ESC inputs must already be limited to [0,1]')


@dataclass(frozen=True)
class NominalModel:
    mass_kg: float
    com_native_m: np.ndarray
    inertia_com_native: np.ndarray
    force_columns_native: np.ndarray
    torque_origin_columns_native: np.ndarray
    gravity_world: np.ndarray
    actuator_kwargs: dict
    initial_hover_mass_kg: float

    def __post_init__(self):
        for k,s in [('com_native_m',(3,)),('inertia_com_native',(3,3)),
                    ('force_columns_native',(3,4)),('torque_origin_columns_native',(3,4)),
                    ('gravity_world',(3,))]:
            object.__setattr__(self,k,frozen_array(getattr(self,k),s))
        if self.mass_kg<=0 or np.linalg.eigvalsh(self.inertia_com_native).min()<=0:
            raise ValueError('invalid nominal mass/inertia')


@dataclass(frozen=True)
class EstimatorSettings:
    window_samples: int = 20
    recent_weight_ratio: float = .90
    residual_scales: tuple = (.0002,.0002,.0002,.002,.002,.002)
    ridge: float = .001
    h0_score_threshold: float = 9.
    minimum_relative_improvement: float = .50
    minimum_relative_margin: float = .15
    minimum_loss: float = .03
    minimum_information: float = 4.
    confirmation_samples: int = 3
    healthy_confirmation_samples: int = 5

    def __post_init__(self):
        if self.window_samples<2 or self.confirmation_samples<1 or self.healthy_confirmation_samples<1:
            raise ValueError('invalid sample counts')
        if not 0<self.recent_weight_ratio<=1 or self.ridge<0:
            raise ValueError('invalid weighting/regularization')
        if len(self.residual_scales)!=6 or min(self.residual_scales)<=0:
            raise ValueError('six positive residual scales required')
        if not np.isfinite(list(self.residual_scales)+[self.h0_score_threshold,self.ridge,self.minimum_information]).all():
            raise ValueError('nonfinite estimator settings')
        for v in (self.minimum_relative_improvement,self.minimum_relative_margin,self.minimum_loss):
            if not 0<=v<=1: raise ValueError('invalid decision fraction')


class MotorEfficiencyEstimator:
    """One update after each observed control interval; all calculations causal.

    y = X 1 + residual. For Hj, loss d=1-alpha minimizes
    sum(weights * ||(residual + X_j*d)/scale||²) + ridge*d².
    The bounded scalar analytic solution is clipped to [0,1]. H0 has d=0.
    """
    def __init__(self, model: NominalModel, settings: EstimatorSettings):
        self.model=model; self.settings=settings
        self.motor=Cf21bFirstOrderActuatorModel(**model.actuator_kwargs)
        self.motor.reset(airborne=True,episode_mass=model.initial_hover_mass_kg,
                         gravity_m_s2=float(np.linalg.norm(model.gravity_world)),randomize=False)
        self.inverse_inertia=np.linalg.inv(model.inertia_com_native)
        self.torque_com=model.torque_origin_columns_native-np.cross(
            np.broadcast_to(model.com_native_m,(4,3)),model.force_columns_native.T).T
        self.window=deque(maxlen=settings.window_samples)
        self.motor_history=deque(maxlen=32)
        self.last_time=None; self.update_count=0
        self.pending=None;self.pending_count=0;self.active=None

    def update(self, before: ObservedState, commands: DeliveredCommands, after: ObservedState):
        if type(before) is not ObservedState or type(after) is not ObservedState or type(commands) is not DeliveredCommands:
            raise TypeError('only explicit observable-state and delivered-command inputs are accepted')
        h=commands.substep_dt;dt=after.time-before.time
        if abs(h-self.motor.dt)>1e-12 or abs(commands.start-before.time)>1e-9 or abs(dt-len(commands.esc_native)*h)>1e-9:
            raise ValueError('commands and observations do not describe the same interval')
        if self.last_time is not None and abs(before.time-self.last_time)>1e-9:
            raise ValueError('noncontiguous or out-of-order update')
        self.last_time=after.time; self.update_count+=1
        f=[]
        for j,u in enumerate(commands.esc_native):
            # Shared existing motor model, driven solely by observed ESC inputs.
            value=self.motor.apply_motor_command(u).f_actual
            f.append(value);self.motor_history.append((commands.start+j*h,value.copy()))
        f=np.array(f)
        q0=before.quaternion_wxyz;q1=after.quaternion_wxyz
        if q0@q1<0:q1=-q1
        R0=rotmat_from_quat_wxyz(q0);R1=rotmat_from_quat_wxyz(q1)
        Xv=np.zeros((3,4))
        for j,fj in enumerate(f):
            # End observation is now available; interpolation is only over the
            # completed interval, never over a future control interval.
            a=j/len(f);q=(1-a)*q0+a*q1;q=q/np.linalg.norm(q)
            Xv += h*(rotmat_from_quat_wxyz(q)@self.model.force_columns_native)*fj/self.model.mass_kg
        # Gyro is the existing velocity-stage sensor cached before the final
        # Euler physics update. Its interval is shifted by one physics sample.
        impulse=np.zeros(4);covered=0.
        for t,fj in self.motor_history:
            overlap=max(0.,min(t+h,after.omega_time)-max(t,before.omega_time))
            impulse+=overlap*fj;covered+=overlap
        dw_time=after.omega_time-before.omega_time
        if dw_time<=0 or abs(covered-dw_time)>1e-8:
            raise ValueError('gyro interval is not covered by causal command history')
        Xw=(self.inverse_inertia@self.torque_com)*impulse
        wmid=(before.omega_native+after.omega_native)/2
        gyroscopic=self.inverse_inertia@np.cross(wmid,self.model.inertia_com_native@wmid)
        c=self.model.com_native_m
        v0=before.velocity_origin_world+R0@np.cross(before.omega_native,c)
        v1=after.velocity_origin_world+R1@np.cross(after.omega_native,c)
        y=np.r_[v1-v0-self.model.gravity_world*dt,
                after.omega_native-before.omega_native+dw_time*gyroscopic]
        X=np.vstack((Xv,Xw)); residual=y-X.sum(axis=1)
        self.window.append((X,residual))
        s=self.settings;scale=np.array(s.residual_scales)
        matrices=np.array([x/scale[:,None] for x,r in self.window])
        residuals=np.array([r/scale for x,r in self.window])
        weights=s.recent_weight_ratio**np.arange(len(self.window)-1,-1,-1)
        weights/=weights.sum()
        energy=np.einsum('n,nci,nci->i',weights,matrices,matrices)
        dot=np.einsum('n,nci,nc->i',weights,matrices,residuals)
        losses=np.clip(-dot/(energy+s.ridge),0,1)
        scores=np.einsum('n,nci,nci->i',weights,
            residuals[:,:,None]+matrices*losses,residuals[:,:,None]+matrices*losses)/6+s.ridge*losses**2/6
        scores=user_from_native(scores);alpha=user_from_native(1-losses)
        information=user_from_native(energy)
        h0=float(np.einsum('n,nc,nc->',weights,residuals,residuals)/6)
        order=np.argsort(scores);best=int(order[0]);second=int(order[1])
        improvement=float((h0-scores[best])/max(h0,1e-15))
        margin=float((scores[second]-scores[best])/max(h0,1e-15))
        enough=len(self.window)==s.window_samples and information[best]>=s.minimum_information
        proposal=None
        if enough:
            if h0<=s.h0_score_threshold:proposal=0
            elif (improvement>=s.minimum_relative_improvement and margin>=s.minimum_relative_margin
                  and 1-alpha[best]>=s.minimum_loss):proposal=best+1
        if proposal is None:self.pending=None;self.pending_count=0
        elif proposal==self.pending:self.pending_count+=1
        else:self.pending=proposal;self.pending_count=1
        count=s.healthy_confirmation_samples if proposal==0 else s.confirmation_samples
        if proposal is not None and self.pending_count>=count:self.active=proposal
        state=('insufficient_data' if not enough else 'healthy' if proposal==self.active==0
               else 'fault' if proposal==self.active and proposal is not None and proposal>0 else 'uncertain')
        eta=np.ones(4)
        if self.active is not None and self.active>0 and state in ('fault','uncertain'):
            eta[self.active-1]=alpha[self.active-1]
        return dict(estimate_time=after.time,gyro_observation_time=after.omega_time,
            estimator_state=state,estimated_motor=(self.active if state=='fault' else 0),
            candidate_motor=(proposal if proposal is not None else -1),
            raw_best_motor=best+1,estimated_efficiency_user=eta,
            hypothesis_alpha_user=alpha,hypothesis_scores=np.r_[h0,scores],
            hypothesis_information=information,relative_improvement=improvement,score_margin=margin,
            persistence_samples=self.pending_count,window_samples=len(self.window),
            nominal_delta_residual=residual,observed_delta=y,normal_prediction_delta=X.sum(axis=1),
            nominal_force_impulse_native=np.sum(f,axis=0)*h,estimator_update_count=self.update_count)

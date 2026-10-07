"""Read-only contract audit using isolated copies of the current MuJoCo model.

No alternative allocator or observation adapter is attached to an evaluation.
All coordinate round trips below are offline algebra / frozen inference tests.
"""
from __future__ import annotations
import argparse
import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from crazyflie_rl.artifacts import _git_metadata
from crazyflie_rl.config import load_config
from crazyflie_rl.controllers import rotmat_from_quat_wxyz
from crazyflie_rl.dr_policy import sha256, training_manifest
from crazyflie_rl.dr_transfer import ROOT, EvaluationAdapter, write_json
from crazyflie_rl.integral_eval import parameter_digest
from crazyflie_rl.integral_validation import Scenario, make_case
from crazyflie_rl.interactive_eval import evaluation_config, Controls
from crazyflie_rl.motor_degradation import apply_motor_effectiveness
from crazyflie_rl.oracle_eval import flat_csv
from crazyflie_rl.payload_motor_eval import DEFAULT_RECORD, RecordedFaultEnv, FaultObserver, condition_config, select_models

DOCS = {
 'free_joint': 'https://mujoco.readthedocs.io/en/stable/overview.html#floating-objects',
 'site_force': 'https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general',
 'camera_axes': 'https://mujoco.readthedocs.io/en/stable/programming/visualization.html',
 'jacobians': 'https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jacsubtreecom',
}


def clean(value):
    return json.loads(json.dumps(value, default=lambda v:v.tolist(), allow_nan=False))


def name(model, kind, index):
    return mujoco.mj_id2name(model, kind, int(index))


def descendants(model, root):
    result=[]
    for i in range(1,model.nbody):
        ancestor=i
        while ancestor not in (0,root):ancestor=int(model.body_parentid[ancestor])
        if ancestor==root:result.append(i)
    return result


def new_env(config, mass=0., offset=(0.,0.)):
    scenario=Scenario('audit',mass,offset)
    env=RecordedFaultEnv(config=condition_config(evaluation_config(config),scenario))
    adapter=EvaluationAdapter(env);adapter.reset_to_case_initial_state(make_case(scenario),42)
    observer=FaultObserver(scenario);observer.on_reset(adapter)
    return env,adapter


def quaternion(rotation):
    q=np.empty(4);mujoco.mju_mat2Quat(q,np.ascontiguousarray(rotation).ravel())
    return q if q[0]>=0 else -q


def orientation(angles):
    result=np.eye(3)
    for axis,angle in zip((2,1,0),np.radians(angles)):
        c,s=np.cos(angle),np.sin(angle)
        if axis==2:r=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        elif axis==1:r=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else:r=np.array([[1,0,0],[0,c,-s],[0,s,c]])
        result=result@r
    return result


def engine_geometry(env, data=None, model=None):
    """Independent of controllers.build_allocation_matrix and oracle helpers."""
    m=env.model if model is None else model;d=env.data if data is None else data;b=env.drone_bid
    R=d.xmat[b].reshape(3,3);origin=d.xpos[b];ids=descendants(m,b)
    masses=m.body_mass[ids];centroid=(masses[:,None]*d.xipos[ids]).sum(0)/masses.sum()
    np.testing.assert_allclose(centroid,d.subtree_com[b],atol=1e-14)
    I=np.zeros((3,3));body_records=[]
    for i in ids:
        Ri=R.T@d.ximat[i].reshape(3,3);ri=R.T@(d.xipos[i]-centroid)
        tensor=Ri@np.diag(m.body_inertia[i])@Ri.T
        I+=tensor+m.body_mass[i]*(ri@ri*np.eye(3)-np.outer(ri,ri))
        body_records.append(dict(id=i,name=name(m,mujoco.mjtObj.mjOBJ_BODY,i),mass_kg=float(m.body_mass[i]),
            com_native_m=R.T@(d.xipos[i]-origin),inertia_about_own_com_native_kg_m2=tensor))
    force=[];moment=[];reaction=[];rotors=[]
    # Derive signed reaction coefficient from the actual actuator implementation,
    # independently of allocator rows; no actuator dynamics are advanced.
    coefficients=env.actuator_model._reaction_torque(np.ones(4),np.zeros(4),np.zeros(4))
    for i,(fa,ta) in enumerate(zip(env.act_force,env.act_torque)):
        sid=int(m.actuator_trnid[fa,0]);tid=int(m.actuator_trnid[ta,0]);parent=int(m.site_bodyid[sid])
        assert m.actuator_trntype[fa]==mujoco.mjtTrn.mjTRN_SITE and parent==b
        assert m.actuator_trntype[ta]==mujoco.mjtTrn.mjTRN_SITE
        for a in (fa,ta):
            assert not m.actuator_ctrllimited[a] and not m.actuator_forcelimited[a]
            assert m.actuator_gainprm[a,0]==1 and not np.any(m.actuator_biasprm[a])
        Rs=R.T@d.site_xmat[sid].reshape(3,3);Rt=R.T@d.site_xmat[tid].reshape(3,3)
        r=R.T@(d.site_xpos[sid]-origin);f=Rs@m.actuator_gear[fa,:3]
        tm=np.cross(r,f)+Rs@m.actuator_gear[fa,3:]
        np.testing.assert_allclose(m.actuator_gear[ta,:3],0,atol=1e-15)
        tq=Rt@m.actuator_gear[ta,3:]*coefficients[i]
        force.append(f);moment.append(tm+tq);reaction.append(tq)
        prop=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'prop{i}')
        joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'prop{i}_hinge')
        rotors.append(dict(allocator_index=i,current_motor_number=i+1,keyboard_key=str(i+1),fault_event_motor_number=i+1,
            comparison_csv_suffix=i,interactive_csv_suffix=i+1,site=name(m,mujoco.mjtObj.mjOBJ_SITE,sid),
            force_actuator=name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,fa),torque_actuator=name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,ta),
            force_actuator_id=int(fa),torque_actuator_id=int(ta),site_parent=name(m,mujoco.mjtObj.mjOBJ_BODY,parent),
            site_local_position_m=m.site_pos[sid],site_local_quaternion_wxyz=m.site_quat[sid],site_to_body_matrix=Rs,
            position_native_m=r,force_per_actual_thrust_native=f,reaction_per_thrust_native_m=tq,
            rotor_body=f'prop{i}',rotor_parent=name(m,mujoco.mjtObj.mjOBJ_BODY,m.body_parentid[prop]),
            rotor_body_local_position_m=m.body_pos[prop],rotor_body_local_quaternion_wxyz=m.body_quat[prop],
            hinge=f'prop{i}_hinge',hinge_axis_local=m.jnt_axis[joint],
            direction_parameter=float(env.motor_direction[i]),actual_reaction_coefficient_m=float(coefficients[i]),
            direction_meaning='signed body reaction torque coefficient along torque-site +Z; not a prescribed propeller joint speed'))
    return dict(force=np.array(force).T,torque_O=np.array(moment).T,reaction=np.array(reaction).T,
        matrix_T=np.vstack((np.array(moment).T,np.ones(4))),rotors=rotors,
        total_mass_kg=float(masses.sum()),whole_com_native_m=R.T@(centroid-origin),whole_com_world_m=centroid,
        drone_body_mass_kg=float(m.body_mass[b]),drone_body_ipos_m=m.body_ipos[b].copy(),
        locked_composite_inertia_native_kg_m2=I,descendants=body_records)


def establish_frames(env,geometry):
    m,d,b=env.model,env.data,env.drone_bid;R=d.xmat[b].reshape(3,3)
    cid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'fpv_cam')
    C=R.T@d.cam_xmat[cid].reshape(3,3)
    forward,right,up=-C[:,2],C[:,0],C[:,1]
    np.testing.assert_allclose(forward,[1,0,0],atol=1e-14)
    np.testing.assert_allclose(right,[0,-1,0],atol=1e-14)
    np.testing.assert_allclose(up,[0,0,1],atol=1e-14)
    # Only establish FLU->FRD after checking the named, model-attached FPV basis.
    S=np.diag([1.,-1.,-1.]);np.testing.assert_allclose(S@S.T,np.eye(3));assert np.linalg.det(S)==1
    locations=((1,-1),(-1,-1),(-1,1),(1,1));names=('front_left','rear_left','rear_right','front_right')
    P=np.zeros((4,4));motor_table=[]
    for rotor in geometry['rotors']:
        r=S@rotor['position_native_m'];u=locations.index(tuple(np.sign(r[:2]).astype(int)))
        P[rotor['allocator_index'],u]=1
        torque=S@rotor['reaction_per_thrust_native_m'];zsign=int(np.sign(torque[2]))
        motor_table.append(dict(rotor,user_motor_number=u+1,user_position=names[u],position_frd_m=r,
            reaction_per_thrust_frd_m=torque,normal_body_reaction_top_view='CW' if zsign>0 else 'CCW',
            implied_propeller_spin_opposite_reaction='CCW' if zsign>0 else 'CW',
            commanded_propeller_spin='not prescribed by XML hinge or Python motor state',
            target_body_yaw_sign=(-1,1,-1,1)[u],matches_user_reaction=zsign==(-1,1,-1,1)[u]))
    return dict(native_world='right-handed Z-up; horizontal compass directions unspecified (not established NED/ENU)',
        native_body='model-defined FLU: +X forward, +Y left, +Z up',
        body_front_evidence='named fpv_cam mounted at +X, optical forward=-camera Z is +body X, up=+camera Y is +body Z',
        front_scope='confirmed model FPV convention; correspondence to physical hardware nose not independently measured',
        fpv_camera_position_body=m.cam_pos[cid],camera_forward_native=forward,camera_right_native=right,camera_up_native=up,
        gravity_world_m_s2=m.opt.gravity.copy(),initial_quaternion_wxyz=env.data.qpos[3:7].copy(),
        S_native_to_frd=S,S_orthogonality_error=float(np.max(np.abs(S@S.T-np.eye(3)))),S_determinant=float(np.linalg.det(S)),
        z_only_flip_determinant=-1.,P_native_from_user=P,permutation_definition='f_native = P @ f_user; eta_native = P @ eta_user',
        motors=motor_table)


def new_data(env, rotation, force):
    m=env.model;d=mujoco.MjData(m);d.qpos[:]=env.data.qpos;d.qpos[3:7]=quaternion(rotation);d.qvel[:]=0
    d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;d.ctrl[env.act_force]=force
    d.ctrl[env.act_torque]=env.actuator_model._reaction_torque(force,np.zeros(4),np.zeros(4))
    mujoco.mj_forward(m,d)
    assert d.ncon==0 and not np.any(d.qfrc_constraint) and not np.any(d.qfrc_passive)
    return d


def inspect_delta(env, baseline, after, geometry, delta_force, S):
    """Compare site r x F with engine generalized forces and Newton-Euler.

    All probe velocities are zero. Thus Jdot*qvel=0 and J*qacc is the
    ordinary world acceleration of the actual point identified by each J.
    """
    m=env.model;b=env.drone_bid;R=baseline.xmat[b].reshape(3,3)
    g=geometry;F=g['force']@delta_force;tauO=g['torque_O']@delta_force;c=g['whole_com_native_m']
    tauC=tauO-np.cross(c,F);dqacc=after.qacc-baseline.qacc
    dqfrc=after.qfrc_actuator-baseline.qfrc_actuator
    engine_F=R.T@dqfrc[:3];engine_tauO=dqfrc[3:6]
    Jc=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,baseline,Jc,b)
    Jo=np.zeros_like(Jc);Jr=np.zeros_like(Jc);mujoco.mj_jacBody(m,baseline,Jo,Jr,b)
    aC=R.T@(Jc@dqacc);aO=R.T@(Jo@dqacc);alpha=R.T@(Jr@dqacc)
    sum_force=np.zeros(3);sum_moment=np.zeros(3);relative_spin_moment=np.zeros(3)
    for item in g['descendants']:
        i=item['id'];Jp=np.zeros_like(Jc);Ji=np.zeros_like(Jc)
        mujoco.mj_jacBodyCom(m,baseline,Jp,Ji,i)
        ai=R.T@(Jp@dqacc);alphai=R.T@(Ji@dqacc);r=item['com_native_m']-c
        inertia=item['inertia_about_own_com_native_kg_m2']
        sum_force+=item['mass_kg']*ai
        sum_moment+=inertia@alphai+np.cross(r,item['mass_kg']*ai)
        relative_spin_moment+=inertia@(alphai-alpha)
    I=g['locked_composite_inertia_native_kg_m2']
    # Free propeller hinges are not RPM actuators. At zero speed, their axial
    # inertia decouples from drone yaw. Retain the full model, do not lock them.
    Ieff=I.copy()
    for item in g['descendants']:
        if item['id']==b:continue
        jid=int(m.body_jntadr[item['id']]);assert m.body_jntnum[item['id']]==1
        axis=R.T@baseline.xaxis[jid]
        tensor=item['inertia_about_own_com_native_kg_m2']
        Ieff-=np.outer(tensor@axis,axis)
    alpha_pred=np.linalg.solve(Ieff,tauC)
    errors=dict(force_map_n=float(np.max(np.abs(engine_F-F))),moment_map_nm=float(np.max(np.abs(engine_tauO-tauO))),
        com_force_newton_n=float(np.max(np.abs(g['total_mass_kg']*aC-F))),
        descendant_force_newton_n=float(np.max(np.abs(sum_force-F))),
        com_moment_euler_nm=float(np.max(np.abs(sum_moment-tauC))),
        inertia_plus_relative_hinges_nm=float(np.max(np.abs(I@alpha+relative_spin_moment-tauC))),
        alpha_from_com_effective_inertia_rad_s2=float(np.max(np.abs(alpha-alpha_pred))),
        com_origin_acceleration_m_s2=float(np.max(np.abs(aC-aO-np.cross(alpha,c)))))
    for k,v in errors.items():assert v < (1e-8 if 'alpha_' in k else 1e-11),(k,v)
    return dict(delta_actual_thrust_n=delta_force,delta_force_native_n=F,delta_force_frd_n=S@F,
        delta_torque_origin_native_nm=tauO,delta_torque_origin_frd_nm=S@tauO,
        delta_torque_com_native_nm=tauC,delta_torque_com_frd_nm=S@tauC,
        engine_delta_force_native_n=engine_F,engine_delta_torque_origin_native_nm=engine_tauO,
        delta_qacc=dqacc,delta_origin_acceleration_world_m_s2=Jo@dqacc,
        delta_whole_com_acceleration_world_m_s2=Jc@dqacc,delta_whole_com_acceleration_native_m_s2=aC,
        delta_angular_acceleration_native_rad_s2=alpha,delta_angular_acceleration_frd_rad_s2=S@alpha,
        delta_angular_acceleration_predicted_from_com_rad_s2=alpha_pred,
        whole_com_native_m=c,drone_body_ipos_native_m=g['drone_body_ipos_m'],
        locked_composite_inertia_native_kg_m2=I,effective_com_inertia_native_kg_m2=Ieff,
        effective_com_inertia_frd_kg_m2=S@Ieff@S.T,relative_hinge_momentum_rate_nm=relative_spin_moment,
        baseline_qacc=baseline.qacc.copy(),after_qacc=after.qacc.copy(),gravity_world_m_s2=m.opt.gravity.copy(),
        contacts_before=int(baseline.ncon),contacts_after=int(after.ncon),errors=errors)


def increments(config,S):
    records=[];mass_records=[]
    for label,mass,offset in [('nominal',0.,(0.,0.)),('payload_5g_pos_x',.005,(.03,0.))]:
        env,adapter=new_env(config,mass,offset)
        try:
            before=adapter.snapshot();model_before=model_digest(env.model)
            # Own an isolated compiled model copy as well as separate MjData.
            isolated=copy.copy(env.model);original=env.model;env.model=isolated
            try:
                for pose,R in [('level',np.eye(3)),('rotated',orientation((37,-17,11)))]:
                    f=np.full(4,.10);baseline=new_data(env,R,f);g=engine_geometry(env,baseline)
                    mass_records.append(dict(condition=label,pose=pose,payload_mass_kg=mass,
                        payload_offset_native_m=[*offset,0],payload_offset_frd_m=S@np.r_[offset,0.],
                        total_mass_kg=g['total_mass_kg'],whole_com_native_m=g['whole_com_native_m'],
                        drone_body_ipos_m=g['drone_body_ipos_m'],locked_composite_inertia_native_kg_m2=g['locked_composite_inertia_native_kg_m2']))
                    for i in range(4):
                        for sign in (-1,1):
                            delta=np.zeros(4);delta[i]=sign*.001
                            after=new_data(env,R,f+delta)
                            row=inspect_delta(env,baseline,after,g,delta,S)
                            row.update(condition=label,pose=pose,current_motor_number=i+1,delta_n=float(delta[i]),
                                physical_path='direct actual force/reaction controls in isolated MjData; actuator lag not bypassed in any rollout')
                            records.append(row)
            finally:env.model=original
            assert before==adapter.snapshot() and model_digest(env.model)==model_before
        finally:env.close()
    return clean(records),clean(mass_records)


def model_digest(model):
    import hashlib
    h=hashlib.sha256()
    for key in ('body_mass','body_ipos','body_inertia','body_iquat','body_pos','site_pos','site_quat','actuator_gear','jnt_axis'):
        h.update(np.ascontiguousarray(getattr(model,key)).tobytes())
    h.update(model.opt.gravity.tobytes());h.update(np.array([model.opt.timestep]).tobytes())
    return h.hexdigest()


def efficiency_loss_checks(config,frames):
    records=[];S=frames['S_native_to_frd'];P=frames['P_native_from_user']
    user_one_index=int(np.flatnonzero(P[:,0])[0])
    for description,index in [('current_motor_1',0),('user_FRD_motor_1',user_one_index)]:
        env,adapter=new_env(config)
        try:
            # Fresh ordinary allocation + motor dynamics first. Then compare
            # one output-loss call at the same physical/RPM state.
            command=np.array([0.,0.,0.,env.mass*env.gravity]);env._apply_control(command)
            baseline_thrust=env._last_f.copy();baseline_reaction=env._last_q_actual.copy()
            baseline=mujoco.MjData(env.model);baseline.qpos[:]=env.data.qpos;baseline.ctrl[:]=env.data.ctrl
            mujoco.mj_forward(env.model,baseline);g=engine_geometry(env,baseline)
            snapshot=adapter.snapshot();omega=env.actuator_model.omega.copy();eta=np.ones(4);eta[index]=.7
            nominal,torque=apply_motor_effectiveness(env,eta)
            np.testing.assert_array_equal(nominal,baseline_thrust);np.testing.assert_array_equal(torque,baseline_reaction)
            np.testing.assert_array_equal(env.actuator_model.omega,omega)
            for key in ('qpos','qvel','position','quaternion','velocity','omega','reference','previous_action'):
                assert adapter.snapshot()[key]==snapshot[key]
            np.testing.assert_array_equal(env._last_f,eta*nominal)
            np.testing.assert_array_equal(env._last_q_actual,eta*torque)
            after=mujoco.MjData(env.model);after.qpos[:]=env.data.qpos;after.ctrl[:]=env.data.ctrl;mujoco.mj_forward(env.model,after)
            row=inspect_delta(env,baseline,after,g,env._last_f-baseline_thrust,S)
            loss=float(baseline_thrust[index]-env._last_f[index]);ell=float(abs(g['rotors'][index]['position_native_m'][0]));k=env.torque_coefficient
            target=np.array([-ell*loss,-ell*loss,k*loss]) if description=='user_FRD_motor_1' else None
            row.update(test=description,current_motor_number=index+1,user_motor_number=g['rotors'][index]['allocator_index'],
                loss_magnitude_n=loss,effectiveness=eta,nominal_thrust_before=nominal,actual_thrust_after=env._last_f.copy(),
                nominal_reaction_before=torque,actual_reaction_after=env._last_q_actual.copy(),
                user_motor1_expected_fault_torque_frd_nm=target,
                user_motor1_expected_minus_observed_nm=None if target is None else target-row['delta_torque_origin_frd_nm'],
                motor_internal_state_unchanged=True,physical_state_and_reference_unchanged=True,
                control_write_once=True)
            row['user_motor_number']=int(np.flatnonzero(P[index])[0])+1
            records.append(row)
        finally:env.close()
    return clean(records)


def state_frame_checks(config,S):
    env,adapter=new_env(config,.005,(.03,0.))
    try:
        R=orientation((37,-17,11));env.data.qpos[3:7]=quaternion(R)
        env.data.qvel[:3]=[.12,-.05,.04];env.data.qvel[3:6]=[.2,-.3,.4];env.data.qvel[6:]=0
        mujoco.mj_forward(env.model,env.data)
        p,q,v,omega=env._read_state();obs=env._obs(p,q,v,omega)
        np.testing.assert_allclose(rotmat_from_quat_wxyz(q),R,atol=1e-14)
        np.testing.assert_allclose(env.data.xmat[env.drone_bid].reshape(3,3),R,atol=1e-14)
        Jp=np.zeros((3,env.model.nv));Jr=np.zeros_like(Jp);Jc=np.zeros_like(Jp)
        mujoco.mj_jacBody(env.model,env.data,Jp,Jr,env.drone_bid)
        mujoco.mj_jacSubtreeCom(env.model,env.data,Jc,env.drone_bid)
        np.testing.assert_allclose(Jp@env.data.qvel,v,atol=1e-14)
        np.testing.assert_allclose(R.T@(Jr@env.data.qvel),omega,atol=1e-14)
        np.testing.assert_allclose(omega,env.data.qvel[3:6],atol=1e-14)
        Rfrd=R@S.T
        np.testing.assert_allclose(Rfrd@S,R,atol=1e-14)
        # This model's native +yaw is CCW about world/body +Z at level. A
        # coordinate rotation changes torque/vector components, not world yaw by decree.
        return clean(dict(quaternion_order='wxyz; sign canonicalized to w>=0 in _read_state',
            rotation_direction='R_world_from_native_body',rotation=R,rotation_world_from_frd=Rfrd,
            position_world=p,velocity_body_origin_world=v,velocity_whole_com_world=Jc@env.data.qvel,
            gyro_native_body=omega,gyro_world=R@omega,gyro_frd=S@omega,
            qvel_linear_world=env.data.qvel[:3].copy(),qvel_angular_native_body=env.data.qvel[3:6].copy(),
            imu_site_quaternion=env.model.site_quat[mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_SITE,'imu')],
            raw_observation=obs,quaternion_matches_engine=True,gyro_matches_body_qvel=True,
            origin_velocity_differs_from_whole_com=bool(np.linalg.norm(v-Jc@env.data.qvel)>1e-5),
            qacc_interpretation='derivative of qvel: translation of free-body origin in world; rotation in body tangent basis; not subtree COM translation',
            diagnostic_method='zero-velocity increment probes use J_subtreeCOM @ delta_qacc; child inertia/hinge acceleration retained'))
    finally:env.close()


def matrix_audit(env,g,frames):
    S=frames['S_native_to_frd'];P=frames['P_native_from_user'];H=np.eye(4);H[:3,:3]=S
    ell=abs(g['rotors'][0]['position_native_m'][0]);k=env.torque_coefficient
    target=np.array([[ell,ell,-ell,-ell],[ell,-ell,-ell,ell],[-k,k,-k,k],[1,1,1,1]])
    physical=g['matrix_T'];user=H@physical@P;allocation=env.B.copy()
    rng=np.random.default_rng(42);roundtrip_errors=[];oracle_errors=[]
    for _ in range(256):
        action=rng.uniform(-1,1,4);w=env.residual_scale*action+np.array([0,0,0,env.mass*env.gravity])
        baseline=np.clip(np.linalg.pinv(allocation)@w,env.thrust_min,env.thrust_max)
        user_command=np.clip(np.linalg.pinv(H@allocation@P)@(H@w),env.thrust_min,env.thrust_max)
        roundtrip_errors.append(np.max(np.abs(P@user_command-baseline)))
        eta_native=np.array([.7,.8,.9,1.]);eta_user=P.T@eta_native
        base=np.clip(np.linalg.pinv(allocation@np.diag(eta_native))@w,env.thrust_min,env.thrust_max)
        transformed=np.clip(np.linalg.pinv((H@allocation@P)@np.diag(eta_user))@(H@w),env.thrust_min,env.thrust_max)
        oracle_errors.append(np.max(np.abs(P@transformed-base)))
    assert max(roundtrip_errors)<1e-12 and max(oracle_errors)<1e-12
    desired=np.array([0.,0.,1e-4,.4]);wrong_commands=np.linalg.solve(target,desired)
    naive_actual=user@wrong_commands
    return clean(dict(wrench_order=['tau_x_Nm','tau_y_Nm','tau_z_Nm','positive_total_thrust_N'],
        moment_origin='drone body origin O for every matrix in this comparison',
        allocator_native=allocation,physical_native_independent=physical,
        physical_user_frd=user,allocator_user_frd=H@allocation@P,user_target=target,
        allocator_minus_physical_native=allocation-physical,physical_user_minus_target=user-target,
        source_axis_arm_m=env.arm_length,xml_axis_arm_m=float(ell),xml_radial_arm_m=float(np.sqrt(2)*ell),
        axis_arm_difference_m=float(ell-env.arm_length),axis_arm_relative_difference_percent=float((ell/env.arm_length-1)*100),
        torque_per_thrust_m=k,native_positive_T_equals_body_Fz=True,frd_body_Fz_equals_negative_T=True,
        physical_user_signed_Fz=np.diag([1,1,1,-1])@user,
        translation_only_transform='world axes untouched; body vectors S, R_world_from_FRD=R_world_from_native@S.T',
        matrix_identity='B_user = blockdiag(S,1) @ B_native @ P',
        motor_command_equivalence_max_n=float(max(roundtrip_errors)),oracle_equivalence_max_n=float(max(oracle_errors)),
        offline_samples=256,naive_target_allocator_only=dict(desired_frd_wrench=desired,
            hypothetical_user_motor_commands_n=wrong_commands,actual_existing_plant_frd_wrench=naive_actual,
            note='algebra only; never connected to plant or evaluation; positive desired yaw would become negative yaw'),
        internal_sign_and_order_match=bool(np.allclose(allocation[2:],physical[2:],atol=1e-15)
                                           and np.array_equal(np.sign(allocation),np.sign(physical))),
        target_roll_pitch_T_match=bool(np.allclose(user[[0,1,3]],target[[0,1,3]],atol=1e-14)),
        target_yaw_reversed=bool(np.allclose(user[2],-target[2],atol=1e-14))))


def policy_audit(config,S,P):
    record=json.loads(DEFAULT_RECORD.read_text());policies=select_models(DEFAULT_RECORD,config)
    previous=json.loads((ROOT/'artifacts/runs/ab-oracle-0r_ekf0x/manifest.json').read_text())
    policy_records=[];samples=[]
    for policy in policies:
        label=policy.provenance['label'];old=next(x for x in previous['models'] if x['label']==label)
        for k in ('path','sha256','normalization','observation_contract'):
            assert policy.provenance[k]==old[k]
        before=parameter_digest(policy);path=Path(policy.provenance['path']);run=path.parent.parent
        mp,md=training_manifest(path);source_path=run/'training_source_record.json';source=json.loads(source_path.read_text())
        source_checks={p:dict(recorded=h,current=sha256(ROOT/p),matches=sha256(ROOT/p)==h)
                       for p,h in source['source_sha256'].items()}
        entry=record['training'][label[0].lower()]
        env,adapter=new_env(config);policy.bind(env)
        try:
            for index,(position,velocity,angles,omega) in enumerate([
                ([0,0,1],[0,0,0],(0,0,0),[0,0,0]),
                ([.02,-.04,1.01],[.1,-.2,.03],(25,-5,8),[.2,-.3,.4]),
                ([.7,-.2,1.2],[-.5,.1,-.1],(-42,20,-13),[-.1,.2,.5]),
                ([-.4,.3,.8],[.4,.2,.1],(135,-15,23),[.4,-.1,-.2]),
            ]):
                R=orientation(angles);q=quaternion(R);raw=env._obs(position,q,velocity,omega)
                # Algebra only: world position/velocity remain world; only the
                # orientation and body angular vector are re-expressed and undone.
                Rfrd=R@S.T;omega_frd=S@np.array(omega);restored=raw.copy()
                restored[6:10]=quaternion(Rfrd@S);restored[10:13]=S.T@omega_frd
                action=policy.predict(raw);action_restored=policy.predict(restored)
                np.testing.assert_allclose(raw,restored,rtol=0,atol=1e-7)
                np.testing.assert_allclose(action,action_restored,rtol=0,atol=1e-6)
                w=env.residual_scale*action+np.array([0,0,0,env.mass*env.gravity])
                H=np.eye(4);H[:3,:3]=S;wu=H@w
                fn=np.clip(env.B_pinv@w,env.thrust_min,env.thrust_max)
                fu=np.clip(np.linalg.pinv(H@env.B@P)@wu,env.thrust_min,env.thrust_max)
                np.testing.assert_allclose(P@fu,fn,atol=1e-12)
                samples.append(dict(label=label,sample=index,raw_observation=raw,restored_observation=restored,
                    action_native=action,action_after_roundtrip=action_restored,wrench_native=w,wrench_frd=wu,
                    motor_command_native_n=fn,motor_command_user_n=fu,
                    observation_max_error=float(np.max(np.abs(raw-restored))),action_max_error=float(np.max(np.abs(action-action_restored))),
                    motor_command_max_error_n=float(np.max(np.abs(P@fu-fn)))))
        finally:env.close()
        assert parameter_digest(policy)==before and sha256(path)==policy.provenance['sha256']
        best_meta=[p for p in (run/'manifests').glob('*best-model*.json') if json.loads(p.read_text()).get('path')==entry['best']['path']]
        policy_records.append(dict(label=label,checkpoint_provenance=policy.provenance,
            timestep=int(policy.model.num_timesteps),parameter_sha256=before,parameters_unchanged=True,
            training_manifest_path=str(mp),training_manifest_sha256=sha256(mp),
            saved_resolved_config=md.get('resolved_config'),training_normalization_record=entry['best']['normalization'],
            source_record_path=str(source_path),source_hash_comparisons=source_checks,
            historical_xml_content_hash_available='resources/mujoco/cf21B_500.xml' in source['source_sha256'],
            historical_xml_limitation='training saved XML path/config; full historical XML content not in training_source_record',
            adapter_connected=False,evaluation_rollout_executed=False))
    return clean(policy_records),clean(samples)


def collect_paths(config_path,xml_path):
    config_chain=[];path=Path(config_path).resolve()
    while True:
        config_chain.append(dict(path=str(path),sha256=sha256(path)))
        obj=yaml.safe_load(path.read_text());parent=obj.get('extends')
        if parent is None:break
        path=(path.parent/parent).resolve()
    xml_records=[];asset_records=[];seen=set()
    def visit(path):
        path=path.resolve()
        if path in seen:return
        seen.add(path);root=ET.parse(path).getroot()
        includes=[(path.parent/n.attrib['file']).resolve() for n in root.iter('include')]
        xml_records.append(dict(path=str(path),sha256=sha256(path),includes=[str(p) for p in includes]))
        compiler=root.find('compiler');meshdir=compiler.get('meshdir','') if compiler is not None else ''
        for mesh in root.findall('./asset/mesh'):
            if mesh.get('file'):
                asset=(path.parent/meshdir/mesh.get('file')).resolve()
                asset_records.append(dict(name=mesh.get('name'),path=str(asset),sha256=sha256(asset)))
        for p in includes:visit(p)
    visit(Path(xml_path))
    return dict(config_inheritance=config_chain,xml_dependencies=xml_records,mesh_assets=asset_records)


def frame_table(config,env):
    return [dict(quantity=q,frame_or_order=f,units=u,meaning=meaning,evidence=evidence) for q,f,u,meaning,evidence in [
        ('world','right-handed Z-up, compass axes unspecified','m','gravity [0,0,-9.81]; no NED/ENU conversion','engine opt.gravity; XML ground plane'),
        ('body','FLU in model FPV basis','m','+X forward / +Y left / +Z up','fpv_cam optical -Z and up +Y expressed in drone frame'),
        ('initial_pose','body axes aligned with world at identity wxyz','m,rad','evaluation p=[0,0,1], q=[1,0,0,0], v/omega=0','dr_transfer.EvaluationAdapter; FaultObserver nominal rotor reset'),
        ('position','world, drone freejoint/body origin O','m','qpos[:3], not whole-aircraft COM','environment._read_state'),
        ('linear_velocity','world, velocity of O','m/s','qvel[:3], not COM velocity if offset and rotating','environment._read_state; Jacobian check'),
        ('quaternion','wxyz, R_world_from_native_body','unit quaternion','normalized; canonical w>=0; body vector world=R@body','controllers.rotmat_from_quat_wxyz; environment._read_state'),
        ('angular_velocity','native body/IMU frame','rad/s','gyro site has identity orientation; qvel[3:6] agrees locally','imu_gyro; state_frames.json'),
        ('position_error_observation','world','m','p-p_target; integral wrapper copies only first three raw channels','environment._obs; integral_controller.prepare_observation'),
        ('velocity_observation_AB','world absolute actual velocity','m/s','both A and B observe v; only B training reward uses velocity error','saved training manifests; velocity_reference.py'),
        ('yaw_error','wrapped world-heading yaw of native body +X','rad','atan2(...)-yaw_target; +yaw about world +Z is CCW at level','environment._yaw_err; quaternion definition'),
        ('integral_xi','world','m','p_cmd=p_target+xi; integrate e_true in world; no FRD rotation while world remains unchanged','integral_controller.py'),
        ('PPO_action','[body tau_x,body tau_y,body tau_z,delta positive T]','normalized [-1,1]','scale [0.0075,0.0075,0.001,0.5] => [Nm,Nm,Nm,N]','environment.step'),
        ('hover_bias','native body +Z force / positive T','N',f'nominal mass*g={env.mass*env.gravity}; not payload or fault trim','environment.step'),
        ('allocator_wrench','native body, moment about O; [tau_x,tau_y,tau_z,T]','Nm,Nm,Nm,N','native Fz=+T; under FRD Fz=-T; do not rename T by changing its sign','controllers.build_allocation_matrix; site gears'),
        ('motor_command_pipeline','native array order 0..3','N -> rad/s -> ESC[0,1] -> rad/s -> N',
         'pinv(B or B_eta) -> rotor clip[0,.20] -> inverse thrust -> ESC clip -> 0.05s lag -> nominal force/reaction -> eta once','environment._apply_control; actuators.apply/_advance; apply_motor_effectiveness'),
        ('rotor_force_application','motor site local +Z; sites parented to drone','N','MuJoCo site transmission creates force at site and r cross F about O','XML motor*_force gears; independent engine differential'),
        ('rotor_reaction_application','torque site local +Z times signed ctrl','Nm','free couple; coefficients +,-,+,-; scales with same eta once','XML motor*_torque; actuators._legacy_reaction_torque'),
        ('external_torque','configured body torque rotated to world before engine application','Nm','dist_torque_body retained; no separate payload gravity torque in step','environment.step; _set_com_bias'),
        ('legacy_wrench_log','native body, about O; last physics interval','Nm,Nm,Nm,N','B0@actual thrust with explicit reaction sum; approximate arm geometry','environment._record_actuator_output'),
        ('oracle_XML_wrench_log','native body, about O; excludes gravity/external torque','Nm,Nm,Nm,N','actual XML site geometry and rotor reaction; more exact than legacy B0 log','oracle_allocation.allocation_diagnostics'),
        ('freejoint_qacc','world origin translation + local-body angular tangent acceleration','m/s2,rad/s2','not directly whole COM acceleration; audit uses subtree COM Jacobian at zero velocities','MuJoCo docs + isolated nonzero-COM tests'),
        ('rotor_visual_joint','prop* hinge local +Z, independent of Python RPM state','rad,rad/s','no hinge motor actuators; no code prescribing alternating spin; passive relative joint motion is not commanded RPM','XML eight site actuators; source scan; visual_state.json'),
    ]]


def recommendations():
    return [
        dict(kind='representation_only',status='proposed_not_connected',
             changes='Expose FRD/user-numbered diagnostics using S and P while preserving world frame and original native observation before frozen PPO.',
             positions='body vectors S; world p/v/target/xi unchanged; R_W_FRD=R_W_native S.T; inertia S I S.T; no Euler sign patch',
             actions='w_FRD=blockdiag(S,1) w_native for positive T; convert back before current path, or transform B consistently offline',
             motors='all commands/eta/events/logs require the same P; user motor 1 maps native index 3',
             tests='raw/normalized observation + action/rotor command and complete nominal/fault rollout equality; actuator and integral history continuity'),
        dict(kind='physical_target_reaction_change',status='proposed_not_performed',
             changes='Current FRD/user-order yaw reaction row is opposite the requested row. If desired hardware spin assignment is authoritative, coordinate/permutation alone cannot fix this.',
             targets=['configs/base.yaml / vehicle.motor_direction and selected profile overrides',
                      'crazyflie_rl/actuators.py::_legacy_reaction_torque/_reaction_torque',
                      'crazyflie_rl/controllers.py::build_allocation_matrix yaw row',
                      'resources/mujoco/cf21B_500.xml motor*_torque gear only if changing torque-axis rather than direction parameter'],
             rule='Choose one physical sign-change location, propagate the resulting sign to allocation/logs; never flip both direction and gear accidentally.',
             policy_risk='A/B trained for native observations and present action-to-plant map. A changed plant/allocator is not automatically checkpoint-compatible; preserve separate legacy mode/contracts and revalidate.',
             tests='repeat independent rotor increment/efficiency loss audit, command-to-actual yaw sign, frozen policy regression before any new evaluation'),
        dict(kind='geometry_rounding',status='observed_no_change',
             changes='allocator axis offset .035355 vs XML .03536m: 5 micrometres per axis. This is separate from FRD numbering and reversed target yaw.',
             targets=['vehicle.arm_length','controllers.build_allocation_matrix','XML motor site positions'],
             policy_risk='do not silently normalize old reports/checkpoints to a changed geometry'),
        dict(kind='propeller_spin_model',status='unconfirmed_physical_hardware',
             changes='Python angular speed is nonnegative actuator state and does not drive XML hinge qvel. CW/CCW propeller rotation must not be inferred from direction or propL/propR mesh names.',
             tests='If physical propeller spin is later specified, separately verify geometry handedness, RPM animation, reaction sign and rotor inertia terms.'),
    ]


def execute_audit(args):
    config=load_config(args.config);resolved=evaluation_config(config)
    previous=json.loads((ROOT/'artifacts/runs/ab-oracle-0r_ekf0x/manifest.json').read_text())
    recovery=json.loads((ROOT/'artifacts/runs/oracle-recovery-m3dh3vij/manifest.json').read_text())
    assert resolved.resolved_dict()==previous['common_resolved_config']==recovery['common_resolved_config']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='coordinate-audit-',dir=args.output_dir));print('results:',directory,flush=True)
    protected_paths=set(json.loads((ROOT/'artifacts/runs/oracle-recovery-m3dh3vij/initial_protected_hashes.json').read_text()))
    for folder in ('crazyflie_rl','configs','resources','docs','tests'):
        protected_paths.update(str(p) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    protected_paths.update(str(p) for p in ROOT.glob('*.py'))
    for folder in ('ab-oracle-0r_ekf0x','oracle-recovery-m3dh3vij'):
        protected_paths.update(str(p) for p in (ROOT/'artifacts/runs'/folder).rglob('*') if p.is_file())
    record=json.loads(DEFAULT_RECORD.read_text())
    for label in ('a','b'):
        root=Path(record['training'][label]['run_dir']);protected_paths.update(str(p) for p in root.rglob('*') if p.is_file())
    protected={p:sha256(p) for p in sorted(protected_paths)};write_json(directory/'protected_hashes_before.json',protected)
    manifest=dict(status='running',mujoco_version=mujoco.__version__,config=str(args.config.resolve()),
        git=_git_metadata(ROOT),source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'crazyflie_rl').glob('*.py')},
        audit_script_sha256=sha256(Path(__file__)),resolved_config=resolved.resolved_dict(),documentation=DOCS,
        no_training=True,no_optimizer_updates=True,production_model_or_control_modified=False,isolated_model_copies=True)
    command=['python','audit_coordinate_contracts.py','--config',str(args.config.resolve()),'--output-dir',str(args.output_dir.resolve())]
    (directory/'rerun.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '+shlex.join(command)+'\n')
    write_json(directory/'manifest.json',manifest)
    try:
        env,adapter=new_env(config)
        try:
            geo=engine_geometry(env);frames=establish_frames(env,geo);table=frame_table(config,env)
            matrices=matrix_audit(env,geo,frames)
            manifest.update(loaded_xml_path=env.xml_path,paths=collect_paths(args.config,env.xml_path),
                            initial_snapshot=adapter.snapshot(),actuator=env.actuator_snapshot())
            # A differential input test cannot treat rendering joint qvel as motor RPM.
            visual=dict(prop_hinge_qpos=env.data.qpos[7:].copy(),prop_hinge_qvel=env.data.qvel[6:].copy(),
                actuator_omega_rad_s=env.actuator_model.omega.copy(),engine_actuated_transmissions=env.model.actuator_trntype.copy(),
                engine_actuator_names=[name(env.model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(env.model.nu)],
                hinge_motor_actuators=0,render_path='viewer copies physics state; no commanded RPM animation',
                physical_spin_direction='not explicitly prescribed; opposite-reaction direction is a model interpretation only')
            # Exercise the existing keyboard-index implementation without stepping.
            keyboard=[]
            for i in range(4):
                controls=Controls();events=[];controls.process([str(i+1)],env,0,events.append)
                eta=np.ones(4);eta[i]=.98;np.testing.assert_array_equal(env.motor_effectiveness,eta)
                keyboard.append(dict(key=str(i+1),allocator_index=i,event_motor=events[-1]['motor'],effectiveness=eta))
        finally:env.close()
        write_json(directory/'frames.json',clean(frames));flat_csv(directory/'frame_table.csv',table)
        write_json(directory/'frame_table.json',table);flat_csv(directory/'motor_mapping.csv',frames['motors'])
        write_json(directory/'motor_mapping.json',clean(frames['motors']));write_json(directory/'matrix_comparison.json',matrices)
        flat_csv(directory/'matrix_comparison.csv',[dict(matrix=k,**{f'row{i}':row for i,row in enumerate(v)})
            for k,v in matrices.items() if isinstance(v,list) and len(v)==4 and isinstance(v[0],list)])
        write_json(directory/'visual_state.json',clean(visual));write_json(directory/'keyboard_index_checks.json',clean(keyboard))
        print('frame/matrix audit done',flush=True)
        probes,masses=increments(config,frames['S_native_to_frd'])
        write_json(directory/'rotor_increments.json',probes);flat_csv(directory/'rotor_increments.csv',probes)
        write_json(directory/'mass_com_inertia.json',masses);flat_csv(directory/'mass_com_inertia.csv',masses)
        losses=efficiency_loss_checks(config,frames);write_json(directory/'efficiency_loss.json',losses);flat_csv(directory/'efficiency_loss.csv',losses)
        state=state_frame_checks(config,frames['S_native_to_frd']);write_json(directory/'state_frames.json',state)
        print('32 rotor increments and two efficiency loss checks done',flush=True)
        policies,samples=policy_audit(resolved,frames['S_native_to_frd'],frames['P_native_from_user'])
        write_json(directory/'policy_contracts.json',policies);write_json(directory/'policy_roundtrip.json',samples)
        flat_csv(directory/'policy_roundtrip.csv',samples);write_json(directory/'change_candidates.json',recommendations())
        maxima={k:max(x['errors'][k] for x in probes+losses) for k in probes[0]['errors']}
        summary=dict(native_world=frames['native_world'],native_body=frames['native_body'],
            front_scope=frames['front_scope'],native_motors_to_user=[r['user_motor_number'] for r in frames['motors']],
            allocator_and_plant_internal_sign_order_agree=matrices['internal_sign_and_order_match'],
            geometry_axis_difference_m=matrices['axis_arm_difference_m'],
            FRD_user_target_roll_pitch_T_match=matrices['target_roll_pitch_T_match'],
            FRD_user_target_yaw_reversed=matrices['target_yaw_reversed'],
            rotor_increment_tests=len(probes),efficiency_loss_tests=len(losses),dynamic_error_maxima=maxima,
            policy_samples=len(samples),observation_roundtrip_max_error=max(x['observation_max_error'] for x in samples),
            policy_action_roundtrip_max_error=max(x['action_max_error'] for x in samples),
            motor_command_roundtrip_max_error_n=matrices['motor_command_equivalence_max_n'],
            oracle_command_roundtrip_max_error_n=matrices['oracle_equivalence_max_n'],
            limitations=['FPV defines model front; real hardware nose/propeller spin not independently measured.',
                'No FRD adapter or new allocator connected to any rollout.',
                'Increment tests use direct actual force/reaction, separate from motor delay; output-loss tests reuse production loss function.',
                'Free prop hinges are retained; locked whole-body inertia alone is insufficient for exact yaw acceleration.',
                'Historical training source record does not hash XML content; saved configuration/path and current engine are distinguished.'])
        write_json(directory/'summary.json',summary)
        assert all(sha256(p)==h for p,h in protected.items())
        manifest.update(status='completed',protected_files_unchanged=True,protected_file_count=len(protected),verification_passed=True)
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        write_json(directory/'manifest.json',clean(manifest));print('results:',directory,flush=True)
    return directory


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/eval_velocity_ab.yaml')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/runs')
    args=parser.parse_args(argv)
    from unittest.mock import patch
    from stable_baselines3 import PPO
    import torch
    def forbidden(*args,**kwargs):raise AssertionError('learning / optimizer update forbidden during audit')
    with patch.object(PPO,'learn',forbidden),patch.object(PPO,'train',forbidden),patch.object(torch.optim.Adam,'step',forbidden):
        execute_audit(args)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

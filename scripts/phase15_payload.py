"""Evaluation-only rigid point payload, massless rigid rod. Controller unchanged."""
import json,copy
import numpy as np
from scipy.spatial.transform import Rotation

def composite(m0,c0,J0,mp,p):
    c0=np.asarray(c0,dtype=np.float64);p=np.asarray(p,dtype=np.float64);J0=np.asarray(J0,dtype=np.float64)
    mass=m0+mp;c=(m0*c0+mp*p)/mass
    def shift(m,d):return m*((d@d)*np.eye(3)-np.outer(d,d))
    J=J0+shift(m0,c0-c)+shift(mp,p-c)
    if mp==0:return mass,c,J,np.diag(J).copy(),np.array([0.,0.,0.,1.])
    values,R=np.linalg.eigh(J)
    if np.linalg.det(R)<0:R[:,-1]*=-1
    assert np.all(values>0)
    return mass,c,J,values,Rotation.from_matrix(R).as_quat()

def theory(cases,m0,c0,J0):
    from crazyflie_isaaclab.tasks.crazyflie.constants import ROTOR_POS,ALLOC_ARM,MOTOR_DIR,TORQUE_RATIO,ACTION_SCALE,BODY_MASS,GRAVITY,THRUST_MAX
    rotor=np.array(ROTOR_POS);a=ALLOC_ARM
    B=np.array([[-a,-a,a,a],[-a,a,a,-a],np.array(MOTOR_DIR)*TORQUE_RATIO,[1.,1.,1.,1.]])
    result=[]
    for case in cases:
        m,c,J,eig,q=composite(m0,c0,J0,case['mass_g']/1000.,[case['x_cm']/100.,0.,0.])
        arms=rotor-c
        A=np.vstack([arms[:,1],-arms[:,0],np.array(MOTOR_DIR)*TORQUE_RATIO,np.ones(4)])
        f=np.linalg.solve(A,[0,0,0,m*GRAVITY]);w=B@f;action=(w-np.array([0,0,0,BODY_MASS*GRAVITY]))/np.array(ACTION_SCALE)
        physical=bool(np.all(f>=-1e-9) and np.all(f<=THRUST_MAX+1e-9))
        pipeline=physical and bool(np.all(np.abs(action)<=1+1e-8))
        result.append(dict(**case,total_mass_kg=m,com_body_m=c.tolist(),inertia_body_kg_m2=J.tolist(),principal_inertia=eig.tolist(),principal_quat_xyzw=q.tolist(),equilibrium_rotor_thrust_N=f.tolist(),equilibrium_nominal_wrench=w.tolist(),equilibrium_normalized_action=action.tolist(),static_level_rotor_feasible=physical,static_level_action_pipeline_feasible=pipeline,
          nominal_hover_initial_az_m_s2=BODY_MASS*GRAVITY/m-GRAVITY))
    return result

def read_engine(raw):
    import warp as wp
    def read(name):return wp.to_torch(getattr(raw.robot.root_view,name)()).detach().cpu().numpy().copy()
    masses=read('get_masses').reshape(-1);coms=read('get_coms').reshape(-1,7);Jp=read('get_inertias').reshape(-1,3,3)
    # get_inertias returns the full tensor in actor/body axes already.
    # get_coms quaternion describes the principal mass frame; do not rotate twice.
    J=Jp.copy()
    return masses,coms,Jp,J

def install(raw,cases):
    import torch
    from crazyflie_isaaclab.tasks.crazyflie.constants import TOTAL_MASS,COM,INERTIA,ROTOR_POS,BODY_MASS,GRAVITY
    from crazyflie_isaaclab.tasks.crazyflie.allocation import rotor_wrench
    masses,coms,Jp,J=read_engine(raw)
    assert np.max(np.abs(masses-TOTAL_MASS))<5e-9
    assert np.max(np.abs(coms[:,:3]-np.array(COM)))<1e-12
    assert np.max(np.abs(J-np.diag(INERTIA)))<5e-12
    expected=theory(cases,float(masses[0]),coms[0,:3],J[0])
    def tensor(x):return torch.tensor(x,dtype=torch.float32,device=raw.device)
    newmass=tensor([x['total_mass_kg'] for x in expected])[:,None]
    newcom=tensor([x['com_body_m']+[0.,0.,0.,1.] for x in expected])[:,None,:]
    newJ=tensor(np.array([np.array(x['inertia_body_kg_m2']).reshape(9) for x in expected]))[:,None,:]
    raw.robot.set_masses_index(masses=newmass)
    raw.robot.set_coms_index(coms=newcom)
    raw.robot.set_inertias_index(inertias=newJ)
    raw._payload_arms=tensor(ROTOR_POS)[None]-newcom[:,:,:3]
    # Force vectors and torques are expressed in the actor/body frame, about the NEW COM.
    # No extra gravity torque is added: gravity is applied by PhysX at composite COM.
    def set_forces(force,reaction):
        f,t=rotor_wrench(force,reaction,raw._payload_arms)
        raw.force_b.copy_(f);raw.torque_b.copy_(t)
        raw.robot.permanent_wrench_composer.set_forces_and_torques_index(forces=f,torques=t,is_global=False)
    raw.set_rotor_forces=set_forces
    original_reset=raw._reset_idx
    def reset(ids):
        original_reset(ids)
        ids=torch.arange(raw.num_envs,device=raw.device) if ids is None else ids
        f,t=rotor_wrench(raw.motors.force[ids],raw.motors.reaction[ids],raw._payload_arms[ids])
        raw.force_b[ids]=f;raw.torque_b[ids]=t
        raw.robot.permanent_wrench_composer.set_forces_and_torques_index(forces=f,torques=t,env_ids=ids,is_global=False)
    raw._reset_idx=reset
    # Engine readback, not only requested configuration.
    rm,rc,rp,rJ=read_engine(raw)
    checks=dict(mass_max_error=float(np.max(np.abs(rm-newmass.cpu().numpy().reshape(-1)))),com_max_error=float(np.max(np.abs(rc[:,:3]-newcom.cpu().numpy()[:,0,:3]))),inertia_body_max_error=float(np.max(np.abs(rJ-np.array([x['inertia_body_kg_m2'] for x in expected])))))
    if checks['inertia_body_max_error']>=2e-10:
        j=int(np.argmax(np.max(np.abs(rJ-np.array([x['inertia_body_kg_m2'] for x in expected])),axis=(1,2))))
        print('INERTIA_DIAGNOSTIC',json.dumps(dict(case=expected[j],engine_com=rc[j].tolist(),engine_inertia=rp[j].tolist(),reconstructed=rJ[j].tolist())),flush=True)
    assert checks['mass_max_error']<5e-9 and checks['com_max_error']<1e-8 and checks['inertia_body_max_error']<2e-10,checks
    return dict(expected=expected,readback=dict(masses_kg=rm.tolist(),com_pose_xyzw=rc.tolist(),inertia_api_body_frame=rp.tolist(),inertia_body=rJ.tolist()),errors=checks,success=True)

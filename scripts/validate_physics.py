"""Actual PhysX checks. Resets between experiments only; no feedback controller."""
import json,math,socket,sys
import numpy as np
import torch
import warp as wp
import omni.physx
from checks import Checks
from crazyflie_isaaclab.tasks.crazyflie.constants import *
from crazyflie_isaaclab.tasks.crazyflie.actuator_model import inverse_thrust
from crazyflie_isaaclab.tasks.crazyflie.frames import wxyz_to_xyzw,xyzw_to_wxyz,rotate

def validate(env,out):
    out.mkdir(parents=True,exist_ok=True)
    c=Checks();n=env.num_envs;device=env.device;robot=env.robot
    engine_dts=[]
    subscription=omni.physx.get_physx_interface().subscribe_physics_on_step_events(
        lambda dt: engine_dts.append(float(dt)),pre_step=False,order=0)
    def tensor(x): return torch.tensor(x,device=device,dtype=torch.float32)
    def zero(): return torch.zeros((n,4),device=device)
    def raw_step():
        env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(DT)
    def setup(q=None):
        env.reset(seed=42)
        if q is not None:
            pose=robot.data.root_link_pose_w.torch.clone()
            pose[:,3:]=wxyz_to_xyzw(tensor(q).expand(n,-1))
            robot.write_root_link_pose_to_sim_index(root_pose=pose)
        robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros((n,6),device=device))
    def set_forces(f):
        f=tensor(f).expand(n,4).clone()
        env.set_rotor_forces(f,f*tensor(MOTOR_DIR)*TORQUE_RATIO)
    def state():
        return torch.cat((robot.data.root_link_pos_w.torch-env.scene.env_origins,
            robot.data.root_link_quat_w.torch,robot.data.root_link_vel_w.torch),dim=-1).clone()
    def read_view(name):
        x=getattr(robot.root_view,name)()
        return wp.to_torch(x) if isinstance(x,wp.array) else x
    c.close('engine_mass',read_view('get_masses'),TOTAL_MASS,5e-9,'kg')
    c.close('engine_com',read_view('get_coms')[...,:3],COM,1e-12,'m')
    c.close('engine_inertia',read_view('get_inertias').reshape(-1,3,3),np.diag(INERTIA),5e-12,'kg m^2')
    c.close('actual_isaacsim_physx',env.sim.physics_manager.__module__=='isaaclab_physx.physics.physx_manager',True,0)
    c.close('physics_dt',env.physics_dt,DT,1e-15,'s')
    c.close('policy_dt',env.step_dt,.01,1e-15,'s')
    c.close('reset_policy_shape',list(env.policy_observation().shape),[n,15],0)
    c.close('action_shape',list(env.actions.shape),[n,4],0)
    c.close('reset_hover_omega',env.motors.omega,1581.521674476,.002,'rad/s')
    c.close('initial_origin_free_position_error',env.policy_observation()[:,:3],0,3e-7,'m')
    # Engine roundtrip exercises actual pose input and data output, all four rotations.
    for name,q in [('identity',[1,0,0,0]),('x90',[2**-.5,2**-.5,0,0]),('y90',[2**-.5,0,2**-.5,0]),('z90',[2**-.5,0,0,2**-.5])]:
        setup(q)
        c.close('engine_quat_'+name,env.policy_observation()[:,6:10],q,2e-7)
        body_f=tensor([.05,.07,.11]).expand(n,-1)
        robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=body_f[:,None],torques=torch.zeros((n,1,3),device=device),is_global=False)
        expected=rotate(tensor(q).expand(n,-1),body_f)/TOTAL_MASS+tensor([0,0,-GRAVITY])
        raw_step()
        c.close('rotated_body_force_'+name,robot.data.root_com_lin_vel_w.torch/DT,expected,3e-5,'m/s^2')
    setup();set_forces([0]*4);raw_step()
    c.close('gravity_only_initial_acceleration',robot.data.root_com_lin_vel_w.torch/DT,[0,0,-GRAVITY],3e-5,'m/s^2')
    # Physical total-mass equilibrium starts level with equilibrium angular speed.
    setup()
    hover=TOTAL_MASS*GRAVITY/4
    env.motors.reset(torch.arange(n,device=device,dtype=torch.int32),torch.full((n,4),hover,device=device))
    action=zero();action[:,3]=(TOTAL_MASS-BODY_MASS)*GRAVITY/ACTION_SCALE[3]
    for _ in range(100): env.step(action)
    c.close('total_mass_hover_velocity_1s',robot.data.root_com_lin_vel_w.torch,0,2e-5,'m/s')
    c.close('total_mass_hover_position_1s',state()[:,:3],[0,0,1],2e-5,'m')
    c.close('total_mass_hover_angular_velocity_1s',robot.data.root_link_ang_vel_w.torch,0,2e-5,'rad/s')
    setup()
    for _ in range(10): env.step(zero())
    expected_az=BODY_MASS*GRAVITY/TOTAL_MASS-GRAVITY
    c.close('zero_action_acceleration_100ms',robot.data.root_com_lin_vel_w.torch[:,2]/.1,expected_az,1.5e-5,'m/s^2',note='Body-mass feedforward preserved; physically below total-weight equilibrium')
    # Native rotor axes and user numbering verified from the actual engine response.
    for user in range(1,5):
        i=4-user;setup();f=np.full(4,hover);f[i]+=.001;set_forces(f);raw_step()
        x,y,_=ROTOR_POS[i]
        alpha=np.array([y*.001,-x*.001,MOTOR_DIR[i]*TORQUE_RATIO*.001])/INERTIA
        c.close(f'user_motor_{user}_angular_acceleration',robot.data.root_link_ang_vel_w.torch/DT,alpha,5e-4,'rad/s^2',note=f'native rotor {i}; signed roll/pitch/yaw response')
    # A repeated set must replace, not integrate, its stored wrench.
    setup();f=np.full(4,hover+.001)
    for _ in range(20): set_forces(f);raw_step()
    c.close('composer_no_accumulation_acceleration',robot.data.root_com_lin_vel_w.torch[:,2]/(20*DT),.004/TOTAL_MASS,3e-5,'m/s^2')
    c.close('composer_force_buffer',robot.permanent_wrench_composer.out_force_b.torch[...,2],sum(f),1e-7,'N')
    # Link velocity is different from COM velocity with offset COM and body rotation.
    setup();vel=torch.zeros((n,6),device=device);vel[:,3]=100
    robot.write_root_link_velocity_to_sim_index(root_velocity=vel)
    c.close('observation_uses_link_velocity',env.policy_observation()[:,3:6],0,1e-9,'m/s')
    c.close('com_velocity_offset',robot.data.root_com_lin_vel_w.torch,[0,-100*COM[2],0],1e-9,'m/s')
    setup();before_motor=env.total_motor_calls;before_sim=env.sim.get_physics_step_count()
    before_time=sum(engine_dts);before_events=len(engine_dts)
    obs,r,terminated,truncated,_=env.step(zero())
    c.close('motor_updates_per_policy',env.total_motor_calls-before_motor,5,0,'calls')
    c.close('physics_updates_per_policy',env.sim.get_physics_step_count()-before_sim,5,0,'calls')
    c.close('elapsed_simulation_time',sum(engine_dts)-before_time,.01,1e-9,'s',note='Sum of actual PhysX post-step event dt values')
    c.close('actual_physx_callback_count',len(engine_dts)-before_events,5,0)
    c.close('actual_physx_callback_dt',engine_dts[-5:],[.002]*5,1e-9,'s')
    c.close('step_observation_shape',list(obs['policy'].shape),[n,15],0)
    c.close('step_finite',torch.isfinite(torch.cat((obs['policy'],r[:,None],env.motors.omega),-1)).all(),True,0)
    setup();env.motors.omega.zero_()
    for _ in range(5): env.step(zero())
    c.close('task_rpm_response_50ms',env.motors.omega,1581.521674476*(1-math.exp(-1)),.002,'rad/s')
    # Selective reset retains every unselected body's state, motor and wrench buffers.
    setup();a=zero();a[:,0]=torch.linspace(-.04,.04,n,device=device);a[:,3]=.02
    for _ in range(3): env.step(a)
    ids=torch.tensor([0],device=device,dtype=torch.int32)
    keep=torch.arange(1,n,device=device)
    before=[state(),env.motors.omega.clone(),env.motors.esc.clone(),env.prev_actions.clone(),
            env.force_b.clone(),env.torque_b.clone(),env.episode_length_buf.clone(),env.motor_updates.clone()]
    env._reset_idx(ids)
    after=[state(),env.motors.omega,env.motors.esc,env.prev_actions,env.force_b,env.torque_b,env.episode_length_buf,env.motor_updates]
    if n>1:
        for name,b,aft in zip(['body','rpm','esc','previous_action','force','torque','episode','motor_count'],before,after):
            c.close('partial_reset_preserve_'+name,aft[keep],b[keep],0)
    c.close('partial_reset_pose',state()[0,:3],[0,0,1],3e-7,'m')
    c.close('partial_reset_velocity',state()[0,7:],0,1e-9)
    c.close('partial_reset_rpm',env.motors.omega[0],1581.521674476,.002,'rad/s')
    c.close('partial_reset_action',env.prev_actions[0],0,0)
    c.close('partial_reset_composer_force',robot.permanent_wrench_composer.out_force_b.torch[0],[[0,0,BODY_MASS*GRAVITY]],1e-7,'N')
    c.close('partial_reset_composer_torque',robot.permanent_wrench_composer.out_torque_b.torch[0],0,1e-9,'Nm')
    c.close('partial_reset_episode',env.episode_length_buf[0],0,0)
    # Returned failure and time-limit flags survive the same-step automatic reset.
    setup();env.episode_length_buf[:]=799
    obs,r,term,trunc,_=env.step(zero())
    c.close('timeout_at_800',trunc,True,0);c.close('timeout_not_failure',term,False,0)
    setup();pose=robot.data.root_link_pose_w.torch.clone();pose[:,2]=env.scene.env_origins[:,2]+.19
    robot.write_root_link_pose_to_sim_index(root_pose=pose)
    obs,r,term,trunc,_=env.step(zero())
    c.close('altitude_failure',term,True,0);c.close('failure_not_timeout',trunc,False,0)
    c.close('failure_auto_reset_pose',state()[:,:3],[0,0,1],3e-7,'m')
    # Explicit action contract and remaining failure boundaries.
    setup();env._pre_physics_step(torch.full((n,4),2.0,device=device))
    c.close('action_upper_clip',env.actions,1,0)
    c.close('action_space_low',env.single_action_space.low,-1,0)
    c.close('action_space_high',env.single_action_space.high,1,0)
    env._pre_physics_step(torch.full((n,4),-2.0,device=device))
    c.close('action_lower_clip',env.actions,-1,0)
    for name,z,x,q in [('high_altitude',2.51,0,None),('position_error',1,1.51,None),
                       ('tilt',1,0,[math.cos(math.radians(61)/2),math.sin(math.radians(61)/2),0,0])]:
        setup(q);pose=robot.data.root_link_pose_w.torch.clone()
        pose[:,0]=env.scene.env_origins[:,0]+x;pose[:,2]=env.scene.env_origins[:,2]+z
        robot.write_root_link_pose_to_sim_index(root_pose=pose)
        term,trunc=env._get_dones()
        c.close('failure_'+name,term,True,0)
    target_before=env.target.clone();env.target[2]=.05;env.reset()
    c.close('ground_reset_motor_zero',env.motors.omega,0,0,'rad/s')
    c.close('ground_reset_force_zero',robot.permanent_wrench_composer.out_force_b.torch,0,0,'N')
    env.target.copy_(target_before)
    # Identical fixed commands, unmodified dynamics; compare against separate N=1 run.
    setup();trace=[]
    for k in range(30):
        a=zero();a[:]=tensor([.012*math.sin(k*.2),-.01*math.cos(k*.13),.008,.01])
        obs,r,term,trunc,_=env.step(a)
        trace.append(torch.cat((state(),env.motors.omega,obs['policy'],r[:,None]),-1).cpu().numpy())
    trace=np.stack(trace);np.save(out/'fixed_action_trace.npy',trace)
    if n>1:
        c.close('parallel_identical_envs',trace,trace[:,:1,:],4e-6,'mixed SI',note='Includes subtraction of nonzero batch origins')
    if n>1:
        setup();independent=[]
        for k in range(30):
            a=zero();a[:]=tensor([.012*math.sin(k*.2),-.01*math.cos(k*.13),.008,.01])
            a[1:]*=-3
            obs,r,term,trunc,_=env.step(a)
            independent.append(torch.cat((state(),env.motors.omega,obs['policy'],r[:,None]),-1)[0].cpu().numpy())
        c.close('env0_unaffected_by_other_commands',np.stack(independent),trace[:,0],1e-7,'mixed SI')
    metadata=dict(hostname=socket.gethostname(),python=sys.executable,num_envs=n,device=device,
        selected_backend='isaacsim_physx',actual_manager=env.sim.physics_manager.__module__+'.'+env.sim.physics_manager.__name__,
        scene_backend=env.scene.physics_backend,sim_time=sum(engine_dts),reward_variant=env.cfg.reward_variant,
        gravity_feedforward_mass=BODY_MASS,physical_mass=TOTAL_MASS,rigid_body_approximation='locked-child composite',optimizer_updates=0)
    (out/'resolved_config.json').write_text(json.dumps(env.cfg.to_dict(),default=str,indent=2)+'\n')
    metadata['engine_dt_events']=len(engine_dts)
    passed=c.save(out,metadata)
    subscription.unsubscribe()
    print('PHYSICS_VALIDATION_COMPLETE',n,passed,flush=True)
    if not passed: raise RuntimeError('Physics validation failed; see results.json')

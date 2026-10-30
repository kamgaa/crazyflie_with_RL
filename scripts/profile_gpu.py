"""Inspect data movement separately from numerical/physics diagnostics."""
import json
import torch
from torch.profiler import profile as torch_profile, ProfilerActivity

def profile(env,out):
    out.mkdir(parents=True,exist_ok=True)
    action=torch.zeros((env.num_envs,4),device=env.device)
    for _ in range(3):env.step(action)
    torch.cuda.synchronize()
    with torch_profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],with_stack=True) as p:
        for _ in range(3):env.step(action)
    p.export_chrome_trace(str(out/'trace.json'))
    interesting=[]
    for e in p.key_averages():
        if any(x in e.key.lower() for x in ['memcpy','nonzero','item','local_scalar','synchronize']):
            interesting.append(dict(name=e.key,count=e.count,cpu_time_us=e.cpu_time_total))
    (out/'summary.json').write_text(json.dumps(dict(num_envs=env.num_envs,steps=3,
        com_parameter_device=str(env.robot.root_view.get_coms().device),events=interesting),indent=2)+'\n')
    # Separate user-task tensor calculations from installed engine/scene dispatch.
    from crazyflie_isaaclab.tasks.crazyflie.allocation import rotor_wrench
    from crazyflie_isaaclab.tasks.crazyflie.observations_rewards import state_observation,reward,failures
    d=env.robot.data
    inputs=[d.root_link_pos_w.torch.clone(),d.root_link_lin_vel_w.torch.clone(),
            d.root_link_quat_w.torch.clone(),d.root_link_ang_vel_w.torch.clone(),
            env.scene.env_origins,env.target]
    torch.cuda.synchronize()
    with torch_profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,with_stack=True) as math_profile:
        for _ in range(3):
            env._pre_physics_step(action)
            for _ in range(5):
                env.motors.advance()
                rotor_wrench(env.motors.force,env.motors.reaction,env.rotor_arms)
            obs=state_observation(*inputs)
            done=failures(obs,inputs[0][:,2]-env.scene.env_origins[:,2])
            reward(obs,done,env.cfg.reward_variant)
    math_profile.export_chrome_trace(str(out/'task_math_trace.json'))
    scalar_details=[]
    for e in math_profile.events():
        if e.name=='aten::_local_scalar_dense':
            parents=[];current=e
            while current is not None:
                parents.append(current.name);current=current.cpu_parent
            scalar_details.append(dict(parents=parents,stack=e.stack))
    (out/'scalar_details.json').write_text(json.dumps(scalar_details,indent=2)+'\n')
    math_events=[dict(name=e.key,count=e.count) for e in math_profile.key_averages()
        if any(x in e.key for x in ['Memcpy HtoD','Memcpy DtoH','aten::nonzero','aten::_local_scalar_dense'])]
    (out/'task_math.json').write_text(json.dumps(dict(steps=3,events=math_events,
        no_environment_python_loops=True,host_device_copy_free=len(math_events)==0),indent=2)+'\n')
    print('TASK_MATH_PROFILE',json.dumps(math_events),flush=True)


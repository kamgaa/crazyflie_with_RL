"""MuJoCo rendering of saved states only; no rollout or estimator recomputation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .integral_eval import read_columns
from .motor_layout import exposed_motor_index


def ffmpeg_binary(explicit=None):
    if explicit:return str(Path(explicit).resolve(strict=True))
    if shutil.which('ffmpeg'):return shutil.which('ffmpeg')
    try:import imageio_ffmpeg
    except ImportError:
        sys.path.insert(0,'/tmp/crazyflie-video-deps')
        try:import imageio_ffmpeg
        except ImportError as e:raise RuntimeError('Install ffmpeg or imageio-ffmpeg, or pass --ffmpeg') from e
    return imageio_ffmpeg.get_ffmpeg_exe()


def font(size,bold=False):
    base=Path('/usr/share/fonts/truetype/dejavu')
    return ImageFont.truetype(str(base/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),size)


def project(scene,xyz,width,height):
    cameras=scene.camera
    pos=np.mean([c.pos for c in cameras],axis=0)
    c=cameras[0];delta=np.asarray(xyz)-pos
    forward=np.array(c.forward);up=np.array(c.up);right=np.cross(forward,up)
    z=delta@forward
    focal=height*c.frustum_near/(c.frustum_top-c.frustum_bottom)
    return np.array([width/2+focal*(delta@right)/z,height/2-focal*(delta@up)/z])


def render(directory,ffmpeg=None,output_dir=None):
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('MESA_SHADER_CACHE_DIR','/tmp/crazyflie-mesa')
    import mujoco
    from .config import load_config
    from .dr_policy import sha256
    from .oracle_eval import flat_csv
    source=Path(directory).resolve()
    directory=Path(output_dir).resolve() if output_dir else source
    if output_dir:
        directory.mkdir(parents=True,exist_ok=True)
        if any(directory.iterdir()):raise FileExistsError('video output directory must be empty')
    manifest=json.loads((source/'manifest.json').read_text())
    info=json.loads((source/'video_run.json').read_text());summary=info['summary']
    assert info['key']=='A_best-motor1_70' and summary['fault_user_motor']==1 and summary['fault_efficiency']==.7
    output=directory/'motor1_fault_estimation.mp4'
    if output.exists():raise FileExistsError(output)
    trace=read_columns(source/info['trace']);replay=np.load(source/info['replay'])
    events=json.loads((source/(info['key']+'-events.json')).read_text())
    assert len(events)==1 and events[0]['user_motor_id']==1 and events[0]['native_motor_index']==3
    event_time=events[0]['simulation_time'];start=2.;end=min(12.,summary['actual_duration_sec'])
    if end<=start:raise RuntimeError('specified rollout terminated before video start; no replacement allowed')
    fps=30;width,height=1280,720;sw,sh=900,590;ox,oy=12,112
    ffmpeg=ffmpeg_binary(ffmpeg)
    config=load_config(manifest['config']['source_path'])
    xml=config.paths.mujoco_xml
    assert sha256(xml)==manifest['nominal_model']['xml_sha256']
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=sh,width=sw)
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    # Camera fitted once to the saved trajectory; it stays fixed for all frames.
    selected=(replay['time']>=start)&(replay['time']<=end)
    positions=np.vstack((replay['qpos'][selected,:3],[0,0,1]))
    center=(positions.min(0)+positions.max(0))/2
    extent=float(np.max(np.ptp(positions,axis=0)))
    camera=mujoco.MjvCamera();camera.lookat[:]=center;camera.distance=max(.46,.42+2*extent)
    camera.azimuth=225;camera.elevation=-40
    # Guard against accidentally advancing the plant during rendering.
    def prohibited(*a,**kw):raise RuntimeError('physics stepping prohibited in saved-state renderer')
    original_step=mujoco.mj_step;mujoco.mj_step=prohibited
    cmd=[ffmpeg,'-y','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','19',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    video_log=(directory/'video_encode.log').open('w')
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=video_log)
    frames=[];n=int(np.ceil((end-start)*fps-1e-9))
    white=(234,242,250);muted=(153,177,201);orange=(255,169,66);cyan=(60,222,216);red=(255,91,106)
    detection=summary['confirmed_detection_time']
    image_checks=[]
    try:
        for i in range(n):
            display_time=start+i/fps
            state_index=int(np.searchsorted(replay['time'],display_time+1e-9,side='right')-1)
            t=float(replay['time'][state_index]);trace_index=int(np.searchsorted(trace.time_post,t+1e-9,side='right')-1)
            assert trace_index>=0 and abs(trace.time_post[trace_index]-t)<1e-8
            qpos=replay['qpos'][state_index]
            np.testing.assert_array_equal(qpos[:3],[trace[f'position_{j}'][trace_index] for j in range(3)])
            q=np.array([trace[f'quaternion_{j}'][trace_index] for j in range(4)])
            assert min(np.max(np.abs(qpos[3:7]-q)),np.max(np.abs(qpos[3:7]+q)))<1e-12
            data.qpos[:]=qpos;data.time=t;mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera,scene_option=opt)
            rgb=renderer.render()
            im=Image.new('RGB',(width,height),(12,21,34));im.paste(Image.fromarray(rgb),(ox,oy));d=ImageDraw.Draw(im)
            d.text((22,14),'Single-motor fault estimation',font=font(31,True),fill=white)
            d.text((23,59),'Shadow mode — estimator not used for control',font=font(21),fill=cyan)
            d.text((1030,22),f't = {t:05.2f} s',font=font(25,True),fill=white)
            d.text((1060,64),'1×  |  A best',font=font(18),fill=muted)
            bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'drone')
            origin=data.xpos[bid];R=data.xmat[bid].reshape(3,3)
            cp=project(renderer.scene,origin,sw,sh)+[ox,oy]
            motor_pixels=[]
            for user in range(1,5):
                native=exposed_motor_index(user,'user_frd')
                sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'motor{native}')
                pixel=project(renderer.scene,data.site_xpos[sid],sw,sh)+[ox,oy];motor_pixels.append(pixel.tolist())
                delta=pixel-cp;delta=delta/max(np.linalg.norm(delta),1.)
                label=pixel+delta*31
                color=orange if user==1 else cyan
                d.line([tuple(pixel),tuple(label)],fill=color,width=2)
                d.ellipse((label[0]-15,label[1]-15,label[0]+15,label[1]+15),fill=(12,21,34),outline=color,width=2)
                d.text(tuple(label-[7,13]),str(user),font=font(20,True),fill=color)
            front=project(renderer.scene,origin+R@np.array([.105,0,0]),sw,sh)+[ox,oy]
            nose=project(renderer.scene,origin+R@np.array([.05,0,.005]),sw,sh)+[ox,oy]
            d.line([tuple(nose),tuple(front)],fill=white,width=3)
            d.text(tuple(front+[-28,-25]),'FRONT',font=font(16,True),fill=white)
            target=project(renderer.scene,[0,0,1],sw,sh)+[ox,oy]
            d.line([(target[0]-9,target[1]),(target[0]+9,target[1])],fill=cyan,width=2)
            d.line([(target[0],target[1]-9),(target[0],target[1]+9)],fill=cyan,width=2)
            d.text((26,126),'USER MOTOR NUMBERS',font=font(17,True),fill=white)
            d.text((26,152),'1 front-left  ·  2 rear-left  ·  3 rear-right  ·  4 front-right',font=font(15),fill=white)
            d.text((26,673),'Cyan cross: fixed target   |   Saved MuJoCo states',font=font(16),fill=white)
            x=935
            d.rounded_rectangle((x-10,112,1266,696),radius=14,fill=(21,35,52))
            d.text((x+5,128),'Motor 1 · front-left',font=font(22,True),fill=white)
            d.text((x+5,160),'Native index 3  |  No payload',font=font(16),fill=muted)
            # At the event boundary use its right-limit efficiency. Physical
            # pose is continuous; estimator remains the latest available result.
            truth=.7 if t>=event_time-1e-9 else 1.
            estimate=float(trace.estimated_efficiency_user_0[trace_index])
            state=str(trace.estimator_state[trace_index]);motor=int(trace.estimated_motor[trace_index])
            d.text((x+5,204),'Ground truth η1',font=font(20),fill=orange)
            d.text((x+223,198),f'{truth:.2f}',font=font(29,True),fill=orange)
            d.text((x+5,251),'Estimated η1',font=font(20),fill=cyan)
            d.text((x+223,245),f'{estimate:.2f}',font=font(29,True),fill=cyan)
            display=('Healthy' if state=='healthy' else f'Fault: motor {motor}' if state=='fault'
                     else 'Insufficient data' if state=='insufficient_data' else 'Uncertain')
            d.text((x+5,306),'Estimator decision',font=font(17),fill=muted)
            d.text((x+5,332),display,font=font(24,True),fill=cyan if state=='fault' else white)
            d.text((x+5,377),'FAULT APPLIED' if truth<1 else 'NO FAULT YET',font=font(20,True),fill=red if truth<1 else muted)
            if detection is not None and t>=detection-1e-9:
                d.text((x+5,408),f'Confirmed at {detection:.2f} s',font=font(17),fill=cyan)
            left,top,right,bottom=x+38,463,1248,625
            def pt(time,value):return (left+(time-start)/(12-start)*(right-left),bottom-value/1.05*(bottom-top))
            for value in (0.,.5,1.):
                yy=pt(start,value)[1];d.line([(left,yy),(right,yy)],fill=(55,73,91));d.text((x+2,yy-8),f'{value:.1f}',font=font(13),fill=muted)
            for tt in (2,5,8,12):
                xx=pt(tt,0)[0];d.text((xx-6,bottom+8),str(tt),font=font(14),fill=muted)
            gt=[pt(start,1.)]
            if t>=event_time:gt += [pt(event_time,1.),pt(event_time,.7)]
            gt.append(pt(t,truth));d.line(gt,fill=orange,width=3)
            use=np.flatnonzero((trace.time_post>=start-1e-9)&(trace.time_post<=t+1e-9))
            curve=[pt(float(trace.time_post[j]),float(trace.estimated_efficiency_user_0[j])) for j in use]
            if len(curve)>1:d.line(curve,fill=cyan,width=2)
            if t>=event_time:
                xx=pt(event_time,0)[0];d.line([(xx,top),(xx,bottom)],fill=red,width=1)
            if detection is not None and t>=detection:
                xx=pt(detection,0)[0];d.line([(xx,top),(xx,bottom)],fill=cyan,width=1)
            d.text((x+36,657),'Simulation time (s)',font=font(15),fill=muted)
            if summary['terminated'] and i==n-1:
                d.rectangle((20,570,890,660),fill=(92,20,33))
                d.text((35,584),f"TERMINATED at {summary['actual_duration_sec']:.2f} s",font=font(26,True),fill=white)
                d.text((35,624),summary['end_reason'],font=font(20),fill=white)
            frame=np.asarray(im)
            proc.stdin.write(frame.tobytes())
            frames.append(dict(frame=i,video_time_s=i/fps,requested_simulation_time_s=display_time,
                state_time_s=t,estimate_time_s=float(trace.estimate_time[trace_index]),trace_row=trace_index,
                replay_row=state_index,truth_eta1=truth,estimated_eta1=estimate,estimator_state=state,
                estimated_motor=motor,qpos_sha256=hashlib.sha256(qpos.tobytes()).hexdigest(),
                motor_pixels=motor_pixels,frame_rgb_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
            fault_frame=int(round((event_time-start)*fps))+1
            if i in (0,fault_frame,n-1):
                name={0:'start',fault_frame:'after_fault',n-1:'end'}[i]
                im.save(directory/f'video_check_{name}.png')
                image_checks.append(dict(frame=i,name=name,mean=float(frame.mean()),std=float(frame.std())))
            if i%60==0:print('render',i,'/',n,flush=True)
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('FFmpeg encode failed; see video_encode.log')
    finally:
        if proc.poll() is None:proc.terminate();proc.wait()
        video_log.close();renderer.close();mujoco.mj_step=original_step
    flat_csv(directory/'video_frames.csv',frames)
    probe=subprocess.run([ffmpeg,'-hide_banner','-i',str(output)],capture_output=True,text=True)
    (directory/'video_probe.txt').write_text(probe.stderr)
    decoded=subprocess.run([ffmpeg,'-v','error','-i',str(output),'-f','null','-'],capture_output=True,text=True)
    assert decoded.returncode==0,decoded.stderr
    assert 'h264' in probe.stderr and 'yuv420p' in probe.stderr and '1280x720' in probe.stderr and '30 fps' in probe.stderr
    assert all(c['std']>10 for c in image_checks)
    # Decode three actual MP4 frames too, checking the delivered video, not only
    # the uncompressed canvas. These are still images, not additional videos.
    for check in image_checks:
        out=directory/f"video_decoded_{check['name']}.png"
        subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(output),'-vf',f"select=eq(n\\,{check['frame']})",'-frames:v','1',str(out)],check=True)
    metadata=dict(run=info['key'],video=output.name,sha256=sha256(output),width=width,height=height,
        fps=fps,frames=n,duration_s=n/fps,requested_simulation_interval=[2.,12.],observed_end=end,
        playback_speed=1.,audio=False,codec='H.264',pixel_format='yuv420p',full_decode_success=True,
        camera=dict(lookat=center.tolist(),distance=camera.distance,azimuth=camera.azimuth,elevation=camera.elevation,fixed=True),
        timeline='causal zero-order hold of 100 Hz saved states/estimates onto 30 fps; no future interpolation; ground truth uses event right limit at 5 s',
        max_state_age_s=max(f['requested_simulation_time_s']-f['state_time_s'] for f in frames),
        estimator_recomputed=False,physics_steps=0,display_smoothing=False,
        source_run_directory=str(source),
        source_trace_sha256=sha256(source/info['trace']),source_replay_sha256=sha256(source/info['replay']),
        event_sha256=sha256(source/(info['key']+'-events.json')),frame_checks=image_checks,
        ffmpeg_version=subprocess.check_output([ffmpeg,'-version'],text=True).splitlines()[0],encode_command=cmd)
    (directory/'video_metadata.json').write_text(json.dumps(metadata,indent=2))
    (directory/'rerender.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(Path.cwd()))+
        '\nrender_dir=$(mktemp -d artifacts/runs/motor-estimation-render-XXXXXX)\nMUJOCO_GL=egl python render_motor_estimation.py --run-dir '+
        shlex.quote(str(source))+' --output-dir "$render_dir"\n')
    print('VIDEO',output,flush=True)
    return metadata


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--ffmpeg');parser.add_argument('--output-dir',type=Path,help='New empty directory; source rollout is preserved')
    args=parser.parse_args(argv);render(args.run_dir,args.ffmpeg,args.output_dir)

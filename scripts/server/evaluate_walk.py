"""Commanded HD1910 walking and stop evaluation; failures survive auto resets."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import mujoco
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp

p=argparse.ArgumentParser()
p.add_argument('--task',default='Mjlab-Walk-Flat-MicroDuck-HD1910')
p.add_argument('--onnx',required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--scenario',choices=['idle','forward','backward','left','right','turn_left','turn_right'],required=True)
p.add_argument('--envs',type=int,default=32);p.add_argument('--seconds',type=float,default=20.)
p.add_argument('--video');a=p.parse_args()
commands={'idle':[0,0,0],'forward':[.2,0,0],'backward':[-.15,0,0],
          'left':[0,.1,0],'right':[0,-.1,0],'turn_left':[0,0,.6],'turn_right':[0,0,-.6]}
cfg=load_env_cfg(a.task,play=True);cfg.scene.num_envs=a.envs;cfg.episode_length_s=a.seconds+1
cfg.commands['twist'].resampling_time_range=(1000.,1000.)
cfg.commands['twist'].rel_standing_envs=0.;cfg.commands['twist'].rel_turn_in_place_envs=0.
env=ManagerBasedRlEnv(cfg,device='cuda:0')
import onnxruntime as ort
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
policy=ort.InferenceSession(a.onnx,sess_options=opts,providers=['CPUExecutionProvider'])
failed=torch.zeros(a.envs,dtype=torch.bool,device=env.device)
vel=[];poses=[];tilts=[];feet=[];first_failure=torch.full((a.envs,),-1.,device=env.device)
writer=renderer=None
try:
    if a.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(a.video,fps=25)
        renderer=mujoco.Renderer(env.sim.mj_model,height=480,width=640)
        cpu=mujoco.MjData(env.sim.mj_model);camera=mujoco.MjvCamera()
        camera.azimuth=150;camera.elevation=-15;camera.distance=.7
    obs,_=env.reset(seed=42);robot=env.scene['robot'];term=env.command_manager.get_term('twist')
    foot_ids,_=robot.find_sites(['left_foot','right_foot'])
    for tick in range(round(a.seconds/env.step_dt)):
        t=tick*env.step_dt
        command=commands[a.scenario] if 2.<=t<a.seconds-5 else [0.,0.,0.]
        term.vel_command_b[:]=torch.tensor(command,device=env.device)
        term.vel_command_w[:]=term.vel_command_b;term.is_standing_env[:]=not any(command)
        obs['actor'][:,48:51]=term.vel_command_b
        action=np.asarray([policy.run(None,{policy.get_inputs()[0].name:x[None]})[0][0]
                           for x in obs['actor'].cpu().numpy()])
        obs,reward,terminated,truncated,_=env.step(torch.tensor(action,device=env.device))
        assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
        first_failure[terminated & ~failed]=t;failed|=terminated
        vel.append(torch.cat((robot.data.root_link_lin_vel_b[:,:2],robot.data.root_link_ang_vel_b[:,2:3]),dim=-1).cpu().numpy())
        poses.append(torch.rad2deg(mdp._servo_joint_pos(env,robot)-robot.data.default_joint_pos[:,mdp._servo_joint_ids(env,robot)]).cpu().numpy())
        tilts.append(torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))).cpu().numpy())
        feet.append(robot.data.site_pos_w[:,foot_ids,2].cpu().numpy())
        if writer and tick%2==0:
            cpu.qpos[:]=env.sim.data.qpos[0].cpu().numpy();mujoco.mj_forward(env.sim.mj_model,cpu)
            camera.lookat[:]=cpu.xpos[env.sim.mj_model.body('robot/trunk_base').id]
            renderer.update_scene(cpu,camera=camera);writer.append_data(renderer.render())
    v=np.asarray(vel);pose=np.asarray(poses)
    moving=v[round(4/env.step_dt):round((a.seconds-5)/env.step_dt)]
    calm=v[round((a.seconds-3)/env.step_dt):]
    error=np.mean(np.abs(moving-np.asarray(commands[a.scenario])),axis=(0,1))
    span=np.ptp(np.asarray(feet)[round(4/env.step_dt):round((a.seconds-5)/env.step_dt)],axis=0)
    result={'scenario':a.scenario,'command':commands[a.scenario],'envs':a.envs,'seconds':a.seconds,
        'failedWorlds':int(failed.sum()),'firstFailureSeconds':first_failure.cpu().tolist(),
        'meanBodyVelocity':moving.mean((0,1)).tolist(),'meanAbsTrackingError':error.tolist(),
        'stopP95LinearSpeedMps':float(np.percentile(np.linalg.norm(calm[:,:,:2],axis=-1),95)),
        'stopP95AngularSpeedRadps':float(np.percentile(np.abs(calm[:,:,2]),95)),
        'stopP95HeadErrorDeg':float(np.percentile(np.abs(pose[round((a.seconds-3)/env.step_dt):,:,5:9]),95)),
        'footHeightSpanMedianM':float(np.median(span)), 'p95TiltDeg':float(np.percentile(tilts,95)),
        'automaticResetsCountAsFailure':True,'hardwareValidated':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
finally:
    if writer:writer.close()
    if renderer:renderer.close()
    env.close()

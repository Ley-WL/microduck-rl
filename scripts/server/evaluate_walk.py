"""Commanded HD1910 walking and stop evaluation; failures survive auto resets."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import mujoco
from gait_metrics import summarize_gait,summarize_heading_path
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp

p=argparse.ArgumentParser()
p.add_argument('--task',default='Mjlab-Walk-Flat-MicroDuck-HD1910')
p.add_argument('--onnx',required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--scenario',choices=['idle','forward','backward','left','right','turn_left','turn_right'],required=True)
p.add_argument('--envs',type=int,default=32);p.add_argument('--seconds',type=float,default=20.)
p.add_argument('--forward-speed',type=float,default=.2,
               help='Forward command in m/s; does not change policy weights')
p.add_argument('--start-moving',action='store_true',help='Diagnostic: omit the initial two-second idle period')
p.add_argument('--trace',type=Path,help='Diagnostic first-world trace including weighted reward terms')
p.add_argument('--checkpoint',help='Compare checkpoint inference against ONNX on rollout observations')
p.add_argument('--head-home-ablation',action='store_true',
               help='Diagnostic intervention: replace head action deltas with zero; never a policy acceptance run')
p.add_argument('--neutral-yaw-ablation',action='store_true',
               help='Diagnostic intervention: neutralize both hip yaw deltas; never a policy acceptance run')
p.add_argument('--video');a=p.parse_args()
commands={'idle':[0,0,0],'forward':[a.forward_speed,0,0],'backward':[-.15,0,0],
          'left':[0,.1,0],'right':[0,-.1,0],'turn_left':[0,0,.6],'turn_right':[0,0,-.6]}
cfg=load_env_cfg(a.task,play=True);cfg.scene.num_envs=a.envs;cfg.episode_length_s=a.seconds+1
cfg.commands['twist'].resampling_time_range=(1000.,1000.)
cfg.commands['twist'].rel_standing_envs=0.;cfg.commands['twist'].rel_turn_in_place_envs=0.
cfg.commands['twist'].rel_heading_envs=0.;cfg.commands['twist'].rel_world_envs=0.
cfg.commands['twist'].rel_forward_envs=0.;cfg.commands['twist'].init_velocity_prob=0.
env=ManagerBasedRlEnv(cfg,device='cuda:0')
import onnxruntime as ort
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
policy=ort.InferenceSession(a.onnx,sess_options=opts,providers=['CPUExecutionProvider'])
checkpoint_policy=None
parity_error=0.
if a.checkpoint:
    from dataclasses import asdict
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg,load_runner_cls
    from rsl_rl.runners import OnPolicyRunner
    agent=load_rl_cfg(a.task)
    runner=(load_runner_cls(a.task) or OnPolicyRunner)(RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions),asdict(agent),device=env.device)
    runner.load(a.checkpoint,map_location=env.device)
    checkpoint_policy=runner.get_inference_policy(device=env.device)
failed=torch.zeros(a.envs,dtype=torch.bool,device=env.device)
vel=[];poses=[];tilts=[];feet=[];first_failure=torch.full((a.envs,),-1.,device=env.device)
foot_xyz=[];contacts=[];air_times=[];heights=[]
root_positions=[];headings=[]
failure_counts={name:0 for name in env.termination_manager.active_terms}
trace=[]
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
        command=commands[a.scenario] if (0. if a.start_moving else 2.)<=t<a.seconds-5 else [0.,0.,0.]
        term.vel_command_b[:]=torch.tensor(command,device=env.device)
        term.vel_command_w[:]=term.vel_command_b;term.is_standing_env[:]=not any(command)
        term.is_heading_env[:]=False;term.is_world_env[:]=False
        obs['actor'][:,48:51]=term.vel_command_b
        action=np.asarray([policy.run(None,{policy.get_inputs()[0].name:x[None]})[0][0]
                           for x in obs['actor'].cpu().numpy()])
        if checkpoint_policy:
            with torch.inference_mode():
                reference=checkpoint_policy(obs).cpu().numpy()
            parity_error=max(parity_error,float(np.max(np.abs(action-reference))))
        if a.head_home_ablation:
            action[:,5:9]=0.
        if a.neutral_yaw_ablation:
            action[:,[0,9]]=0.
        obs,reward,terminated,truncated,_=env.step(torch.tensor(action,device=env.device))
        assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
        ended=terminated|truncated
        if a.trace:
            trace.append({'time':t,'command':command,'ended':bool(ended[0]),
                'velocity':robot.data.root_link_lin_vel_b[0,:2].cpu().tolist(),
                'yawRate':float(robot.data.root_link_ang_vel_b[0,2]),
                'straightQuality':float(torch.exp(-(robot.data.root_link_ang_vel_b[0,2]/.25).square()
                    -(robot.data.root_link_lin_vel_b[0,1]/.04).square())),
                'tiltDeg':float(torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[0,2]).clamp(-1,1)))),
                'weightedRewardTerms':{name:float(env.reward_manager._step_reward[0,i]*env.step_dt)
                    for i,name in enumerate(env.reward_manager.active_terms)}})
        for name in failure_counts:
            failure_counts[name]+=int(env.termination_manager.get_term(name).sum())
        first_failure[ended & ~failed]=t;failed|=ended
        vel.append(torch.cat((robot.data.root_link_lin_vel_b[:,:2],robot.data.root_link_ang_vel_b[:,2:3]),dim=-1).cpu().numpy())
        poses.append(torch.rad2deg(mdp._servo_joint_pos(env,robot)-robot.data.default_joint_pos[:,mdp._servo_joint_ids(env,robot)]).cpu().numpy())
        tilts.append(torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))).cpu().numpy())
        feet.append(robot.data.site_pos_w[:,foot_ids,2].cpu().numpy())
        foot_xyz.append(robot.data.site_pos_w[:,foot_ids].cpu().numpy().copy())
        foot_sensor=env.scene['feet_ground_contact'].data
        contacts.append((foot_sensor.found>0).cpu().numpy().copy())
        air_times.append(foot_sensor.last_air_time.cpu().numpy().copy())
        heights.append(env.scene['foot_height_scan'].data.heights.cpu().numpy().copy())
        root_positions.append(robot.data.root_link_pos_w.cpu().numpy().copy())
        headings.append(robot.data.heading_w.cpu().numpy().copy())
        if writer and tick%2==0:
            cpu.qpos[:]=env.sim.data.qpos[0].cpu().numpy();mujoco.mj_forward(env.sim.mj_model,cpu)
            camera.lookat[:]=cpu.xpos[env.sim.mj_model.body('robot/trunk_base').id]
            renderer.update_scene(cpu,camera=camera);writer.append_data(renderer.render())
    v=np.asarray(vel);pose=np.asarray(poses)
    moving=v[round(4/env.step_dt):round((a.seconds-5)/env.step_dt)]
    calm=v[round((a.seconds-3)/env.step_dt):]
    error=np.mean(np.abs(moving-np.asarray(commands[a.scenario])),axis=(0,1))
    span=np.ptp(np.asarray(feet)[round(4/env.step_dt):round((a.seconds-5)/env.step_dt)],axis=0)
    active=slice(round(4/env.step_dt),round((a.seconds-5)/env.step_dt))
    gait=summarize_gait(np.asarray(foot_xyz)[active],np.asarray(contacts)[active],
                        np.asarray(air_times)[active],np.asarray(heights)[active],env.step_dt)
    result={'scenario':a.scenario,'command':commands[a.scenario],'envs':a.envs,'seconds':a.seconds,
        'failedWorlds':int(failed.sum()),'firstFailureSeconds':first_failure.cpu().tolist(),
        'terminationCounts':failure_counts,
        'meanBodyVelocity':moving.mean((0,1)).tolist(),'meanAbsTrackingError':error.tolist(),
        'stopP95LinearSpeedMps':float(np.percentile(np.linalg.norm(calm[:,:,:2],axis=-1),95)),
        'stopP95AngularSpeedRadps':float(np.percentile(np.abs(calm[:,:,2]),95)),
        'stopP95HeadErrorDeg':float(np.percentile(np.abs(pose[round((a.seconds-3)/env.step_dt):,:,5:9]),95)),
        'movingMeanHeadHomeErrorDeg':pose[active,:,5:9].mean((0,1)).tolist(),
        'movingP95HeadHomeErrorDeg':np.percentile(np.abs(pose[active,:,5:9]),95,axis=(0,1)).tolist(),
        'footHeightSpanMedianM':float(np.median(span)), 'p95TiltDeg':float(np.percentile(tilts,95)),
        'automaticResetsCountAsFailure':True,'hardwareValidated':False,
        'headHomeAblation':a.head_home_ablation,
        'neutralYawAblation':a.neutral_yaw_ablation,
        'checkpointOnnxMaxAbsError':parity_error if checkpoint_policy else None}
    result['gait']=gait
    motion=slice(round(2/env.step_dt),round((a.seconds-5)/env.step_dt))
    result['path']=summarize_heading_path(np.asarray(root_positions)[motion],np.asarray(headings)[motion],
                                          env.step_dt,commands[a.scenario][2])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    if a.trace:
        a.trace.parent.mkdir(parents=True,exist_ok=True)
        a.trace.write_text(json.dumps(trace),encoding='utf-8')
finally:
    if writer:writer.close()
    if renderer:renderer.close()
    env.close()

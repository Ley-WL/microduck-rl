"""Independent repeated/strong push evaluation, retaining failures across resets."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp

p=argparse.ArgumentParser()
p.add_argument("--task",default="Mjlab-PushRecovery-Flat-MicroDuck-HD1910")
p.add_argument("--onnx",required=True)
p.add_argument("--scenario",choices=("idle","repeated","alternating","strong"),required=True)
p.add_argument("--seconds",type=float,default=20.)
p.add_argument("--envs",type=int,default=32)
p.add_argument("--strength",type=float,default=.3)
p.add_argument("--seed",type=int,default=42)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--video")
a=p.parse_args()
cfg=load_env_cfg(a.task,play=True)
cfg.scene.num_envs=a.envs
cfg.episode_length_s=a.seconds+1
cfg.events["reset_base"].params["pose_range"].update(roll=(-.03,.03),pitch=(-.03,.03))
cfg.events["reset_robot_joints"].params["position_range"]=(-.01,.01)
env=ManagerBasedRlEnv(cfg,device="cuda:0")
import onnxruntime as ort
options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
policy=ort.InferenceSession(a.onnx,sess_options=options,providers=["CPUExecutionProvider"])
times={"idle":[],"repeated":[2.,2.4,2.8,8.,8.4,8.8],
       "alternating":[2.,2.4,2.8,3.2,8.,8.4,8.8,9.2],"strong":[2.,8.]}[a.scenario]
push_ticks={round(t/env.step_dt):i for i,t in enumerate(times)}
renderer=writer=cpu_data=camera=None
try:
    if a.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(a.video,fps=25)
        renderer=mujoco.Renderer(env.sim.mj_model,height=480,width=640)
        cpu_data=mujoco.MjData(env.sim.mj_model)
        camera=mujoco.MjvCamera();camera.azimuth=150;camera.elevation=-15;camera.distance=.65
    obs,_=env.reset(seed=a.seed)
    robot=env.scene["robot"]
    names=list(robot.joint_names)
    failed=torch.zeros(a.envs,dtype=torch.bool,device=env.device)
    first_failure=torch.full((a.envs,),-1.,device=env.device)
    tilt_history=[];height_history=[];pose_history=[];angular_history=[];action_history=[]
    for tick in range(round(a.seconds/env.step_dt)):
        if tick in push_ticks:
            index=push_ticks[tick]
            # Four direction cohorts; identical direction within a repeated burst.
            cohort=torch.arange(a.envs,device=env.device)%4
            if a.scenario=="alternating":
                cohort=(cohort+index)%4
            angles=cohort*torch.pi/2
            q=robot.data.root_link_quat_w
            yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
            velocity=robot.data.root_link_vel_w.clone()
            velocity[:,0]+=a.strength*torch.cos(angles+yaw)
            velocity[:,1]+=a.strength*torch.sin(angles+yaw)
            robot.write_root_link_velocity_to_sim(velocity)
            mdp._hd1910_push_state(env)
            env._hd1910_last_push[:]=env.common_step_counter
        values=[policy.run(None,{policy.get_inputs()[0].name:x[None]})[0][0] for x in obs["actor"].cpu().numpy()]
        action_history.append(np.degrees(np.array(values)))
        obs,reward,terminated,truncated,_=env.step(torch.tensor(np.array(values),device=env.device))
        assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
        first_failure[terminated & ~failed]=tick*env.step_dt
        failed|=terminated
        tilt=torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1.,1.)))
        tilt_history.append(tilt.cpu().numpy())
        height_history.append((robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]).cpu().numpy())
        pose_history.append(torch.rad2deg(robot.data.joint_pos-robot.data.default_joint_pos).cpu().numpy())
        angular_history.append(robot.data.root_link_ang_vel_b.norm(dim=-1).cpu().numpy())
        if writer and tick%2==0:
            cpu_data.qpos[:]=env.sim.data.qpos[0].cpu().numpy()
            mujoco.mj_forward(env.sim.mj_model,cpu_data)
            camera.lookat[:]=cpu_data.xpos[env.sim.mj_model.body("robot/trunk_base").id]
            renderer.update_scene(cpu_data,camera=camera)
            writer.append_data(renderer.render())
    tilt=np.array(tilt_history)
    settled=np.array(pose_history)[round(max(0,a.seconds-5)/env.step_dt):]
    leg=[i for i,n in enumerate(names) if n.startswith(("left_","right_"))]
    yaw_roll=[i for i in leg if "yaw" in names[i] or "roll" in names[i]]
    head=[i for i,n in enumerate(names) if n.startswith(("neck_","head_"))]
    calm_start=round(max(0,a.seconds-5)/env.step_dt)
    target_span=np.ptp(np.array(action_history)[calm_start:],axis=0)
    pose_span=np.ptp(settled,axis=0)
    mirror=np.stack([settled[:,:,names.index('left_'+n)]+settled[:,:,names.index('right_'+n)] for n in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')],axis=-1)
    result={"scenario":a.scenario,"strengthMps":a.strength,"pushTimesSeconds":times,
        "task":a.task,"settledP95AbsHeadErrorDeg":float(np.percentile(np.abs(settled[:,:,head]),95)),
        "settledP95JointTargetSpanDeg":float(np.percentile(target_span,95)),
        "settledP95JointPositionSpanDeg":float(np.percentile(pose_span,95)),
        "envs":a.envs,"seed":a.seed,"seconds":a.seconds,"failedWorlds":int(failed.sum()),
        "firstFailureSeconds":first_failure.cpu().tolist(),"maxTiltDeg":float(tilt.max()),
        "p95TiltDeg":float(np.percentile(tilt,95)),"settledP95AbsLegErrorDeg":float(np.percentile(np.abs(settled[:,:,leg]),95)),
        "settledP95AngularSpeedRadps":float(np.percentile(np.array(angular_history)[round(max(0,a.seconds-5)/env.step_dt):],95)),
        "settledP95YawRollErrorDeg":float(np.percentile(np.abs(settled[:,:,yaw_roll]),95)),
        "settledP95MirrorErrorDeg":float(np.percentile(np.abs(mirror),95)),
        "settledMeanHomeErrorDeg":dict(zip(names,settled.mean((0,1)).tolist())),
        "bodyGroundMeshes":sum("body_ground_" in env.sim.mj_model.geom(i).name for i in range(env.sim.mj_model.ngeom)),
        "automaticResetsCountAsFailure":True,"hardwareValidated":False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("RECOVERY_EVALUATION_JSON="+json.dumps(result))
finally:
    if writer:writer.close()
    if renderer:renderer.close()
    env.close()

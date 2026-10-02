"""Headless stand-hold evaluation; keep fall counts across automatic resets."""
import argparse, json
from dataclasses import replace
import mujoco
import numpy as np
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv

p = argparse.ArgumentParser()
p.add_argument("--onnx")
p.add_argument("--seconds", type=float, default=3.0)
p.add_argument("--envs", type=int, default=16)
p.add_argument("--push", action="store_true")
p.add_argument("--video")
p.add_argument("--voltage", type=float, default=7.4)
a = p.parse_args()
cfg = load_env_cfg("Mjlab-StandHold-Flat-MicroDuck-HD1910", play=True)
cfg.scene.num_envs = a.envs
robot_cfg = cfg.scene.entities["robot"]
robot_cfg.articulation.actuators = (replace(robot_cfg.articulation.actuators[0], vin_range=(a.voltage,a.voltage)),)
cfg.episode_length_s = a.seconds + 1
cfg.events["reset_base"].params["pose_range"].update(roll=(-0.03,0.03),pitch=(-0.03,0.03))
cfg.events["reset_robot_joints"].params["position_range"] = (-0.01,0.01)
if a.push:
    from mjlab.managers import EventTermCfg
    from mjlab.envs import mdp
    cfg.events["eval_push"] = EventTermCfg(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(4.0,4.0), params={"velocity_range":{"x":(-0.08,0.08),"y":(-0.08,0.08)}})
env = ManagerBasedRlEnv(cfg, device="cuda:0")
renderer = writer = cpu_data = camera = None
if a.video:
    import imageio.v2 as imageio
    renderer = mujoco.Renderer(env.sim.mj_model, height=480, width=640)
    cpu_data = mujoco.MjData(env.sim.mj_model)
    camera = mujoco.MjvCamera()
    camera.azimuth = 180; camera.elevation = -15; camera.distance = 0.65
    writer = imageio.get_writer(a.video, fps=25)
session = None
if a.onnx:
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(a.onnx, sess_options=options, providers=["CPUExecutionProvider"])
try:
    obs,_=env.reset(seed=42)
    assert obs["actor"].shape == (a.envs,61)
    failed=torch.zeros(a.envs,dtype=torch.bool,device=env.device)
    tilts=[];heights=[];joint_errors=[]
    joint_names=list(env.scene["robot"].joint_names)
    for step in range(round(a.seconds/env.step_dt)):
        actions=torch.zeros((a.envs,14),device=env.device)
        if session:
            # Exporter fixes batch=1; run each observation independently.
            values=[session.run(None,{session.get_inputs()[0].name:x[None]})[0][0] for x in obs["actor"].cpu().numpy()]
            actions=torch.tensor(np.array(values),device=env.device)
        obs,reward,terminated,truncated,_=env.step(actions)
        failed |= terminated
        assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
        robot=env.scene["robot"]
        joint_errors.append(torch.rad2deg(robot.data.joint_pos-robot.data.default_joint_pos).cpu().numpy())
        tilt=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))*180/torch.pi
        if writer and step % 2 == 0:
            cpu_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            mujoco.mj_forward(env.sim.mj_model, cpu_data)
            camera.lookat[:] = cpu_data.xpos[env.sim.mj_model.body("robot/trunk_base").id]
            renderer.update_scene(cpu_data, camera=camera)
            writer.append_data(renderer.render())
        tilts.append(tilt.cpu().numpy());heights.append((robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]).cpu().numpy())
    tilts=np.array(tilts);heights=np.array(heights)
    settled=np.array(joint_errors)[len(joint_errors)//2:]
    leg_ids=[i for i,n in enumerate(joint_names) if n.startswith(("left_","right_"))]
    symmetry=np.stack([settled[:,:,joint_names.index('left_'+n)]+settled[:,:,joint_names.index('right_'+n)] for n in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')],axis=-1)
    print("EVALUATION_JSON="+json.dumps({"seconds":a.seconds,"voltage":a.voltage,"envs":a.envs,"failedWorlds":int(failed.sum()),"maxTiltDeg":float(tilts.max()),"p95TiltDeg":float(np.percentile(tilts,95)),"endMeanHeight":float(heights[-1].mean()),"massKg":float(env.sim.mj_model.body_mass.sum()),"settledMeanHomeErrorDeg":dict(zip(joint_names,settled.mean(axis=(0,1)).tolist())),"settledP95AbsLegErrorDeg":float(np.percentile(np.abs(settled[:,:,leg_ids]),95)),"settledP95MirrorErrorDeg":float(np.percentile(np.abs(symmetry),95))}))
finally:
    if writer: writer.close()
    if renderer: renderer.close()
    env.close()

"""Frozen, seeded dance rollout; latch failures across resets and save actual motion."""
import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import mujoco
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
p.add_argument('--robust', action='store_true')
p.add_argument('--seed', type=int, default=81371)
p.add_argument('--envs', type=int, default=32)
p.add_argument('--video', action='store_true')
p.add_argument('--stress', action='store_true')
p.add_argument('--task',default='Mjlab-Dance10-Flat-MicroDuck-HD1910')
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
task = a.task
torch.manual_seed(a.seed); np.random.seed(a.seed)
cfg = load_env_cfg(task, play=not a.robust)
cfg.scene.num_envs = a.envs
cfg.episode_length_s = 17.0
for name in ('head_pose','body_pose'):
    cfg.commands[name].ramp_steps=0  # evaluate full amplitude even in robust mode
if a.stress:
    cfg.events['reset_base'].params['pose_range'].update(roll=(-.08,.08),pitch=(-.10,.10))
    cfg.events['reset_robot_joints'].params['position_range']=(-.04,.04)
    cfg.events['push_robot'].params['velocity_range']={'x':(-.16,.16),'y':(-.16,.16)}
    cfg.events['push_robot'].interval_range_s=(3.,3.)
env = ManagerBasedRlEnv(cfg, device='cuda:0')
writer = renderer = None
try:
    runner = load_runner_cls(task)(RslRlVecEnvWrapper(env), asdict(load_rl_cfg(task)), device=env.device)
    runner.load(a.checkpoint, load_cfg={'actor': True}, map_location=env.device)
    actor = runner.alg.actor; actor.eval()
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    obs, _ = env.reset(seed=a.seed)
    initial_qpos=env.sim.data.qpos.detach().cpu().numpy().copy()
    initial_qvel=env.sim.data.qvel.detach().cpu().numpy().copy()
    initial_obs=obs['actor'].detach().cpu().numpy().copy()
    robot = env.scene['robot']
    ids, _ = robot.find_joints_by_actuator_names(['.*neck_pitch.*', '.*head_pitch.*', '.*head_yaw.*', '.*head_roll.*'])
    failed = torch.zeros(a.envs, dtype=torch.bool, device=env.device)
    first = torch.full((a.envs,), -1., device=env.device)
    frames = {k: [] for k in ('time','tilt','position','head','headCommand','bodyCommand','failed','action','yaw','roll')}
    if a.video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(env.sim.mj_model, height=480, width=640)
        cpu = mujoco.MjData(env.sim.mj_model)
        camera = mujoco.MjvCamera(); camera.azimuth=135; camera.elevation=-18; camera.distance=.65
        writer = imageio.get_writer(str(out/'rollout.mp4'), fps=25)
    for tick in range(800):
        head_cmd = env.command_manager.get_command('head_pose').clone()
        body_cmd = env.command_manager.get_command('body_pose').clone()
        with torch.inference_mode():
            action = actor(obs)
            obs, reward, done, timeout, _ = env.step(action)
        q = robot.data.root_link_quat_w
        tilt = torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1))
        yaw = torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
        roll = torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2))
        ended = done | timeout | (tilt > torch.pi/4)
        first = torch.where(ended & ~failed, torch.full_like(first,(tick+1)*env.step_dt), first)
        failed |= ended
        assert torch.isfinite(obs['actor']).all() and torch.isfinite(action).all()
        values = dict(time=(tick+1)*env.step_dt, tilt=tilt,
            position=robot.data.root_link_pos_w-env.scene.env_origins,
            head=robot.data.joint_pos[:,ids]-robot.data.default_joint_pos[:,ids],
            headCommand=head_cmd,bodyCommand=body_cmd,failed=failed.clone(),action=action,yaw=yaw,roll=roll)
        for key,v in values.items(): frames[key].append(v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v)
        if writer and tick%2==0:
            cpu.qpos[:]=env.sim.data.qpos[0].cpu().numpy(); mujoco.mj_forward(env.sim.mj_model,cpu)
            camera.lookat[:]=cpu.xpos[env.sim.mj_model.body('robot/trunk_base').id]
            renderer.update_scene(cpu,camera=camera); writer.append_data(renderer.render())
    assert all(torch.equal(v, actor.state_dict()[k]) for k,v in before.items())
    frames={k:np.array(v) for k,v in frames.items()}
    np.savez_compressed(out/'trace.npz',**frames,initialQpos=initial_qpos,initialQvel=initial_qvel,initialActorObservation=initial_obs)
    alive=~frames['failed']
    report=dict(seed=a.seed,robust=a.robust,stress=a.stress,worlds=a.envs,seconds=16,
        survived=int((~failed).sum()), firstFailureSeconds=first.cpu().tolist(),
        headRmseRad=np.sqrt(np.mean((frames['head']-frames['headCommand'])**2,axis=(0,2))).tolist(),
        headYawPeakToPeakRad=np.ptp(frames['head'][:,:,2],axis=0).tolist(),
        driftMaxM=np.linalg.norm(frames['position'][:,:,:2],axis=-1).max(axis=0).tolist(),
        maxTiltDeg=np.rad2deg(frames['tilt']).max(axis=0).tolist(),
        yawMaxAbsDeg=np.rad2deg(np.abs(frames['yaw'])).max(axis=0).tolist(),
        initialQposSha256=hashlib.sha256(initial_qpos.tobytes()).hexdigest(),
        initialQvelSha256=hashlib.sha256(initial_qvel.tobytes()).hexdigest(),
        initialActorObservationSha256=hashlib.sha256(initial_obs.tobytes()).hexdigest(),
        actorUnchanged=True, optimizerSteps=0, accepted=False,
        note='Motion metrics include post-reset segments for failed worlds; use trace failure mask. GPU screening, not hardware validation.')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
finally:
    if writer: writer.close()
    if renderer: renderer.close()
    env.close()

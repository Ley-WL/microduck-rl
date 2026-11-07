"""Nominal bidirectional/heading-response screen; no hardware I/O."""
import argparse
import hashlib
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.swizzle_observer_hd1910 import make_cfg


def run(args):
    commands=[(.4,0.),(-.4,0.),(.3,-.25),(.3,.25),(-.3,-.25),(-.3,.25),(0.,-.5),(0.,.5)]
    labels=['Forward','Backward','Forward turn -','Forward turn +','Backward turn -','Backward turn +','Zero push turn -','Zero push turn +']
    cfg=make_cfg(play=True);cfg.seed=241079;cfg.scene.num_envs=len(commands)*args.envs
    cfg.episode_length_s=12.;cfg.curriculum.clear();cfg.observations['actor'].enable_corruption=False
    for key in ('push_robot','randomize_com','randomize_head_com','randomize_mass_inertia','randomize_armature','randomize_joint_friction','encoder_bias','imu_orientation'):
        cfg.events.pop(key,None)
    cfg.observations['actor'].terms['joint_pos'].params['biased']=False
    for a in cfg.scene.entities['robot'].articulation.actuators:
        a.vin_range=(7.4,7.4);a.vin_drop_gain_range=(0.,0.)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    net=ort.InferenceSession(args.policy,sess_options=options,providers=['CPUExecutionProvider'])
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    renderer=None;writer=None
    try:
        renderer=mujoco.Renderer(env.sim.mj_model,height=240,width=384)
        cpu=mujoco.MjData(env.sim.mj_model)
        camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance=.85;camera.azimuth=125;camera.elevation=-20
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
        writer=imageio.get_writer(str(out/f'swizzle-{args.updates}.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1)
        obs,_=env.reset(seed=241079);robot=env.scene['robot'];term=env.command_manager.get_term('twist')
        cmd=torch.tensor([[x,0.,h] for x,h in commands],device=env.device).repeat_interleave(args.envs,0)
        bad=torch.zeros(cfg.scene.num_envs,dtype=torch.bool,device=env.device)
        vx=[];wz=[];tilts=[]
        for tick in range(500):
            term.vel_command_b[:]=cmd;term.time_left.fill_(100.)
            rows=obs['actor'].cpu().numpy();rows[:,48:51]=cmd.cpu().numpy();rows[:,51:61]=0.
            action=np.concatenate([net.run(None,{net.get_inputs()[0].name:row[None]})[0] for row in rows],axis=0)
            if action.shape!=(cfg.scene.num_envs,14) or not np.isfinite(action).all():raise RuntimeError('Invalid ONNX action')
            with torch.no_grad():obs,_,done,timeout,_=env.step(torch.as_tensor(action,device=env.device))
            tilt=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))
            bad|=done|timeout|(tilt>np.deg2rad(45))
            if tick%2==0:
                canvas=Image.new('RGB',(1536,608),'#17212b');draw=ImageDraw.Draw(canvas)
                draw.text((12,8),f'HD1910 Swizzle | {args.updates} updates | {(tick+1)*.02:.2f}s | 1x speed | nominal simulation, seed 241079',font=font,fill='white')
                positions=env.sim.data.qpos.cpu().numpy()
                origins=env.scene.env_origins.cpu().numpy()
                for i,(throttle,heading) in enumerate(commands):
                    x=(i%4)*384;y=32+(i//4)*288
                    cpu.qpos[:]=positions[i];cpu.qpos[:3]-=origins[i];mujoco.mj_forward(env.sim.mj_model,cpu)
                    camera.lookat[:]=cpu.xpos[env.sim.mj_model.body('robot/trunk_base').id]
                    renderer.update_scene(cpu,camera=camera)
                    canvas.paste(Image.fromarray(renderer.render()),(x,y+48))
                    draw.text((x+8,y+2),f'{labels[i]}  push {throttle:+.2f}  h {heading:+.2f}',font=font,fill='white')
                    v=float(robot.data.root_link_lin_vel_b[i,0]);w=float(robot.data.root_link_ang_vel_b[i,2])
                    draw.text((x+8,y+24),f'v {v:+.2f} m/s   yaw {w:+.2f} rad/s'+('  FAILED' if bad[i] else ''),font=font,fill='#ffad78' if bad[i] else '#82e6bd')
                writer.append_data(np.asarray(canvas))
                if tick in (100,250,400):canvas.save(out/f'preview-{tick}.jpg')
            if tick>=100:
                vx.append(robot.data.root_link_lin_vel_b[:,0].cpu().numpy().copy())
                wz.append(robot.data.root_link_ang_vel_b[:,2].cpu().numpy().copy())
                tilts.append(np.rad2deg(tilt.cpu().numpy()))
        failed=bad.cpu().numpy();v=np.mean(vx,axis=0);w=np.mean(wz,axis=0)
        report=[]
        for i,(x,h) in enumerate(commands):
            sl=slice(i*args.envs,(i+1)*args.envs);valid=~failed[sl]
            report.append(dict(throttle=x,headingErrorRad=h,worlds=args.envs,survived=int(valid.sum()),
                meanForwardSpeedSurvivors=float(v[sl][valid].mean()) if valid.any() else None,
                meanYawRateSurvivors=float(w[sl][valid].mean()) if valid.any() else None,
                forwardSpeedByWorld=v[sl].tolist(),yawRateByWorld=w[sl].tolist(),failed=failed[sl].tolist()))
        (out/'results.json').write_text(json.dumps(dict(nominalOnly=True,seconds=10,transientSeconds=2,
            commandSemantics='throttle and constant heading error; not velocity targets',rows=report),indent=2),encoding='utf-8')
        (out/'render-audit.json').write_text(json.dumps(dict(policySha256=hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
            video=f'swizzle-{args.updates}.mp4',seed=241079,worlds=8,seconds=10,fps=25,playbackSpeed=1,nominalOnly=True,
            newRollout=True,notOriginal240WorldEvaluation=True,trainingModified=False,hardwareAccess=False),indent=2),encoding='utf-8')
    finally:
        if writer:writer.close()
        if renderer:renderer.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--output',required=True);p.add_argument('--envs',type=int,default=1)
    p.add_argument('--updates',type=int,default=3001)
    args=p.parse_args()
    if args.envs!=1:p.error('render uses one world per command')
    run(args)

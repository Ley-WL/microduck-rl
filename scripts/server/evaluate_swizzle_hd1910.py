"""Nominal bidirectional/heading-response screen; no hardware I/O."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.swizzle_observer_hd1910 import make_cfg


def run(args):
    commands=[(0.,0.),(.2,0.),(.4,0.),(.6,0.),(-.2,0.),(-.4,0.),(-.6,0.),(.3,-.25),(.3,.25),(.3,-.5),(.3,.5),(-.3,-.25),(-.3,.25),(0.,-.5),(0.,.5)]
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
    try:
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
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--output',required=True);p.add_argument('--envs',type=int,default=16)
    args=p.parse_args()
    if args.envs<1:p.error('envs must be positive')
    run(args)

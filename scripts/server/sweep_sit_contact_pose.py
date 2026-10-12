"""GPU static pose audit before training; all worlds sit, no policy or resets."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.ground_contact import lowest_mesh_z
from mjlab_microduck.tasks.microduck_hd1910_sit_contact_env_cfg import get_sit_contact_spec


def sweep(output):
    cfg=load_env_cfg('Mjlab-SitStandContact-Flat-MicroDuck-HD1910',play=True)
    cfg.scene.num_envs=64
    cfg.curriculum.clear()
    cfg.episode_length_s=10000.
    cfg.terminations.pop('time_out',None)
    cfg.commands['twist'].resampling_time_range=(10000.,10000.)
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    model=get_sit_contact_spec().compile()
    data=mujoco.MjData(model)
    rows=[]
    try:
        robot=env.scene['robot']
        for knee in (1.2,1.25,1.3):
            for hip in (.3079,.3579,.4079,.4579):
                reset=env.event_manager.get_term_cfg('set_ground_state')
                override=dict(reset.params['sit_overrides'])
                override.update({2:-hip,11:hip,3:knee,12:-knee})
                target=robot.data.default_joint_pos.clone()
                for index,value in override.items():
                    target[:,index]=value
                data.qpos[7:]=target[0].cpu().numpy()
                mujoco.mj_forward(model,data)
                heights=list(reset.params['spawn_heights'])
                heights[1]=float(data.qpos[2]-lowest_mesh_z(model,data))
                reset.params.update(sit_overrides=override,pose_probabilities=(0.,1.,0.,0.,0.,0.,0.,0.),spawn_heights=tuple(heights),joint_noise=.025,tilt_noise_deg=2.)
                env.reset(seed=5109)
                action=target-robot.data.default_joint_pos
                env.action_manager.get_term('joint_pos')._filtered_actions.copy_(action)
                twist=env.command_manager.get_term('twist')
                twist.vel_command_b[:]=0
                twist.vel_command_b[:,0]=1
                twist._alpha[:]=1
                samples=[]
                for tick in range(round(5./env.step_dt)):
                    _,_,terminated,truncated,_=env.step(action)
                    if terminated.any() or truncated.any():
                        raise RuntimeError('Unexpected reset in static hold')
                    if tick>=round(4./env.step_dt):
                        tilt=torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1.,1.)))
                        height=robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
                        error=(robot.data.joint_pos-target).abs().amax(-1)
                        samples.append(torch.stack((tilt,height,error),-1).cpu().numpy())
                values=np.array(samples)
                row={'knee':knee,'hip':hip,'worlds':64,'uprightWorlds':int((values[:,:,0].max(0)<20.).sum()),
                     'tiltP95Deg':float(np.percentile(values[:,:,0],95)),
                     'heightMedianM':float(np.median(values[:,:,1])),
                     'targetErrorP95Rad':float(np.percentile(values[:,:,2],95))}
                rows.append(row)
                print(json.dumps(row),flush=True)
                output.write_text(json.dumps({'hardwareValidated':False,'policyEvaluation':False,'rows':rows},indent=2))
    finally:
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sweep(args.output)

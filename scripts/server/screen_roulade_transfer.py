"""Independent frozen starts, genuine forward-roll completion, full traces."""
import argparse,json,math
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
import mjlab_microduck.tasks

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[90173,120173])
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
modelcfg={k:v for k,v in asdict(load_rl_cfg('Mjlab-RouladeTransferControl-Flat-MicroDuck').actor).items()
          if k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,**modelcfg)
actor.load_state_dict(state['actor_state_dict']);actor=actor.cuda().eval()
reports=[]
for voltage in (7.4,6.4):
    cfg=load_env_cfg('Mjlab-RouladeTransferControl-Flat-MicroDuck',play=True);cfg.scene.num_envs=64
    cfg.events['set_roulade_state'].params.update(standing_prob=1.,midroll_prob=0.,real_probability=0.)
    for name in list(cfg.events):
        if name.startswith('randomize_'):cfg.events.pop(name)
    if 'encoder_bias' in cfg.events:cfg.events['encoder_bias'].params['bias_range']=(0.,0.)
    cfg.events['reset_robot_joints'].params['position_range']=(-.025,.025)
    cfg.events['foot_friction'].params['ranges']=(1.,1.)
    cfg.scene.entities['robot'].articulation.actuators[0].vin_range=(voltage,voltage)
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        for seed in a.seeds:
            torch.manual_seed(seed);obs,_=env.reset(seed=seed)
            initial=env.sim.data.qpos.cpu().numpy().copy()
            fields={k:[] for k in ('tilt','height','accum','head','targets','qpos','gyro','velocity','done')}
            with torch.no_grad():
                for step in range(150):
                    action=actor(TensorDict({'actor':obs['actor']},batch_size=[64]))
                    obs,_,done,timeout,_=env.step(action)
                    robot=env.scene['robot']
                    for key,value in dict(tilt=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))*180/math.pi,
                        height=robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2],
                        accum=env._roulade_accum,head=env._roulade_head_latch,
                        targets=env.action_manager.get_term('joint_pos').position_target,
                        qpos=env.sim.data.qpos,gyro=robot.data.root_link_ang_vel_b,
                        velocity=robot.data.root_link_lin_vel_w,done=done|timeout).items():
                        fields[key].append(value.cpu().numpy().copy())
            f={k:np.stack(v) for k,v in fields.items()}
            stable=(f['tilt']<20)&(f['height']>.095)&(f['accum']>=math.radians(330))&f['head']
            run=np.zeros(64,dtype=int);time=np.full(64,np.inf)
            for step in range(150):
                run=np.where(stable[step],run+1,0)
                newly=(run>=10)&~np.isfinite(time);time[newly]=(step+1)*.02
            valid=~f['done'].any(0)&np.isfinite(f['qpos']).all(axis=(0,2))
            success=valid&np.isfinite(time)
            frame=np.degrees(np.abs(np.diff(f['targets'],axis=0)))
            accel=np.linalg.norm(np.diff(f['velocity'],axis=0)/.02,axis=2)
            row=dict(seed=seed,voltage=voltage,worlds=64,success=success.tolist(),
                completionSeconds=[float(x) if np.isfinite(x) else None for x in time],
                successWithinOneSecond=int((success&(time<=1)).sum()),
                successWithinTwoSeconds=int((success&(time<=2)).sum()),
                targetFrameMaxDeg=float(frame.max()),targetFrameP95Deg=float(np.quantile(frame,.95)),
                baseAccelerationMax=float(accel.max()),initialQpos=initial.tolist())
            reports.append(row);np.savez_compressed(a.output/f'v{voltage}-seed{seed}.npz',initialQpos=initial,**f)
            print(json.dumps({k:v for k,v in row.items() if k not in ('initialQpos','success','completionSeconds')}),flush=True)
    finally:env.close()
(a.output/'summary.json').write_text(json.dumps(dict(results=reports,simulationOnly=True,hardwareAccess=False,
    completion='supported sagittal rotation >=330deg + head-top latch + upright20deg,height>.095m for200ms',
    deployment=False,limitations=['scalar voltage approximation','frozen simulator starts, not real robot acceptance',
        'accelerations and target frame deltas are diagnostic proxies, not measured impact or joint tracking']),indent=2))

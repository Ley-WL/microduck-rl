"""Compare frozen ONNX against bounded feedback references, with reward audit.

References are diagnostics, not trained policies or hardware instructions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def run(args):
    cfg = load_env_cfg('Mjlab-SitStandContact-Flat-MicroDuck-HD1910', play=True)
    cfg.scene.num_envs = 64
    cfg.curriculum.clear()
    cfg.episode_length_s = 1000.
    cfg.commands['twist'].resampling_time_range = (1000.,1000.)
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    actor = ort.InferenceSession(str(args.onnx), providers=['CPUExecutionProvider'])
    results = []
    try:
        for gain in ([None] if args.actor_only else [None,1.,2.,3.]):
            obs,_ = env.reset(seed=args.seed)
            robot=env.scene['robot']
            home=robot.data.default_joint_pos.clone()
            seated=home.clone()
            for j,value in cfg.events['set_ground_state'].params['sit_overrides'].items():
                seated[:,j]=value
            initially_sit=robot.data.joint_pos[:,3]>.7
            alpha=initially_sit.float()
            action_term=env.action_manager.get_term('joint_pos')
            command=env.command_manager.get_term('twist')
            rows=[]
            demonstrations=[]
            for tick in range(round(20/env.step_dt)):
                t=tick*env.step_dt
                phase=0 if t<2 else 1 if t<8 else 2 if t<14 else 3
                sit=initially_sit if phase%2==0 else ~initially_sit
                command.vel_command_b[:]=0.
                command.vel_command_b[:,0]=sit.float()
                obs['actor'][:,48:]=0.
                obs['actor'][:,48]=sit.float()
                alpha+=(sit.float()-alpha).clamp(-env.step_dt/2,env.step_dt/2)
                reference=home+alpha[:,None]*(seated-home)
                if gain is None:
                    observed=obs['actor'].cpu().numpy()
                    action=torch.tensor(np.concatenate([actor.run(None,
                        {actor.get_inputs()[0].name:row[None]})[0] for row in observed]),device=env.device)
                    if args.collect:
                        demonstrations.append((observed.copy(),action.cpu().numpy()))
                else:
                    action=reference+(gain-1)*(reference-robot.data.joint_pos)-home
                obs,reward,terminated,truncated,_=env.step(action)
                if terminated.any() or truncated.any():
                    raise RuntimeError('Diagnostic terminated/reset')
                z=robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
                tilt=torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1.,1.)))
                target=torch.where(sit[:,None],seated,home)
                legs=[0,1,2,3,4,9,10,11,12,13]
                legerr=(robot.data.joint_pos[:,legs]-target[:,legs]).abs().mean(-1)
                goalz=torch.where(sit,cfg.commands['twist'].sit_z,cfg.commands['twist'].stand_z)
                stable=(tilt<20)&((z-goalz).abs()<.012)&(legerr<.25)
                costs={}
                for name in ['blocked_target_error','target_overdrive','posture_height_l1','posture_pose_legs','rise_bootstrap']:
                    term=env.reward_manager.get_term_cfg(name)
                    costs[name]=(term.func(env,**term.params)*term.weight).cpu().tolist()
                rows.append({'time':t,'phase':phase,'stable':stable.cpu().tolist(),
                    'height':z.cpu().tolist(),'tilt':tilt.cpu().tolist(),
                    'kneeActual':robot.data.joint_pos[:,[3,12]].cpu().tolist(),
                    'kneeTarget':action_term.position_target[:,[3,12]].cpu().tolist(),
                    'rewardRate':(reward/env.step_dt).cpu().tolist(),'terms':costs})
            summary=[]
            for end in [8,14,20]:
                window=[r for r in rows if end-2<=r['time']<end]
                valid=np.mean([r['stable'] for r in window],axis=0)>=.95
                for label,mask in [('stand-start',~initially_sit),('sit-start',initially_sit)]:
                    mask=mask.cpu().numpy()
                    summary.append({'end':end,'start':label,'worlds':int(mask.sum()),
                        'stableWorlds':int(valid[mask].sum()),
                        'heightMean':float(np.mean([np.array(r['height'])[mask] for r in window]))})
            label='actor' if gain is None else f'reference-gain-{gain}'
            (args.output/f'{label}.json').write_text(json.dumps({'summary':summary,'trace':rows}))
            if demonstrations:
                np.savez_compressed(args.output/'demonstrations.npz',
                    observations=np.concatenate([x[0] for x in demonstrations]),
                    actions=np.concatenate([x[1] for x in demonstrations]))
            results.append({'controller':label,'summary':summary})
            print(json.dumps(results[-1]),flush=True)
            (args.output/'summary.json').write_text(json.dumps({'diagnosticOnly':True,'results':results},indent=2))
    finally:
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--actor-only',action='store_true');p.add_argument('--collect',action='store_true')
    p.add_argument('--seed',type=int,default=1009)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);run(args)

"""Frozen transfer-context diagnostic, no optimizer update or hardware access.

This is not CPU/GPU dynamics parity or reconstruction of historical training.
"""
import json,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab_microduck.tasks.walk_surface_transfer import make_cfg,rl_cfg

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence'
records=[]
for surface in (False,True):
    cfg=make_cfg(surface);cfg.scene.num_envs=32;cfg.episode_length_s=26
    cfg.observations['actor'].enable_corruption=False
    cfg.commands['twist'].resampling_time_range=(1000.,1000.)
    for name in ('rel_standing_envs','rel_turn_in_place_envs','rel_heading_envs','rel_world_envs','rel_forward_envs','init_velocity_prob'):
        setattr(cfg.commands['twist'],name,0.)
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        agent=asdict(rl_cfg());agent['upload_model']=False
        runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=None,device=env.device)
        starts={}
        for label in ('Baseline','Control','Surface'):
            runner.load(str(ROOT/f'inputs/{label}.pt'),load_cfg={'actor':True},map_location=env.device)
            actor=runner.alg.actor;actor.eval()
            before={k:v.clone() for k,v in actor.state_dict().items()}
            for speed in (.1,.2,.25):
                # Explicit reset at unchanged common_step_counter needs fresh caches.
                for name in ('_walk_velocity_step','_walk_twist_step'):
                    if hasattr(env,name):delattr(env,name)
                obs,_=env.reset(seed=97133)
                robot=env.scene['robot'];term=env.command_manager.get_term('twist')
                start=robot.data.root_link_pos_w.clone()
                if speed in starts:assert torch.equal(start,starts[speed]),'Unmatched initial root position'
                else:starts[speed]=start.clone()
                failed=torch.zeros(32,dtype=torch.bool,device=env.device)
                first=torch.full((32,),-1.,device=env.device)
                peak=torch.zeros(32,device=env.device)
                vsum=torch.zeros(32,device=env.device);vcount=torch.zeros(32,device=env.device)
                filtered=torch.zeros((32,3),device=env.device)
                for tick in range(1250):
                    t=tick*.02;command=[speed if 2<=t<22 else 0.,0.,0.]
                    desired=torch.tensor(command,device=env.device)
                    term.vel_command_b[:]=desired;term.vel_command_w[:]=desired
                    term.is_standing_env[:]=not any(command);term.is_heading_env[:]=False;term.is_world_env[:]=False
                    filtered=.8*filtered+.2*desired
                    obs['actor'][:,48:51]=filtered
                    gravity=obs['actor'][:,3:6]
                    tilt=torch.rad2deg(torch.acos((-gravity[:,2]).clamp(-1,1)))
                    peak=torch.maximum(peak,torch.where(~failed,tilt,torch.zeros_like(tilt)))
                    with torch.inference_mode():action=actor(obs)
                    obs,_,done,timeout,_=env.step(action)
                    ended=done|timeout|(tilt>=60)
                    first=torch.where(ended&~failed,torch.full_like(first,t),first)
                    failed|=ended
                    if 4<=t<22:
                        live=~failed;vsum+=torch.where(live,robot.data.root_link_lin_vel_b[:,0],0.);vcount+=live
                velocity=(vsum/vcount.clamp_min(1)).cpu().numpy()
                records.append(dict(policy=label,surface=surface,speed=speed,seed=97133,
                    survival=(~failed).cpu().tolist(),firstFailureSeconds=first.cpu().tolist(),
                    movingMeanVx=velocity.tolist(),maxTiltDeg=peak.cpu().tolist(),
                    effectiveForward=((~failed).cpu().numpy()&(velocity>=max(.02,.3*speed))).tolist()))
                print(json.dumps(dict(policy=label,surface=surface,speed=speed,
                    surviving=int((~failed).sum()),forward=int(records[-1]['effectiveForward'].count(True)))),flush=True)
            assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
            assert not runner.alg.optimizer.state,'Diagnostic checkpoints must have empty optimizer'
    finally:env.close()
(OUT/'summary.json').write_text(json.dumps(dict(results=records,actorNormalizerUnchanged=True,
    optimizerSteps=0,hardwareAccess=False,accepted=False,
    boundary='32world frozen transfer context, observation noise disabled, seed97133; not CPU/GPU dynamics parity'),indent=2))

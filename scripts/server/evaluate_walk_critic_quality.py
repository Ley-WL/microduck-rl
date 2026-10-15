"""Frozen on-policy value calibration against long discounted reward tails."""
import argparse
import importlib
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence'
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,default=Path('inputs/candidate.pt'))
p.add_argument('--collect-training',action='store_true',help='Separate seeds and whole-world critic dataset; never reuse calibration seeds for fitting')
p.add_argument('--task',choices=['walk_low_speed_mean_tracking','walk_mean_motion_tracking','walk_motion_critic_context'],default='walk_low_speed_mean_tracking')
a=p.parse_args()
module=importlib.import_module('mjlab_microduck.tasks.'+a.task)
make_cfg,rl_cfg=module.make_cfg,module.rl_cfg
checkpoint=ROOT/a.checkpoint
reports=[]
for seed in ((239101,239107) if a.collect_training else (235151,235157)):
    torch.manual_seed(seed)
    cfg=make_cfg()
    cases=[(v,yaw) for v in (-.2,-.05,.05,.2) for yaw in (-.6,0.,.6)]
    cfg.scene.num_envs=len(cases)*4
    # First 30s are scored; another 30s reduces missing-tail discount to gamma^1500.
    # Training episode ends at 32s; the first scored cycle retains the same commands.
    cfg.episode_length_s=64.
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        agent=asdict(rl_cfg());agent['upload_model']=False
        runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=None,device=env.device)
        runner.load(str(checkpoint),load_cfg={'actor':True,'critic':True},map_location=env.device)
        actor,critic=runner.alg.actor.eval(),runner.alg.critic.eval()
        before={label:{k:v.detach().clone() for k,v in model.state_dict().items()}
                for label,model in [('actor',actor),('critic',critic)]}
        gamma=runner.alg.gamma
        obs,_=env.reset(seed=seed)
        command=env.command_manager.get_term('twist')
        grid=torch.tensor(cases,device=env.device).repeat_interleave(4,dim=0)
        command.stop_speed[:]=grid[:,0];command.stop_yaw[:]=grid[:,1]
        alive=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        reward_rows=[];value_rows=[];done_rows=[];live_rows=[];commands=[]
        observation_rows=[]
        with torch.inference_mode():
            for tick in range(3000):
                if a.collect_training and tick<1500:
                    observation_rows.append(obs['critic'].detach().cpu().clone())
                value_rows.append(critic(obs).squeeze(-1).clone())
                live_rows.append(alive.clone())
                commands.append(env.command_manager.get_command('twist').clone())
                actions=actor(obs,stochastic_output=True)
                obs,reward,terminated,truncated,_=env.step(actions)
                done=terminated|truncated
                reward_rows.append(reward.clone());done_rows.append(done.clone())
                alive &= ~done
                if tick%500==0:
                    (OUT/'critic-progress.json').write_text(json.dumps(dict(seed=seed,tick=tick,alive=int(alive.sum()))))
        rewards=torch.stack(reward_rows);values=torch.stack(value_rows)
        dones=torch.stack(done_rows);live=torch.stack(live_rows)
        returns=torch.zeros_like(rewards);tail=torch.zeros(env.num_envs,device=env.device)
        for tick in reversed(range(3000)):
            tail=rewards[tick]+gamma*(~dones[tick])*tail
            returns[tick]=tail
        assert torch.isfinite(rewards).all() and torch.isfinite(values).all()
        for label,model in [('actor',actor),('critic',critic)]:
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in before[label].items())
        result=[]
        for name,start,end in [('stand',0,1),('forward',5,11),('stopFirst',11,16),
                               ('reverse',20,26),('stopSecond',26,30),('fullCycle',0,30)]:
            lo,hi=round(start/.02),round(end/.02)
            mask=live[lo:hi];target=returns[lo:hi][mask];prediction=values[lo:hi][mask]
            residual=prediction-target;variance=target.var(unbiased=False)
            result.append(dict(phase=name,samples=int(mask.sum()),
                meanReturn=float(target.mean()),meanPrediction=float(prediction.mean()),
                rmse=float(residual.square().mean().sqrt()),bias=float(residual.mean()),
                explainedVariance=float(1-residual.var(unbiased=False)/variance) if variance>1e-12 else None,
                targetStd=float(variance.sqrt())))
        np.savez_compressed(OUT/f'critic-{seed}.npz',rewards=rewards.cpu().numpy(),
            values=values.cpu().numpy(),returns=returns.cpu().numpy(),dones=dones.cpu().numpy(),
            live=live.cpu().numpy(),commands=torch.stack(commands).cpu().numpy(),dt=.02,gamma=gamma)
        if a.collect_training:
            torch.save(dict(observations=torch.stack(observation_rows),
                returns=returns[:1500].cpu(),live=live[:1500].cpu(),
                validationWorlds=torch.arange(env.num_envs)%4==3,seed=seed,
                checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
                OUT/f'critic-data-{seed}.pt')
        reports.append(dict(seed=seed,worlds=env.num_envs,survivedFirst30=int(live[1499].sum()),
            survived60=int(alive.sum()),gamma=gamma,unobservedTailDiscount=gamma**1500,phases=result))
        (OUT/'critic-quality.json').write_text(json.dumps(dict(results=reports,
            optimizationSteps=0,actorAndCriticUnchanged=True,stochasticTrainingPolicy=True,task=a.task,
            checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            limitation='Frozen critic on its own final actor distribution; 64s diagnostic episode differs from 32s training timeout, first 30s only scored. Value fit does not prove policy gradient causality.',
            hardwareAccess=False,accepted=False),indent=2))
    finally:
        env.close()

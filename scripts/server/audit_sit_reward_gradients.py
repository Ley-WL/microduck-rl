"""Exact additive GAE and frozen PPO policy-gradient reward attribution."""
import copy,hashlib,json,os,subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from audit_sit_gradient_contributions import vector,cosine
ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'evidence/reward-gradients'

def gae_parts(rewards,values,dones,timeouts,last,gamma,lam):
    rewards=rewards.double();values=values.double();last=last.double()
    next_values=torch.cat([values[1:],last.unsqueeze(0)],0)
    critic=gamma*next_values*(1-dones.double())-values+gamma*values*timeouts.double()
    deltas=torch.cat([rewards,critic.unsqueeze(-1)],-1)
    parts=torch.zeros_like(deltas);following=torch.zeros_like(deltas[0])
    for t in reversed(range(len(deltas))):
        following=deltas[t]+gamma*lam*(1-dones[t].double()).unsqueeze(-1)*following
        parts[t]=following
    raw=parts.sum(-1)
    normalized=(parts-parts.mean((0,1),keepdim=True))/(raw.std()+1e-8)
    return parts,normalized

def run(label,seed):
    task='Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910'
    cfg=load_env_cfg(task);cfg.scene.num_envs=64
    agent=load_rl_cfg(task);agent.num_steps_per_env=512
    torch.manual_seed(seed);np.random.seed(seed)
    env=ManagerBasedRlEnv(cfg,device='cuda:0');wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    runner=load_runner_cls(task)(wrapped,asdict(agent),device=env.device)
    runner.load(str(ROOT/'checkpoints'/f'{label}.pt'),map_location=env.device)
    alg=runner.alg;alg.train_mode();params=list(alg.actor.parameters())
    before={k:v.clone() for k,v in alg.actor.state_dict().items()}
    names=list(env.reward_manager.active_terms)+['critic_bootstrap_value']
    rewards=[];timeouts=[];reward_errors=[]
    try:
        obs,_=wrapped.reset()
        with torch.no_grad():
            for _ in range(512):
                action=alg.act(obs);obs,reward,done,extras=wrapped.step(action)
                scale=env.step_dt if env.reward_manager._scale_by_dt else 1.
                term_rewards=env.reward_manager._step_reward.clone()*scale
                reward_errors.append(float((term_rewards.sum(-1)-reward).abs().max()))
                torch.testing.assert_close(term_rewards.sum(-1),reward,atol=1e-5,rtol=1e-5)
                rewards.append(term_rewards);timeouts.append(extras.get('time_outs',torch.zeros_like(done)).clone())
                alg.process_env_step(obs,reward,done,extras)
            last=alg.critic(obs).detach().squeeze(-1)
            alg.compute_returns(obs)
        st=alg.storage
        parts,normalized=gae_parts(torch.stack(rewards),st.values.squeeze(-1),st.dones.squeeze(-1),torch.stack(timeouts),last,alg.gamma,alg.lam)
        actual=st.advantages.squeeze(-1)
        difference=float((normalized.sum(-1)-actual).abs().max())
        torch.testing.assert_close(normalized.sum(-1).float(),actual,atol=2e-4,rtol=2e-4)
        assert not alg.normalize_advantage_per_mini_batch and alg.entropy_coef==0 and alg.symmetry is None
        all_obs=st.observations['actor'].flatten(0,1);actions=st.actions.flatten(0,1)
        old_log=st.actions_log_prob.flatten(0,1).squeeze(-1);advantages=actual.flatten();contributions=normalized.flatten(0,1).float()
        batches=[];order=torch.randperm(len(all_obs),device=env.device)
        for ids in order.chunk(4):
            alg.actor(alg._td(all_obs[ids]),stochastic_output=True)
            ratio=(alg.actor.get_output_log_prob(actions[ids]).squeeze(-1)-old_log[ids]).exp()
            ratio_error=float((ratio-1).abs().max());assert ratio_error<1e-4
            actual_loss=torch.max(-advantages[ids]*ratio,-advantages[ids]*ratio.clamp(1-alg.clip_param,1+alg.clip_param)).mean()
            total=vector(params,torch.autograd.grad(actual_loss,params,retain_graph=True,allow_unused=True))
            terms={}
            for i,name in enumerate(names):
                loss=-(contributions[ids,i]*ratio).mean()
                terms[name]=vector(params,torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True))
            summed=sum(terms.values());relative=float((summed-total).norm()/(total.norm()+1e-12))
            assert relative<2e-4,relative
            batches.append({'totalPpoRawNorm':float(total.norm()),'sumGradientRelativeError':relative,'maxRatioError':ratio_error,
                'terms':{k:{'norm':float(g.norm()),'cosineWithTotal':cosine(g,total),'signedTotalProjectionFraction':float(torch.dot(g,total)/(total.square().sum()+1e-20))} for k,g in terms.items()}})
            if len(batches)==1:
                np.savez_compressed(FOLDER/f'gradients-{label}-{seed}.npz',total=total.cpu().numpy(),**{k:g.cpu().numpy() for k,g in terms.items()})
        for k,v in before.items():torch.testing.assert_close(v,alg.actor.state_dict()[k],atol=0,rtol=0)
        np.savez_compressed(FOLDER/f'rollout-{label}-{seed}.npz',observations=st.observations['actor'].cpu().numpy(),actions=st.actions.cpu().numpy(),values=st.values.cpu().numpy(),dones=st.dones.cpu().numpy(),termRewards=torch.stack(rewards).cpu().numpy(),timeouts=torch.stack(timeouts).cpu().numpy(),lastValues=last.cpu().numpy(),advantages=actual.cpu().numpy(),normalizedParts=normalized.cpu().numpy())
        return {'label':label,'seed':seed,'rewardNames':names,'maxRewardSumError':max(reward_errors),'maxNormalizedAdvantageReconstructionError':difference,'actorIncludingNormalizerExactlyPreserved':True,'batches':batches}
    finally:env.close()

if __name__=='__main__':
    FOLDER.mkdir(parents=True,exist_ok=True)
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
        usage=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert usage<200
        manifest=json.loads((FOLDER/'launch.json').read_text())
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        result={'hardwareAccess':False,'training':False,'formalAccepted':False,'scope':'Frozen same-run Main0/Main99 WeakProcess reward attribution; old EMA contract, not current kissqy platform','runs':[]}
        for label in ('Main0','Main99'):
            for seed in (9017,12031):
                result['runs'].append(run(label,seed));(FOLDER/'summary.json').write_text(json.dumps(result,indent=2))
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        result['sourceHashesExactlyPreserved']=True;(FOLDER/'summary.json').write_text(json.dumps(result,indent=2));(FOLDER/'job.exit').write_text('0')
    except BaseException:(FOLDER/'job.exit').write_text('1');raise

"""Frozen 1024-world, 24-step independent rollout attribution, no optimizer."""
import hashlib,json,os,subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from audit_sit_reward_gradients import gae_parts
from audit_sit_gradient_contributions import vector,cosine
ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'evidence/formal24'

def run(label,seed):
    task='Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910'
    cfg=load_env_cfg(task);cfg.scene.num_envs=1024
    agent=load_rl_cfg(task)
    assert agent.num_steps_per_env==24 and agent.algorithm.num_mini_batches==4
    torch.manual_seed(seed);np.random.seed(seed)
    env=ManagerBasedRlEnv(cfg,device='cuda:0');wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    runner=load_runner_cls(task)(wrapped,asdict(agent),device=env.device)
    runner.load(str(ROOT/'checkpoints'/f'{label}.pt'),map_location=env.device)
    alg=runner.alg;alg.train_mode();params=list(alg.actor.parameters())
    before={k:v.clone() for k,v in alg.actor.state_dict().items()}
    stdmask=torch.cat([torch.full((p.numel(),),'distribution' in n,device=env.device,dtype=torch.bool) for n,p in alg.actor.named_parameters()])
    names=list(env.reward_manager.active_terms)+['critic_bootstrap_value']
    results=[];seen_edge=False;previous_flag=None
    try:
        obs,_=wrapped.reset()
        for block in range(21):
            rewards=[];timeouts=[];reward_errors=[];edges=0
            with torch.no_grad():
                for step in range(24):
                    flag=obs['actor'][:,48]>.5
                    if previous_flag is not None:edges+=int((flag!=previous_flag).sum())
                    previous_flag=flag.clone()
                    action=alg.act(obs);obs,reward,done,extras=wrapped.step(action)
                    scale=env.step_dt if env.reward_manager._scale_by_dt else 1.
                    terms=env.reward_manager._step_reward.clone()*scale
                    reward_errors.append(float((terms.sum(-1)-reward).abs().max()))
                    torch.testing.assert_close(terms.sum(-1),reward,atol=1e-5,rtol=1e-5)
                    rewards.append(terms);timeouts.append(extras.get('time_outs',torch.zeros_like(done)).clone())
                    alg.process_env_step(obs,reward,done,extras)
                last=alg.critic(obs).detach().squeeze(-1)
                alg.compute_returns(obs)
            phase='initial' if block==0 else ('first_edge' if edges and not seen_edge else ('late' if block==20 else None))
            if phase=='first_edge':seen_edge=True
            if phase is None:alg.storage.clear();continue
            st=alg.storage
            _,normalized=gae_parts(torch.stack(rewards),st.values.squeeze(-1),st.dones.squeeze(-1),torch.stack(timeouts),last,alg.gamma,alg.lam)
            actual=st.advantages.squeeze(-1)
            difference=float((normalized.sum(-1)-actual).abs().max())
            torch.testing.assert_close(normalized.sum(-1).float(),actual,atol=2e-4,rtol=2e-4)
            assert not alg.normalize_advantage_per_mini_batch and alg.entropy_coef==0 and alg.symmetry is None
            all_obs=st.observations['actor'].flatten(0,1);actions=st.actions.flatten(0,1)
            old_log=st.actions_log_prob.flatten(0,1).squeeze(-1);advantages=actual.flatten();contributions=normalized.flatten(0,1).float()
            batches=[];order=torch.randperm(len(all_obs),device=env.device)
            for index,ids in enumerate(order.chunk(4)):
                assert len(ids)==6144
                alg.actor(alg._td(all_obs[ids]),stochastic_output=True)
                ratio=(alg.actor.get_output_log_prob(actions[ids]).squeeze(-1)-old_log[ids]).exp()
                ratio_error=float((ratio-1).abs().max());assert ratio_error<1e-4
                total=vector(params,torch.autograd.grad(-(advantages[ids]*ratio).mean(),params,retain_graph=True,allow_unused=True))
                terms={name:vector(params,torch.autograd.grad(-(contributions[ids,i]*ratio).mean(),params,retain_graph=True,allow_unused=True)) for i,name in enumerate(names)}
                relative=float((sum(terms.values())-total).norm()/(total.norm()+1e-12));assert relative<2e-4
                groups={}
                for group,mask in [('meanNetwork',~stdmask),('distribution',stdmask)]:
                    g=total[mask]
                    groups[group]={'norm':float(g.norm()),'terms':{k:{'norm':float(v[mask].norm()),'cosineWithTotal':cosine(v[mask],g),'signedProjection':float(torch.dot(v[mask],g)/(g.square().sum()+1e-20))} for k,v in terms.items()}}
                batches.append({'samples':len(ids),'sumGradientRelativeError':relative,'maxRatioError':ratio_error,'totalRawNorm':float(total.norm()),'groups':groups})
                np.savez_compressed(FOLDER/f'vectors-{label}-{seed}-{phase}-{index}.npz',total=total.cpu().numpy(),sampleIndices=ids.cpu().numpy(),**{k:v.cpu().numpy() for k,v in terms.items()})
            np.savez_compressed(FOLDER/f'rollout-{label}-{seed}-{phase}.npz',observations=st.observations['actor'].cpu().numpy(),actions=st.actions.cpu().numpy(),values=st.values.cpu().numpy(),dones=st.dones.cpu().numpy(),termRewards=torch.stack(rewards).cpu().numpy(),timeouts=torch.stack(timeouts).cpu().numpy(),lastValues=last.cpu().numpy(),advantages=actual.cpu().numpy(),normalizedParts=normalized.cpu().numpy())
            results.append({'phase':phase,'startStep':block*24,'commandEdges':edges,'maxRewardSumError':max(reward_errors),'maxAdvantageReconstructionError':difference,'batches':batches})
            for k,v in before.items():torch.testing.assert_close(v,alg.actor.state_dict()[k],atol=0,rtol=0)
            alg.storage.clear()
        assert seen_edge and len(results)==3
        return {'label':label,'seed':seed,'windows':results,'actorIncludingNormalizerExactlyPreserved':True,'criticNormalizerScope':'normal updates during rollout, fixed during each gradient audit'}
    finally:env.close()

if __name__=='__main__':
    FOLDER.mkdir(parents=True,exist_ok=True)
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
        usage=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert usage<200
        manifest=json.loads((FOLDER/'launch.json').read_text())
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        result={'hardwareAccess':False,'training':False,'formalAccepted':False,'scope':'Independent frozen 1024-world rollout in 24-step blocks, four randomized 6144-sample batches; first frozen-ratio update point only, not historical training minibatches or later clipped epochs. Old EMA contract, not current kissqy platform.','runs':[]}
        for label in ('Main0','Main99'):
            for seed in (9017,12031):
                result['runs'].append(run(label,seed));(FOLDER/'summary.json').write_text(json.dumps(result,indent=2))
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        result['sourceHashesExactlyPreserved']=True;(FOLDER/'summary.json').write_text(json.dumps(result,indent=2));(FOLDER/'job.exit').write_text('0')
    except BaseException:(FOLDER/'job.exit').write_text('1');raise

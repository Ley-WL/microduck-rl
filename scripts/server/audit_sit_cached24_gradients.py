"""CPU cached-context policy gradients; no simulator/optimizer/GPU access."""
import hashlib,json,os
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.tasks.registry import load_rl_cfg
import mjlab_microduck.tasks
from audit_sit_reward_gradients import gae_parts
from audit_sit_gradient_contributions import vector,cosine
ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'evidence/cached24'

def windows(obs):
    changes=np.flatnonzero(np.any(np.diff(obs[:,:,48]>.5,axis=0),axis=1))+1
    assert len(changes)>0
    return [('initial',0),('first_edge',int(changes[0])//24*24),('late',480)]

def run(label,seed):
    cache=np.load(ROOT/'inputs'/f'rollout-{label}-{seed}.npz')
    cfg=load_rl_cfg('Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910')
    assert cfg.num_steps_per_env==24
    actor_cfg=asdict(cfg.actor)
    actor_cfg={k:v for k,v in actor_cfg.items() if k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
    td=TensorDict({'actor':torch.tensor(cache['observations'][0])},batch_size=[64])
    actor=MLPModel(td,{'actor':['actor']},'actor',14,**actor_cfg)
    state=torch.load(ROOT/'checkpoints'/f'{label}.pt',map_location='cpu',weights_only=False)['actor_state_dict']
    actor.load_state_dict(state,strict=True);actor.eval()
    original={k:v.clone() for k,v in actor.state_dict().items()};params=list(actor.parameters())
    rewards=torch.from_numpy(cache['termRewards']);values=torch.from_numpy(cache['values'][...,0]);dones=torch.from_numpy(cache['dones'][...,0]);timeouts=torch.from_numpy(cache['timeouts']);last=torch.from_numpy(cache['lastValues'])
    _,reconstruction=gae_parts(rewards,values,dones,timeouts,last,.99,.95)
    torch.testing.assert_close(reconstruction,torch.from_numpy(cache['normalizedParts']),atol=2e-12,rtol=2e-12)
    names=json.loads((ROOT/'inputs/summary.json').read_text())['runs'][0]['rewardNames']
    rows=[]
    for phase,start in windows(cache['observations']):
        end=start+24
        obs=TensorDict({'actor':torch.from_numpy(cache['observations'][start:end]).flatten(0,1)},batch_size=[24*64])
        actions=torch.from_numpy(cache['actions'][start:end]).flatten(0,1)
        following=values[end] if end<len(values) else last
        actor(obs,stochastic_output=True)
        log_prob=actor.get_output_log_prob(actions).squeeze(-1)
        ratio=(log_prob-log_prob.detach()).exp() # frozen on-policy evaluation point
        assert bool((ratio==1).all())
        gradients={}
        for lam in (.95,1.):
            _,parts=gae_parts(rewards[start:end],values[start:end],dones[start:end],timeouts[start:end],following,.99,lam)
            components=parts.flatten(0,1).float();advantage=components.sum(-1)
            total=vector(params,torch.autograd.grad(-(advantage*ratio).mean(),params,retain_graph=True,allow_unused=True))
            terms={}
            for i,name in enumerate(names):
                loss=-(components[:,i]*ratio).mean()
                terms[name]=vector(params,torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True))
            relative=float((sum(terms.values())-total).norm()/(total.norm()+1e-12));assert relative<2e-4
            gradients[lam]=total
            rows.append({'phase':phase,'startStep':start,'horizonSteps':24,'lambda':lam,'totalRawNorm':float(total.norm()),'sumRelativeError':relative,
                'terms':{k:{'norm':float(g.norm()),'cosineWithTotal':cosine(g,total),'signedTotalProjectionFraction':float(torch.dot(g,total)/(total.square().sum()+1e-20))} for k,g in terms.items()}})
            if lam==.95:
                np.savez_compressed(FOLDER/f'vectors-{label}-{seed}-{phase}.npz',total=total.numpy(),**{k:v.numpy() for k,v in terms.items()})
        rows[-1]['cosineWithLambda095']=cosine(gradients[.95],gradients[1.])
    for k,v in original.items():torch.testing.assert_close(v,actor.state_dict()[k],atol=0,rtol=0)
    return {'label':label,'seed':seed,'actorIncludingNormalizerExactlyPreserved':True,'windows':rows}

if __name__=='__main__':
    FOLDER.mkdir(parents=True,exist_ok=True)
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        torch.set_num_threads(2)
        manifest=json.loads((FOLDER/'launch.json').read_text())
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        result={'hardwareAccess':False,'gpuAccess':False,'training':False,'formalAccepted':False,'scope':'CPU 24-step cuts of cached 64-world trajectories at frozen checkpoint; not original 1024-world training minibatches. Lambda1 is counterfactual gradient only, not trained policy or acceptance.','runs':[]}
        for label in ('Main0','Main99'):
            for seed in (9017,12031):
                result['runs'].append(run(label,seed));(FOLDER/'summary.json').write_text(json.dumps(result,indent=2))
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        result['sourceHashesExactlyPreserved']=True;(FOLDER/'summary.json').write_text(json.dumps(result,indent=2));(FOLDER/'job.exit').write_text('0')
    except BaseException:(FOLDER/'job.exit').write_text('1');raise

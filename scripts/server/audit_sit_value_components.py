"""Split current state baseline from future value in cached formal24, CPU only."""
import copy,hashlib,json,os
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
ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'evidence/formal24'

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2)
    assert (FOLDER/'job.exit').read_text()=='0'
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FOLDER.glob('*.npz'))}
    result={'training':False,'gpuAccess':False,'hardwareAccess':False,'scope':'CPU decomposition of same cached formal24 frozen first-update gradients. Current baseline expected-zero claim assumes on-policy sampling and independence; finite-sample projection is not proof of faulty value learning. Future remainder includes GAE next-value, boundary bootstrap, resets and timeout correction; not solely terminal value.','sourceHashes':source,'runs':[]}
    cfg=asdict(load_rl_cfg('Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910').actor)
    cfg={k:v for k,v in cfg.items() if k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
    for label in ('Main0','Main99'):
        actor=MLPModel(TensorDict({'actor':torch.zeros(1024,61)},batch_size=[1024]),{'actor':['actor']},'actor',14,**copy.deepcopy(cfg))
        actor.load_state_dict(torch.load(ROOT/'checkpoints'/f'{label}.pt',map_location='cpu',weights_only=False)['actor_state_dict'],strict=True);actor.eval()
        original={k:v.clone() for k,v in actor.state_dict().items()};params=list(actor.parameters())
        mask=torch.cat([torch.full((p.numel(),),'distribution' not in n,dtype=torch.bool) for n,p in actor.named_parameters()])
        for seed in (9017,12031):
            for phase in ('initial','first_edge','late'):
                data=np.load(FOLDER/f'rollout-{label}-{seed}-{phase}.npz')
                values=torch.from_numpy(data['values'][...,0])
                parts,normalized=gae_parts(torch.from_numpy(data['termRewards']),values,torch.from_numpy(data['dones'][...,0]),torch.from_numpy(data['timeouts']),torch.from_numpy(data['lastValues']),.99,.95)
                torch.testing.assert_close(normalized,torch.from_numpy(data['normalizedParts']),atol=2e-12,rtol=2e-12)
                raw=parts.sum(-1);baseline=-values.double();future=parts[...,-1]-baseline;reward=parts[...,:-1].sum(-1)
                components=torch.stack([baseline,future,reward],-1)
                components=((components-components.mean((0,1),keepdim=True))/(raw.std()+1e-8)).flatten(0,1).float()
                torch.testing.assert_close(components.sum(-1).reshape(raw.shape),torch.from_numpy(data['advantages']),atol=2e-4,rtol=2e-4)
                observations=torch.from_numpy(data['observations']).flatten(0,1);actions=torch.from_numpy(data['actions']).flatten(0,1)
                for index in range(4):
                    saved=np.load(FOLDER/f'vectors-{label}-{seed}-{phase}-{index}.npz');ids=torch.from_numpy(saved['sampleIndices'])
                    actor(TensorDict({'actor':observations[ids]},batch_size=[len(ids)]),stochastic_output=True)
                    log=actor.get_output_log_prob(actions[ids]).squeeze(-1);ratio=(log-log.detach()).exp()
                    total=vector(params,torch.autograd.grad(-(components[ids].sum(-1)*ratio).mean(),params,retain_graph=True,allow_unused=True))
                    gradients={name:vector(params,torch.autograd.grad(-(components[ids,i]*ratio).mean(),params,retain_graph=True,allow_unused=True)) for i,name in enumerate(('current_state_baseline','future_value_remainder','reward_gae'))}
                    error=float((total-torch.from_numpy(saved['total'])).norm()/(total.norm()+1e-12));assert error<1e-3
                    sum_error=float((sum(gradients.values())-total).norm()/(total.norm()+1e-12));assert sum_error<2e-4
                    row={'label':label,'seed':seed,'phase':phase,'batch':index,'cpuGpuTotalRelativeError':error,'sumRelativeError':sum_error,'meanNetwork':{name:{'norm':float(g[mask].norm()),'signedProjection':float(torch.dot(g[mask],total[mask])/(total[mask].square().sum()+1e-20)),'cosineWithTotal':cosine(g[mask],total[mask])} for name,g in gradients.items()}}
                    result['runs'].append(row)
                    np.savez_compressed(FOLDER/f'valueparts-{label}-{seed}-{phase}-{index}.npz',total=total.numpy(),**{k:v.numpy() for k,v in gradients.items()})
                    (FOLDER/'value-components.json').write_text(json.dumps(result,indent=2))
        for k,v in original.items():torch.testing.assert_close(v,actor.state_dict()[k],atol=0,rtol=0)
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in source.items())
    result['sourceHashesExactlyPreserved']=True;result['actorIncludingNormalizerExactlyPreserved']=True
    (FOLDER/'value-components.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    try:run();(FOLDER/'value-components.exit').write_text('0')
    except BaseException:(FOLDER/'value-components.exit').write_text('1');raise

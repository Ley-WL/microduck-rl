"""Initialize a single actor from real rollouts of the complementary experts.

Preserve the seed normalizer, critic and deployment contract; reset PPO state.
This is supervised initialization, not a claim of motion qualification.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import onnxruntime as ort
from rsl_rl.modules.normalization import EmpiricalNormalization


def main(a):
    if a.output.exists():raise ValueError('Preserve existing seed')
    torch.manual_seed(2027)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state=checkpoint['actor_state_dict']
    std=state['distribution.std_param']
    if std.shape!=(14,) or not torch.isfinite(std).all() or (std<=0).any():
        raise ValueError('Unexpected exploration parameter encoding')
    normalizer=EmpiricalNormalization(61).to(a.device)
    normalizer.load_state_dict({k.removeprefix('obs_normalizer.'):v for k,v in state.items() if k.startswith('obs_normalizer.')})
    normalizer.eval()
    actor=nn.Sequential(nn.Linear(61,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,14)).to(a.device)
    actor.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')})
    data=np.load(a.data)
    observations=data['observations'];actions=data['actions']
    if observations.shape[1:]!=(61,) or actions.shape!=(len(observations),14) or not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError('Invalid demonstration contract')
    if len(observations)%64:raise ValueError('Expected tick-major 64-world trajectories')
    # Hold out entire worlds, avoiding adjacent-time leakage in validation.
    validation=np.arange(len(observations))%64<8
    x=torch.tensor(observations,device=a.device);y=torch.tensor(actions,device=a.device)
    with torch.no_grad():x=normalizer(x)
    session=ort.InferenceSession(str(a.seed_onnx),providers=['CPUExecutionProvider'])
    with torch.no_grad():original=actor(x[:16]).cpu().numpy()
    exported=np.concatenate([session.run(None,{session.get_inputs()[0].name:row[None]})[0] for row in observations[:16]])
    np.testing.assert_allclose(original,exported,rtol=2e-5,atol=2e-5)
    train=torch.tensor(np.where(~validation)[0],device=a.device);valid=torch.tensor(np.where(validation)[0],device=a.device)
    optimizer=torch.optim.Adam(actor.parameters(),lr=3e-4)
    def errors():
        with torch.no_grad():
            pred=actor(x[valid]);flags=torch.tensor(observations[validation,48]>.5,device=a.device)
            return {name:float((pred[mask]-y[valid][mask]).square().mean().sqrt())
                for name,mask in [('sit',flags),('stand',~flags)]}
    initial=errors();best=max(initial.values());best_state=copy.deepcopy(actor.state_dict());history=[]
    for epoch in range(60):
        shuffled=train[torch.randperm(len(train),device=a.device)]
        for ids in shuffled.split(4096):
            loss=(actor(x[ids])-y[ids]).square().mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        metrics=errors();score=max(metrics.values())
        if score<best:best=score;best_state=copy.deepcopy(actor.state_dict())
        history.append({'epoch':epoch+1,'validationActionRmseRad':metrics})
        if epoch%10==9:print(json.dumps(history[-1]),flush=True)
    if best>=max(initial.values()):raise RuntimeError('Neither direction improved')
    for key,value in best_state.items():state['mlp.'+key]=value.cpu()
    state['distribution.std_param'].clamp_(min=.1,max=.2)
    checkpoint['optimizer_state_dict']['state']={}
    for group in checkpoint['optimizer_state_dict']['param_groups']:
        group['lr']=3e-4
    checkpoint['iter']=0
    checkpoint['infos']['env_state']['common_step_counter']=0
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(checkpoint,a.output)
    proof={'kind':'single-actor-imitation-initialization','sourceCheckpointSha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'demonstrationSha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'checkpointSha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
        'samples':len(observations),'heldOutWorlds':8,'initialValidationActionRmseRad':initial,
        'bestWorstDirectionValidationActionRmseRad':best,'normalizerPreserved':True,'criticPreserved':True,
        'optimizerMomentsReset':True,'iterationAndCurriculumReset':True,'hardwareValidated':False,'history':history}
    a.output.with_suffix('.json').write_text(json.dumps(proof,indent=2));print(json.dumps({k:v for k,v in proof.items() if k!='history'}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['checkpoint','seed-onnx','data','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');main(p.parse_args())

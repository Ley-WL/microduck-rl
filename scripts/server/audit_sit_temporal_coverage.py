"""CPU frozen actors: auxiliary gradients on paired cached inputs, no optimizer."""
import hashlib,json,os
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.tasks.registry import load_rl_cfg
import mjlab_microduck.tasks
ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'evidence/temporal-coverage'

def flat(grads,parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(grads,parameters)])

def alignment(a,b):
    denominator=float(a.norm()*b.norm())
    return None if denominator<1e-15 else float(torch.dot(a,b)/denominator)

def actor_from(path,obs):
    cfg=load_rl_cfg('Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910')
    settings={k:v for k,v in asdict(cfg.actor).items() if k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
    actor=MLPModel(TensorDict({'actor':obs},batch_size=[len(obs)]),{'actor':['actor']},'actor',14,**settings)
    actor.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['actor_state_dict'],strict=True);actor.eval()
    return actor

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2)
    manifest=json.loads((FOLDER/'launch.json').read_text())
    def hashes_match():
        for name,value in manifest['sourceHashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name
    hashes_match()
    cache=np.load(ROOT/'inputs/temporal-coverage-inputs.npz')
    # One reproducible 512-pair draw, without replacement; not historical PPO batch.
    ids=np.random.default_rng(73109).choice(len(cache['current']),512,replace=False)
    current=torch.from_numpy(cache['current'][ids]);previous=torch.from_numpy(cache['previous'][ids])
    np.save(FOLDER/'sampled-pair-indexes.npy',ids)
    data=np.load(ROOT/'transfer-data.npz');real=torch.from_numpy(data['real']);pairs=torch.from_numpy(data['pair_indexes']).long()
    assert len(real)==525 and len(pairs)==517 and bool((pairs>0).all())
    td=lambda x:TensorDict({'actor':x},batch_size=[len(x)])
    anchor=actor_from(ROOT/'checkpoints/baseline.pt',current)
    for parameter in anchor.parameters():parameter.requires_grad_(False)
    with torch.no_grad():expected=anchor(td(current));expected_previous=anchor(td(previous))
    torch.testing.assert_close(expected,torch.from_numpy(cache['teacher'][ids]),atol=3e-6,rtol=1e-5)
    torch.testing.assert_close(expected_previous,torch.from_numpy(cache['teacher_previous'][ids]),atol=3e-6,rtol=1e-5)
    rows=[]
    for label in ('baseline','WeakProcess','CriticalReplay'):
        actor=actor_from(ROOT/'checkpoints'/f'{label}.pt',current)
        initial={k:v.clone() for k,v in actor.state_dict().items()};parameters=list(actor.parameters())
        prediction=actor(td(current));old=actor(td(previous));physical=actor(td(real))
        session=ort.InferenceSession(str(ROOT/'onnx'/f'{label}.onnx'),providers=['CPUExecutionProvider'])
        exported=np.concatenate([session.run(None,{session.get_inputs()[0].name:ob[None]})[0] for ob in current.numpy()[:16]])
        torch.testing.assert_close(prediction[:16].detach(),torch.from_numpy(exported),atol=3e-6,rtol=1e-5)
        change=prediction-old;teacher_change=expected-expected_previous
        knee_squared=change[:,[3,12]].square().flatten()
        tail_indexes=torch.topk(knee_squared.detach(),max(1,int(np.ceil(len(knee_squared)*.05)))).indices
        objectives={
            'real_temporal_weighted':.3*(physical[pairs]-physical[pairs-1]).square().mean(),
            'critical_retention_weighted':10.*(prediction-expected).square().mean(),
            'critical_absolute_smooth_same_weight':.3*change.square().mean(),
            'critical_teacher_delta_same_weight':.3*(change-teacher_change).square().mean(),
            'critical_knee_tail_proxy':knee_squared[tail_indexes].mean(),
        }
        gradients={name:flat(torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True),parameters).detach() for name,value in objectives.items()}
        proxy=gradients['critical_knee_tail_proxy']
        np.savez_compressed(FOLDER/f'vectors-{label}.npz',**{k:v.numpy() for k,v in gradients.items()})
        row={'label':label,'pairs':512,'actualSampleSeed':73109,'actorIncludingNormalizerExactlyPreserved':True,'parameters':sum(p.numel() for p in parameters),'gradientTerms':{k:{'weightedLoss':float(objectives[k].detach()),'norm':float(v.norm()),'cosineWithCounterfactualKneeTail':alignment(v,proxy)} for k,v in gradients.items()},'orthogonalScope':'Positive cosine means descending both frozen-input objectives locally aligns. Not an optimizer step, PPO comparison, closed-loop response, or proof of deployment improvement.'}
        for key,value in initial.items():torch.testing.assert_close(value,actor.state_dict()[key],atol=0,rtol=0)
        rows.append(row)
    hashes_match()
    result={'gpuAccess':False,'hardwareAccess':False,'training':False,'deployment':False,'rows':rows,'sourceHashesExactlyPreserved':True,'anchorExportParityPassed':True,'scope':'Same fitted successful baseline adjacent inputs, bounded within trial/window; real pairs as original training. Frozen CPU actor gradients, sampled pair tail proxy not actual screen frame gate. Baseline naturally moving outputs must not all be flattened.'}
    (FOLDER/'summary.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    try:main();(FOLDER/'job.exit').write_text('0')
    except BaseException:(FOLDER/'job.exit').write_text('1');raise

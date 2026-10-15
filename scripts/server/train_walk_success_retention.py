"""Bounded CPU supervised retention from two frozen teachers, one deployable graph.

Not PPO, not a dual-actor runtime, and fitted cases are not holdout acceptance.
"""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import onnx
from onnx import numpy_helper
import onnxruntime as ort

R=Path(__file__).resolve().parents[2];I=R/'inputs';E=R/'evidence'
torch.set_num_threads(1);torch.manual_seed(171337)
graph=onnx.load(I/'Surface.onnx')
tensors={v.name:numpy_helper.to_array(v).copy() for v in graph.graph.initializer}
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(61,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
            nn.Linear(256,128),nn.ELU(),nn.Linear(128,14))
        self.register_buffer('mean',torch.tensor(tensors['obs_normalizer._mean']))
        self.register_buffer('denominator',torch.tensor(tensors['onnx::Div_24']))
        self.mlp.load_state_dict({k[4:]:torch.tensor(v) for k,v in tensors.items() if k.startswith('mlp.')})
    def forward(self,x):return self.mlp((x-self.mean)/self.denominator)
actor=Actor();normalizer={k:v.clone() for k,v in actor.named_buffers()}
pool=np.load(I/'retention-pool.npz')
anchor_x=torch.tensor(pool['anchorObs']);anchor_y=torch.tensor(pool['anchorRaw'])
keep_x=torch.tensor(pool['preserveObs']);keep_y=torch.tensor(pool['preserveRaw'])
assert anchor_x.shape[1]==keep_x.shape[1]==61 and anchor_y.shape[1]==keep_y.shape[1]==14
assert all(torch.isfinite(x).all() for x in (anchor_x,anchor_y,keep_x,keep_y))
opts=ort.SessionOptions();opts.intra_op_num_threads=1
def infer(path,rows):
    session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    return np.concatenate([session.run(None,{session.get_inputs()[0].name:x[None]})[0] for x in rows])
probe=np.random.default_rng(171337).normal(0,.4,(128,61)).astype(np.float32)
with torch.no_grad():error=float(abs(actor(torch.tensor(probe)).numpy()-infer(I/'Surface.onnx',probe)).max())
assert error<2e-5,error
optimizer=torch.optim.Adam(actor.parameters(),lr=1e-5)
generator=torch.Generator().manual_seed(171337);audits=[]
for step in range(1,101):
    a=torch.randint(len(anchor_x),(256,),generator=generator)
    b=torch.randint(len(keep_x),(512,),generator=generator)
    anchor_loss=(actor(anchor_x[a])-anchor_y[a]).square().mean()
    preserve_loss=(actor(keep_x[b])-keep_y[b]).square().mean()
    params=list(actor.parameters())
    ga=torch.autograd.grad(anchor_loss,params,retain_graph=True)
    gp=torch.autograd.grad(preserve_loss,params,retain_graph=True)
    av=torch.cat([v.flatten() for v in ga]);pv=torch.cat([v.flatten() for v in gp])
    optimizer.zero_grad();(anchor_loss+preserve_loss).backward()
    grad_norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
    optimizer.step()
    audits.append(dict(step=step,anchorLoss=float(anchor_loss.detach()),preserveLoss=float(preserve_loss.detach()),
        anchorGradientNorm=float(av.norm()),preserveGradientNorm=float(pv.norm()),combinedRawNorm=grad_norm,
        gradientCosine=float(torch.dot(av,pv)/(av.norm()*pv.norm()).clamp_min(1e-20))))
    if step in (20,50,100):
        assert all(torch.equal(v,dict(actor.named_buffers())[k]) for k,v in normalizer.items())
        assert all(torch.isfinite(v).all() for v in actor.state_dict().values())
        output=E/f'Step{step}';output.mkdir(exist_ok=False)
        result=copy.deepcopy(graph);state=actor.state_dict()
        for index,t in enumerate(result.graph.initializer):
            if t.name.startswith('mlp.'):result.graph.initializer[index].CopyFrom(numpy_helper.from_array(state[t.name].detach().numpy(),t.name))
        path=output/'policy.onnx';onnx.checker.check_model(result);onnx.save(result,path)
        with torch.no_grad():expected=actor(torch.tensor(probe)).numpy()
        err=float(abs(infer(path,probe)-expected).max());assert err<2e-5,err
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        meta=json.loads((I/'Surface.metadata.json').read_text())
        meta.update(policySha256=sha,artifactType='supervised-success-retention-candidate',hardwareValidated=False)
        meta['walkExecution']['policySha256']=sha
        meta['successRetention']=dict(steps=step,seed=171337,accepted=False,normalizerUnchanged=True,
            criticOptimizerSteps=0,source='Surface20; baseline lost-success teacher + Surface successful trajectories')
        path.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2))
        torch.save(dict(actor=actor.state_dict(),optimizer=optimizer.state_dict(),steps=step),output/'checkpoint.pt')
        (output/'audit.json').write_text(json.dumps(dict(steps=step,finite=True,normalizerUnchanged=True,
            onnxTorchMaxError=err,policySha256=sha,accepted=False,hardwareAccess=False),indent=2))
(E/'gradient-audit.json').write_text(json.dumps(dict(rows=audits,seed=171337,steps=100,
    learningRate=1e-5,anchorWeight=1,preserveWeight=1,anchorBatch=256,preserveBatch=512,
    actorOnnxInitialError=error,normalizerUnchanged=True,accepted=False),indent=2))

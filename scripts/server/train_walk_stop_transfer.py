"""One bounded transfer stage; baked actor normalizer frozen, critic learns."""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import onnxruntime as ort
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
import mjlab_microduck.tasks
from mjlab_microduck.tasks.walk_stop_transfer import make_cfg,rl_cfg

p=argparse.ArgumentParser();p.add_argument('--surface',action='store_true')
p.add_argument('--envs',type=int,required=True);p.add_argument('--iterations',type=int,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert (a.envs,a.iterations) in [(64,5),(1024,100)]
ROOT=Path(__file__).resolve().parents[2];a.output.mkdir(parents=True,exist_ok=False)
torch.manual_seed(68131);cfg=make_cfg(a.surface);cfg.scene.num_envs=a.envs
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    rl=asdict(rl_cfg());rl.update(upload_model=False,logger='tensorboard')
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),rl,log_dir=str(a.output/'logs'),device='cuda:0')
    runner.load(str(ROOT/'inputs/seed.pt'),map_location='cuda:0')
    env.common_step_counter=0;runner.current_learning_iteration=0
    actor=runner.alg.actor
    normalizer={k:v.clone() for k,v in actor.state_dict().items() if k.startswith('obs_normalizer.')}
    # Retain the baked deployment distribution; only critic normalization learns.
    actor.update_normalization=lambda *args,**kwargs:None
    assert not runner.alg.optimizer.state
    runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
    assert all(torch.equal(actor.state_dict()[k],v) for k,v in normalizer.items()),'Actor normalizer drift'
    steps={int(v['step']) for v in runner.alg.optimizer.state.values()}
    assert steps=={a.iterations*8},steps
    checkpoint=a.output/'final.pt';runner.save(str(checkpoint))
    runner.export_policy_to_onnx(str(a.output),'policy.onnx')
    for model in (runner.alg.actor,runner.alg.critic):
        assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    rows=np.random.default_rng(68331).normal(0,.4,(128,61)).astype(np.float32)
    actor.eval()
    with torch.no_grad():expected=actor(TensorDict({'actor':torch.tensor(rows,device=env.device)},batch_size=[128])).cpu().numpy()
    session=ort.InferenceSession(str(a.output/'policy.onnx'),providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
    actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:r[None]})[0] for r in rows])
    error=float(abs(actual-expected).max());assert error<2e-5,error
    meta=json.loads((ROOT/'inputs/walk.metadata.json').read_text())
    sha=hashlib.sha256((a.output/'policy.onnx').read_bytes()).hexdigest()
    meta.update(policySha256=sha,artifactType='diagnostic-transfer-policy',hardwareValidated=False)
    meta['walkExecution']['policySha256']=sha
    meta['transferExperiment']=dict(surface=a.surface,seed=68131,worlds=a.envs,iterations=a.iterations,
        originalDonorResume=False,critic='fresh-initialized',actorNormalizerFrozen=True,accepted=False)
    (a.output/'policy.metadata.json').write_text(json.dumps(meta,indent=2))
    (a.output/'stage-audit.json').write_text(json.dumps(dict(seed=68131,surface=a.surface,worlds=a.envs,
        iterations=a.iterations,rolloutSteps=a.envs*24*a.iterations,adamSteps=sorted(steps),actorNormalizerUnchanged=True,
        finite=True,onnxTorchMaxError=error,policySha256=sha,hardwareAccess=False,accepted=False),indent=2))
finally:env.close()

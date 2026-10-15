"""Export a frozen walking actor through the runner, preserving normalization."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab_microduck.tasks.walk_tracking_transfer import make_cfg, rl_cfg

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[2]
source=root/a.checkpoint;out=root/a.output;out.mkdir(parents=True,exist_ok=False)
saved=torch.load(source,map_location='cpu',weights_only=False)
seed=torch.load(root/'inputs/seed.pt',map_location='cpu',weights_only=False)
for key,value in saved['actor_state_dict'].items():
    assert torch.isfinite(value).all()
    if key.startswith('obs_normalizer.'):
        assert torch.equal(value,seed['actor_state_dict'][key]),key
cfg=make_cfg();cfg.scene.num_envs=1
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    agent=asdict(rl_cfg());agent['upload_model']=False
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=None,device=env.device)
    runner.load(str(source),load_cfg={'actor':True},map_location=env.device)
    actor=runner.alg.actor;actor.eval()
    assert all(torch.equal(v.cpu(),saved['actor_state_dict'][k]) for k,v in actor.state_dict().items())
    runner.export_policy_to_onnx(str(out),'policy.onnx')
    rows=np.random.default_rng(258017).normal(0,.4,(128,61)).astype(np.float32)
    with torch.inference_mode():
        expected=actor(TensorDict({'actor':torch.tensor(rows,device=env.device)},batch_size=[128])).cpu().numpy()
    session=ort.InferenceSession(str(out/'policy.onnx'),providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
    actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:r[None]})[0] for r in rows])
    error=float(abs(actual-expected).max());assert error<2e-5
    sha=hashlib.sha256((out/'policy.onnx').read_bytes()).hexdigest()
    meta=json.loads((root/'inputs/walk.metadata.json').read_text())
    meta.update(policySha256=sha,artifactType='diagnostic-walk-intermediate',hardwareValidated=False)
    meta['walkExecution']['policySha256']=sha
    (out/'policy.metadata.json').write_text(json.dumps(meta,indent=2))
    steps=sorted({int(v['step']) for v in saved['optimizer_state_dict']['state'].values()})
    (out/'audit.json').write_text(json.dumps(dict(sourceCheckpoint=str(a.checkpoint),
        sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),savedIteration=saved['iter'],
        savedAdamSteps=steps,actorExactlyLoaded=True,actorNormalizerUnchanged=True,
        finite=True,onnxTorchMaxError=error,policySha256=sha,optimizationSteps=0,accepted=False),indent=2))
finally:
    env.close()

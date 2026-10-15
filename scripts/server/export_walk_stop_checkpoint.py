"""Export one saved early checkpoint with runner normalizer and numerical audit."""
import json,hashlib
from pathlib import Path
from dataclasses import asdict
import torch,numpy as np,onnxruntime as ort
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab_microduck.tasks.walk_stop_transfer import make_cfg,rl_cfg
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'evidence/checkpoint25-export';out.mkdir(exist_ok=False)
cfg=make_cfg();cfg.scene.num_envs=1;env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
 agent=asdict(rl_cfg());agent['upload_model']=False
 runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=None,device=env.device)
 runner.load(str(ROOT/'evidence/main/logs/model_25.pt'),load_cfg={'actor':True},map_location=env.device)
 runner.export_policy_to_onnx(str(out),'policy.onnx');actor=runner.alg.actor;actor.eval()
 rows=np.random.default_rng(191017).normal(0,.4,(128,61)).astype(np.float32)
 with torch.inference_mode():expected=actor(TensorDict({'actor':torch.tensor(rows,device=env.device)},batch_size=[128])).cpu().numpy()
 sess=ort.InferenceSession(str(out/'policy.onnx'),providers=['CPUExecutionProvider']);actual=np.concatenate([sess.run(None,{sess.get_inputs()[0].name:r[None]})[0] for r in rows]);error=float(np.max(abs(expected-actual)));assert error<2e-5
 sha=hashlib.sha256((out/'policy.onnx').read_bytes()).hexdigest();meta=json.loads((ROOT/'inputs/walk.metadata.json').read_text());meta.update(policySha256=sha,artifactType='diagnostic-stop-transfer-checkpoint25',hardwareValidated=False);meta['walkExecution']['policySha256']=sha
 (out/'policy.metadata.json').write_text(json.dumps(meta,indent=2));(out/'audit.json').write_text(json.dumps(dict(policySha256=sha,onnxTorchMaxError=error,sourceCheckpoint='model_25.pt',hardwareValidated=False)))
finally:env.close()

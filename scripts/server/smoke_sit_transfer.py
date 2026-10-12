"""Exercise transition kicks and verify frozen-normalizer seed training."""
import json,os
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

root=Path(__file__).resolve().parents[2]
cfg=load_env_cfg(os.environ.get('MICRODUCK_CHECK_TASK','Mjlab-SitStandTransfer-Flat-MicroDuck-HD1910'))
cfg.scene.num_envs=64
cfg.commands['twist'].resampling_time_range=(.4,.6)
env=ManagerBasedRlEnv(cfg,device='cuda:0')
options=ort.SessionOptions();options.intra_op_num_threads=1
session=ort.InferenceSession(str(root/'weight-1p0.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
finite=True
try:
    obs,_=env.reset(seed=8111)
    for _ in range(160):
        values=np.concatenate([session.run(None,{session.get_inputs()[0].name:row[None]})[0] for row in obs['actor'].cpu().numpy()])
        obs,reward,terminated,truncated,_=env.step(torch.from_numpy(values).to(env.device))
        finite&=bool(torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all())
    counts=env._sit_transfer_kick_count.cpu().numpy()
    assert finite and int(counts.sum())>0 and obs['actor'].shape==(64,61)
    logs=list((root/'logs/rsl_rl'/os.environ.get('MICRODUCK_CHECK_EXPERIMENT','hd1910_sitstand_transfer_v1')).glob('*'+os.environ.get('MICRODUCK_CHECK_RUN','transfer-smoke-v2')+'/model_4.pt'))
    assert len(logs)==1
    source=torch.load(root/'seed.pt',map_location='cpu',weights_only=False)['actor_state_dict']
    result=torch.load(logs[0],map_location='cpu',weights_only=False)['actor_state_dict']
    for key in source:
        if key.startswith('obs_normalizer.'):
            torch.testing.assert_close(source[key],result[key],rtol=0,atol=0)
    report={'finite':finite,'worlds':64,'steps':160,'transitionKickCount':int(counts.sum()),
            'worldsKicked':int((counts>0).sum()),'normalizerExactlyPreservedInSmoke':True,
            'hardwareAccess':False,'motionAcceptance':False}
    (root/'evidence/smoke/contract-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
finally:env.close()

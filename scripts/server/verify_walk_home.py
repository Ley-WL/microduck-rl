"""Check the actual replica HOME equilibrium before spending a gait run."""
import json
from pathlib import Path
import numpy as np
import torch
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

cfg=load_env_cfg('Mjlab-Walk-Flat-MicroDuck-HD1910',play=True)
cfg.scene.num_envs=8
cfg.commands['twist'].rel_standing_envs=1.
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    env.reset(seed=42)
    tilted=[];failed=0
    for _ in range(round(3/env.step_dt)):
        _,_,terminated,_,_=env.step(torch.zeros((8,14),device=env.device))
        failed+=int(terminated.sum())
        tilted.append(torch.rad2deg(torch.acos((-env.scene['robot'].data.projected_gravity_b[:,2]).clamp(-1,1))).cpu().numpy())
    result={'seconds':3,'envs':8,'terminations':failed,'tiltMaxDeg':float(np.max(tilted)),
            'tiltLastP95Deg':float(np.percentile(tilted[-1],95)),'hardwareValidated':False}
    print(json.dumps(result),flush=True)
    (Path(__file__).resolve().parents[3]/'setup/hd1910-walk-v2-home.json').write_text(json.dumps(result,indent=2))
    assert failed==0 and result['tiltLastP95Deg']<10,result
finally:
    env.close()

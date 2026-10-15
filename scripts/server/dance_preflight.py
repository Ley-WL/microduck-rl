"""Verify noisy HOME equilibrium in the actual assembled HD1910 GPU model."""
import json
from pathlib import Path
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv

cfg = load_env_cfg('Mjlab-Dance10-Flat-MicroDuck-HD1910', play=True)
cfg.scene.num_envs = 32
torch.manual_seed(71421)
env = ManagerBasedRlEnv(cfg, device='cuda:0')
try:
    obs, _ = env.reset(seed=71421)
    assert obs['actor'].shape == (32, 61)
    resets = 0
    worst_tilt = 0.
    for i in range(150):
        obs, reward, terminated, truncated, _ = env.step(torch.zeros(32, 14, device='cuda:0'))
        resets += int((terminated | truncated).sum())
        q = env.scene['robot'].data.root_link_quat_w
        tilt = torch.acos((1-2*(q[:, 1]**2+q[:, 2]**2)).clamp(-1, 1))
        worst_tilt = max(worst_tilt, float(tilt.max()))
        assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
    report = dict(worlds=32, seconds=3, resets=resets, maxTiltRad=worst_tilt,
                  equilibriumPassed=resets==0 and worst_tilt < .25,
                  danceAccepted=False)
    Path('evidence/preflight.json').write_text(json.dumps(report, indent=2))
    assert report['equilibriumPassed'], report
finally:
    env.close()

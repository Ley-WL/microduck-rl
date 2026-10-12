"""Split saved gradients using verified actual actor parameter ordering, CPU only."""
import json, os
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.tasks.registry import load_rl_cfg
import mjlab_microduck.tasks

root = Path(__file__).resolve().parents[2]
folder = root / 'evidence/cached24'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
cfg = asdict(load_rl_cfg('Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910').actor)
cfg = {k:v for k,v in cfg.items() if k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
actor = MLPModel(TensorDict({'actor':torch.zeros(64,61)}, batch_size=[64]), {'actor':['actor']}, 'actor', 14, **cfg)
offset = 0
layout = []
std_mask = []
for name, parameter in actor.named_parameters():
    layout.append({'name':name,'start':offset,'end':offset+parameter.numel()})
    std_mask.extend(['distribution' in name] * parameter.numel())
    offset += parameter.numel()
std_mask = np.array(std_mask)
assert std_mask.any() and (~std_mask).any()
rows = []
for path in sorted(folder.glob('vectors-*.npz')):
    data = np.load(path)
    total = data['total']
    assert len(total) == offset
    row = {'file':path.name,'stdSquaredNormFraction':float((total[std_mask]**2).sum()/(total**2).sum()),'groups':{}}
    for group, mask in [('distribution',std_mask),('meanNetwork',~std_mask)]:
        g = total[mask]; denominator = float(g@g)
        row['groups'][group] = {'totalNorm':float(np.linalg.norm(g)), 'terms':{name:{'norm':float(np.linalg.norm(data[name][mask])), 'signedProjection':float(data[name][mask]@g/(denominator+1e-20))} for name in data.files if name!='total'}}
    rows.append(row)
(folder/'parameter-split.json').write_text(json.dumps({'scope':'Frozen cached 24-step CPU gradients, no optimizer step; Gaussian distribution parameters separated from deterministic mean output network.','layout':layout,'rows':rows},indent=2))
print(json.dumps({'files':len(rows),'parameters':offset,'distributionParameters':int(std_mask.sum())}))

"""Independently verify full-return targets and whole-world holdout evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('evidence',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
reports=[]
for seed in (239101,239107):
    path=a.evidence/f'critic-data-{seed}.pt'
    data=torch.load(path,map_location='cpu',weights_only=False)
    with np.load(a.evidence/f'critic-{seed}.npz') as raw:
        target=np.zeros(raw['rewards'].shape,dtype=np.float64)
        tail=np.zeros(target.shape[1],dtype=np.float64)
        for tick in reversed(range(len(target))):
            tail=raw['rewards'][tick]+float(raw['gamma'])*(~raw['dones'][tick])*tail
            target[tick]=tail
        error=float(np.max(np.abs(target[:1500]-data['returns'].numpy())))
        assert error<1e-4
        expected_live=np.concatenate((np.ones((1,48),dtype=bool),
            np.logical_and.accumulate(~raw['dones'],axis=0)[:-1]),axis=0)
        assert np.array_equal(data['live'].numpy(),expected_live[:1500])
        assert np.array_equal(data['live'].numpy(),raw['live'][:1500])
        assert data['observations'].shape==(1500,48,86)
        assert torch.isfinite(data['observations']).all()
        assert torch.equal(data['validationWorlds'],torch.arange(48)%4==3)
        assert data['seed']==seed
        reports.append(dict(seed=seed,maxReturnError=error,
            trainingSamples=int(data['live'][:,~data['validationWorlds']].sum()),
            validationSamples=int(data['live'][:,data['validationWorlds']].sum()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checkpointSha256=data['checkpointSha256']))
assert len({r['checkpointSha256'] for r in reports})==1
a.output.write_text(json.dumps(dict(passed=True,reports=reports,
    hardwareAccess=False,gaitAccepted=False),indent=2),encoding='utf-8')
print(a.output)

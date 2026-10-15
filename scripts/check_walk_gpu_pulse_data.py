"""Validate frozen pulse recordings before any imitation dataset is assembled."""
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
report=json.loads((a.evidence/'pulse-teacher-screen.json').read_text())
assert report['pulseTeacher'] and report['yawGrid']==[0.]
assert len(report['results'])==4
rows=[]
for row in report['results']:
    path=a.evidence/row['demonstrationFile']
    data=torch.load(path,map_location='cpu',weights_only=False)
    frames=800 if row['policy']=='candidate' else 1500
    worlds=len(report['cases'])*4
    x,y,live=data['observation'],data['target'],data['live']
    assert x.shape==(frames,worlds,61) and y.shape==(frames,worlds,14)
    assert live.shape==(frames,worlds) and live.dtype==torch.bool
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert not ((~live[:-1]) & live[1:]).any(),'Reset frames must never reenter the dataset'
    assert torch.equal(data['validationWorlds'],torch.arange(worlds)%4==3)
    assert data['seed']==row['seed'] and data['policy']==row['policy']
    assert [list(case) for case in data['cases']]==report['cases'] and data['clockAtRuntime'] is False
    if row['policy']=='candidate':assert (x[:,:,48][live]>=-1e-6).all(),'Reverse teacher is not accepted'
    with np.load(a.evidence/row['motionTrace']) as trace:
        # smoothed_twist applies old + .2*(command-old), once per control step.
        # Actor observations precede the step; the trace contains the next raw command.
        filtered=x[0,:,48:51].numpy().copy()
        raw_commands=trace['values'][:frames-1,:,3:6]
        for tick in range(frames-1):
            filtered=filtered+np.float32(.2)*(raw_commands[tick]-filtered)
            active=live[tick+1].numpy()
            np.testing.assert_allclose(x[tick+1,:,48:51].numpy()[active],filtered[active],atol=5e-8,rtol=1e-6)
    assert np.array_equal(np.asarray(row['phaseStepCount']).sum(1),row['stepCount'])
    rows.append(dict(seed=row['seed'],policy=row['policy'],frames=frames,worlds=worlds,
        liveSamples=int(live.sum()),validationWorlds=int(data['validationWorlds'].sum()),
        checkpointSha256=data['checkpointSha256'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert len({r['checkpointSha256'] for r in rows})==1
a.output.write_text(json.dumps(dict(passed=True,rows=rows,optimizationSteps=0,
    accepted=False,hardwareAccess=False),indent=2),encoding='utf-8')
print(a.output)

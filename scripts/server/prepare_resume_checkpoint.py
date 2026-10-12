"""Resume after the saved iteration without repeating its already-applied update."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('source',type=Path)
p.add_argument('destination',type=Path)
p.add_argument('--total-iterations',type=int,required=True)
a=p.parse_args()
if a.destination.exists():
    raise RuntimeError('Destination exists; preserve existing checkpoint')
checkpoint=torch.load(a.source,map_location='cpu',weights_only=False)
last=int(checkpoint['iter'])
next_iteration=last+1
remaining=a.total_iterations-next_iteration
if remaining<=0:
    raise ValueError('Source already meets or exceeds the requested training budget')
counter=checkpoint['infos']['env_state']['common_step_counter']
if counter!=next_iteration*24:
    raise ValueError(f'Unexpected 24-step rollout counter: {counter}')
# Preserve actor/critic, normalizers, Adam state, and the curriculum counter.
checkpoint['iter']=next_iteration
a.destination.parent.mkdir(parents=True,exist_ok=True)
torch.save(checkpoint,a.destination)
proof={'source':str(a.source),'sourceSha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
       'destination':str(a.destination),'lastCompletedIteration':last,'nextIteration':next_iteration,
       'commonStepCounter':counter,'totalIterations':a.total_iterations,'remainingUpdates':remaining}
a.destination.with_suffix('.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof),flush=True)

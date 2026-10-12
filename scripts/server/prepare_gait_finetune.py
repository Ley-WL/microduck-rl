"""Preserve actor/critic/normalizers; use gentle exploration for gait repair."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
a=p.parse_args()
if a.destination.exists():raise RuntimeError('Preserve existing seed')
checkpoint=torch.load(a.source,map_location='cpu',weights_only=False)
std=checkpoint['actor_state_dict']['distribution.std_param']
if std.shape!=(14,) or not torch.isfinite(std).all() or (std<=0).any():
    raise ValueError('Unexpected exploration parameter; do not guess its encoding')
old=std.tolist();std.clamp_(max=.2)
# Keep learned parameters and statistics; discard momentum of the old objective.
checkpoint['optimizer_state_dict']['state']={}
a.destination.parent.mkdir(parents=True,exist_ok=True);torch.save(checkpoint,a.destination)
proof={'source':str(a.source),'sourceSha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
    'actorCriticAndNormalizersPreserved':True,'oldExplorationStd':old,
    'newExplorationStd':std.tolist(),'optimizerMomentsReset':True,'hardwareValidated':False}
a.destination.with_suffix('.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))

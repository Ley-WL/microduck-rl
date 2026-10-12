"""Relearn head control from HOME without erasing an existing leg policy.

This is initialization only. All 14 outputs remain trainable and unfrozen.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch


def prepare(checkpoint,head_retain=0.):
    if not 0.<=head_retain<=1.:raise ValueError('head_retain must be within [0,1]')
    result=copy.deepcopy(checkpoint)
    actor=result['actor_state_dict']
    weight=actor['mlp.6.weight'];bias=actor['mlp.6.bias']
    std=actor['distribution.std_param']
    if weight.shape!=(14,128) or bias.shape!=(14,) or std.shape!=(14,):
        raise ValueError('Unexpected actor architecture; do not guess head output rows')
    weight[5:9]*=head_retain;bias[5:9]*=head_retain;std[5:9]=.1
    # The only 14-row optimizer tensors are this actor's output layer/std.
    # Clear changed rows so inherited Adam momentum cannot restore old droop.
    for state in result['optimizer_state_dict']['state'].values():
        for value in state.values():
            if isinstance(value,torch.Tensor) and value.shape in ((14,128),(14,)):
                value[5:9]=0.
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    parser.add_argument('--head-retain',type=float,default=0.)
    args=parser.parse_args()
    if args.destination.exists():raise RuntimeError('Destination exists; preserve previous artifacts')
    source=torch.load(args.source,map_location='cpu',weights_only=False)
    prepared=prepare(source,args.head_retain)
    args.destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(prepared,args.destination)
    proof={'source':str(args.source),'sourceSha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
           'headOutputRows':[5,6,7,8],'headExplorationStdRad':.1,
           'retainedHeadOutputFraction':args.head_retain,
           'legWeightsPreserved':True,'normalizersPreserved':True,
           'outputsRemainTrainable':True,'hardwareValidated':False}
    args.destination.with_suffix('.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof))

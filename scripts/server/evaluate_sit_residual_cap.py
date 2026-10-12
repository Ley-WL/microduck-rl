"""Causal CPU probe of pointwise learned drift from the original actor.

Both actors see the same live feedback; only their difference is bounded. This
is a two-policy diagnostic, not a deployable/exported single actor or acceptance.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--anchor',type=Path,required=True);p.add_argument('--residual-cap-deg',type=float,required=True)
p.add_argument('--residual-group',choices=['all','legs','head'],default='all')
a,remaining=p.parse_known_args();assert 0<a.residual_cap_deg<=2
candidate=Path(remaining[remaining.index('--onnx')+1]);output=Path(remaining[remaining.index('--output')+1])
assert not (output/'summary.json').exists(),'Inspect existing result'
anchor_metadata=json.loads(a.anchor.with_suffix('.metadata.json').read_text())
candidate_metadata=json.loads(candidate.with_suffix('.metadata.json').read_text())
names=candidate_metadata['jointOrder']
mask=np.asarray([a.residual_group=='all' or (name.startswith(('left_','right_')) if a.residual_group=='legs' else name.startswith(('neck_','head_'))) for name in names],dtype=np.float32)
for name in ('jointOrder','homeRadians','actionEmaOldWeight','trainingTargetClipRadians',
             'simulation','skillPhysicsRevision','lastActionObservation'):
    assert anchor_metadata[name]==candidate_metadata[name],name
original=ort.InferenceSession
options=ort.SessionOptions();options.intra_op_num_threads=2
anchor=original(str(a.anchor),options,providers=['CPUExecutionProvider'])
stats={'frames':0,'clippedByJoint':np.zeros(14,dtype=int),'maxDeltaRad':0.}

class CappedSession:
    def __init__(self,*args,**kwargs):self.candidate=original(*args,**kwargs)
    def get_inputs(self):return self.candidate.get_inputs()
    def get_outputs(self):return self.candidate.get_outputs()
    def run(self,outputs,feed):
        value=self.candidate.run(outputs,feed)[0]
        reference=anchor.run(None,{anchor.get_inputs()[0].name:next(iter(feed.values()))})[0]
        delta=value-reference;bound=np.deg2rad(a.residual_cap_deg)
        stats['frames']+=len(value);stats['clippedByJoint']+=(np.abs(delta)>bound).sum(axis=0)
        stats['maxDeltaRad']=max(stats['maxDeltaRad'],float(np.abs(delta).max()))
        return [reference+np.clip(delta,-bound,bound)*mask]

ort.InferenceSession=CappedSession
sys.argv=[str(Path(__file__).resolve().parents[1]/'evaluate_skills_cpu.py'),*remaining]
try:runpy.run_path(sys.argv[0],run_name='__main__')
finally:ort.InferenceSession=original
report=json.loads((output/'summary.json').read_text());report['accepted']=False
report['diagnosticComposite']={'anchorSha256':hashlib.sha256(a.anchor.read_bytes()).hexdigest(),
    'candidateSha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
    'residualCapDeg':a.residual_cap_deg,'scope':'Pointwise residual cap, same live feedback. Not a standalone policy export.',
    'residualGroup':a.residual_group,
    'frames':stats['frames'],'clippedByJoint':stats['clippedByJoint'].tolist(),
    'maxUncappedResidualDeg':float(np.rad2deg(stats['maxDeltaRad']))}
report['policySha256']=None
(output/'summary.json').write_text(json.dumps(report,indent=2))

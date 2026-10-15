"""Pure inference/arithmetic replay, never updates or drives hardware."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort
import evaluate_walk_runtime_cpu as scan

p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True)
p.add_argument('--frames',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();before=hashlib.sha256(a.onnx.read_bytes()).hexdigest()
opts=ort.SessionOptions();opts.intra_op_num_threads=1
session=ort.InferenceSession(str(a.onnx),sess_options=opts,providers=['CPUExecutionProvider'])
rows=json.loads(a.frames.read_text());errors={k:[] for k in ['raw','filter','target','history']}
for i,r in enumerate(rows):
    obs=np.asarray(r['observation'],dtype=np.float32)
    raw=session.run(None,{session.get_inputs()[0].name:obs[None]})[0][0]
    errors['raw'].append(float(np.max(np.abs(raw-r['rawActionRadians']))))
    adjusted=np.asarray(r['rawActionRadians'],dtype=np.float32)*np.float32(r['voltageScale'])
    filtered=scan.action_step(np.asarray(r['emaPreviousRadians'],dtype=np.float32),adjusted,i>0)
    errors['filter'].append(float(np.max(np.abs(filtered-r['filteredActionRadians']))))
    target=scan.mechanical.encode_target(np.asarray(r['homeRadians'])+filtered)
    errors['target'].append(float(np.max(np.abs(np.degrees(target)-r['sentTargetDegrees']))))
    if i:errors['history'].append(float(np.max(np.abs(obs[34:48]-rows[i-1]['rawActionRadians']))))
assert max(errors['raw'])<2e-5 and max(errors['filter'])<2e-6 and max(errors['target'])<1e-4 and max(errors['history'])<2e-6,errors
assert before==hashlib.sha256(a.onnx.read_bytes()).hexdigest()
a.output.write_text(json.dumps(dict(frames=len(rows),policySha256=before,maxErrors={k:max(v) for k,v in errors.items()},
    hardwareAccess=False,modelUpdated=False,limits='observed first600 frames only; missing failure tail cannot be replayed',
    accepted=False),indent=2))

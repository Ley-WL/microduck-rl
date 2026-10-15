"""Frozen parameter-update magnitude probe, not runtime action scaling."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

R=Path(__file__).resolve().parents[2];I=R/'inputs';E=R/'evidence'
base=onnx.load(I/'Baseline.onnx');repair=onnx.load(I/'Surface.onnx')
a={v.name:numpy_helper.to_array(v) for v in base.graph.initializer}
b={v.name:numpy_helper.to_array(v) for v in repair.graph.initializer}
assert set(a)==set(b)
for name in a:
    assert a[name].shape==b[name].shape
    if not name.startswith('mlp.'):np.testing.assert_array_equal(a[name],b[name])
rows=np.random.default_rng(97331).normal(0,.4,(128,61)).astype(np.float32)
opts=ort.SessionOptions();opts.intra_op_num_threads=1
def outputs(path):
    s=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    return np.concatenate([s.run(None,{s.get_inputs()[0].name:x[None]})[0] for x in rows])
checks=[]
for fraction in (0.,.25,.5,.75,1.):
    graph=copy.deepcopy(repair)
    for index,tensor in enumerate(graph.graph.initializer):
        if tensor.name.startswith('mlp.'):
            values=a[tensor.name]+np.float32(fraction)*(b[tensor.name]-a[tensor.name])
            graph.graph.initializer[index].CopyFrom(numpy_helper.from_array(values,tensor.name))
    path=I/f'Fraction{fraction:g}.onnx';assert not path.exists()
    onnx.checker.check_model(graph);onnx.save(graph,path)
    actual=outputs(path);assert np.isfinite(actual).all()
    if fraction in (0.,1.):
        error=float(abs(actual-outputs(I/('Baseline.onnx' if fraction==0 else 'Surface.onnx'))).max())
        assert error<2e-5,error
    else:error=None
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    meta=json.loads((I/'Baseline.metadata.json').read_text())
    meta.update(policySha256=sha,hardwareValidated=False,artifactType='frozen-update-fraction-probe')
    meta['walkExecution']['policySha256']=sha
    meta['updateFractionProbe']=dict(fraction=fraction,accepted=False,optimizerSteps=0,
        runtimeActionScaleUnchanged=True,normalizerUnchanged=True)
    path.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2))
    checks.append(dict(fraction=fraction,finite=True,endpointMaxError=error,sha256=sha))
(E/'preparation-audit.json').write_text(json.dumps(dict(checks=checks,normalizerIdentical=True,
    baselineSha256=hashlib.sha256((I/'Baseline.onnx').read_bytes()).hexdigest(),
    repairSha256=hashlib.sha256((I/'Surface.onnx').read_bytes()).hexdigest(),
    optimizerSteps=0,hardwareAccess=False,accepted=False),indent=2))

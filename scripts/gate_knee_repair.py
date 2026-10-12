"""Diagnostic knee-head correction with a continuous angular-speed gate.

Preserve the source actor exactly below the gate. No sensor sample is dropped
or replaced. Thresholds are diagnostic hypotheses, not servo protection.
Same raw-action history / HOME / target clipping / EMA contract as the source.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
import onnxruntime as ort


def main(a):
    assert a.start<a.full and a.start>=0
    assert not a.output.exists(),'Preserve existing diagnostic'
    source=onnx.load(a.source);repair=onnx.load(a.repair)
    tensors=lambda model:{t.name:numpy_helper.to_array(t) for t in model.graph.initializer}
    old,new=tensors(source),tensors(repair)
    assert old.keys()==new.keys()
    for name in old:
        if name not in ('mlp.6.weight','mlp.6.bias'):np.testing.assert_array_equal(old[name],new[name])
    nonknees=[i for i in range(14) if i not in (3,12)]
    for name in ('mlp.6.weight','mlp.6.bias'):np.testing.assert_array_equal(old[name][nonknees],new[name][nonknees])
    graph=copy.deepcopy(source)
    tail=graph.graph.node[-1]
    assert tail.op_type=='Gemm' and list(tail.input[1:])==['mlp.6.weight','mlp.6.bias']
    output=tail.output[0];tail.output[0]='knee_base_output'
    hidden=tail.input[0]
    constants={'knee_delta_weight':new['mlp.6.weight']-old['mlp.6.weight'],
               'knee_delta_bias':new['mlp.6.bias']-old['mlp.6.bias'],
               'gyro_slice_start':np.array([0],dtype=np.int64),'gyro_slice_end':np.array([3],dtype=np.int64),
               'gyro_axes':np.array([1],dtype=np.int64),
               'gate_start':np.array(a.start,dtype=np.float32),'gate_width':np.array(a.full-a.start,dtype=np.float32),
               'gate_zero':np.array(0.,dtype=np.float32),'gate_one':np.array(1.,dtype=np.float32)}
    graph.graph.initializer.extend(numpy_helper.from_array(value,name) for name,value in constants.items())
    delta=copy.deepcopy(tail);delta.name='knee_delta_gemm';delta.input[:]=[hidden,'knee_delta_weight','knee_delta_bias'];delta.output[:]=['knee_delta_output']
    graph.graph.node.extend([
        delta,
        helper.make_node('Slice',[graph.graph.input[0].name,'gyro_slice_start','gyro_slice_end','gyro_axes'],['gate_gyro']),
        helper.make_node('Mul',['gate_gyro','gate_gyro'],['gate_gyro_sq']),
        helper.make_node('ReduceSum',['gate_gyro_sq','gyro_axes'],['gate_norm_sq'],keepdims=1),
        helper.make_node('Sqrt',['gate_norm_sq'],['gate_norm']),
        helper.make_node('Sub',['gate_norm','gate_start'],['gate_numerator']),
        helper.make_node('Div',['gate_numerator','gate_width'],['gate_fraction']),
        helper.make_node('Clip',['gate_fraction','gate_zero','gate_one'],['gate']),
        helper.make_node('Mul',['gate','knee_delta_output'],['gated_delta']),
        helper.make_node('Add',['knee_base_output','gated_delta'],[output])])
    onnx.checker.check_model(graph)
    a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(graph,a.output)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    sessions=[ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']) for path in (a.source,a.repair,a.output)]
    dataset=np.load(a.data)
    samples=np.concatenate([dataset['simulation'][::20],dataset['validation'][::20]])
    maximum=0.;exact=0
    for sample in samples:
        results=[s.run(None,{s.get_inputs()[0].name:sample[None]})[0] for s in sessions]
        gate=np.clip((np.linalg.norm(sample[:3])-a.start)/(a.full-a.start),0,1)
        expected=results[0]+gate*(results[1]-results[0])
        maximum=max(maximum,float(np.abs(expected-results[2]).max()))
        np.testing.assert_allclose(results[2],expected,atol=2e-6,rtol=1e-5)
        np.testing.assert_array_equal(results[0][:,nonknees],results[2][:,nonknees])
        if gate==0:np.testing.assert_array_equal(results[0],results[2]);exact+=1
    metadata=json.loads(a.source.with_suffix('.metadata.json').read_text())
    proof={'kind':'diagnostic-angular-gated-knee-head-repair','sourceSha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
       'repairSha256':hashlib.sha256(a.repair.read_bytes()).hexdigest(),'gyroNormStartRadS':a.start,'gyroNormFullRadS':a.full,
       'sameHiddenLayers':True,'nonKneeOutputRowsExactlyUnchanged':True,'sampleCount':len(samples),
       'exactSourceBelowGateSamples':exact,'maxNumericCheckDifference':maximum,'accepted':False,'hardwareValidated':False}
    metadata['deploymentProvenance']=proof;metadata['policySha256']=hashlib.sha256(a.output.read_bytes()).hexdigest()
    a.output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','repair','data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--start',type=float,required=True);p.add_argument('--full',type=float,required=True)
    main(p.parse_args())

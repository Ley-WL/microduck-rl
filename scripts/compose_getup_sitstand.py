"""Explicit cross-skill composition, preserving each baked normalizer.

Getup was trained with floor-only contacts and no command. Evaluate both
experts under the sit v3 physics; do not pretend they share training physics.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import compose,helper,numpy_helper
import onnxruntime as ort


def build(a):
    experts={};metadata={}
    for name,path in [('stand',a.getup),('sit',a.sitstand)]:
        meta=json.loads(path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['policySha256']
        model=onnx.load(path);onnx.checker.check_model(model)
        assert [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]==[1,61]
        assert [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]==[1,14]
        experts[name]=model;metadata[name]=meta
    assert metadata['stand']['skillKind']=='getup' and metadata['sit']['skillKind']=='sitstand'
    assert metadata['sit']['skillPhysicsRevision']=='sit-self-contact-bounded-target-v3'
    for key in ['jointOrder','homeRadians','robotSourceHashes','actuatorParamsSha256','simulation','identifiedCommandDelaySeconds',
                'actionEmaOldWeight','actionEmaNewWeight','lastActionObservation','controlHz',
                'actorObservationDim','actionDim','actionSemantics']:
        assert metadata['stand'][key]==metadata['sit'][key],f'Incompatible shared runtime contract: {key}'
    assert [(v.domain,v.version) for v in experts['stand'].opset_import]==[(v.domain,v.version) for v in experts['sit'].opset_import]
    inp=copy.deepcopy(experts['sit'].graph.input[0]);inp.name='obs'
    out=copy.deepcopy(experts['sit'].graph.output[0]);out.name='actions'
    initializers=[numpy_helper.from_array(np.array([1.]*48+[0.]*13,dtype=np.float32)[None],'getup_command_mask'),
                  numpy_helper.from_array(np.array([48],dtype=np.int64),'flag_index'),
                  numpy_helper.from_array(np.array(.5,dtype=np.float32),'flag_threshold')]
    nodes=[helper.make_node('Mul',['obs','getup_command_mask'],['getup_obs'])];info=[];outputs={}
    for name,model in experts.items():
        prefixed=compose.add_prefix(model,name+'/')
        nodes.append(helper.make_node('Identity',['getup_obs' if name=='stand' else 'obs'],[prefixed.graph.input[0].name]))
        nodes.extend(prefixed.graph.node);initializers.extend(prefixed.graph.initializer);info.extend(prefixed.graph.value_info)
        outputs[name]=prefixed.graph.output[0].name
    nodes.extend([helper.make_node('Gather',['obs','flag_index'],['flag'],axis=1),
                  helper.make_node('Greater',['flag','flag_threshold'],['sit_requested']),
                  helper.make_node('Where',['sit_requested',outputs['sit'],outputs['stand']],['actions'])])
    model=helper.make_model(helper.make_graph(nodes,'getup_as_stand_sitstand_composition',[inp],[out],
        initializer=initializers,value_info=info),opset_imports=list(experts['sit'].opset_import))
    model.ir_version=max(v.ir_version for v in experts.values());onnx.checker.check_model(model)
    assert not a.output.exists(),'Preserve earlier composed weights'
    a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,a.output)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    combined=ort.InferenceSession(str(a.output),sess_options=options,providers=['CPUExecutionProvider'])
    sessions={name:ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']) for name,path in [('stand',a.getup),('sit',a.sitstand)]}
    rng=np.random.default_rng(3007);error=0.
    for flag in (0.,1.):
        for _ in range(32):
            x=rng.normal(0,.3,(1,61)).astype(np.float32);x[:,5]=-1.;x[:,48:]=0.;x[:,48]=flag
            expert='sit' if flag else 'stand';mapped=x.copy()
            if expert=='stand':mapped[:,48:]=0.
            expected=sessions[expert].run(None,{sessions[expert].get_inputs()[0].name:mapped})[0]
            actual=combined.run(None,{'obs':x})[0]
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
            error=max(error,float(np.max(np.abs(actual-expected))))
    meta=copy.deepcopy(metadata['sit']);meta.update(policySha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        artifactType='cross-skill-command-routed-candidate',hardwareValidated=False)
    meta['construction']={'kind':'getup-as-standing-expert-with-sitstand-sitting-expert','commandIndex':48,
        'getupCommandsZeroed':True,'sharedRawPreviousAction':True,'sharedEma':True,'expertNormalizersBaked':True,
        'runtimeAndEvaluationContactContract':metadata['sit']['skillPhysicsRevision'],
        'experts':{name:{'policySha256':m['policySha256'],'task':m['task'],'trainingContactContract':m['skillPhysicsRevision']} for name,m in metadata.items()},
        'routingVerification':{'inputs':64,'maxAbsError':error},'accepted':False,
        'scope':'Standing expert has different training contacts; composition needs new closed-loop screening, not source qualification reuse.'}
    a.output.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'sha256':meta['policySha256'],'routing':meta['construction']['routingVerification']}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('getup','sitstand','output'):p.add_argument('--'+name,type=Path,required=True)
    build(p.parse_args())

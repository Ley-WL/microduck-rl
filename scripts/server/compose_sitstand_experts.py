"""Build one ONNX from complementary experts; validate identical physical contracts.

No observation/EMA/history change and no deployment. Both normalizers stay baked.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import compose, helper, numpy_helper


def build(stand_path, sit_path, output):
    experts=[]
    for name,path in [('stand',stand_path),('sit',sit_path)]:
        meta=json.loads(path.with_suffix('.metadata.json').read_text())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != meta['policySha256']:
            raise ValueError(f'{name} policy hash mismatch')
        model=onnx.load(path)
        onnx.checker.check_model(model)
        if len(model.graph.input)!=1 or len(model.graph.output)!=1 or model.functions:
            raise ValueError('Expected standalone one-input/output graphs without local functions')
        if [x.dim_value for x in model.graph.input[0].type.tensor_type.shape.dim]!=[1,61]:
            raise ValueError('Expected 1x61 observations')
        if [x.dim_value for x in model.graph.output[0].type.tensor_type.shape.dim]!=[1,14]:
            raise ValueError('Expected 1x14 raw actions')
        experts.append((name,path,meta,model))
    stand,sit=experts
    for key in ['task','jointOrder','homeRadians','simulation','robotSourceHashes',
        'actuatorParamsSha256','sitJointOverrides','skillPhysicsRevision',
        'trainingTargetClipRadians','actionEmaOldWeight','actionEmaNewWeight',
        'lastActionObservation','commandSemantics','nominalTrunkHeightM',
        'nominalSeatedTrunkHeightM','identifiedCommandDelaySeconds',
        'contactContractSourceSha256','selfCollisionAssemblyExcludes',
        'controlHz','actorObservationDim','actionDim','actionSemantics']:
        if stand[2].get(key)!=sit[2].get(key) or key not in stand[2]:
            raise ValueError(f'Incompatible physical/observation contract: {key}')
    if [(x.domain,x.version) for x in stand[3].opset_import] != [(x.domain,x.version) for x in sit[3].opset_import]:
        raise ValueError('Different opsets; explicit conversion required')
    inp=copy.deepcopy(stand[3].graph.input[0]);inp.name='obs'
    out=copy.deepcopy(stand[3].graph.output[0]);out.name='actions'
    nodes=[];initializers=[];value_info=[];branches={}
    for name,path,meta,model in experts:
        prefixed=compose.add_prefix(model,name+'/')
        nodes.append(helper.make_node('Identity',['obs'],[prefixed.graph.input[0].name],name=name+'/shared_obs'))
        nodes.extend(prefixed.graph.node)
        initializers.extend(prefixed.graph.initializer)
        value_info.extend(prefixed.graph.value_info)
        branches[name]=prefixed.graph.output[0].name
    initializers.extend([numpy_helper.from_array(np.array([48],dtype=np.int64),'posture_index'),
        numpy_helper.from_array(np.array(.5,dtype=np.float32),'posture_threshold')])
    nodes.extend([helper.make_node('Gather',['obs','posture_index'],['posture_flag'],axis=1),
        helper.make_node('Greater',['posture_flag','posture_threshold'],['is_sit']),
        helper.make_node('Where',['is_sit',branches['sit'],branches['stand']],['actions'])])
    model=helper.make_model(helper.make_graph(nodes,'command_routed_sitstand',[inp],[out],
        initializer=initializers,value_info=value_info),opset_imports=list(stand[3].opset_import))
    model.ir_version=max(stand[3].ir_version,sit[3].ir_version)
    onnx.checker.check_model(model)
    if output.exists():
        raise ValueError('Preserve existing composed artifact')
    output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,output)
    meta=copy.deepcopy(sit[2])
    meta['artifactType']='command-routed-policy'
    meta['policySha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    meta['hardwareValidated']=False
    meta['construction']={'kind':'command-routed-two-experts','commandIndex':48,
        'sitWhenGreaterThan':.5,'sharedRawPreviousAction':True,
        'sharedRuntimeEma':True,'bothExpertNormalizersBaked':True,
        'experts':{name:{'path':str(path),'policySha256':metadata['policySha256'],
            'skillEnvironmentConfigSha256':metadata.get('skillEnvironmentConfigSha256')}
            for name,path,metadata,_ in experts}}
    # Graph-level proof over both commands, nonzero previous actions and noisy states.
    import onnxruntime as ort
    merged=ort.InferenceSession(str(output),providers=['CPUExecutionProvider'])
    sessions={name:ort.InferenceSession(str(path),providers=['CPUExecutionProvider']) for name,path,_,_ in experts}
    rng=np.random.default_rng(1009);maximum_error=0.
    for flag in [0.,1.]:
        for _ in range(32):
            obs=rng.normal(0,.2,(1,61)).astype(np.float32);obs[:,48:]=0;obs[:,48]=flag
            obs[:,5]=-1.
            session=sessions['sit' if flag else 'stand']
            expected=session.run(None,{session.get_inputs()[0].name:obs})[0]
            actual=merged.run(None,{'obs':obs})[0]
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
            assert np.isfinite(actual).all()
            maximum_error=max(maximum_error,float(np.max(np.abs(actual-expected))))
    meta['construction']['routingVerification']={'inputs':64,'maxAbsActionError':maximum_error}
    output.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'policySha256':meta['policySha256'],'routingProof':meta['construction']['routingVerification']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stand',type=Path,required=True);p.add_argument('--sit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.stand,a.sit,a.output)

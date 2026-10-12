"""Routing changes policy outputs: prove branches and reject incompatible experts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper, TensorProto

spec=importlib.util.spec_from_file_location('compose_experts',Path(__file__).parents[1]/'scripts/server/compose_sitstand_experts.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def expert(path,bias):
    constants=[numpy_helper.from_array(np.array([v],dtype=np.int64),n) for n,v in [('start',6),('end',20),('axis',1)]]
    constants.append(numpy_helper.from_array(np.array(bias,dtype=np.float32),'bias'))
    graph=helper.make_graph([helper.make_node('Slice',['obs','start','end','axis'],['joints']),
        helper.make_node('Add',['joints','bias'],['actions'])], 'expert',
        [helper.make_tensor_value_info('obs',TensorProto.FLOAT,[1,61])],
        [helper.make_tensor_value_info('actions',TensorProto.FLOAT,[1,14])],constants)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);model.ir_version=8
    onnx.save(model,path)
    keys=['task','jointOrder','homeRadians','simulation','robotSourceHashes','actuatorParamsSha256',
        'sitJointOverrides','skillPhysicsRevision','trainingTargetClipRadians','actionEmaOldWeight',
        'actionEmaNewWeight','lastActionObservation','commandSemantics','nominalTrunkHeightM',
        'nominalSeatedTrunkHeightM','identifiedCommandDelaySeconds',
        'contactContractSourceSha256','selfCollisionAssemblyExcludes',
        'controlHz','actorObservationDim','actionDim','actionSemantics']
    meta={k:'same-physical-contract' for k in keys}
    meta.update(policySha256=hashlib.sha256(path.read_bytes()).hexdigest(),skillEnvironmentConfigSha256=str(bias))
    path.with_suffix('.metadata.json').write_text(json.dumps(meta))


def test_one_graph_routes_both_directions_without_changing_previous_actions(tmp_path):
    stand=tmp_path/'stand.onnx';sit=tmp_path/'sit.onnx';output=tmp_path/'composed.onnx'
    expert(stand,.1);expert(sit,.2)
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in [stand,sit]]
    module.build(stand,sit,output)
    session=ort.InferenceSession(str(output),providers=['CPUExecutionProvider'])
    obs=np.arange(61,dtype=np.float32).reshape(1,61)/100
    for flag,bias in [(0,.1),(1,.2),(0,.1),(1,.2)]:
        obs[:,48]=flag
        np.testing.assert_allclose(session.run(None,{'obs':obs})[0],obs[:,6:20]+bias,atol=1e-6)
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in [stand,sit]]==hashes
    meta=json.loads(output.with_suffix('.metadata.json').read_text())
    assert meta['construction']['routingVerification']['maxAbsActionError']==0
    assert meta['artifactType']=='command-routed-policy'
    assert meta['hardwareValidated'] is False
    with pytest.raises(ValueError,match='Preserve existing'):
        module.build(stand,sit,output)


@pytest.mark.parametrize('key',['homeRadians','simulation','trainingTargetClipRadians','actionEmaOldWeight','sitJointOverrides','robotSourceHashes'])
def test_mismatched_expert_contract_is_rejected(tmp_path,key):
    stand=tmp_path/'stand.onnx';sit=tmp_path/'sit.onnx';output=tmp_path/'composed.onnx'
    expert(stand,.1);expert(sit,.2)
    path=sit.with_suffix('.metadata.json');meta=json.loads(path.read_text());meta[key]='different'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='Incompatible'):
        module.build(stand,sit,output)
    assert not output.exists()


def test_replaced_source_policy_is_rejected(tmp_path):
    stand=tmp_path/'stand.onnx';sit=tmp_path/'sit.onnx'
    expert(stand,.1);expert(sit,.2)
    path=sit.with_suffix('.metadata.json');meta=json.loads(path.read_text());meta['policySha256']='wrong'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='hash mismatch'):
        module.build(stand,sit,tmp_path/'composed.onnx')

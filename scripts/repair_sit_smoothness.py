"""Diagnostic actor repair with simulation retention and real temporal loss.

No robot access. A recorded trajectory is not an independent physical test.
Retain the source ONNX graph/normalizer and EMA contract, change MLP only.
Closed-loop tests are required after fitting; replay improvement is not success.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from torch import nn


def main(a):
    torch.set_num_threads(4)
    torch.manual_seed(3019)
    graph = onnx.load(a.source)
    tensors = {t.name:numpy_helper.to_array(t).copy() for t in graph.graph.initializer}
    mean = torch.tensor(tensors['obs_normalizer._mean'])
    divisor = torch.tensor(tensors['onnx::Div_24'])
    actor = nn.Sequential(nn.Linear(61,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
                          nn.Linear(256,128),nn.ELU(),nn.Linear(128,14))
    actor.load_state_dict({k.removeprefix('mlp.'):torch.tensor(v) for k,v in tensors.items() if k.startswith('mlp.')})
    source_state = copy.deepcopy(actor.state_dict())
    knees_only=getattr(a,'knees_only',False)
    dimensions=[3,12] if knees_only else list(range(14))
    if knees_only:
        # Preserve all hidden features and the other twelve output rows.
        # This is a narrow trainable-head control, not an output EMA override.
        for parameter in actor.parameters():parameter.requires_grad_(False)
        final=actor[-1]
        final.weight.requires_grad_(True);final.bias.requires_grad_(True)
        mask=torch.zeros(14);mask[dimensions]=1.
        final.weight.register_hook(lambda gradient:gradient*mask[:,None])
        final.bias.register_hook(lambda gradient:gradient*mask)
    session = ort.InferenceSession(str(a.source),providers=['CPUExecutionProvider'])
    normalize = lambda x:(x-mean)/divisor
    files=[]
    if getattr(a,'retention_data',None):
        dataset=np.load(a.retention_data)
        sim=torch.tensor(dataset['simulation'],dtype=torch.float32)
        valid=torch.tensor(dataset['validation'],dtype=torch.float32)
        if not len(sim) or not len(valid) or sim.shape[1]!=61 or valid.shape[1]!=61:
            raise ValueError('Need distinct complete-rollout train/validation retention data')
    else:
        files = sorted(a.demonstrations.glob('*.policy.json'))
        if len(files) < 2:
            raise ValueError('Need separate simulation trajectories for retention validation')
        examples = [json.loads(p.read_text(encoding='utf-8')) for p in files]
        # Entire final rollout is held out, rather than adjacent frames.
        sim = torch.tensor(np.array([r['observation'] for rows in examples[:-1] for r in rows],dtype=np.float32))
        valid = torch.tensor(np.array([r['observation'] for r in examples[-1]],dtype=np.float32))
    trace = json.loads(a.trace.read_text(encoding='utf-8'))
    rows = [r for r in trace['frames'] if r['type']=='inference' and r['kind'] in ('sitstand_sit','sitstand_stand')]
    real = torch.tensor(np.array([r['observation'] for r in rows],dtype=np.float32))
    pairs = [i for i in range(1,len(rows)) if rows[i]['kind']==rows[i-1]['kind'] and 0 < rows[i]['mono']-rows[i-1]['mono'] < .04]
    ids = torch.tensor(pairs)
    with torch.no_grad():
        sim_y, valid_y, real_y = actor(normalize(sim)),actor(normalize(valid)),actor(normalize(real))
    reference = session.run(None,{session.get_inputs()[0].name:real[:1].numpy()})[0]
    np.testing.assert_allclose(real_y[:1].numpy(),reference,rtol=2e-5,atol=2e-5)
    # Retain all recorded outputs as an anchor. Temporal cost only compares
    # same-command fresh pairs and never bridges a command switch.
    a.output.mkdir(parents=True,exist_ok=True)
    report = {'sourceSha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
              'traceSha256':hashlib.sha256(a.trace.read_bytes()).hexdigest(),
              'simulationFiles':[str(p) for p in files], 'simulationTrainSamples':len(sim),
              'simulationHeldOutSamples':len(valid), 'realSamples':len(real),
              'stepsPerCandidate':a.steps, 'hardwareValidated':False,
              'gyroConsistencyWeight':getattr(a,'gyro_consistency_weight',0.),
              'gyroConsistencyAugmentationRadS':.2,
              'scope':'One physical trajectory used for fitting; replay is in-sample. Separate simulation rollout held out.',
              'candidates':[]}
    report['trainableOutputDimensions']=dimensions
    report['hiddenLayersFrozen']=knees_only
    if getattr(a,'retention_data',None):
        report['retentionDatasetSha256']=hashlib.sha256(a.retention_data.read_bytes()).hexdigest()
        report['scope']='Successful complete simulation trajectories; disjoint seed retained for validation. Real trajectory used for fitting is not independent physical validation.'
    for weight in a.weights:
        actor.load_state_dict(source_state)
        optimizer = torch.optim.Adam(actor.parameters(),lr=2e-5)
        for step in range(a.steps):
            batch = torch.randint(len(sim),(512,))
            anchor = (actor(normalize(sim[batch]))[:,dimensions]-sim_y[batch][:,dimensions]).square().mean()
            pred = actor(normalize(real))
            retention = (pred[:,dimensions]-real_y[:,dimensions]).square().mean()
            smooth = (pred[ids][:,dimensions]-pred[ids-1][:,dimensions]).square().mean()
            consistency=torch.zeros(())
            if getattr(a,'gyro_consistency_weight',0.)>0:
                jitter=real.clone()
                jitter[:,1]+=torch.empty(len(real)).uniform_(-.2,.2)
                # Local sensor-input robustness augmentation; it is not a
                # physical intervention. Retain central values, price slope.
                consistency=(actor(normalize(jitter))-pred.detach()).square().mean()
            loss = anchor*10+retention+weight*smooth+getattr(a,'gyro_consistency_weight',0.)*consistency
            optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(),1.)
            optimizer.step()
        with torch.no_grad():
            v=actor(normalize(valid));p=actor(normalize(real))
        if knees_only:
            for key,value in actor.state_dict().items():
                if key.startswith('6.'):
                    other=[i for i in range(14) if i not in dimensions]
                    torch.testing.assert_close(value[other],source_state[key][other],atol=0,rtol=0)
                else:torch.testing.assert_close(value,source_state[key],atol=0,rtol=0)
        tag = 'weight-'+str(weight).replace('.','p')
        path=a.output/(tag+'.onnx')
        if path.exists():
            raise ValueError('Preserve existing experiment: '+str(path))
        candidate=copy.deepcopy(graph)
        replacements={'mlp.'+k:v.detach().numpy() for k,v in actor.state_dict().items()}
        for tensor in candidate.graph.initializer:
            if tensor.name in replacements:
                tensor.CopyFrom(numpy_helper.from_array(replacements[tensor.name],tensor.name))
        onnx.checker.check_model(candidate);onnx.save(candidate,path)
        checked=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        check=checked.run(None,{checked.get_inputs()[0].name:real[-1:].numpy()})[0]
        np.testing.assert_allclose(p[-1:].numpy(),check,rtol=2e-5,atol=2e-5)
        proof={'weight':weight,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'simulationHeldOutActionRmseRad':float((v-valid_y).square().mean().sqrt()),
               'recordedActionRmseRad':float((p-real_y).square().mean().sqrt()),
               'recordedTemporalRmsRad':float((p[ids]-p[ids-1]).square().mean().sqrt()),
               'baselineRecordedTemporalRmsRad':float((real_y[ids]-real_y[ids-1]).square().mean().sqrt())}
        meta=json.loads(a.source.with_suffix('.metadata.json').read_text(encoding='utf-8'))
        meta['policySha256']=proof['sha256']
        meta['deploymentProvenance']={'kind':'diagnostic-temporal-regularized-actor',
                                     'sourceSha256':report['sourceSha256'],'hardwareValidated':False,
                                     'accepted':False,'trainingSteps':a.steps,'temporalWeight':weight}
        meta['deploymentProvenance']['gyroConsistencyWeight']=getattr(a,'gyro_consistency_weight',0.)
        meta['deploymentProvenance']['trainableOutputDimensions']=dimensions
        meta['deploymentProvenance']['hiddenLayersFrozen']=knees_only
        path.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
        report['candidates'].append(proof)
        print(json.dumps(proof),flush=True)
    (a.output/'fit-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','trace','output'):
        p.add_argument('--'+name,type=Path,required=True)
    data=p.add_mutually_exclusive_group(required=True)
    data.add_argument('--demonstrations',type=Path)
    data.add_argument('--retention-data',type=Path)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--weights',type=float,nargs='+',default=[.1,.3,1.])
    p.add_argument('--gyro-consistency-weight',type=float,default=0.,help='Diagnostic local pitch-gyro robustness augmentation; requires closed-loop screening')
    p.add_argument('--knees-only',action='store_true',help='Freeze hidden layers and twelve non-knee output rows; requires closed-loop screening')
    main(p.parse_args())

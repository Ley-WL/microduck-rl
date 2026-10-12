"""Frozen-input actor differences, not a skill score or a policy export.

Allows dynamic batching only on an in-memory diagnostic ONNX graph and checks
against the original fixed-batch session. Never writes a modified policy.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch

def session(path):
    options=ort.SessionOptions();options.intra_op_num_threads=2
    model=onnx.load(path)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param='diagnostic_batch'
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param='diagnostic_batch'
    return ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider'])

def predict(s,x):
    return np.concatenate([s.run(None,{s.get_inputs()[0].name:x[i:i+1024]})[0] for i in range(0,len(x),1024)])

def main(a):
    original=a.experiment/'evidence/boundary-gain-control'
    policies={'baseline':original/'baseline.onnx','smoke5':original/'BoundaryGain/smoke.onnx',
              'final100':original/'BoundaryGain/train.onnx'}
    sessions={k:session(p) for k,p in policies.items()}
    names=json.loads(policies['final100'].with_suffix('.metadata.json').read_text())['jointOrder']
    seed=torch.load(a.experiment/'seed.pt',map_location='cpu',weights_only=False)['actor_state_dict']
    checkpoints=list((a.experiment/'logs').rglob('model_99.pt'));assert len(checkpoints)==1
    state=torch.load(checkpoints[0],map_location='cpu',weights_only=False)
    actor=state['actor_state_dict']
    normalizer=all(torch.equal(actor[k],v) for k,v in seed.items() if k.startswith('obs_normalizer.'))
    assert normalizer and all(torch.isfinite(v).all() for v in actor.values())
    assert all(torch.isfinite(v).all() for v in state['critic_state_dict'].values())
    contracts={}
    before=json.loads(policies['baseline'].with_suffix('.metadata.json').read_text())
    after=json.loads(policies['final100'].with_suffix('.metadata.json').read_text())
    for k in before:
        if k not in ('task','policySha256','deploymentProvenance','skillEnvironmentConfigSha256'):
            contracts[k]=before[k]==after.get(k)
    assert all(contracts.values()),[k for k,v in contracts.items() if not v]
    result={'hardwareAccess':False,'policyModified':False,'normalizerExactlyPreserved':normalizer,
        'finiteActorCritic':True,'metadataContractChecks':contracts,
        'environmentConfigHashes':{k:json.loads(p.with_suffix('.metadata.json').read_text()).get('skillEnvironmentConfigSha256') for k,p in policies.items()},
        'sourceHashes':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in policies.items()},
        'scope':'Frozen observations from fitting/diagnostic cohorts. Not fresh closed-loop acceptance.', 'cohorts':{}}
    transfer=np.load(a.experiment/'transfer-data.npz')
    datasets={'fit-simulation':transfer['simulation'],'fit-real':transfer['real']}
    for condition in ('nominal','stress','gain-low','gain-high'):
        folder=a.retention/condition
        report=json.loads((folder/'summary.json').read_text())
        frames=[]
        for g in report['groups']:
            for r in g['results']:
                if r['success']:
                    rows=json.loads((folder/f"{g['scenario']}-{g['seed']}-{r['trial']}.policy.json").read_text())
                    frames.extend(row['observation'] for row in rows)
        datasets['baseline-success-'+condition]=np.asarray(frames,dtype=np.float32)
    for label,x in datasets.items():
        x=np.asarray(x,dtype=np.float32);assert x.shape[1]==61
        y={k:predict(s,x) for k,s in sessions.items()}
        for k,p in policies.items():
            fixed=ort.InferenceSession(str(p),providers=['CPUExecutionProvider'])
            check=np.concatenate([fixed.run(None,{fixed.get_inputs()[0].name:row[None]})[0] for row in x[:16]])
            np.testing.assert_allclose(y[k][:16],check,atol=2e-6,rtol=1e-5)
        stats={}
        for k in ('smoke5','final100'):
            d=y[k]-y['baseline'];deg=np.rad2deg(d)
            stats[k]={'meanSquareRad':float(np.mean(d*d)),
                'rmsDeg':float(np.sqrt(np.mean(deg*deg))),
                'p99AbsByJointDeg':dict(zip(names,np.percentile(np.abs(deg),99,axis=0).tolist())),
                'maxAbsDeg':float(np.abs(deg).max())}
        result['cohorts'][label]={'frames':len(x),'differences':stats}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({'normalizer':normalizer,
        'contracts':all(contracts.values()),'cohorts':{k:{p:round(v['rmsDeg'],4) for p,v in r['differences'].items()} for k,r in result['cohorts'].items()}}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--retention',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())

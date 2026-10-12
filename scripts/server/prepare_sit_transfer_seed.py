"""Reconstruct a PPO seed from the deployed temporal actor, keeping provenance."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
import torch


def main(a):
    if a.output.exists():raise ValueError('Preserve existing seed')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    tensors={x.name:numpy_helper.to_array(x).copy() for x in onnx.load(a.model).graph.initializer}
    actor=state['actor_state_dict']
    np.testing.assert_allclose(actor['obs_normalizer._mean'].numpy(),tensors['obs_normalizer._mean'],atol=1e-7)
    np.testing.assert_allclose(actor['obs_normalizer._std'].numpy()+.01,tensors['onnx::Div_24'],atol=1e-7)
    for name,value in tensors.items():
        if name.startswith('mlp.'):actor[name]=torch.tensor(value)
    actor['distribution.std_param'].fill_(.1)
    state['optimizer_state_dict']['state']={}
    for group in state['optimizer_state_dict']['param_groups']:group['lr']=1e-5
    state['iter']=0;state['infos']['env_state']['common_step_counter']=0
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(state,a.output)
    simulations=[]
    for path in sorted(a.simulation.glob('*.policy.json'))[:-1]:
        simulations.extend(r['observation'] for r in json.loads(path.read_text(encoding='utf-8')))
    trace=json.loads(a.trace.read_text(encoding='utf-8'))
    rows=[r for r in trace['frames'] if r['type']=='inference' and r['kind'] in ('sitstand_sit','sitstand_stand')]
    pairs=[i for i in range(1,len(rows)) if rows[i]['kind']==rows[i-1]['kind'] and 0<rows[i]['mono']-rows[i-1]['mono']<.04]
    np.savez_compressed(a.data,simulation=np.asarray(simulations,dtype=np.float32),
                        real=np.array([r['observation'] for r in rows],dtype=np.float32),pair_indexes=np.array(pairs,dtype=np.int64))
    payload={'actorSourceSha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
             'seedSha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
             'dataSha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
             'simulationSamples':len(simulations),'realSamples':len(rows),'pairs':len(pairs),
             'normalizerPreserved':True,'criticPreserved':True,'optimizerReset':True,'hardwareAccess':False}
    a.output.with_suffix('.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8');print(json.dumps(payload))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('checkpoint','model','output','simulation','trace','data'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())

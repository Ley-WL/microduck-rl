"""Read-only held-out imitation diagnostics; action fit is not rollout acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper


def predictions(path, observations):
    graph = onnx.load(str(path))
    assert [n.op_type for n in graph.graph.node] == [
        'Sub', 'Div', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm']
    weights = {p.name: torch.tensor(numpy_helper.to_array(p).copy())
               for p in graph.graph.initializer}
    denominator = weights[graph.graph.node[1].input[1]]
    result = []
    with torch.no_grad():
        for batch in torch.as_tensor(observations).split(2048):
            x = (batch - weights['obs_normalizer._mean']) / denominator
            for layer in (0, 2, 4, 6):
                x = torch.nn.functional.linear(x, weights[f'mlp.{layer}.weight'],
                                               weights[f'mlp.{layer}.bias'])
                if layer != 6:
                    x = torch.nn.functional.elu(x)
            result.append(x.numpy())
    result = np.concatenate(result)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    ids = np.linspace(0, len(observations)-1, 32, dtype=int)
    oracle = np.concatenate([session.run(None, {session.get_inputs()[0].name:
                            observations[i:i+1]})[0] for i in ids])
    error = float(np.max(np.abs(result[ids] - oracle)))
    assert error < 2e-5, error
    return result, error


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--datasets', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    report = dict(simulationOnly=True,rolloutAcceptance=False,datasets=[])
    for path in a.datasets:
        if path.suffix == '.npz':
            data = np.load(path)
            x = data['observation'].reshape(-1,1500,61).transpose(1,0,2)
            y = data['target'].reshape(-1,1500,14).transpose(1,0,2)
            mask = data['validation'].reshape(-1,1500)
            assert np.all(mask == mask[:,:1])
            heldout = mask[:,0]
        else:
            data = torch.load(path,map_location='cpu',weights_only=False)
            x,y = data['observation'].numpy(),data['target'].numpy()
            heldout = data['validationWorlds'].numpy()
        x,y = x[:,heldout],y[:,heldout]
        moving = np.linalg.norm(x[:,:,48:51],axis=-1) >= .02
        onset = moving.copy(); onset[1:] &= ~moving[:-1]
        startup = np.zeros_like(moving)
        for tick,world in zip(*np.nonzero(onset)):
            startup[tick:tick+100,world] = True
        masks = {'stop':~moving, 'startupFirst2s':startup & moving,
                 'sustained':moving & ~startup}
        straight = np.abs(x[:,:,50]) < .01
        masks.update({key+'Straight':value & straight for key,value in list(masks.items())})
        row = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   validationWorlds=int(heldout.sum()),models={})
        for label,policy in [('baseline',a.baseline),('candidate',a.candidate)]:
            pred,oracle_error = predictions(policy,x.reshape(-1,61))
            squared = np.square(pred-y.reshape(-1,14)).mean(-1)
            row['models'][label] = dict(sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                oracleError=oracle_error,phases={key:dict(frames=int(mask.sum()),
                rms=float(np.sqrt(squared[mask.flatten()].mean())) if mask.any() else None)
                for key,mask in masks.items()})
        report['datasets'].append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

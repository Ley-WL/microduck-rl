"""Record final normalizer preservation and evaluation revision separately."""
import argparse,hashlib,json
from pathlib import Path
import torch

root=Path(__file__).resolve().parents[2]
evidence=root/'evidence/train'
launch=json.loads((evidence/'launch.json').read_text())
p=argparse.ArgumentParser()
p.add_argument('--process-v2',action='store_true')
a=p.parse_args()
experiment='hd1910_sitstand_process_v2' if a.process_v2 else 'hd1910_sitstand_transfer_v1'
run='process1000' if a.process_v2 else 'retained-smooth1000'
checkpoints=list((root/'logs/rsl_rl'/experiment).glob('*'+run+'/model_999.pt'))
assert len(checkpoints)==1
source=torch.load(root/'seed.pt',map_location='cpu',weights_only=False)['actor_state_dict']
final=torch.load(checkpoints[0],map_location='cpu',weights_only=False)['actor_state_dict']
normalizers=[key for key in source if key.startswith('obs_normalizer.')]
assert normalizers
for key in normalizers:torch.testing.assert_close(source[key],final[key],rtol=0,atol=0)
core=['src/mjlab_microduck/tasks/mdp.py','src/mjlab_microduck/tasks/sit_transfer_ppo.py',
      'src/mjlab_microduck/tasks/microduck_hd1910_sit_transfer_env_cfg.py','scripts/server/run_sit_transfer.sh']
if a.process_v2:
    core=['src/mjlab_microduck/tasks/mdp.py','src/mjlab_microduck/tasks/sit_transfer_ppo.py',
          'src/mjlab_microduck/tasks/microduck_hd1910_sit_process_env_cfg.py',
          'src/mjlab_microduck/tasks/sit_process_cost.py','scripts/server/run_sit_process.sh']
hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in core}
assert all(hashes[name]==launch['sourceHashes'][name] for name in core)
evaluation=['scripts/evaluate_skills_cpu.py','scripts/server/evaluate_skills.py','scripts/server/finalize_sit_transfer.py']
payload={'normalizerExactlyPreservedThrough1000Iterations':True,'trainingSourcesUnchangedSinceLaunch':True,
         'trainingSourceHashes':hashes,'evaluationRevisionHashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in evaluation},
         'note':'Evaluation scripts refined after launch; these hashes record final checks, not the training implementation.',
         'hardwareValidated':False}
if a.process_v2:
    import numpy as np
    import onnx
    import onnxruntime as ort
    paths=[evidence/'policy.onnx',evidence/'candidate-999.onnx']
    tensors=[{t.name:onnx.numpy_helper.to_array(t) for t in onnx.load(path).graph.initializer} for path in paths]
    assert tensors[0].keys()==tensors[1].keys()
    for key in tensors[0]:np.testing.assert_array_equal(tensors[0][key],tensors[1][key])
    samples=np.load(root/'transfer-data.npz')['simulation'][:3000].astype(np.float32)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    sessions=[ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']) for path in paths]
    predictions=[np.concatenate([s.run(None,{s.get_inputs()[0].name:x[None]})[0] for x in samples]) for s in sessions]
    np.testing.assert_allclose(*predictions,rtol=0,atol=0)
    payload['equivalentFinalExports']={'paths':[str(p) for p in paths],
        'sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
        'allInitializersExactlyEqual':True,'sampleCount':len(samples),'maxOutputDifference':0.}
(evidence/'final-contract-check.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload))

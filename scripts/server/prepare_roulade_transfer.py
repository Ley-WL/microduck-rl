"""Recover deployed actor; initialize a fresh task-specific critic/Adam."""
import json, hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
import mjlab_microduck.tasks
from mjlab_microduck.tasks.getup_mechanical_contract import encode_target

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/roulade-transfer'
TASK='Mjlab-RouladeTransferControl-Flat-MicroDuck'
torch.manual_seed(62173)
cfg=load_env_cfg(TASK); cfg.scene.num_envs=64
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    obs,_=env.reset(seed=62173)
    assert obs['actor'].shape==(64,61)
    term=env.action_manager.get_term('joint_pos')
    targets=[];history=[]
    for tick in range(40):
        action=np.random.default_rng(63100+tick).normal(0,1.6,(64,14)).astype(np.float32)
        expected=encode_target(action.astype(np.float64)+term._home64.cpu().numpy())
        obs,_,done,timeout,_=env.step(torch.tensor(action,device=env.device))
        # Manager resets targets for completed worlds. Compare surviving worlds.
        keep=~(done|timeout).cpu().numpy()
        targets.append(float(np.max(np.abs(term.position_target.cpu().numpy()[keep]-expected[keep]))))
        history.append(float(np.max(np.abs(obs['actor'][keep,34:48].cpu().numpy()-action[keep]))))
    assert max(targets)<2e-6 and max(history)<2e-6
    rl=asdict(load_rl_cfg(TASK)); rl['upload_model']=False
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),rl,log_dir=None,device='cuda:0')
    actor=runner.alg.actor
    tensors={a.name:numpy_helper.to_array(a).copy() for a in onnx.load(ROOT/'inputs/roulade.onnx').graph.initializer}
    state=actor.state_dict()
    for name,value in tensors.items():
        if name.startswith('mlp.') or name=='obs_normalizer._mean':
            state[name]=torch.as_tensor(value,device=env.device)
    std=torch.tensor(tensors['onnx::Div_24'],device=env.device)-.01
    assert bool((std>=-1e-7).all()); std=std.clamp_min(0)
    state['obs_normalizer._std']=std; state['obs_normalizer._var']=std.square()
    # ONNX has no sample count. Explicit transfer assumption, not donor lineage.
    state['obs_normalizer.count'].fill_(1_000_000)
    state['distribution.std_param'].fill_(.08)
    actor.load_state_dict(state); actor.eval()
    rows=np.random.default_rng(62173).normal(0,.4,(128,61)).astype(np.float32)
    from tensordict import TensorDict
    with torch.no_grad(): actual=actor(TensorDict({'actor':torch.tensor(rows,device=env.device)},batch_size=[128])).cpu().numpy()
    session=ort.InferenceSession(str(ROOT/'inputs/roulade.onnx'),providers=['CPUExecutionProvider'])
    expected=np.concatenate([session.run(None,{session.get_inputs()[0].name:r[None]})[0] for r in rows])
    error=float(np.max(abs(actual-expected))); assert error<2e-5,error
    assert not runner.alg.optimizer.state
    env.common_step_counter=0;runner.current_learning_iteration=0
    folder=ROOT/'logs/rsl_rl/roulade_mechanical_transfer/roll_seed';folder.mkdir(parents=True,exist_ok=True)
    reused=(folder/'seed.pt').exists()
    if not reused:runner.save(str(folder/'seed.pt'))
    (OUT/'seed-audit.json').write_text(json.dumps(dict(actorOnnxError=error,actorNormalizerMeanStdRestored=True,
        normalizerCountSynthesized=1000000,critic='fresh-random-task-specific',optimizer='fresh-empty-Adam',
        targetError=max(targets),rawHistoryError=max(history),worlds=64,steps=40,
        actorDim=61,criticDim=int(obs['critic'].shape[-1]),explorationStd=.08,
        seedSha256=hashlib.sha256((folder/'seed.pt').read_bytes()).hexdigest(),reusedSeed=reused,
        contactSensorMaxMatch=256,hardwareAccess=False),indent=2))
finally:env.close()

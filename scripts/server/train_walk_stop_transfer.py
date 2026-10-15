"""One bounded transfer stage; baked actor normalizer frozen, critic learns."""
import argparse,json,hashlib,copy
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import onnxruntime as ort
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
import mjlab_microduck.tasks
from walk_transfer_task import make_cfg,rl_cfg,TASK

p=argparse.ArgumentParser();p.add_argument('--surface',action='store_true')
p.add_argument('--envs',type=int,required=True);p.add_argument('--iterations',type=int,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert (a.envs,a.iterations) in ([(64,5),(1024,50)] if TASK!='walk_stop_transfer' else [(64,5),(1024,100)])
actor_only=TASK in ('walk_sustained_transfer','walk_low_speed_stride','walk_low_speed_control','walk_low_speed_focused','walk_low_speed_focused_control','walk_low_speed_air_window','walk_low_speed_mean_tracking')
ROOT=Path(__file__).resolve().parents[2];a.output.mkdir(parents=True,exist_ok=False)
torch.manual_seed(68131);cfg=make_cfg(a.surface);cfg.scene.num_envs=a.envs
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    rl=asdict(rl_cfg());rl.update(upload_model=False,logger='tensorboard')
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),rl,log_dir=str(a.output/'logs'),device='cuda:0')
    runner.load(str(ROOT/'inputs/seed.pt'),load_cfg={'actor':True} if actor_only else None,map_location='cuda:0')
    if actor_only:
        assert not runner.alg.optimizer.state
        assert all(group['lr']==1e-5 for group in runner.alg.optimizer.param_groups)
    env.common_step_counter=0;runner.current_learning_iteration=0
    actor=runner.alg.actor
    normalizer={k:v.clone() for k,v in actor.state_dict().items() if k.startswith('obs_normalizer.')}
    # Retain the baked deployment distribution; only critic normalization learns.
    actor.update_normalization=lambda *args,**kwargs:None
    anchor_stats=[]
    if TASK!='walk_stop_transfer':
        anchor=copy.deepcopy(actor).eval()
        for parameter in anchor.parameters():parameter.requires_grad_(False)
        def retain(optimizer,args,kwargs):
            rows=runner.alg.storage.observations['actor'].flatten(0,1)
            sample=rows[torch.randint(len(rows),(512,),device=env.device)]
            td=TensorDict({'actor':sample},batch_size=[len(sample)])
            with torch.no_grad():target=anchor(td)
            loss=(actor(td)-target).square().mean()
            ((5. if actor_only else 10.)*loss).backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),runner.alg.max_grad_norm)
            anchor_stats.append(float(loss.detach()))
        hook=runner.alg.optimizer.register_step_pre_hook(retain)
    assert not runner.alg.optimizer.state
    runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
    if TASK!='walk_stop_transfer':
        hook.remove()
        assert len(anchor_stats)==a.iterations*8
        (a.output/'retention.json').write_text(json.dumps(dict(coefficient=5. if actor_only else 10.,
            loss=anchor_stats,source='input seed actor; on-policy samples',
            notHistoricalCheckpointResume=True)))
    assert all(torch.equal(actor.state_dict()[k],v) for k,v in normalizer.items()),'Actor normalizer drift'
    steps={int(v['step']) for v in runner.alg.optimizer.state.values()}
    assert steps=={a.iterations*8},steps
    checkpoint=a.output/'final.pt';runner.save(str(checkpoint))
    runner.export_policy_to_onnx(str(a.output),'policy.onnx')
    for model in (runner.alg.actor,runner.alg.critic):
        assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    rows=np.random.default_rng(68331).normal(0,.4,(128,61)).astype(np.float32)
    actor.eval()
    with torch.no_grad():expected=actor(TensorDict({'actor':torch.tensor(rows,device=env.device)},batch_size=[128])).cpu().numpy()
    session=ort.InferenceSession(str(a.output/'policy.onnx'),providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
    actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:r[None]})[0] for r in rows])
    error=float(abs(actual-expected).max());assert error<2e-5,error
    meta=json.loads((ROOT/'inputs/walk.metadata.json').read_text())
    sha=hashlib.sha256((a.output/'policy.onnx').read_bytes()).hexdigest()
    meta.update(policySha256=sha,artifactType='diagnostic-transfer-policy',hardwareValidated=False)
    meta['walkExecution']['policySha256']=sha
    meta['transferExperiment']=dict(surface=a.surface,seed=68131,worlds=a.envs,iterations=a.iterations,
        originalDonorResume=False,critic='fresh-initialized',actorNormalizerFrozen=True,accepted=False)
    (a.output/'policy.metadata.json').write_text(json.dumps(meta,indent=2))
    (a.output/'stage-audit.json').write_text(json.dumps(dict(seed=68131,surface=a.surface,worlds=a.envs,
        iterations=a.iterations,rolloutSteps=a.envs*24*a.iterations,adamSteps=sorted(steps),actorNormalizerUnchanged=True,
        finite=True,onnxTorchMaxError=error,policySha256=sha,hardwareAccess=False,accepted=False),indent=2))
finally:env.close()

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
p.add_argument('--output',type=Path,required=True)
p.add_argument('--noise-schedule',choices=['fixed','anneal','low'])
p.add_argument('--critic-only',action='store_true')
p.add_argument('--critic-checkpoint',type=Path)
p.add_argument('--resume-checkpoint',type=Path);a=p.parse_args()
assert not a.resume_checkpoint or (not a.critic_checkpoint and not a.critic_only and a.noise_schedule=='anneal' and TASK=='walk_motion_critic_context')
assert not (a.critic_only and a.noise_schedule)
assert a.noise_schedule!='low' or (TASK=='walk_continuous_coverage' and a.critic_checkpoint and not a.resume_checkpoint)
assert not a.critic_checkpoint or (TASK in ('walk_low_speed_mean_tracking','walk_mean_motion_tracking','walk_motion_critic_context','walk_continuous_coverage') and (a.critic_only or a.noise_schedule))
if a.noise_schedule or a.critic_only:
    assert TASK in ('walk_low_speed_mean_tracking','walk_mean_motion_tracking','walk_motion_critic_context','walk_continuous_coverage')
    assert (a.envs,a.iterations) in [(64,5),(1024,200)]
else:
    assert (a.envs,a.iterations) in ([(64,5),(1024,50)] if TASK!='walk_stop_transfer' else [(64,5),(1024,100)])
actor_only=TASK in ('walk_sustained_transfer','walk_low_speed_stride','walk_low_speed_control','walk_low_speed_focused','walk_low_speed_focused_control','walk_low_speed_air_window','walk_low_speed_mean_tracking','walk_mean_motion_tracking','walk_motion_critic_context','walk_continuous_coverage')
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
    critic_source=None
    if a.critic_checkpoint:
        critic_path=ROOT/a.critic_checkpoint
        before_load={k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
        runner.load(str(critic_path),load_cfg={'critic':True},map_location='cuda:0')
        assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in before_load.items())
        assert not runner.alg.optimizer.state
        critic_source=dict(sha256=hashlib.sha256(critic_path.read_bytes()).hexdigest(),
            actorUnchangedOnLoad=True,optimizerStatesLoaded=False,iterationLoaded=False)
        (a.output/'critic-load-audit.json').write_text(json.dumps(critic_source,indent=2))
    env.common_step_counter=0;runner.current_learning_iteration=0
    resume_source=None;initial_adam_steps=0;completed_iterations=0
    if a.resume_checkpoint:
        resume_path=ROOT/a.resume_checkpoint
        saved=torch.load(resume_path,map_location='cuda:0',weights_only=False)
        runner.load(str(resume_path),map_location='cuda:0')
        loaded_optimizer=runner.alg.optimizer.state_dict()
        assert loaded_optimizer['param_groups']==saved['optimizer_state_dict']['param_groups']
        for index,state in saved['optimizer_state_dict']['state'].items():
            for name,value in state.items():
                actual=loaded_optimizer['state'][index][name]
                assert torch.equal(value,actual) if torch.is_tensor(value) else value==actual
        for key,model in [('actor_state_dict',runner.alg.actor),('critic_state_dict',runner.alg.critic)]:
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in saved[key].items())
        steps_loaded={int(v['step']) for v in runner.alg.optimizer.state.values()}
        assert len(steps_loaded)==1
        initial_adam_steps=steps_loaded.pop();assert initial_adam_steps==1600
        completed_iterations=initial_adam_steps//8
        assert saved['iter']==completed_iterations-1 and env.common_step_counter==completed_iterations*24
        assert all(group['lr']==1e-5 for group in runner.alg.optimizer.param_groups)
        # The checkpoint records the last iteration index, not completed count.
        runner.current_learning_iteration=completed_iterations
        resume_source=dict(sha256=hashlib.sha256(resume_path.read_bytes()).hexdigest(),
            optimizerStepsLoaded=initial_adam_steps,completedIterations=completed_iterations,
            environmentStateRestored=False,commonStepCounter=env.common_step_counter)
    actor=runner.alg.actor
    frozen_actor=None
    if a.critic_only:
        frozen_actor={k:v.clone() for k,v in actor.state_dict().items()}
        for parameter in actor.parameters():parameter.requires_grad_(False)
        runner.alg.learning_rate=3e-4
        for group in runner.alg.optimizer.param_groups:group['lr']=3e-4
    normalizer={k:v.clone() for k,v in actor.state_dict().items() if k.startswith('obs_normalizer.')}
    # Retain the baked deployment distribution; only critic normalization learns.
    actor.update_normalization=lambda *args,**kwargs:None
    anchor_stats=[]
    if TASK!='walk_stop_transfer' and not a.critic_only:
        anchor=copy.deepcopy(actor).eval()
        if a.resume_checkpoint:
            anchor.load_state_dict(torch.load(ROOT/'inputs/seed.pt',map_location='cuda:0',weights_only=False)['actor_state_dict'])
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
    assert bool(runner.alg.optimizer.state)==bool(a.resume_checkpoint)
    noise=None
    if a.noise_schedule:
        from walk_noise_schedule import RolloutNoise
        assert rl['num_steps_per_env']==24 and abs(env.step_dt-.02)<1e-9
        if a.noise_schedule=='low':
            assert torch.allclose(actor.distribution.std_param,torch.full_like(actor.distribution.std_param,.005))
        noise=RolloutNoise(runner.alg,a.noise_schedule)
        noise.iteration=completed_iterations
    value_stats=[]
    if a.critic_only or a.critic_checkpoint or a.resume_checkpoint:
        original_update=runner.alg.update
        def value_update():
            storage=runner.alg.storage
            target=storage.returns.detach();prediction=storage.values.detach()
            residual=prediction-target;variance=target.var(unbiased=False)
            value_stats.append(dict(iteration=len(value_stats),
                meanBootstrappedTarget=float(target.mean()),meanPrediction=float(prediction.mean()),
                rmse=float(residual.square().mean().sqrt()),
                explainedVariance=float(1-residual.var(unbiased=False)/variance) if variance>1e-12 else None))
            return original_update()
        runner.alg.update=value_update
    runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
    if value_stats:
        assert len(value_stats)==a.iterations
        (a.output/'value-learning.json').write_text(json.dumps(value_stats,indent=2))
    if a.critic_only:
        assert all(torch.equal(actor.state_dict()[k],v) for k,v in frozen_actor.items())
        assert all(parameter not in runner.alg.optimizer.state for parameter in actor.parameters())
        assert len(value_stats)==a.iterations
        (a.output/'critic-warmup-audit.json').write_text(json.dumps(dict(actorExactlyUnchanged=True,
            actorOptimizerStates=0,criticLearningRate=3e-4,iterations=value_stats,
            limitation='Training targets are bootstrapped; independent frozen reward-tail validation still required.'),indent=2))
    if noise:
        assert len(noise.audit)==a.iterations and noise.calls==0
        (a.output/'noise-audit.json').write_text(json.dumps(dict(mode=a.noise_schedule,
            distributionFrozen=True,deterministicExport=True,iterations=noise.audit),indent=2))
    if TASK!='walk_stop_transfer' and not a.critic_only:
        hook.remove()
        assert len(anchor_stats)==a.iterations*8
        (a.output/'retention.json').write_text(json.dumps(dict(coefficient=5. if actor_only else 10.,
            loss=anchor_stats,source='input seed actor; on-policy samples',
            notHistoricalCheckpointResume=not bool(a.resume_checkpoint))))
    assert all(torch.equal(actor.state_dict()[k],v) for k,v in normalizer.items()),'Actor normalizer drift'
    steps={int(v['step']) for v in runner.alg.optimizer.state.values()}
    assert steps=={initial_adam_steps+a.iterations*8},steps
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
    meta['transferExperiment']=dict(task=TASK,surface=a.surface,seed=68131,worlds=a.envs,iterations=a.iterations,
        originalDonorResume=False,critic='resumed' if resume_source else ('pretrained' if critic_source else 'fresh-initialized'),actorNormalizerFrozen=True,
        criticInitializer=critic_source,resumeInitializer=resume_source,
        criticOnly=a.critic_only,accepted=False)
    (a.output/'policy.metadata.json').write_text(json.dumps(meta,indent=2))
    (a.output/'stage-audit.json').write_text(json.dumps(dict(task=TASK,seed=68131,surface=a.surface,worlds=a.envs,
        iterations=a.iterations,rolloutSteps=a.envs*24*a.iterations,adamSteps=sorted(steps),actorNormalizerUnchanged=True,
        finite=True,onnxTorchMaxError=error,policySha256=sha,hardwareAccess=False,
        criticOnly=a.critic_only,criticInitializer=critic_source,resumeInitializer=resume_source,
        actorExactlyUnchanged=bool(a.critic_only),accepted=False),indent=2))
finally:env.close()

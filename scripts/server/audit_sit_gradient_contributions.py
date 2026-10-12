"""Frozen live-rollout gradient audit; only copied Adam parameters may update.

Intercept the installed PPO clipping and step, retaining its actual minibatch
loss, return computation and auxiliary sampling. No learner or hardware writes.
"""
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from tensordict import TensorDict
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'evidence/gradient-contributions'


def vector(parameters, gradients=None):
    parameters = list(parameters)
    gradients = [p.grad for p in parameters] if gradients is None else gradients
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p, g in zip(parameters, gradients)])


def cosine(a, b):
    denominator = float(a.norm() * b.norm())
    return None if denominator < 1e-20 else float(torch.dot(a, b) / denominator)


def clipped(v, maximum):
    return v * min(1., maximum / (float(v.norm()) + 1e-6))


def copy_model_with_detached_caches(model):
    # Gaussian output caches can retain non-leaf tensors after PPO forward.
    # Diagnostic copies retain values, without copying their autograd graph.
    memo, seen = {}, set()
    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if torch.is_tensor(value):
            if not value.is_leaf:
                memo[id(value)] = value.detach().clone()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif hasattr(value, '__dict__') and not isinstance(value, type):
            visit(vars(value))
    visit(model)
    return copy.deepcopy(model, memo)


def shadow_update(alg, gradient, critic_gradient, observations):
    """Exact inherited Adam state on copies, with no auxiliary pre-hook."""
    actor, critic = copy_model_with_detached_caches(alg.actor), copy_model_with_detached_caches(alg.critic)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = type(alg.optimizer)(params, lr=alg.learning_rate)
    optimizer.load_state_dict(copy.deepcopy(alg.optimizer.state_dict()))
    for group in optimizer.param_groups:
        group['lr'] = alg.learning_rate
    offset = 0
    for p in actor.parameters():
        p.grad = gradient[offset:offset+p.numel()].reshape_as(p).clone()
        offset += p.numel()
    offset = 0
    for p in critic.parameters():
        p.grad = critic_gradient[offset:offset+p.numel()].reshape_as(p).clone()
        offset += p.numel()
    before = vector(actor.parameters(), list(actor.parameters()))
    with torch.no_grad():
        prediction_before = actor(observations).clone()
    optimizer.step()
    with torch.no_grad():
        delta = actor(observations) - prediction_before
    displacement = vector(actor.parameters(), list(actor.parameters())) - before
    return {'parameterDeltaNorm': float(displacement.norm()),
            'meanJointOutputDeltaDeg': torch.rad2deg(delta.mean(0)).cpu().tolist(),
            'rmsJointOutputDeltaDeg': torch.rad2deg(delta.square().mean(0).sqrt()).cpu().tolist()}, displacement


def run_seed(seed, checkpoint='seed.pt', label='Initial'):
    task = 'Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910'
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 64
    agent = load_rl_cfg(task)
    agent.num_steps_per_env = 512
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(task)(wrapped, asdict(agent), device=env.device)
    runner.load(str(ROOT/checkpoint), map_location=env.device)
    alg = runner.alg
    alg.train_mode()
    assert alg.symmetry is None and alg.rnd is None, 'Unaccounted extra actor objective'
    assert alg.entropy_coef == 0., 'Entropy needs separate attribution'
    actor_initial = {k:v.clone() for k,v in alg.actor.state_dict().items()}
    optimizer_initial = copy.deepcopy(alg.optimizer.state_dict())
    captures = []
    original_clip = torch.nn.utils.clip_grad_norm_
    actor_parameters = list(alg.actor.parameters())
    critic_parameters = list(alg.critic.parameters())
    pending = {}
    original_step = alg.optimizer.step
    try:
        obs, _ = wrapped.reset()
        with torch.no_grad():
            for _ in range(512):
                actions = alg.act(obs)
                obs, rewards, dones, extras = wrapped.step(actions)
                alg.process_env_step(obs, rewards, dones, extras)
            alg.compute_returns(obs)
        for k,v in actor_initial.items():
            torch.testing.assert_close(v, alg.actor.state_dict()[k], atol=0, rtol=0)
        critic_initial = {k:v.clone() for k,v in alg.critic.state_dict().items()}
        commands=alg.storage.observations['actor'][...,48].detach().cpu().numpy()
        np.savez_compressed(FOLDER/f'rollout-{label}-{seed}.npz',
            observations=alg.storage.observations['actor'].cpu().numpy(),
            actions=alg.storage.actions.cpu().numpy(),
            advantages=alg.storage.advantages.cpu().numpy(),
            returns=alg.storage.returns.cpu().numpy())
        # PPO itself calls this before optimizer.step. Capture pre/post values.
        def clip_hook(parameters, maximum, *args, **kwargs):
            parameters = list(parameters)
            is_actor = parameters[0] is actor_parameters[0]
            if is_actor:
                pending['ppoRaw'] = vector(parameters)
            result = original_clip(parameters, maximum, *args, **kwargs)
            if is_actor:
                pending['ppoClipped'] = vector(parameters)
            else:
                pending['criticClipped'] = vector(parameters)
            return result

        def frozen_step(*args, **kwargs):
            ids = torch.randint(len(alg.demonstrations), (512,), device=alg.device)
            x = alg.demonstrations[ids]
            all_current = alg.storage.observations['actor'].flatten(0, 1)
            current = all_current[torch.randint(len(all_current), (512,), device=alg.device)]
            with torch.no_grad():
                expected = alg.anchor(alg._td(x))
                expected_current = alg.anchor(alg._td(current))
            retain = (alg.actor(alg._td(x))-expected).square().mean()
            onpolicy = (alg.actor(alg._td(current))-expected_current).square().mean()
            prediction = alg.actor(alg._td(alg.real))
            temporal = (prediction[alg.pairs]-prediction[alg.pairs-1]).square().mean()
            terms = {'offlineRetention': alg.retention_coef*retain,
                     'onpolicyRetention': alg.onpolicy_retention_coef*onpolicy,
                     'temporal': alg.temporal_coef*temporal}
            gradients = {k:vector(actor_parameters, torch.autograd.grad(
                loss, actor_parameters, allow_unused=True)) for k,loss in terms.items()}
            raw, ppo = pending['ppoRaw'], pending['ppoClipped']
            combined = ppo + sum(gradients.values())
            final = clipped(combined, alg.max_grad_norm)
            row = {'ppoRawNorm': float(raw.norm()), 'ppoFirstClippedNorm': float(ppo.norm()),
                   'combinedPreSecondClipNorm': float(combined.norm()),
                   'combinedPostSecondClipNorm': float(final.norm()),
                   'terms': {k:{'weightedLoss':float(terms[k].detach()), 'norm':float(g.norm()),
                                'cosineWithPpo':cosine(g,ppo)} for k,g in gradients.items()},
                   'termPairCosines': {a+'__'+b:cosine(gradients[a],gradients[b])
                                       for a in gradients for b in gradients if a<b}}
            # One exact copied-optimizer step per seed, all gradients from same state.
            if not captures:
                probes = alg._td(current[:128])
                row['shadowPpoOnly'], displacement = shadow_update(alg,ppo,pending['criticClipped'],probes)
                row['shadowCombined'], combined_displacement = shadow_update(alg,final,pending['criticClipped'],probes)
                row['shadowStepCosine'] = cosine(displacement,combined_displacement)
            captures.append(row)
            # Do not call original optimizer.step or its auxiliary hook.
            return None

        alg._aux_hook.remove()
        alg.optimizer.step = frozen_step
        torch.nn.utils.clip_grad_norm_ = clip_hook
        alg.update()
        for model, state in ((alg.actor, actor_initial), (alg.critic, critic_initial)):
            for k,v in state.items():
                torch.testing.assert_close(v, model.state_dict()[k], atol=0, rtol=0)
        assert optimizer_initial.keys() == alg.optimizer.state_dict().keys()
        # Adam optimizer state stays untouched; every actual step was suppressed.
        for key,state in optimizer_initial['state'].items():
            for name,value in state.items():
                other=alg.optimizer.state_dict()['state'][key][name]
                if torch.is_tensor(value):torch.testing.assert_close(value,other,atol=0,rtol=0)
                else:assert value == other
        return {'seed':seed, 'checkpointLabel':label, 'checkpointSha256':hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest(), 'worlds':64, 'rolloutSteps':512, 'minibatches':captures,
                'sampledSitCommandFraction':float(np.mean(commands>.5)),
                'commandEdges':int(np.count_nonzero(np.diff(commands>.5,axis=0))),
                'actorIncludingNormalizerExactlyPreserved':True,
                'criticExactlyPreservedDuringGradientAudit':True,
                'optimizerStateExactlyPreserved':True,
                'criticNormalizerUpdatedNormallyDuringRollout':True}
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        alg.optimizer.step = original_step
        env.close()


if __name__ == '__main__':
    FOLDER.mkdir(parents=True, exist_ok=True)
    try:
        memory=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert memory < 200, 'GPU6 occupied; never interfere'
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '6'
        manifest=json.loads((FOLDER/'launch.json').read_text())
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        result={'hardwareAccess':False, 'training':False, 'formalAccepted':False,
                'scope':'WeakProcess task actual PPO returns/minibatches; fixed actor. Old EMA contract, not current kissqy platform. Copied Adam steps are diagnostic, not learned/deployable weights.',
                'runs':[]}
        for label in ('Main0','Main99'):
            for seed in (9017,12031):
                result['runs'].append(run_seed(seed,'checkpoints/'+label+'.pt',label))
                (FOLDER/'summary.json').write_text(json.dumps(result,indent=2))
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        result['sourceHashesExactlyPreserved']=True
        (FOLDER/'summary.json').write_text(json.dumps(result,indent=2))
        (FOLDER/'job.exit').write_text('0')
        print(json.dumps(result),flush=True)
    except BaseException:
        (FOLDER/'job.exit').write_text('1')
        raise

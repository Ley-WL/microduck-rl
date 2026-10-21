"""ONNX transfer, then checkpoint-continuous PPO with the published curriculum."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch
from flamingo_hd1910_trial import restore, save
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab_microduck.tasks.flamingo_hd1910_curriculum import make_cfg, rl_cfg


def run(args):
    assert 0 <= args.start < args.end <= 3000
    assert bool(args.resume) == (args.start > 0)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = make_cfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.envs
    rl = asdict(rl_cfg())
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        runner = MjlabOnPolicyRunner(
            RslRlVecEnvWrapper(env), rl, log_dir=str(out/'logs'), device=env.device
        )
        actor = runner.alg.actor
        if args.resume:
            runner.load(args.resume)
            assert runner.current_learning_iteration == args.start
            assert env.common_step_counter == args.start * rl['num_steps_per_env']
            source = Path(args.resume)
            error = None
        else:
            update_normalization = actor.update_normalization
            source = Path(args.policy)
            error = restore(actor, source, env.device)
            # The published initial exploration, not the previous trial's 0.08.
            with torch.no_grad():
                actor.distribution.std_param.fill_(rl['actor']['distribution_cfg']['init_std'])
            before = {k: v.clone() for k, v in actor.state_dict().items()}
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            runner.learn(10, init_at_random_ep_len=True)
            assert all(torch.equal(v, actor.state_dict()[k]) for k, v in before.items())
            for parameter in actor.parameters():
                parameter.requires_grad_(True)
            actor.update_normalization = update_normalization
            # Precalibration must not consume the curriculum's first ten iterations.
            runner.current_learning_iteration = 0
            env.common_step_counter = 0
            with torch.inference_mode():
                env.reset(seed=args.seed)
            runner.alg.learning_rate = rl['algorithm']['learning_rate']
            for group in runner.alg.optimizer.param_groups:
                group['lr'] = runner.alg.learning_rate
        save(out/'contract.json', dict(
            sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            resumed=bool(args.resume), start=args.start, end=args.end, worlds=args.envs,
            seed=args.seed, actorOnnxMaxError=error,
            ppo=rl['algorithm'], actor=rl['actor'],
            normalizer='restored mean/std with synthetic count 1000000; updates enabled after critic warmup',
            criticWarmup=10 if not args.resume else 0,
            curriculum='published hard stages, no collapsing or pacing changes',
            hardware=False,
        ))
        runner.learn(args.end-args.start, init_at_random_ep_len=True)
        assert env.common_step_counter == args.end * rl['num_steps_per_env']
        assert all(torch.isfinite(v).all() for v in actor.state_dict().values())
        # Save the number of completed updates, not the runner's zero-based last index.
        runner.current_learning_iteration = args.end
        runner.save(str(out/'final.pt'))
        runner.export_policy_to_onnx(str(out), 'policy.onnx')
        save(out/'complete.json', dict(
            completedUpdates=args.end, commonStepCounter=env.common_step_counter,
            normalizerCount=float(actor.obs_normalizer.count),
            learningRate=runner.alg.learning_rate, hardware=False,
        ))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--envs', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=211022)
    run(parser.parse_args())

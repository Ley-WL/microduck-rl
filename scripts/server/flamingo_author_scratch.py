"""Published CycleHard PPO from scratch; only HD1910/limits adaptation and logging."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'evidence'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def train(smoke=False):
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from mjlab_microduck.tasks.flamingo_hd1910_curriculum import make_cfg, rl_cfg

    seed = 211023
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = make_cfg()
    cfg.seed = seed
    cfg.scene.num_envs = 64 if smoke else 4096
    rl = asdict(rl_cfg())
    rounds = 5 if smoke else rl['max_iterations']
    name = 'smoke' if smoke else 'train'
    destination = OUT/name
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        runner = MicroduckOnPolicyRunner(RslRlVecEnvWrapper(env), rl,
                                         log_dir=str(destination/'logs'), device=env.device)
        assert float(runner.alg.actor.obs_normalizer.count) == 0
        assert float(runner.alg.critic.obs_normalizer.count) == 0
        assert not runner.alg.optimizer.state
        assert env.common_step_counter == 0
        save(destination/'contract.json', dict(
            fromScratch=True, onnxImported=False, checkpointLoaded=False, criticWarmup=0,
            syntheticNormalizerCount=False, initialActorNormalizerCount=0,
            initialCriticNormalizerCount=0, initialOptimizerStates=0,
            seed=seed, worlds=cfg.scene.num_envs, iterations=rounds,
            actor=rl['actor'], critic=rl['critic'], ppo=rl['algorithm'],
            runner='MicroduckOnPolicyRunner/VelocityOnPolicyRunner',
            curriculum={k:v.params for k,v in cfg.curriculum.items()}, hardware=False,
        ))
        original_save = runner.save
        evaluated = set()
        comparisons = {}
        baseline = json.loads((OUT/'baseline/results.json').read_text())

        def save_and_evaluate(path, *args, **kwargs):
            original_save(path, *args, **kwargs)
            completed = env.common_step_counter // rl['num_steps_per_env']
            save(OUT/'progress.json', dict(stage=name, completedUpdates=completed,
                                          target=rounds, hardware=False))
            # Public save_interval=100 uses zero-based iteration labels.
            if smoke or completed not in (501, 1001, 1501, 2001, 2501, 3000) or completed in evaluated:
                return
            model_dir = OUT/f'model{completed}'
            model_dir.mkdir(exist_ok=True)
            runner.export_policy_to_onnx(str(model_dir), 'policy.onnx')
            save(OUT/'progress.json', dict(stage=f'eval{completed}',
                                          completedUpdates=completed, hardware=False))
            with (OUT/f'eval{completed}.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT/'scripts/server/flamingo_hd1910_trial.py'),
                    'eval', '--policy', str(model_dir/'policy.onnx'), '--output', str(OUT/f'eval{completed}'),
                    '--envs', '16'], cwd=ROOT, env=os.environ.copy(), stdout=log,
                    stderr=subprocess.STDOUT, check=True)
            rows = json.loads((OUT/f'eval{completed}/results.json').read_text())
            for key in baseline:
                assert baseline[key]['initialQposQvelSha256'] == rows[key]['initialQposQvelSha256'], key
            comparisons[str(completed)] = {m:sum(r[m] for r in rows.values())
                for m in ('holdSuccess', 'survived', 'lowerSuccess')}
            save(OUT/'comparison.json', dict(
                baseline={m:sum(r[m] for r in baseline.values()) for m in ('holdSuccess','survived','lowerSuccess')},
                candidates=comparisons, totalCases=96, accepted=False,
            ))
            evaluated.add(completed)
            save(OUT/'progress.json', dict(stage=name, completedUpdates=completed,
                                          target=rounds, hardware=False))

        runner.save = save_and_evaluate
        # One continuous standard learning loop: no task switches or artificial resets.
        runner.learn(rounds, init_at_random_ep_len=True)
        assert env.common_step_counter == rounds * rl['num_steps_per_env']
        assert all(torch.isfinite(v).all() for v in runner.alg.actor.state_dict().values())
        runner.export_policy_to_onnx(str(destination), 'policy.onnx')
        save(destination/'complete.json', dict(completedUpdates=rounds,
            commonStepCounter=env.common_step_counter,
            normalizerCount=float(runner.alg.actor.obs_normalizer.count), hardware=False))
    finally:
        env.close()


def pipeline():
    environment = dict(os.environ, PYTHONPATH=str(ROOT/'src'),
        CUDA_VISIBLE_DEVICES=os.environ.get('FLAMINGO_GPU','1'), MUJOCO_GL='egl',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', WANDB_MODE='offline', PYTHONUNBUFFERED='1')
    environment.pop('MICRODUCK_WARM_START', None)
    try:
        for name in ('smoke', 'train'):
            save(OUT/'progress.json', dict(stage=name, hardware=False))
            with (OUT/f'{name}.log').open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), name], cwd=ROOT,
                    env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        save(OUT/'progress.json', dict(stage='complete', completedUpdates=3000,
                                      hardware=False, accepted=False, needsReview=True))
        (OUT/'job.exit').write_text('0', encoding='utf-8')
    except BaseException as error:
        save(OUT/'progress.json', dict(stage='failed', error=repr(error), hardware=False))
        (OUT/'job.exit').write_text('1', encoding='utf-8')
        raise


def launch():
    OUT.mkdir(exist_ok=True)
    with (OUT/'launch.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    gpu = os.environ.get('FLAMINGO_GPU', '1')
    free, util = map(int, subprocess.check_output(['nvidia-smi','-i',gpu,
        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'], text=True).strip().split(','))
    assert free > 32000 and util < 15, (free, util)
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src','scripts','tests') for p in (ROOT/folder).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts}
    save(OUT/'source-manifest.json', hashes)
    with (OUT/'main.log').open('w') as log:
        child = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'pipeline'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=child.pid,gpu=gpu,worlds=4096,iterations=3000,fromScratch=True,
                  startedAt=time.time(),root=str(ROOT),hardware=False)
    save(OUT/'launch.json',record)
    print(json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('launch','pipeline','smoke','train'))
    mode = parser.parse_args().mode
    if mode == 'launch': launch()
    elif mode == 'pipeline': pipeline()
    else: train(smoke=mode=='smoke')

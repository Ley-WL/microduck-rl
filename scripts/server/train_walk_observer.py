"""Train official walking from scratch with observer limits and HD1910."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence'
TASK = 'Mjlab-Velocity-Flat-MicroDuck'
RECIPE_BLOB = '45a1f6d5b7f5d81aa58dab777ed34f458a2f847c'


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2), encoding='utf-8')


def verify_recipe():
    data = (ROOT / 'src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py').read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == RECIPE_BLOB


def stage(name):
    import numpy as np
    import onnxruntime as ort
    import mjlab_microduck.tasks
    from mjlab.scripts.train import TrainConfig, run_train
    from mjlab_microduck.robot.hardware_profile import HD1910_PARAMS, provenance

    verify_recipe()
    cfg = TrainConfig.from_task(TASK)
    from mjlab_microduck.tasks.walk_observer_contract import apply_observer_contract
    from mjlab_microduck.tasks.getup_mechanical_contract import LIMITS
    apply_observer_contract(cfg.env)
    # Only resource budget and logging change; no task or PPO hyperparameters.
    cfg.env.scene.num_envs = 64 if name == 'smoke' else 4096
    cfg.agent.max_iterations = 5 if name == 'smoke' else 5000
    cfg.agent.upload_model = False
    cfg.agent.logger = 'tensorboard'
    assert not cfg.agent.resume
    assert cfg.agent.algorithm.learning_rate == .001
    assert cfg.agent.algorithm.schedule == 'adaptive'
    assert cfg.agent.algorithm.num_learning_epochs == 5
    assert cfg.agent.num_steps_per_env == 24
    assert cfg.agent.actor.distribution_cfg['init_std'] == 1.
    assert cfg.env.commands['twist'].rel_turn_in_place_envs == .15
    assert cfg.env.actions['joint_pos'].__class__.__name__ == 'JointPositionActionCfg'
    assert cfg.env.actions['joint_pos'].scale == 1.
    assert not any(key.startswith('turn_') for key in cfg.env.rewards)
    assert 'action_rate_weight' in cfg.env.curriculum
    assert 'standing_envs' in cfg.env.curriculum
    assert cfg.env.scene.entities['robot'].spec_fn.__name__ == 'get_observer_walk_spec'
    for actuator in cfg.env.scene.entities['robot'].articulation.actuators:
        assert Path(actuator.json_path).resolve() == HD1910_PARAMS.resolve()
        assert actuator.kp_fw == 5.
    save(name + '-preflight.json', dict(hardware=provenance(), task=TASK,
        recipeGitBlob=RECIPE_BLOB, fromScratch=True, worlds=cfg.env.scene.num_envs,
        iterations=cfg.agent.max_iterations, curriculum=list(cfg.env.curriculum),
        actionScale=1., actionFiltering=False, originalPpo=True,
        jointLimitsRadians=LIMITS, targetClipRadians=LIMITS,
        officialCommit='b02fdea4902c2b93b7d34a83a6f05da4ececb7d3'))
    save('progress.json', dict(stage=name, startedAt=time.time()))
    run_train(TASK, cfg, OUT / name)
    exports = sorted((OUT / name).rglob('*.onnx'))
    assert exports, 'Official runner did not export ONNX'
    for path in exports:
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        obs = session.get_inputs()[0]
        assert obs.shape[-1] == 61, obs.shape
        result = session.run(None, {obs.name: np.zeros((1, 61), dtype=np.float32)})[0]
        assert result.shape == (1, 14) and np.isfinite(result).all()
    save(name + '-passed.json', dict(exports=len(exports), finite=True,
        note='Export smoke check only; not a gait-quality acceptance test.'))


def pipeline():
    code = 1
    try:
        for name in ('smoke', 'main'):
            with (OUT / (name + '.log')).open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), name],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            log_text = (OUT / (name + '.log')).read_text()
            assert 'collision overflow' not in log_text and 'contact match overflow' not in log_text
        manifest = json.loads((OUT / 'source-manifest.json').read_text())
        assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in manifest.items())
        save('progress.json', dict(stage='evaluating', accepted=False))
        with (OUT / 'evaluation.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'scripts/server/eval_walk_observer.py')],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        save('progress.json', dict(stage='complete-evaluated', accepted=False))
        code = 0
    except BaseException as error:
        save('progress.json', dict(stage='failed', error=repr(error)))
        raise
    finally:
        (OUT / 'job.exit').write_text(str(code))


def launch():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'launch.json').exists(), 'Do not duplicate an existing launch'
    verify_recipe()
    free, util = map(int, subprocess.check_output(['nvidia-smi', '-i', '4',
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
    assert free > 30000 and util < 10, (free, util)
    files = [p for folder in ('src', 'scripts') for p in (ROOT / folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    save('source-manifest.json', {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    env = os.environ.copy()
    env.pop('MICRODUCK_WARM_START', None)
    env.update(PYTHONPATH=str(ROOT / 'src'), CUDA_VISIBLE_DEVICES='4', MUJOCO_GL='egl',
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', WANDB_MODE='offline')
    with (OUT / 'launcher.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'pipeline'],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    report = dict(pid=process.pid, root=str(ROOT), gpu=4, fromScratch=True,
        task=TASK, worlds=4096, iterations=5000, startedAt=time.time(),
        hardwareAccess=False, automaticDeployment=False)
    save('launch.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'pipeline', 'smoke', 'main'])
    mode = parser.parse_args().mode
    if mode == 'launch':
        launch()
    elif mode == 'pipeline':
        pipeline()
    else:
        stage(mode)

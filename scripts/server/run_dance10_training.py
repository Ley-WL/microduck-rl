"""One bounded isolated experiment: equilibrium, smoke/export, 1000 iterations."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence'
TASK = 'Mjlab-Dance10-Flat-MicroDuck-HD1910'
ENV = os.environ.copy()
ENV.update(PYTHONPATH=str(ROOT/'src'), CUDA_VISIBLE_DEVICES='1', MUJOCO_GL='egl',
           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', WANDB_MODE='offline', MICRODUCK_WARM_START='0')


def status(stage, **kw):
    (OUT/'progress.json').write_text(json.dumps(dict(stage=stage, time=time.time(), **kw), indent=2))


def gate():
    value = subprocess.check_output(['nvidia-smi', '--id=1',
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    free, util = map(int, value.strip().split(','))
    assert free > 12288 and util < 30, ('GPU1 busy; no training started', free, util)


def run(args, name):
    status(name)
    with (OUT/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=ENV,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    manifest=json.loads((ROOT/'source-manifest.json').read_text())
    for p,h in manifest.items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h, p
    gate()
    run(['scripts/server/dance_preflight.py'], 'preflight')
    for name, worlds, iterations in [('smoke', 64, 5), ('first1000', 2048, 1000)]:
        gate()
        run(['-m', 'mjlab_microduck.train_cli', TASK, '--env.scene.num-envs', worlds,
             '--agent.max-iterations', iterations, '--agent.run-name', name,
             '--agent.seed', 71421, '--agent.logger', 'tensorboard'], name)
        checkpoints = list(ROOT.glob('logs/**/model_*.pt'))
        checkpoint = sorted([p for p in checkpoints if name in str(p.parent)],
                            key=lambda p:int(p.stem.split('_')[-1]))[-1]
        run(['scripts/export.py', TASK, '--checkpoint-file', checkpoint,
             '--onnx-file', OUT/(name+'.onnx'), '--num-envs', 1, '--device', 'cuda:0'], name+'-export')
        import onnxruntime as ort
        import numpy as np
        import onnx
        session = ort.InferenceSession(str(OUT/(name+'.onnx')), providers=['CPUExecutionProvider'])
        model=onnx.load(str(OUT/(name+'.onnx')))
        assert any('normalizer' in x.name for x in model.graph.initializer)
        assert session.get_inputs()[0].shape[-1] == 61
        for row in np.random.default_rng(71422).normal(0,.1,(16,61)).astype('float32'):
            action=session.run(None,{session.get_inputs()[0].name:row[None]})[0]
            assert action.shape==(1,14) and np.isfinite(action).all()
        (OUT/(name+'-export-check.json')).write_text(json.dumps(dict(finite=True,
            normalizerIncluded=True, checkpoint=str(checkpoint), danceAccepted=False)))
    status('training-complete-awaiting-rollout-review', danceAccepted=False, deployment=False)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    try:
        main()
    except BaseException as error:
        status('failed', error=repr(error))
        (OUT/'job.exit').write_text('1\n')
        raise
    (OUT/'job.exit').write_text('0\n')

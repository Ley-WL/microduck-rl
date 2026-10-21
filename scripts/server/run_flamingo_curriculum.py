"""Bounded 3000-update transfer with staged, paired evaluations; never deploys."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'evidence'
GPU = os.environ.get('FLAMINGO_GPU', '1')
MILESTONES = (100, 500, 1000, 1500, 2000, 3000)


def save(name, value):
    path = OUT/name
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def pipeline():
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), CUDA_VISIBLE_DEVICES=GPU,
               MUJOCO_GL='egl', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               WANDB_MODE='offline', PYTHONUNBUFFERED='1')
    env.pop('MICRODUCK_WARM_START', None)

    def stage(name, script, arguments):
        save('progress.json', dict(stage=name, startedAt=time.time(), hardware=False))
        with (OUT/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/server'/script), *arguments],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        logtext = (OUT/(name+'.log')).read_text()
        for marker in ('contact buffer overflow', 'contact_sensor_maxmatch exceeded', 'NaN detected'):
            if marker.lower() in logtext.lower():
                raise RuntimeError(f'{name}: invalid run: {marker}')

    def train(name, start, end, worlds, resume=None):
        args = ['--policy', str(ROOT/'inputs/policy.onnx'), '--output', str(OUT/name),
                '--start', str(start), '--end', str(end), '--envs', str(worlds)]
        if resume:
            args += ['--resume', str(OUT/resume/'final.pt')]
        stage(name, 'train_flamingo_curriculum.py', args)

    try:
        train('smoke', 0, 5, 64)
        train('smoke_resume', 5, 7, 64, 'smoke')
        baseline = json.loads((OUT/'baseline/results.json').read_text())
        comparisons = {}
        previous = None
        completed = 0
        for target in MILESTONES:
            name = f'train{target}'
            train(name, completed, target, 4096, previous)
            evaluation = f'eval{target}'
            stage(evaluation, 'flamingo_hd1910_trial.py', [
                'eval', '--policy', str(OUT/name/'policy.onnx'),
                '--output', str(OUT/evaluation), '--envs', '16',
            ])
            results = json.loads((OUT/evaluation/'results.json').read_text())
            for key, row in baseline.items():
                assert row['initialQposQvelSha256'] == results[key]['initialQposQvelSha256'], key
            comparisons[str(target)] = {
                metric: sum(row[metric] for row in results.values())
                for metric in ('holdSuccess', 'survived', 'lowerSuccess')
            }
            save('comparison.json', dict(
                baseline={m: sum(r[m] for r in baseline.values()) for m in ('holdSuccess', 'survived', 'lowerSuccess')},
                candidates=comparisons, totalCases=96, accepted=False,
            ))
            completed, previous = target, name
        save('progress.json', dict(stage='complete', completedUpdates=completed,
                                   hardware=False, accepted=False, needsReview=True))
        (OUT/'job.exit').write_text('0', encoding='utf-8')
    except BaseException as error:
        save('progress.json', dict(stage='failed', error=repr(error), hardware=False))
        (OUT/'job.exit').write_text('1', encoding='utf-8')
        raise


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'pipeline':
        pipeline()
    else:
        # Atomic guard remains after failure: retries need explicit investigation.
        with (OUT/'launch.lock').open('x') as lock:
            lock.write(str(os.getpid()))
        free, util = map(int, subprocess.check_output([
            'nvidia-smi', '-i', GPU, '--query-gpu=memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip().split(','))
        assert free > 32000 and util < 15, (free, util)
        hashes = {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src', 'scripts', 'inputs', 'tests')
            for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
        }
        save('source-manifest.json', hashes)
        with (OUT/'main.log').open('w') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'pipeline'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
        report = dict(pid=child.pid, gpu=GPU, root=str(ROOT), startedAt=time.time(),
                      worlds=4096, iterations=3000, hardware=False)
        save('launch.json', report)
        print(json.dumps(report))

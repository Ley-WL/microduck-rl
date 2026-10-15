"""One finite, isolated speed/yaw/stride experiment on an idle permitted GPU."""
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence'; OUT.mkdir(exist_ok=True)
if '--worker' not in sys.argv:
    assert not (OUT / 'launch.json').exists(), 'Inspect existing job; do not restart'
    manifest = dict(hardwareAccess=False, autoDeployment=False, smoke=[64, 5], main=[1024, 50],
        task='walk_tracking_transfer', learningRate=3e-6, retentionCoefficient=10.,
        sourceHashes={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['src', 'scripts', 'inputs'] for p in (ROOT / folder).rglob('*')
            if p.is_file() and '__pycache__' not in str(p)})
    (OUT / 'launch.json').write_text(json.dumps(manifest, indent=2))
    with (OUT / 'job.log').open('w') as log:
        process = subprocess.Popen([sys.executable, __file__, '--worker'], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (OUT / 'pid').write_text(str(process.pid)); print(json.dumps({'pid': process.pid})); sys.exit()
env = os.environ.copy()
env.update(PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT / 'scripts'),
    MICRODUCK_WALK_SURFACE_INPUTS=str(ROOT / 'inputs'), MICRODUCK_WALK_TRANSFER_TASK='walk_tracking_transfer',
    CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', WANDB_MODE='offline')

def run(stage, script, *args):
    memory, usage = map(int, subprocess.check_output(['nvidia-smi', '--id=1',
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
    assert memory > 12288 and usage < 30, (stage, memory, usage)
    (OUT / 'progress.json').write_text(json.dumps(dict(stage=stage, updated=time.time(), gpu=1)))
    with (OUT / (stage + '.log')).open('w') as log:
        subprocess.run([sys.executable, 'scripts/server/' + script, *args], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (OUT / (stage + '.log')).read_text()
    assert 'contact match overflow' not in log and 'collision overflow' not in log

status = 1
try:
    run('tests', 'test_walk_tracking_transfer.py')
    run('prepare', 'prepare_walk_stop_transfer.py')
    shutil.copy2(OUT / 'seed.pt', ROOT / 'inputs/seed.pt')
    run('smoke', 'train_walk_stop_transfer.py', '--envs', '64', '--iterations', '5', '--output', 'evidence/smoke')
    run('main', 'train_walk_stop_transfer.py', '--envs', '1024', '--iterations', '50', '--output', 'evidence/main')
    run('screen', 'evaluate_walk_tracking_transfer.py')
    screen = json.loads((OUT / 'tracking-screen.json').read_text())
    if screen['accepted']:
        run('holdout', 'evaluate_walk_tracking_transfer.py', '--seeds', '191081', '191083', '--output', 'tracking-holdout.json')
    manifest = json.loads((OUT / 'launch.json').read_text())
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in manifest['sourceHashes'].items())
    (OUT / 'source-check.json').write_text(json.dumps({'passed': True}))
    (OUT / 'progress.json').write_text(json.dumps(dict(stage='complete', screenAccepted=screen['accepted'], deploymentAccepted=False)))
    status = 0
finally:
    (OUT / 'job.exit').write_text(str(status))

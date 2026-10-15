"""Independent frozen diagnostic, capacity guarded on authorized GPUs 0/1."""
import os,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'evidence';E.mkdir(exist_ok=True)
assert not (E/'launch.json').exists(),'Follow existing PID'
rows=[list(map(int,x.split(','))) for x in subprocess.check_output(['nvidia-smi','--id=0,1',
    '--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
ready=[x for x in rows if x[1]>12288 and x[2]<30];assert ready,'No spare capacity; no launch created'
gpu=max(ready,key=lambda x:x[1])[0]
manifest=dict(gpu=gpu,capacity=rows,training=False,hardwareAccess=False,
    sourceHashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src','scripts','inputs') for p in (ROOT/folder).rglob('*')
        if p.is_file() and '__pycache__' not in str(p)})
wrapper=ROOT/'run-frozen.py'
wrapper.write_text('''import subprocess,sys,json,hashlib,tarfile
from pathlib import Path
r=Path(__file__).resolve().parent;e=r/'evidence'
status=subprocess.run([sys.executable,'scripts/server/evaluate_walk_surface_frozen_gpu.py'],cwd=r).returncode
log=(e/'job.log').read_text(errors='replace')
if 'collision overflow' in log or 'contact match overflow' in log:status=1
m=json.loads((e/'launch.json').read_text())
changed=[p for p,h in m['sourceHashes'].items() if hashlib.sha256((r/p).read_bytes()).hexdigest()!=h]
if changed:status=1
(e/'source-check.json').write_text(json.dumps({'passed':not changed,'changed':changed}))
(e/'job.exit').write_text(str(status))
with tarfile.open(r/'complete.tar.gz','w:gz') as a:
    for name in ('src','scripts','inputs','evidence','run-frozen.py'):a.add(r/name,arcname=name)
(r/'complete.sha256').write_text(hashlib.sha256((r/'complete.tar.gz').read_bytes()).hexdigest())
sys.exit(status)
''')
(E/'launch.json').write_text(json.dumps(manifest,indent=2))
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),MICRODUCK_WALK_SURFACE_INPUTS=str(ROOT/'inputs'),
    CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline')
with (E/'job.log').open('wb') as stream:
    p=subprocess.Popen([os.sys.executable,str(wrapper)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,start_new_session=True)
(E/'pid').write_text(str(p.pid));print(json.dumps(dict(pid=p.pid,gpu=gpu,training=False)))

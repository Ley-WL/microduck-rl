"""Single bounded CPU actor repair with locked fitted data; no deployment."""
import os,json,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'evidence';E.mkdir(exist_ok=True)
assert not (E/'launch.json').exists(),'Track existing PID'
m=dict(training=True,kind='CPU-supervised-actor-only',seed=171337,optimizerStepsBudget=100,
    hardwareAccess=False,accepted=False,
    sourceHashes={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src','scripts','inputs') for p in (R/folder).rglob('*')
        if p.is_file() and '__pycache__' not in str(p)})
wrapper=R/'run-retention.py'
wrapper.write_text('''import sys,json,subprocess,hashlib,tarfile
from pathlib import Path
r=Path(__file__).resolve().parent;e=r/'evidence';m=json.loads((e/'launch.json').read_text())
s=subprocess.run([sys.executable,'scripts/server/train_walk_success_retention.py'],cwd=r).returncode
changed=[p for p,h in m['sourceHashes'].items() if hashlib.sha256((r/p).read_bytes()).hexdigest()!=h]
if changed:s=1
(e/'source-check.json').write_text(json.dumps({'passed':not changed,'changed':changed}))
(e/'job.exit').write_text(str(s))
with tarfile.open(r/'complete.tar.gz','w:gz') as a:
    for n in ('src','scripts','inputs','evidence','run-retention.py'):a.add(r/n,arcname=n)
(r/'complete.sha256').write_text(hashlib.sha256((r/'complete.tar.gz').read_bytes()).hexdigest())
sys.exit(s)
''')
(E/'launch.json').write_text(json.dumps(m,indent=2))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (E/'job.log').open('wb') as stream:
    p=subprocess.Popen([os.sys.executable,str(wrapper)],cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,start_new_session=True)
(E/'pid').write_text(str(p.pid));print(json.dumps(dict(pid=p.pid,training=True,kind=m['kind'])))

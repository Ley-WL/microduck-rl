"""Audit before learning; only authorized GPU0/1 with spare capacity."""
import json,hashlib,subprocess,os
from pathlib import Path
from dataclasses import asdict
import mjlab_microduck.tasks
from mjlab_microduck.tasks.walk_surface_transfer import make_cfg,rl_cfg

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence'
assert not (OUT/'prepare-launch.json').exists(),'Track the existing prepare PID'
a=asdict(make_cfg());b=asdict(make_cfg(True))
ta=a['scene'].pop('terrain');tb=b['scene'].pop('terrain')
def simple(v):
    if isinstance(v,dict):return {str(k):simple(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [simple(x) for x in v]
    if callable(v):return v.__module__+'.'+v.__qualname__
    if hasattr(v,'tolist'):return v.tolist()
    if isinstance(v,(str,float,int,bool)) or v is None:return v
    return str(v)
assert simple(a)==simple(b),'Unexpected non-terrain branch difference'
rl=rl_cfg();assert rl.algorithm.num_mini_batches==4 and rl.algorithm.num_learning_epochs==2
assert rl.num_steps_per_env==24 and rl.algorithm.learning_rate==1e-5 and rl.max_iterations==20
(OUT/'config-tests.json').write_text(json.dumps(dict(passed=6,branchDifference='terrain only',
    controlTerrain=simple(ta),repairTerrain=simple(tb),commonCfg=simple(a),rl=simple(asdict(rl)),
    originalDonorResume=False,hardwareAccess=False),indent=2))
rows=[list(map(int,x.split(','))) for x in subprocess.check_output(['nvidia-smi','--id=0,1',
    '--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
ready=[r for r in rows if r[1]>12288 and r[2]<30]
preferred=os.environ.get('MICRODUCK_WALK_AUDIT_GPU')
if preferred is not None:
    assert preferred in ('0','1'),'Only authorized GPU0/1'
    ready=[r for r in ready if r[0]==int(preferred)]
assert ready,'GPU capacity unavailable; no launch/PID created'
gpu=max(ready,key=lambda r:r[1])[0]
sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in ('src','scripts','inputs') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
wrapper=ROOT/'prepare-wrapper.py'
wrapper.write_text('''import os,subprocess,json,hashlib,sys
from pathlib import Path
r=Path(__file__).resolve().parent
s=subprocess.run([sys.executable,'scripts/server/prepare_walk_surface_transfer.py'],cwd=r).returncode
log=(r/'evidence/prepare.log').read_text(errors='replace')
if 'collision overflow' in log or 'contact match overflow' in log:
    print('Rejected collision-capacity overflow');s=1
if s==0:
    m=json.loads((r/'evidence/prepare-launch.json').read_text())
    changed=[p for p,h in m['sourceHashes'].items() if hashlib.sha256((r/p).read_bytes()).hexdigest()!=h]
    if changed:print('Source changed:',changed);s=1
    else:(r/'evidence/prepare-source-check.json').write_text(json.dumps({'passed':True,'files':len(m['sourceHashes'])}))
(r/'evidence/prepare.exit').write_text(str(s))
sys.exit(s)
''')
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES=str(gpu),
    MICRODUCK_WALK_SURFACE_INPUTS=str(ROOT/'inputs'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline')
manifest=dict(gpu=gpu,capacity=rows,training=False,hardwareAccess=False,sourceHashes=sources)
(OUT/'prepare-launch.json').write_text(json.dumps(manifest,indent=2))
with (OUT/'prepare.log').open('wb') as stream:
    process=subprocess.Popen([os.sys.executable,str(wrapper)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,start_new_session=True)
(OUT/'prepare.pid').write_text(str(process.pid));print(json.dumps(dict(pid=process.pid,gpu=gpu,training=False,hardwareAccess=False)))

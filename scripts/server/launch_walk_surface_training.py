"""Matched smoke/main stages; no automatic deployment or hardware access."""
import os,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
assert not (OUT/'launch.json').exists(),'Follow existing PID'
prepare=json.loads((ROOT/'inputs/prepare-audit.json').read_text())
assert prepare['learningPerformed'] is False
assert hashlib.sha256((ROOT/'inputs/seed.pt').read_bytes()).hexdigest()==prepare['seedSha256']
assert json.loads((ROOT/'inputs/prepare-source-check.json').read_text())['passed']
files=[p for folder in ('src','scripts','inputs') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
roots=os.environ.get('MICRODUCK_WALK_TRAIN_ROOTS','Control,Surface').split(',')
assert roots in (['Control','Surface'],['Control'],['Surface'])
manifest=dict(training=True,hardwareAccess=False,autoDeployment=False,seed=68131,
    roots=roots,smoke=[64,5],main=[1024,20],onlyBranchDifference='terrain generator',
    actorPositionQuantized4096=bool(prepare.get('actorPositionQuantized4096',False)),
    actorNormalizerFrozen=True,critic='fresh-for-transfer, not original resume',
    sourceHashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
wrapper=ROOT/'run-training.py'
wrapper.write_text('''import os,sys,json,subprocess,time,hashlib,tarfile
from pathlib import Path
r=Path(__file__).resolve().parent;e=r/'evidence';manifest=json.loads((e/'launch.json').read_text())
env=os.environ.copy();env.update(PYTHONPATH=str(r/'src'),MICRODUCK_WALK_SURFACE_INPUTS=str(r/'inputs'),
    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline')
def capacity(stage):
    while True:
        rows=[list(map(int,x.split(','))) for x in subprocess.check_output(['nvidia-smi','--id=0,1',
            '--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
        ready=[x for x in rows if x[1]>12288 and x[2]<30]
        preferred=os.environ.get('MICRODUCK_WALK_TRAIN_GPU')
        if preferred is not None:
            assert preferred in ('0','1')
            ready=[x for x in ready if x[0]==int(preferred)]
        if ready:
            gpu=max(ready,key=lambda x:x[1])[0];env['CUDA_VISIBLE_DEVICES']=str(gpu)
            (e/'progress.json').write_text(json.dumps({'stage':stage,'gpu':gpu,'capacity':rows}));return
        (e/'progress.json').write_text(json.dumps({'stage':'waiting-capacity','next':stage,'capacity':rows}));time.sleep(20)
status=1
try:
    for stage,worlds,iters in [('smoke',64,5),('main',1024,20)]:
        for label in manifest['roots']:
            surface=label=='Surface'
            name=stage+'-'+label;capacity(name)
            args=[sys.executable,'scripts/server/train_walk_surface_transfer.py','--envs',str(worlds),
                '--iterations',str(iters),'--output','evidence/'+name]
            if surface:args+=['--surface']
            with (e/(name+'.log')).open('w') as stream:subprocess.run(args,cwd=r,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            text=(e/(name+'.log')).read_text()
            assert 'contact match overflow' not in text and 'collision overflow' not in text,'Invalid contact capacity'
    changed=[p for p,h in manifest['sourceHashes'].items() if hashlib.sha256((r/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    (e/'source-check.json').write_text(json.dumps({'passed':True,'files':len(manifest['sourceHashes'])}))
    (e/'progress.json').write_text(json.dumps({'stage':'training-complete-awaiting-frozen-paired-screen','accepted':False}))
    status=0
finally:
    (e/'job.exit').write_text(str(status))
    with tarfile.open(r/'complete.tar.gz','w:gz') as a:
        for name in ('src','scripts','inputs','evidence','run-training.py'):a.add(r/name,arcname=name)
    (r/'complete.sha256').write_text(hashlib.sha256((r/'complete.tar.gz').read_bytes()).hexdigest())
sys.exit(status)
''')
(OUT/'launch.json').write_text(json.dumps(manifest,indent=2))
with (OUT/'job.log').open('wb') as stream:
    process=subprocess.Popen([os.sys.executable,str(wrapper)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,start_new_session=True)
(OUT/'pid').write_text(str(process.pid));print(json.dumps(dict(pid=process.pid,training=True,hardwareAccess=False,seed=68131)))

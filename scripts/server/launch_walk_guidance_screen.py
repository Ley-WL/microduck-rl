"""Frozen teacher feasibility only; no optimizer or hardware access."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
if '--worker' not in sys.argv:
    assert not (OUT/'launch.json').exists()
    files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['src','scripts','inputs'] for p in (ROOT/folder).rglob('*')
        if p.is_file() and '__pycache__' not in str(p)}
    (OUT/'launch.json').write_text(json.dumps(dict(training=False,hardwareAccess=False,sourceHashes=files),indent=2))
    with (OUT/'job.log').open('w') as log:
        process=subprocess.Popen([sys.executable,__file__,'--worker'],cwd=ROOT,stdout=log,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (OUT/'pid').write_text(str(process.pid));print(json.dumps({'pid':process.pid}));sys.exit()
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'),
    MICRODUCK_WALK_SURFACE_INPUTS=str(ROOT/'inputs'),CUDA_VISIBLE_DEVICES='0',
    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline')
status=1
try:
    cache_args=[]
    if (ROOT/'inputs/baseline-cache.json').exists():
        provenance=json.loads((ROOT/'inputs/baseline-provenance.json').read_text())
        for name,digest in provenance['sourceHashes'].items():
            if name.startswith(('src/','scripts/contracts/','inputs/walk','inputs/seed.pt')) or name=='scripts/kissqy_walk_physics.py':
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        cache_args=['--baseline-cache','inputs/baseline-cache.json']
    for stage,args in [('tests',['scripts/server/test_walk_guidance_map.py']),
        ('screen',['scripts/server/evaluate_walk_tracking_transfer.py','--candidate','inputs/seed.pt',
            '--teacher-map','inputs/guidance-map.json','--seeds','191111','191117','--output','guidance-screen.json',*cache_args])]:
        memory,usage=map(int,subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        assert memory>12288 and usage<30,(memory,usage)
        (OUT/'progress.json').write_text(json.dumps(dict(stage=stage,gpu=0,training=False,updated=time.time())))
        with (OUT/(stage+'.log')).open('w') as log:subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(OUT/(stage+'.log')).read_text();assert 'contact match overflow' not in text and 'collision overflow' not in text
    original=json.loads((OUT/'launch.json').read_text());assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in original['sourceHashes'].items())
    (OUT/'source-check.json').write_text(json.dumps({'passed':True}))
    result=json.loads((OUT/'guidance-screen.json').read_text())
    (OUT/'progress.json').write_text(json.dumps(dict(stage='complete',screenAccepted=result['accepted'],deploymentAccepted=False)))
    status=0
finally:(OUT/'job.exit').write_text(str(status))

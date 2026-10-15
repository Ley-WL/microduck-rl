"""Isolated experimental student learning, no command mapping at deployment."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
if '--worker' not in sys.argv:
    assert not (OUT/'launch.json').exists()
    prepare=json.loads((ROOT/'inputs/prepare-audit.json').read_text())
    assert hashlib.sha256((ROOT/'inputs/seed.pt').read_bytes()).hexdigest()==prepare['seedSha256']
    reference=json.loads((ROOT/'inputs/reference-screen.json').read_text())
    assert all(all(row['survival']) for row in reference['results'] if row['policy']=='candidate')
    assert reference['teacherMapSha256']==hashlib.sha256((ROOT/'inputs/guidance-map.json').read_bytes()).hexdigest()
    manifest=dict(training=True,hardwareAccess=False,teacherMapUsedAtRuntime=False,
        sourceHashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['src','scripts','inputs'] for p in (ROOT/folder).rglob('*')
            if p.is_file() and '__pycache__' not in str(p)})
    (OUT/'launch.json').write_text(json.dumps(manifest,indent=2))
    with (OUT/'job.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,__file__,'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    (OUT/'pid').write_text(str(proc.pid));print(json.dumps({'pid':proc.pid}));sys.exit()
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'),
    MICRODUCK_WALK_SURFACE_INPUTS=str(ROOT/'inputs'),CUDA_VISIBLE_DEVICES='0',
    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline')
status=1
try:
    for stage,script in [('tests','test_walk_guidance_map.py'),('student','train_walk_guided_student.py')]:
        memory,usage=map(int,subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        assert memory>12288 and usage<30,(memory,usage)
        (OUT/'progress.json').write_text(json.dumps(dict(stage=stage,updated=time.time(),gpu=0)))
        with (OUT/(stage+'.log')).open('w') as log:subprocess.run([sys.executable,'scripts/server/'+script],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(OUT/(stage+'.log')).read_text();assert 'contact match overflow' not in text and 'collision overflow' not in text
    manifest=json.loads((OUT/'launch.json').read_text());assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest['sourceHashes'].items())
    (OUT/'source-check.json').write_text(json.dumps({'passed':True}))
    (OUT/'progress.json').write_text(json.dumps(dict(stage='complete',deploymentAccepted=False)))
    status=0
finally:(OUT/'job.exit').write_text(str(status))

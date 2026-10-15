"""Bounded sustained-tracking refinement from the audited v6 student."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
if '--worker' not in sys.argv:
    assert not (OUT/'launch.json').exists(),'Inspect existing handle; do not restart'
    prepare=json.loads((ROOT/'inputs/prepare-audit.json').read_text())
    assert hashlib.sha256((ROOT/'inputs/seed.pt').read_bytes()).hexdigest()==prepare['seedSha256']
    assert prepare['actorSource']=='guided-student-v6-step3000'
    manifest=dict(hardwareAccess=False,autoDeployment=False,stageIterations=150,maxIterations=600,
        actorSource=prepare['actorSource'],
        sourceHashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['src','scripts','inputs'] for p in (ROOT/folder).rglob('*')
            if p.is_file() and '__pycache__' not in str(p)})
    (OUT/'launch.json').write_text(json.dumps(manifest,indent=2))
    with (OUT/'job.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,__file__,'--worker'],cwd=ROOT,stdout=log,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (OUT/'pid').write_text(str(proc.pid));print(json.dumps({'pid':proc.pid}));sys.exit()
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'),
    MICRODUCK_WALK_SURFACE_INPUTS=str(ROOT/'inputs'),MICRODUCK_WALK_TRANSFER_TASK='walk_sustained_transfer',
    CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline')
status=1
try:
    for stage,script,args in [('friction-test','test_bam_friction_parity.py',[]),
        ('reward-test','test_walk_sustained_transfer.py',[]),
        ('smoke','train_walk_stop_transfer.py',['--envs','64','--iterations','5','--output','evidence/smoke']),
        ('training','train_walk_progressive.py',['--output','evidence/sustained'])]:
        memory,usage=map(int,subprocess.check_output(['nvidia-smi','--id=0',
            '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        assert memory>12288 and usage<30,(memory,usage)
        (OUT/'progress.json').write_text(json.dumps(dict(stage=stage,updated=time.time(),gpu=0)))
        with (OUT/(stage+'.log')).open('w') as log:
            subprocess.run([sys.executable,'scripts/server/'+script,*args],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        logtext=(OUT/(stage+'.log')).read_text()
        assert 'contact match overflow' not in logtext and 'collision overflow' not in logtext
    original=json.loads((OUT/'launch.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in original['sourceHashes'].items())
    (OUT/'source-check.json').write_text(json.dumps({'passed':True}))
    (OUT/'progress.json').write_text(json.dumps(dict(stage='complete',deploymentAccepted=False)))
    status=0
finally:(OUT/'job.exit').write_text(str(status))

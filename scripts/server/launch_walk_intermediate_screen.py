"""Screen already-trained checkpoints; no optimizer steps or hardware access."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
if '--worker' not in sys.argv:
    assert not (OUT/'launch.json').exists(),'Inspect existing job; do not restart'
    spec=json.loads((ROOT/'inputs/intermediate-spec.json').read_text())
    for name,sha in spec['checkpointHashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    (OUT/'launch.json').write_text(json.dumps(dict(hardwareAccess=False,autoDeployment=False,
        optimizationSteps=0,sourceHashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src','scripts','inputs') for p in (ROOT/folder).rglob('*')
            if p.is_file() and '__pycache__' not in str(p)}),indent=2))
    with (OUT/'job.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,__file__,'--worker'],cwd=ROOT,stdout=log,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (OUT/'pid').write_text(str(proc.pid));print(json.dumps(dict(pid=proc.pid)));sys.exit()
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'),
    MICRODUCK_WALK_SURFACE_INPUTS=str(ROOT/'inputs'),CUDA_VISIBLE_DEVICES='0',
    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline')
status=1
try:
    spec=json.loads((ROOT/'inputs/intermediate-spec.json').read_text())
    stages=[]
    for iteration in spec['iterations']:
        checkpoint=f'inputs/model_{iteration}.pt'
        stages.append((f'export{iteration}','scripts/server/export_walk_actor_checkpoint.py',
            ['--checkpoint',checkpoint,'--output',f'evidence/step{iteration}']))
        stages.append((f'screen{iteration}','scripts/server/evaluate_walk_tracking_transfer.py',
            ['--candidate',checkpoint,'--long-hold','--seeds',*[str(s) for s in spec['seeds']],
             '--speed-grid',*[str(s) for s in spec['speeds']],
             '--output',f'screen{iteration}.json']))
    for name,path,args in stages:
        for attempt in range(30):
            memory,usage=map(int,subprocess.check_output(['nvidia-smi','--id=0',
                '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            if memory>12288 and usage<30:break
            (OUT/'progress.json').write_text(json.dumps(dict(stage='waiting-gpu',nextStage=name,freeMiB=memory,utilization=usage)))
            time.sleep(10)
        else:raise RuntimeError('GPU busy; inspect completed stages')
        (OUT/'progress.json').write_text(json.dumps(dict(stage=name)))
        with (OUT/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,path,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    hashes=json.loads((OUT/'launch.json').read_text())['sourceHashes']
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (OUT/'source-check.json').write_text(json.dumps(dict(passed=True)))
    (OUT/'progress.json').write_text(json.dumps(dict(stage='complete',deploymentAccepted=False)))
    status=0
finally:(OUT/'job.exit').write_text(str(status))

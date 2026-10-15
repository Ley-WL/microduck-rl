"""Walking refinement initialized with independently calibrated critic only."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
if '--worker' not in sys.argv:
    assert not (OUT/'launch.json').exists(),'Inspect the existing job instead of restarting'
    proof=json.loads((ROOT/'inputs/relative-proof.json').read_text())
    assert proof['relativeRankingNonRegression'] and not proof['training'] and not proof['accepted']
    seed=json.loads((ROOT/'inputs/prepare-audit.json').read_text())
    assert seed['actorSource']=='pulse-v55-step500'
    assert hashlib.sha256((ROOT/'inputs/seed.pt').read_bytes()).hexdigest()==seed['seedSha256']
    critic_proof=json.loads((ROOT/'inputs/critic-proof.json').read_text())
    assert critic_proof['passed'] and not critic_proof['gaitAccepted']
    assert critic_proof['task']=='walk_relative_cycle_context'
    assert json.loads((ROOT/'inputs/dataset-audit.json').read_text())['passed']
    assert hashlib.sha256((ROOT/'inputs/warm-critic.pt').read_bytes()).hexdigest()==critic_proof['checkpointSha256']
    (OUT/'launch.json').write_text(json.dumps(dict(hardwareAccess=False,autoDeployment=False,
        feedbackUsedAtRuntime=False,sourceHashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
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
    task='walk_relative_cycle_context'
    stages=[('coverage-test','scripts/server/test_walk_relative_cycle_context.py',[],task),('noise-test','scripts/server/test_walk_noise_schedule.py',[],task)]
    retention_path=ROOT/'inputs/retention-spec.json'
    retention=json.loads(retention_path.read_text()) if retention_path.exists() else None
    extra=[]
    if retention is not None:
        assert retention['coefficient']==.01 and retention['mode']=='gaussian-kl'
        assert len(retention['screenSeeds'])==2 and len(set(retention['screenSeeds']))==2
        extra=['--retention-kl',str(retention['coefficient'])]
        stages.append(('retention-test','scripts/server/test_walk_actor_retention.py',[],task))
    for label,envs,iterations in [('smoke',64,5),('main',1024,200)]:
        stages.append((label,'scripts/server/train_walk_stop_transfer.py',
            ['--envs',str(envs),'--iterations',str(iterations),'--noise-schedule','low',
             '--critic-checkpoint','inputs/warm-critic.pt','--output','evidence/'+label,*extra],task))
    stages.append(('screen','scripts/server/evaluate_walk_tracking_transfer.py',
        ['--candidate','evidence/main/final.pt','--long-hold','--seeds',
         *[str(s) for s in (retention['screenSeeds'] if retention is not None else [256151,256157])],
         '--speed-grid','-.4','-.2','-.075','-.05','-.035','0','.025','.035','.05','.075','.1','.2','.4',
         '--output','relative-cycle-screen.json'],task))
    for name,path,args,task in stages:
        if task:env['MICRODUCK_WALK_TRANSFER_TASK']=task
        for attempt in range(30):
            memory,usage=map(int,subprocess.check_output(['nvidia-smi','--id=0',
                '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            if memory>12288 and usage<30:break
            (OUT/'progress.json').write_text(json.dumps(dict(stage='waiting-gpu',nextStage=name,freeMiB=memory,utilization=usage)))
            time.sleep(10)
        else:raise RuntimeError('GPU busy; inspect completed stages before resuming')
        (OUT/'progress.json').write_text(json.dumps(dict(stage=name)))
        with (OUT/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,path,*args],cwd=ROOT,env=env,stdout=log,
                stderr=subprocess.STDOUT,check=True)
    (OUT/'progress.json').write_text(json.dumps(dict(stage='complete',deploymentAccepted=False)))
    hashes=json.loads((OUT/'launch.json').read_text())['sourceHashes']
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (OUT/'source-check.json').write_text(json.dumps(dict(passed=True)))
    status=0
finally:(OUT/'job.exit').write_text(str(status))

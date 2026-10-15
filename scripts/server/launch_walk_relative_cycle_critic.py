"""Calibrate the large-step actor critic under continuous command coverage; actor frozen."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
if '--worker' not in sys.argv:
    assert not (OUT/'launch.json').exists(),'Inspect the existing job instead of restarting'
    proof=json.loads((ROOT/'inputs/relative-ranking.json').read_text())
    assert proof['relativeRankingNonRegression'] and not proof['training'] and not proof['accepted']
    seed=json.loads((ROOT/'inputs/prepare-audit.json').read_text())
    assert seed['actorSource']=='pulse-v55-step500'
    assert hashlib.sha256((ROOT/'inputs/seed.pt').read_bytes()).hexdigest()==seed['seedSha256']
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
    stages=[('context-test','scripts/server/test_walk_relative_cycle_context.py',[],task),
        ('collect','scripts/server/evaluate_walk_critic_quality.py',
        ['--checkpoint','inputs/context.pt','--task',task,'--collect-training','--spec','inputs/critic-spec.json'],task),
        ('fit','scripts/server/train_walk_mc_critic.py',[],task),
        ('critic-audit','scripts/server/evaluate_walk_critic_quality.py',[],task)]
    for name,path,args,task in stages:
        if name=='critic-audit':args=['--checkpoint',(OUT/'mc-selected.txt').read_text(),'--task',task,'--spec','inputs/critic-spec.json']
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

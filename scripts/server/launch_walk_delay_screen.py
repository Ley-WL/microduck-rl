"""Queue a frozen 12-case paired screen after one already launched training job."""
from pathlib import Path
import json, shutil, subprocess, sys, os, time, hashlib, tarfile
import numpy as np

base=Path('/home/leiwu30/microduck-training')
root=base/'walk-delay16-screen-v2-20261013'
train=base/'walk-delay16-transfer-20261013'
prior=base/'walk-delay-parity-probe-20261013'
encoder=base/'walk-encoder-probe-20261013'

def run():
    e=root/'evidence'
    status=1
    try:
        while not (train/'evidence/job.exit').exists():
            (e/'progress.json').write_text(json.dumps({'stage':'waiting-existing-training','trainingPid':2251965}))
            time.sleep(20)
        assert (train/'evidence/job.exit').read_text().strip()=='0'
        assert json.loads((train/'evidence/source-check.json').read_text())['passed']
        models=list((train/'evidence/main-Surface').rglob('*.onnx'))
        assert len(models)==1,models
        shutil.copy2(models[0],root/'inputs/Delay16.onnx')
        shutil.copy2(models[0].with_suffix('.metadata.json'),root/'inputs/Delay16.metadata.json')
        manifest=json.loads((e/'launch.json').read_text())
        manifest['sourceHashes']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src','scripts','inputs') for p in (root/folder).rglob('*')
            if p.is_file() and '__pycache__' not in str(p)}
        (e/'launch.json').write_text(json.dumps(manifest,indent=2))
        commands=json.loads((encoder/'evidence/launch.json').read_text())['commands']
        for lag in (7,8):
            for original in commands:
                args=original.copy()
                args[args.index('--onnx')+1]='inputs/Delay16.onnx'
                old_out=args[args.index('--output')+1]
                args[args.index('--output')+1]=old_out.replace('Encoder-',f'Delay16at{lag*2}ms-')
                args+=['--command-delay-steps',str(lag)]
                (e/'progress.json').write_text(json.dumps({'stage':args[args.index('--output')+1]}))
                env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
                with (e/'evaluation.log').open('a') as log:
                    subprocess.run([sys.executable]+args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        refs=json.loads((prior/'inputs/delay-reference.json').read_text())
        rows=[]
        def forward(r,speed):
            return r['completed'] and r['movingMeanVx'] is not None and r['movingMeanVx']>=max(.02,.3*speed)
        for folder in e.glob('Delay16at*ms-*'):
            label,terrain=folder.name.split('-',1)
            lag=7 if '14ms' in label else 8
            summary=json.loads((folder/'summary.json').read_text())
            assert summary['manifest']['effectiveCommandDelaySteps']==lag
            for result in summary['results']:
                seed,trial,speed=result['seed'],result['trial'],result['speed']
                name=f'seed{seed}-trial{trial}-speed{speed:g}.json'
                trace=json.loads((folder/name).read_text())
                key=f'{terrain}/{seed}/{trial}/{speed:g}'
                comparisons={}
                for policy in ('Baseline','Encoder'):
                    if lag==8:
                        reference=refs[f'{policy}14ms/{key}']
                    else:
                        old_folder=prior/'evidence'/f'{policy}14ms-{terrain}'
                        old_trace=json.loads((old_folder/name).read_text())
                        old_rows=json.loads((old_folder/'summary.json').read_text())['results']
                        old_result=next(r for r in old_rows if (r['seed'],r['trial'],r['speed'])==(seed,trial,speed))
                        reference=dict(initialQpos=old_trace['initialQpos'],initialVelocity=old_trace['frames'][0]['velocity'],result=old_result)
                    np.testing.assert_array_equal(trace['initialQpos'],reference['initialQpos'])
                    np.testing.assert_array_equal(trace['frames'][0]['velocity'],reference['initialVelocity'])
                    old_result=reference['result']
                    comparisons[policy]=dict(reference=old_result,
                        lostSurvival=old_result['completed'] and not result['completed'],
                        lostForward=forward(old_result,speed) and not forward(result,speed),
                        gainedForward=not forward(old_result,speed) and forward(result,speed))
                rows.append(dict(key=key,delayMs=2*lag,result=result,comparisons=comparisons))
        assert len(rows)==12
        (e/'comparison.json').write_text(json.dumps(dict(rows=rows,initialStatesMatched=True,
            accepted=False,hardwareAccess=False,optimizerSteps=0,
            boundary='Selected fitted diagnostic cases, synthetic terrain, not formal or physical acceptance'),indent=2))
        bad=[p for p,h in manifest['sourceHashes'].items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
        assert not bad,bad
        (e/'source-check.json').write_text(json.dumps(dict(passed=True,files=len(manifest['sourceHashes']))))
        status=0
    finally:
        (e/'job.exit').write_text(str(status))
        with tarfile.open(root/'complete.tar.gz','w:gz') as a:
            for folder in ('src','scripts','inputs','evidence'):a.add(root/folder,arcname=folder)
        (root/'complete.sha256').write_text(hashlib.sha256((root/'complete.tar.gz').read_bytes()).hexdigest())

if '--worker' in sys.argv:
    run()
else:
    assert not root.exists(),'Track existing screen PID'
    assert (train/'evidence/launch.json').exists()
    root.mkdir()
    for folder in ('src','scripts','inputs'):shutil.copytree(prior/folder,root/folder)
    (root/'evidence').mkdir()
    worker=root/'scripts/server/launch_walk_delay_screen.py'
    shutil.copy2(Path(__file__),worker)
    (root/'evidence/launch.json').write_text(json.dumps(dict(training=False,hardwareAccess=False,
        waitForExistingTraining=str(train),cases=12,delaysMs=[14,16]),indent=2))
    with (root/'evidence/job.log').open('w') as log:
        process=subprocess.Popen([sys.executable,str(worker),'--worker'],cwd=root,stdout=log,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (root/'evidence/pid').write_text(str(process.pid))
    print(json.dumps(dict(pid=process.pid,training=False,hardwareAccess=False)))

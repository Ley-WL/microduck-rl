"""Expand fitted regression without re-running any cached completed cases."""
from pathlib import Path
import json,shutil,subprocess,sys,os,hashlib,tarfile
import numpy as np

base=Path('/home/leiwu30/microduck-training')
root=base/'walk-delay16-fullscreen-20261013'
probe=base/'walk-delay16-screen-v2-20261013'
delay=base/'walk-delay-parity-probe-20261013'
screen=base/'walk-surface-screen-20261013'
terrains={'flat':(0,.02),'pile3-soft.02':(3,.02),'pile6-soft.04':(6,.04)}

def worker():
    e=root/'evidence';status=1
    try:
        rows=[];provenance=[]
        for lag in (7,8):
            for policy in ('Baseline','Delay16'):
                for terrain,(height,contact) in terrains.items():
                    output=e/f'{policy}{2*lag}ms-{terrain}';output.mkdir()
                    collected=[]
                    for seed in (130031,240017):
                        for speed in (.1,.15,.2,.25):
                            names=[f'seed{seed}-trial{trial}-speed{speed:g}.json' for trial in (0,1)]
                            cached=None
                            if policy=='Baseline' and lag==8:
                                cached=screen/'evidence'/f'Baseline-{terrain}'
                            elif policy=='Baseline':
                                cached=delay/'evidence'/f'Baseline14ms-{terrain}'
                            else:
                                cached=probe/'evidence'/f'Delay16at{2*lag}ms-{terrain}'
                            if not all((cached/name).exists() for name in names):
                                cached=None
                            if cached:
                                summary=json.loads((cached/'summary.json').read_text())
                                source=cached
                            else:
                                source=e/f'new-{policy}{2*lag}ms-{terrain}-{seed}-{speed:g}'
                                args=['scripts/evaluate_walk_runtime_cpu.py','--onnx',f'inputs/{policy}.onnx',
                                    '--body-contract','inputs/body-contract.json','--output',str(source),
                                    '--seconds','25','--backlash','--voltage-compensation',
                                    '--terrain-height-mm',str(height),'--contact-timeconst',str(contact),
                                    '--seeds',str(seed),'--speeds',str(speed),'--trials','2',
                                    '--command-delay-steps',str(lag)]
                                (e/'progress.json').write_text(json.dumps(dict(stage=str(source.name),reused=len(provenance))))
                                env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
                                with (e/'evaluation.log').open('a') as log:
                                    subprocess.run([sys.executable]+args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                                summary=json.loads((source/'summary.json').read_text())
                            for trial,name in enumerate(names):
                                result=next(r for r in summary['results'] if (r['seed'],r['trial'],r['speed'])==(seed,trial,speed))
                                shutil.copy2(source/name,output/name)
                                if cached:
                                    provenance.append(dict(trace=str(source/name),sha256=hashlib.sha256((source/name).read_bytes()).hexdigest()))
                                collected.append(result)
                    (output/'summary.json').write_text(json.dumps(dict(results=collected)))
        def forward(r):return r['completed'] and r['movingMeanVx'] is not None and r['movingMeanVx']>=max(.02,.3*r['speed'])
        for lag in (7,8):
            for terrain in terrains:
                folders={p:e/f'{p}{2*lag}ms-{terrain}' for p in ('Baseline','Delay16')}
                reports={p:json.loads((folder/'summary.json').read_text())['results'] for p,folder in folders.items()}
                for original in reports['Baseline']:
                    key=(original['seed'],original['trial'],original['speed'])
                    candidate=next(r for r in reports['Delay16'] if (r['seed'],r['trial'],r['speed'])==key)
                    name=f'seed{key[0]}-trial{key[1]}-speed{key[2]:g}.json'
                    traces={p:json.loads((folder/name).read_text()) for p,folder in folders.items()}
                    np.testing.assert_array_equal(traces['Baseline']['initialQpos'],traces['Delay16']['initialQpos'])
                    np.testing.assert_array_equal(traces['Baseline']['frames'][0]['velocity'],traces['Delay16']['frames'][0]['velocity'])
                    metrics={}
                    for p,t in traces.items():
                        frame=np.abs(np.diff(np.asarray([f['target'] for f in t['frames']]),axis=0))*180/np.pi
                        metrics[p]=dict(frameP95Deg=np.percentile(frame,95,axis=0).tolist(),frameMaxDeg=frame.max(axis=0).tolist())
                    rows.append(dict(delayMs=2*lag,terrain=terrain,key=key,baseline=original,candidate=candidate,
                        lostSurvival=original['completed'] and not candidate['completed'],
                        lostForward=forward(original) and not forward(candidate),
                        gainedForward=not forward(original) and forward(candidate),metrics=metrics))
        assert len(rows)==96
        (e/'comparison.json').write_text(json.dumps(dict(rows=rows,initialStatesMatched=True,accepted=False,
            boundary='Fitted full regression, synthetic carpet proxy, not independent or physical acceptance'),indent=2))
        (e/'reuse-provenance.json').write_text(json.dumps(provenance,indent=2))
        manifest=json.loads((e/'launch.json').read_text())
        assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in manifest['sourceHashes'].items())
        (e/'source-check.json').write_text(json.dumps(dict(passed=True,files=len(manifest['sourceHashes']))))
        status=0
    finally:
        (e/'job.exit').write_text(str(status))
        with tarfile.open(root/'complete.tar.gz','w:gz') as archive:
            for folder in ('src','scripts','inputs','evidence'):archive.add(root/folder,arcname=folder)
        (root/'complete.sha256').write_text(hashlib.sha256((root/'complete.tar.gz').read_bytes()).hexdigest())

if '--worker' in sys.argv:worker()
else:
    assert not root.exists(),'Track existing PID'
    assert (probe/'evidence/job.exit').read_text().strip()=='0'
    root.mkdir()
    for folder in ('src','scripts','inputs'):shutil.copytree(probe/folder,root/folder)
    (root/'evidence').mkdir()
    script=root/'scripts/server/launch_walk_delay_fullscreen.py';shutil.copy2(Path(__file__),script)
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src','scripts','inputs') for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    (root/'evidence/launch.json').write_text(json.dumps(dict(training=False,hardwareAccess=False,
        totalCases=192,newCases=126,reusedCases=66,delaysMs=[14,16],sourceHashes=files),indent=2))
    with (root/'evidence/job.log').open('w') as log:
        process=subprocess.Popen([sys.executable,str(script),'--worker'],cwd=root,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    (root/'evidence/pid').write_text(str(process.pid));print(json.dumps(dict(pid=process.pid,training=False,hardwareAccess=False)))

"""Frozen-policy gain sensitivity diagnostic, CPU only, no new training."""
import concurrent.futures,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[2]
folder=root/'evidence/gain-curve'
python=root.parent/'microduck-rl/.venv/bin/python'
environment=os.environ.copy();environment.update(PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',MUJOCO_GL='egl')
scales=(.8,.82,.85,.9,.95,1.,1.1,1.2)
policies={'baseline':root/'evidence/arrival-gate-control/baseline.onnx',
          'ArrivalGate':root/'evidence/arrival-gate-control/ArrivalGate/train.onnx'}


def sweep(name,policy):
    output={}
    for scale in scales:
        target=folder/name/str(scale);target.mkdir(parents=True,exist_ok=True)
        with target.with_name(target.name+'.log').open('w') as log:
            subprocess.run([str(python),'scripts/evaluate_skills_cpu.py','--onnx',str(policy),
                '--output',str(target),'--scenario','stand_cycle','--seeds','9001','--trials','4',
                '--actuator-kp-scale',str(scale)],cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((target/'summary.json').read_text())
        assert report['policySha256']==hashlib.sha256(policy.read_bytes()).hexdigest()
        assert len(report['groups'])==1 and len(report['groups'][0]['results'])==4
        output[str(scale)]=report['groups'][0]['results']
        print(json.dumps({'policy':name,'scale':scale,'successes':sum(r['success'] for r in output[str(scale)]),'trials':4}),flush=True)
    return output


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={n:pool.submit(sweep,n,p) for n,p in policies.items()}
        rows={n:f.result() for n,f in futures.items()}
    output={'hardwareAccess':False,'newTraining':False,'formalAccepted':False,
            'scope':'One seed, standing-start repeated sit/stand, 4 trials per gain, both policies. Same initial state between policies at each gain; initial settling itself depends on gain, so cross-gain differences include that effect.',
            'scales':[]}
    for scale in scales:
        base,current=rows['baseline'][str(scale)],rows['ArrivalGate'][str(scale)]
        for old,new in zip(base,current):
            assert old['trial']==new['trial']
            np.testing.assert_allclose(old['initialQpos'],new['initialQpos'],atol=1e-12)
        item={'scale':scale,'baseline':sum(r['success'] for r in base),'ArrivalGate':sum(r['success'] for r in current),
              'lostSuccessfulTrials':[r['trial'] for r,c in zip(base,current) if r['success'] and not c['success']],
              'initialStatesPaired':True}
        for name in policies:
            tilted=[]
            for trial in range(4):
                trace=json.loads((folder/name/str(scale)/f'stand_cycle-9001-{trial}.json').read_text())['trace']
                tilted.append(float(np.mean([r['tiltDeg'] for r in trace if 11<=r['time']<14])))
            item[name+'FinalStandTiltDeg']=tilted
        output['scales'].append(item)
    manifest=json.loads((folder/'launch.json').read_text())
    assert manifest['sourceSha256']==hashlib.sha256((root/'scripts/evaluate_skills_cpu.py').read_bytes()).hexdigest()
    (folder/'summary.json').write_text(json.dumps(output,indent=2))
    (folder/'completed.txt').write_text('complete\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        (folder/'job.exit').write_text('1');raise
    else:(folder/'job.exit').write_text('0')

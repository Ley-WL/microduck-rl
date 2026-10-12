"""Capture only successful complete seed rollouts under four conditions.

No relabeling of failure trajectories. Seed 9001 is held out as entire rollouts.
These are fitting cohorts; independent regression must use other seeds later.
"""
import concurrent.futures,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
folder=root/'evidence/broad-retention';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'completed.txt').exists(),'Preserve existing collection'
python=root.parent/'microduck-rl/.venv/bin/python'
conditions={'nominal':[],
 'stress':['--scenario','stand_cycle','sit_cycle','--transition-stress','--joint-velocity-lag-steps','1'],
 'gain-low':['--scenario','stand_cycle','sit_cycle','--actuator-kp-scale','.8'],
 'gain-high':['--scenario','stand_cycle','sit_cycle','--actuator-kp-scale','1.2']}


def collect(name,extra):
    path=folder/name
    environment=os.environ.copy();environment['OMP_NUM_THREADS']='1'
    with (folder/(name+'.log')).open('w') as log:
        subprocess.run([str(python),'scripts/evaluate_skills_cpu.py','--onnx','weight-1p0.onnx',
            '--output',str(path),'--trials','4','--seeds','8111','9001','--capture-policy-frames',*extra],
            cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((path/'summary.json').read_text())
    accepted=[]
    for group in report['groups']:
        for row in group['results']:
            if row['success']:
                trajectory=path/f"{group['scenario']}-{group['seed']}-{row['trial']}.policy.json"
                rows=json.loads(trajectory.read_text())
                observations=np.asarray([r['observation'] for r in rows],dtype=np.float32)
                assert len(observations)==(1000 if group['scenario'].endswith('cycle') else 750)
                accepted.append((group['seed'],observations,str(trajectory),hashlib.sha256(trajectory.read_bytes()).hexdigest()))
    print(json.dumps({'condition':name,'successes':len(accepted)}),flush=True)
    return accepted


if __name__=='__main__':
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={name:pool.submit(collect,name,extra) for name,extra in conditions.items()}
            all_rows=[(name,row) for name,future in futures.items() for row in future.result()]
        training=np.concatenate([row[1] for name,row in all_rows if row[0]==8111])
        validation=np.concatenate([row[1] for name,row in all_rows if row[0]==9001])
        np.savez_compressed(folder/'retention.npz',simulation=training,validation=validation)
        manifest={'sourceSha256':hashlib.sha256((root/'weight-1p0.onnx').read_bytes()).hexdigest(),
            'trainSeed':8111,'validationSeed':9001,'trainSamples':len(training),'validationSamples':len(validation),
            'successfulTrajectories':[{'condition':name,'seed':row[0],'path':row[2],'sha256':row[3]} for name,row in all_rows],
            'independentEvaluation':False,'hardwareValidated':False}
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
        (folder/'completed.txt').write_text('complete\n');(folder/'job.exit').write_text('0')
    except BaseException:
        (folder/'job.exit').write_text('1');raise

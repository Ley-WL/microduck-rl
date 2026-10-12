"""Evaluate intermediate gait checkpoints so final-round regression is visible."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

run=sys.argv[1]
task=sys.argv[2] if len(sys.argv)>2 else 'Mjlab-Walk-Flat-MicroDuck-HD1910'
if not re.fullmatch(r'[a-zA-Z0-9_-]+',run):
    raise ValueError('Invalid run name')
root=Path(__file__).resolve().parents[2];base=root.parent
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',MUJOCO_GL='egl',OMP_NUM_THREADS='4')
results=[]
report=base/'setup'/(run+'-checkpoint-audit.json')
for iteration in (100,200,300,400,499):
    while True:
        paths=list((root/'logs/rsl_rl/microduck_hd1910_walk').glob('*_'+run+'/model_'+str(iteration)+'.pt'))
        if paths:
            break
        exit_file=base/'runs'/(run+'.exit')
        if exit_file.exists():
            print('Training stopped before checkpoint',iteration,flush=True)
            sys.exit(0)
        time.sleep(10)
    time.sleep(2)
    policy=base/'exports'/(run+'-checkpoint'+str(iteration)+'.onnx')
    output=base/'setup'/(run+'-checkpoint'+str(iteration)+'.json')
    with (base/'runs'/(run+'-checkpoint'+str(iteration)+'.log')).open('w') as log:
        subprocess.run(['bash','scripts/server/export.sh',task,str(paths[0]),str(policy)],
            cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(root/'.venv/bin/python'),'scripts/server/evaluate_walk.py','--task',task,'--onnx',str(policy),
            '--scenario','forward','--envs','8','--seconds','20','--output',str(output)],
            cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads(output.read_text());result['checkpointIteration']=iteration
    results.append(result)
    eligible=[r for r in results if r['failedWorlds']==0]
    def score(r):
        return (max(abs(x) for x in r['movingMeanHeadHomeErrorDeg'])
            +.2*r['path']['headingDriftP95Deg']+1000*max(0.,.075-r['gait']['medianStrideM'])
            +100*r['stopP95LinearSpeedMps'])
    best=min(eligible,key=score) if eligible else None
    report.write_text(json.dumps({'run':run,'results':results,
        'bestStableCheckpointIteration':best['checkpointIteration'] if best else None,
        'note':'8-world comparison only; best candidate still requires full validation'},indent=2))
    print('checkpoint',iteration,'failed',result['failedWorlds'],
        'mean head',result['movingMeanHeadHomeErrorDeg'],'best stable',best['checkpointIteration'] if best else None,flush=True)

"""Compare identical .1m/s commands across source/intermediate/final policies."""
import json, os, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];base=root.parent
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',MUJOCO_GL='egl',OMP_NUM_THREADS='4')
out=base/'setup'/'v13-regression';out.mkdir(exist_ok=True)
uv=str(Path.home()/'.local/bin/uv')
run=next((root/'logs/rsl_rl/microduck_hd1910_walk_slow').glob('*_hd1910-walk-v13-slow1000'))
cases=[('v12','Mjlab-WalkSlow-Flat-MicroDuck-HD1910',base/'exports/hd1910-walk-v12-progress1000.onnx',False)]
for n in (0,250,500,750):
    policy=out/f'v13-{n}.onnx'
    with (out/f'export-{n}.log').open('w') as log:
        subprocess.run(['bash','scripts/server/export.sh','Mjlab-WalkSlow-Flat-MicroDuck-HD1910',
            str(run/f'model_{n}.pt'),str(policy)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    cases.append((f'v13-{n}','Mjlab-WalkSlow-Flat-MicroDuck-HD1910',policy,False))
cases.extend([('v13-999','Mjlab-WalkSlow-Flat-MicroDuck-HD1910',base/'exports/hd1910-walk-v13-slow1000.onnx',False),
    ('v13-999-immediate','Mjlab-WalkSlow-Flat-MicroDuck-HD1910',base/'exports/hd1910-walk-v13-slow1000.onnx',True)])
results=[]
for name,task,policy,immediate in cases:
    target=out/f'{name}.json'
    args=[uv,'run','--locked','python','scripts/server/evaluate_walk.py','--task',task,
        '--onnx',str(policy),'--scenario','forward','--forward-speed','.1','--envs','8',
        '--seconds','12','--output',str(target),'--trace',str(out/f'{name}-trace.json')]
    if immediate:args.append('--start-moving')
    with (out/f'{name}.log').open('w') as log:
        subprocess.run(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads(target.read_text());result['case']=name;results.append(result)
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    print(name,'failed',result['failedWorlds'],'first failure',result['firstFailureSeconds'],
        'stride',result['gait']['medianStrideM'],'air',result['gait']['medianLandingAirTimeSec'],flush=True)

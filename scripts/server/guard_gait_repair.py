"""Audit saved gait checkpoints and stop only this run on measured regression."""
import argparse,json,os,re,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('baseline',type=Path)
p.add_argument('task',nargs='?',default='Mjlab-WalkRepair-Flat-MicroDuck-HD1910')
p.add_argument('experiment',nargs='?',default='microduck_hd1910_walk_repair')
p.add_argument('--total-iterations',type=int,default=1000)
p.add_argument('--interval',type=int,default=100)
p.add_argument('--observe-only',action='store_true')
a=p.parse_args();run=a.run;baseline=a.baseline
if not re.fullmatch(r'[A-Za-z0-9_-]+',run):raise ValueError('Invalid run')
root=Path(__file__).resolve().parents[2];base=root.parent
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='6',MUJOCO_GL='egl',OMP_NUM_THREADS='4')
task=a.task;experiment=a.experiment
uv=str(Path.home()/'.local/bin/uv')
reference=json.loads(baseline.read_text());report=base/'runs'/(run+'.guard-status.json')
results=[]


def stop_this_run():
    # Match this exact run argument; never touch unrelated training processes.
    listing=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    victims=[]
    for line in listing.splitlines()[1:]:
        pid,args=line.strip().split(None,1)
        tokens=args.split()
        if '--agent.run-name' in tokens and '/bin/train' in args:
            i=tokens.index('--agent.run-name')
            if i+1<len(tokens) and tokens[i+1]==run:victims.append(int(pid))
    for pid in victims:os.kill(pid,signal.SIGTERM)
    return victims


for n in range(max(200,a.interval),a.total_iterations,a.interval):
    while True:
        paths=list((root/'logs/rsl_rl'/experiment).glob('*_'+run+f'/model_{n}.pt'))
        if paths:break
        if (base/'runs'/(run+'.exit')).exists():
            report.write_text(json.dumps({'stage':'training_ended','results':results},indent=2));sys.exit(0)
        time.sleep(5)
    time.sleep(2)
    policy=base/'exports'/f'{run}-checkpoint{n}.onnx';output=base/'setup'/f'{run}-checkpoint{n}.json'
    with (base/'runs'/f'{run}-checkpoint{n}.log').open('w') as log:
        subprocess.run(['bash','scripts/server/export.sh',task,str(paths[0]),str(policy)],
            cwd=root,env={**env,'MICRODUCK_GPU':'6'},stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([uv,'run','--locked','python','scripts/server/evaluate_walk.py','--task',task,
            '--onnx',str(policy),'--scenario','forward','--forward-speed','.1','--envs','32',
            '--seconds','20','--output',str(output)],cwd=root,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    value=json.loads(output.read_text());value['checkpointIteration']=n;results.append(value)
    regression=(value['failedWorlds']>max(reference['failedWorlds']+1,2)
        or value['gait']['medianStrideM']<.75*reference['gait']['medianStrideM']
        or value['gait']['medianSwingPeakM']<.75*reference['gait']['medianSwingPeakM']
        or value['gait']['medianLandingAirTimeSec']<.85*reference['gait']['medianLandingAirTimeSec']
        or value['meanAbsTrackingError'][0]>reference['meanAbsTrackingError'][0]+.025
        or value['path']['headingDriftP95Deg']>reference['path']['headingDriftP95Deg']*1.5+5)
    report.write_text(json.dumps({'stage':'regression' if regression else 'checked',
        'checkpoint':n,'results':results},indent=2))
    print(n,'failed',value['failedWorlds'],'stride',value['gait']['medianStrideM'],flush=True)
    if regression and not a.observe_only:
        victims=stop_this_run()
        report.write_text(json.dumps({'stage':'stopped_for_regression','checkpoint':n,
            'stoppedPids':victims,'results':results},indent=2));sys.exit(0)

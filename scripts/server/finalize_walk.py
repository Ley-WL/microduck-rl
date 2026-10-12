"""Wait for the full walk run, export, then evaluate commanded gait and stopping."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

run=sys.argv[1]
if not re.fullmatch(r"[a-zA-Z0-9_-]+",run):
    raise ValueError("Invalid run name")
root=Path(__file__).resolve().parents[2]
base=root.parent
status=base/"runs"/(run+".finalize-status.json")
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=env.get("MICRODUCK_GPU","7"),MUJOCO_GL="egl",OMP_NUM_THREADS="4")
uv=str(Path.home()/".local/bin/uv")
task=sys.argv[2] if len(sys.argv)>2 else "Mjlab-Walk-Flat-MicroDuck-HD1910"
iterations=int(sys.argv[3]) if len(sys.argv)>3 else 500
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_rl_cfg
experiment=load_rl_cfg(task).experiment_name

def report(stage,**details):
    status.write_text(json.dumps({"run":run,"stage":stage,**details},indent=2),encoding="utf-8")
    print(stage,details,flush=True)

def execute(args):
    subprocess.run(args,cwd=root,env=env,check=True)

try:
    report("waiting_for_training")
    exit_file=base/"runs"/(run+".exit")
    while not exit_file.exists():
        time.sleep(15)
    code=int(exit_file.read_text().strip())
    if code!=0:
        guard=base/'runs'/(run+'.guard-status.json')
        if guard.exists():
            reason=json.loads(guard.read_text())
            if reason.get('stage')=='stopped_for_regression':
                report('stopped_for_regression',checkpoint=reason.get('checkpoint'),
                       accepted=False,hardwareValidated=False)
                (base/'runs'/(run+'.finalize-exit')).write_text('2\n',encoding='utf-8')
                sys.exit(0)
        raise RuntimeError(f"Training exited {code}; no success export")
    runs=list((root/"logs/rsl_rl"/experiment).glob("*_"+run))
    if len(runs)!=1:
        raise RuntimeError("Expected exactly one training run directory")
    checkpoint=runs[0]/f"model_{iterations-1}.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"Full {iterations}-iteration checkpoint missing")
    policy=base/"exports"/(run+".onnx")
    report("exporting",checkpoint=str(checkpoint))
    execute(["bash","scripts/server/export.sh",task,str(checkpoint),str(policy)])
    execute([uv,"run","--locked","python","scripts/server/verify_onnx.py",str(policy)])
    execute([uv,"run","--locked","python","scripts/server/write_policy_metadata.py",str(policy),task])
    results=[]
    slow_task=task in ('Mjlab-WalkSlow-Flat-MicroDuck-HD1910','Mjlab-WalkRepair-Flat-MicroDuck-HD1910',
                      'Mjlab-WalkSwing-Flat-MicroDuck-HD1910','Mjlab-WalkSwingFocus-Flat-MicroDuck-HD1910')
    for scenario in (('idle','forward') if slow_task else ('idle','forward','backward','left','right','turn_left','turn_right')):
        output=base/'exports'/(run+'-'+scenario+'.json')
        report('evaluating',scenario=scenario)
        command=[uv,'run','--locked','python','scripts/server/evaluate_walk.py','--task',task,
                 '--onnx',str(policy),'--scenario',scenario,'--envs','32','--seconds','20','--output',str(output)]
        if slow_task: command+=['--forward-speed','.1']
        if scenario=='forward': command+=['--video',str(base/'exports'/(run+'.mp4'))]
        execute(command);results.append(json.loads(output.read_text()))
    accepted=all(r['failedWorlds']==0 and max(r['meanAbsTrackingError'][:2])<=.08
                 and r['p95TiltDeg']<=25
                 and r['meanAbsTrackingError'][2]<=.2 and r['stopP95LinearSpeedMps']<=.03
                 and r['stopP95AngularSpeedRadps']<=.1 and r['stopP95HeadErrorDeg']<=10
                 and max(abs(x) for x in r['movingMeanHeadHomeErrorDeg'])<=8
                 and max(r['movingP95HeadHomeErrorDeg'])<=20
                 and (r['scenario'] in ('idle','turn_left','turn_right') or
                      r['path']['headingDriftP95Deg']<=10)
                 and (r['scenario'] not in ('forward','backward') or r['path']['crossTrackP95M']<=.08)
                 and (r['scenario']!='forward' or r['path']['projectedForwardTravelMeanM']>=1.5)
                 and (r['scenario']!='backward' or r['path']['projectedForwardTravelMeanM']<=-1.)
                 and (r['scenario'] not in ('forward','backward') or
                      (r['gait']['medianStrideM']>=(.075 if r['scenario']=='forward' else .045) and r['gait']['medianLandingAirTimeSec']>=.1
                       and r['gait']['medianSwingPeakM']>=.01))
                 and (r['scenario'] in ('idle','turn_left','turn_right') or r['footHeightSpanMedianM']>=.008)
                 for r in results)
    if slow_task:
        accepted=all(r['failedWorlds']==0 and r['p95TiltDeg']<=15
            and r['stopP95LinearSpeedMps']<=.03 and r['stopP95AngularSpeedRadps']<=.1
            and r['stopP95HeadErrorDeg']<=10
            and max(abs(x) for x in r['movingMeanHeadHomeErrorDeg'])<=8
            and max(r['movingP95HeadHomeErrorDeg'])<=20
            and (r['scenario']=='idle' or (
                .08<=r['meanBodyVelocity'][0]<=.12
                and r['meanAbsTrackingError'][0]<=.03
                and r['path']['headingDriftP95Deg']<=10
                and r['path']['crossTrackP95M']<=.08
                and r['path']['projectedForwardTravelMeanM']>=1.
                and r['gait']['medianStrideM']>=.04
                and r['gait']['medianLandingAirTimeSec']>=.1
                and r['gait']['medianSwingPeakM']>=.01)) for r in results)
    proof={'run':run,'task':task,'checkpoint':str(checkpoint),'iterationsCompleted':iterations,
           'acceptedInSimulation':accepted,'results':results,'hardwareValidated':False,
           'criteria':'32 worlds x20s per scenario, no falls; tracking/stop/head; moving mean head HOME error<=8deg and P95<=20deg; straight heading drift P95<=10deg, cross track<=80mm; forward travel>=1.5m, stride>=75mm; backward travel<=-1m, stride>=45mm; air>=100ms, lift>=10mm',
           'metadata':json.loads(policy.with_suffix('.metadata.json').read_text())}
    if slow_task:
        proof['criteria']='0.1m/s forward + idle, 32 worlds x20s, no failures; actual forward mean .08-.12m/s, mean absolute speed error<=.03; heading P95<=10deg, cross track P95<=80mm; travel>=1m, same-foot stride>=40mm, air>=100ms, lift>=10mm; tilt P95<=15deg; head mean<=8deg/P95<=20deg; stop linear<=.03/yaw<=.1/head<=10deg. Video review still required; other commands not accepted by this specialization.'
    (base/"setup"/(run+"-verification.json")).write_text(json.dumps(proof,indent=2),encoding="utf-8")
    report("accepted_in_simulation" if accepted else "needs_more_training",accepted=accepted)
    (base/"runs"/(run+".finalize-exit")).write_text("0\n",encoding="utf-8")
except Exception as error:
    report("failed",error=str(error))
    (base/"runs"/(run+".finalize-exit")).write_text("1\n",encoding="utf-8")
    raise

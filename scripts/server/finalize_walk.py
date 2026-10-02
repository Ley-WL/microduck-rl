"""Wait for the full run, export, then evaluate bursts and strong pushes."""
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
iterations=int(sys.argv[3]) if len(sys.argv)>3 else 6000
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
    for scenario in ('idle','forward','backward','left','right','turn_left','turn_right'):
        output=base/'exports'/(run+'-'+scenario+'.json')
        report('evaluating',scenario=scenario)
        command=[uv,'run','--locked','python','scripts/server/evaluate_walk.py','--task',task,
                 '--onnx',str(policy),'--scenario',scenario,'--envs','32','--seconds','20','--output',str(output)]
        if scenario=='forward': command+=['--video',str(base/'exports'/(run+'.mp4'))]
        execute(command);results.append(json.loads(output.read_text()))
    accepted=all(r['failedWorlds']==0 and max(r['meanAbsTrackingError'][:2])<=.08
                 and r['meanAbsTrackingError'][2]<=.2 and r['stopP95LinearSpeedMps']<=.03
                 and r['stopP95AngularSpeedRadps']<=.1 and r['stopP95HeadErrorDeg']<=10
                 and (r['scenario'] in ('idle','turn_left','turn_right') or r['footHeightSpanMedianM']>=.008)
                 for r in results)
    proof={'run':run,'task':task,'checkpoint':str(checkpoint),'iterationsCompleted':iterations,
           'acceptedInSimulation':accepted,'results':results,'hardwareValidated':False,
           'criteria':'32 worlds x20s per scenario, no falls; commanded tracking and foot lift, stop speed/heading/head posture',
           'metadata':json.loads(policy.with_suffix('.metadata.json').read_text())}
    (base/"setup"/(run+"-verification.json")).write_text(json.dumps(proof,indent=2),encoding="utf-8")
    report("accepted_in_simulation" if accepted else "needs_more_training",accepted=accepted)
    (base/"runs"/(run+".finalize-exit")).write_text("0\n",encoding="utf-8")
except Exception as error:
    report("failed",error=str(error))
    (base/"runs"/(run+".finalize-exit")).write_text("1\n",encoding="utf-8")
    raise

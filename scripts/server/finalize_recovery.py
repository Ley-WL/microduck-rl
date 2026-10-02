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
task=sys.argv[2] if len(sys.argv)>2 else "Mjlab-PushRecovery-Flat-MicroDuck-HD1910"
iterations=int(sys.argv[3]) if len(sys.argv)>3 else 3000
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
    for scenario,strength in (("idle",.3),("repeated",.15),("repeated",.3),("alternating",.3),("strong",.5)):
        tag=f"{scenario}-{strength:.2f}"
        output=base/"exports"/(run+"-"+tag+".json")
        report("evaluating",scenario=scenario,strength=strength)
        command=[uv,"run","--locked","python","scripts/server/evaluate_recovery.py",
                 "--task",task,
                 "--onnx",str(policy),"--scenario",scenario,"--strength",str(strength),
                 "--seconds","20","--envs","32","--output",str(output)]
        if scenario=="repeated" and strength==.3:
            command += ["--video",str(base/"exports"/(run+".mp4"))]
        execute(command)
        results.append(json.loads(output.read_text()))
    accepted=all(r["failedWorlds"]==0 and r["settledP95YawRollErrorDeg"]<=5
                 and r["settledP95MirrorErrorDeg"]<=5 and r["settledP95AbsLegErrorDeg"]<=10
                 and r["settledP95AngularSpeedRadps"]<=.5 for r in results)
    if "PushSettle" in task:
        accepted=accepted and all(r["settledP95AbsHeadErrorDeg"]<=5 and r["settledP95JointTargetSpanDeg"]<=2
                                 and r["settledP95JointPositionSpanDeg"]<=2 and r["settledP95AngularSpeedRadps"]<=.1 for r in results)
    proof={"run":run,"task":task,"checkpoint":str(checkpoint),"iterationsCompleted":iterations,
           "acceptedInSimulation":accepted,"results":results,"hardwareValidated":False,
           "criteria":"32 worlds per scenario, zero failures incl. resets; calm last 5s yaw/roll and mirror p95 <=5deg, leg p95 <=10deg, root angular speed p95 <=0.5rad/s",
           "metadata":json.loads(policy.with_suffix(".metadata.json").read_text())}
    if "PushSettle" in task:
        proof["additionalSettleCriteria"]="last 5s: head p95 error <=5deg; joint target and feedback span p95 <=2deg; root angular speed p95 <=0.1rad/s"
    (base/"setup"/(run+"-verification.json")).write_text(json.dumps(proof,indent=2),encoding="utf-8")
    report("accepted_in_simulation" if accepted else "needs_more_training",accepted=accepted)
    (base/"runs"/(run+".finalize-exit")).write_text("0\n",encoding="utf-8")
except Exception as error:
    report("failed",error=str(error))
    (base/"runs"/(run+".finalize-exit")).write_text("1\n",encoding="utf-8")
    raise

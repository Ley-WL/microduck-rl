"""Frozen broad speed/yaw CPU screen; reports failures and non-starting low speeds."""
import argparse,subprocess,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
status=1
try:
    for yaw in (0.,-.6,.6):
        with (a.output/'run.log').open('a') as log:
            subprocess.run([sys.executable,'-X','utf8','scripts/evaluate_walk_runtime_cpu.py','--onnx',str(a.onnx),'--physics-profile','kissqy','--output',str(a.output/f'cpu-yaw{yaw:g}'),'--speeds','-.4','-.2','-.05','.05','.2','.4','--seeds','222171','222177','--trials','1','--seconds','30','--transport-ms','0','--turn-rate',str(yaw)],stdout=log,stderr=subprocess.STDOUT,check=True)
        (a.output/'progress.json').write_text(json.dumps(dict(completedYaw=yaw)))
    subprocess.run([sys.executable,'-X','utf8','scripts/analyze_walk_runtime.py',str(a.output),'--seconds','30','--window','17','27'],check=True)
    status=0
finally:(a.output/'job.exit').write_text(str(status))

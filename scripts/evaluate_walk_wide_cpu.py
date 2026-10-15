"""Frozen broad speed/yaw CPU screen; reports failures and non-starting low speeds."""
import argparse,subprocess,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--seeds',type=int,nargs='+',default=[222171,222177])
p.add_argument('--speeds',type=float,nargs='+',default=[-.4,-.2,-.05,.05,.2,.4]);a=p.parse_args()
assert len(a.seeds)>0 and len(a.seeds)==len(set(a.seeds))
assert len(a.speeds)>0 and len(a.speeds)==len(set(a.speeds))
assert all(abs(v)<=.4 for v in a.speeds)
a.output.mkdir(parents=True,exist_ok=False)
status=1
try:
    for yaw in (0.,-.6,.6):
        with (a.output/'run.log').open('a') as log:
            subprocess.run([sys.executable,'-X','utf8','scripts/evaluate_walk_runtime_cpu.py','--onnx',str(a.onnx),'--physics-profile','kissqy','--output',str(a.output/f'cpu-yaw{yaw:g}'),'--speeds',*[str(v) for v in a.speeds],'--seeds',*[str(s) for s in a.seeds],'--trials','1','--seconds','30','--transport-ms','0','--turn-rate',str(yaw)],stdout=log,stderr=subprocess.STDOUT,check=True)
        (a.output/'progress.json').write_text(json.dumps(dict(completedYaw=yaw)))
    subprocess.run([sys.executable,'-X','utf8','scripts/analyze_walk_runtime.py',str(a.output),'--seconds','30','--window','17','27'],check=True)
    status=0
finally:(a.output/'job.exit').write_text(str(status))

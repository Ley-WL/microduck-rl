"""Intermediate startup gate; never treats small-step startup as final acceptance."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence'
name=a.stage.name+'-startup-gpu.json'
subprocess.run([sys.executable,'scripts/server/evaluate_walk_tracking_transfer.py','--candidate',str(a.stage/'checkpoint.pt'),'--long-hold','--speed-grid','-.05','.05','--seeds','225161','225167','--output',name],check=True)
gpu=json.loads((OUT/name).read_text())
gstats={}
for label in ('baseline','candidate'):
    rows=[r for r in gpu['results'] if r['policy']==label]
    m=np.concatenate([r['metrics'] for r in rows]);stop=np.concatenate([r['stop'] for r in rows])
    steps=np.concatenate([r['stepCount'] for r in rows]).sum(-1)
    started=((m[:,:,0]*m[:,:,1]>0)&(np.abs(m[:,:,1])>.03)&(steps[:,None]>=8))
    gstats[label]=dict(vxError=float(np.abs(m[:,:,1]-m[:,:,0]).mean()),yawError=float(np.abs(m[:,:,3]-m[:,:,2]).mean()),started=int(started.sum()),total=int(started.size),survived=all(all(r['survival']) for r in rows),maxStop=stop.max(0).tolist())

def cpu(policy,destination):
    sha=hashlib.sha256(policy.read_bytes()).hexdigest()
    cached=destination/'cpu-body-tracking.json'
    if cached.exists():
        assert (destination/'actor.sha256').read_text()==sha
        return json.loads(cached.read_text())['results']
    for yaw in (-.6,0.,.6):
        subprocess.run([sys.executable,'scripts/evaluate_walk_runtime_cpu.py','--onnx',str(policy),'--physics-profile','kissqy','--output',str(destination/f'cpu-yaw{yaw:g}'),'--speeds','-.05','.05','--seeds','225171','225177','--trials','1','--seconds','30','--transport-ms','0','--turn-rate',str(yaw)],check=True)
    subprocess.run([sys.executable,'scripts/analyze_walk_runtime.py',str(destination),'--seconds','30','--window','17','27'],check=True)
    (destination/'actor.sha256').write_text(sha)
    return json.loads(cached.read_text())['results']
cstats={};cpu_rows={}
for label,policy,folder in [('baseline',ROOT/'inputs/student.onnx',OUT/'startup-cpu-baseline'),('candidate',a.stage/'policy.onnx',a.stage/'startup-cpu')]:
    rows=cpu(policy,folder);cpu_rows[label]=rows
    cstats[label]=dict(vxError=float(np.mean([r['rolling1sMAE'][0] for r in rows])),yawError=float(np.mean([r['rolling1sMAE'][1] for r in rows])),started=sum(r['meanCommand'][0]*r['meanActual'][0]>0 and abs(r['meanActual'][0])>.03 and r['stepCount']>=4 for r in rows),total=len(rows),survived=all(r['completed'] for r in rows),maxStop=max(r['stopMeanHorizontalSpeed'] for r in rows))

def improved(stats,increment):
    b,c=stats['baseline'],stats['candidate']
    return c['survived'] and c['vxError']<b['vxError']*.8 and c['yawError']<=b['yawError']+.03 and c['started']>=min(c['total'],b['started']+increment)
def phase_bias(case,label):
    m=np.asarray(case[label]['phases'])
    return np.abs(m[:,[1,3]]-m[:,[0,2]]).mean(0)
gpu_directions=all(np.all(phase_bias(c,'candidate')<=phase_bias(c,'baseline')+[.005,.03])
    for comparison in gpu['comparisons'] for c in comparison['perCase'] if c['paired'])
assert [r['source'] for r in cpu_rows['baseline']]==[r['source'] for r in cpu_rows['candidate']]
cpu_directions=all(c['rolling1sMAE'][0]<=b['rolling1sMAE'][0]+.005 and c['rolling1sMAE'][1]<=b['rolling1sMAE'][1]+.05 for b,c in zip(cpu_rows['baseline'],cpu_rows['candidate']))
gpu_ok=gpu_directions and improved(gstats,4) and all(c<=limit for c,limit in zip(gstats['candidate']['maxStop'],[.015,.08]))
cpu_ok=cpu_directions and improved(cstats,2) and cstats['candidate']['maxStop']<=.01
result=dict(gpuDirectionsPassed=bool(gpu_directions),cpuDirectionsPassed=bool(cpu_directions),gpu=gstats,cpu=cstats,gpuPassed=gpu_ok,cpuPassed=cpu_ok,continueTraining=bool(gpu_ok and cpu_ok),score=gstats['candidate']['vxError']+cstats['candidate']['vxError'],finalLargeStrideAccepted=False)
(a.stage/'startup-screen.json').write_text(json.dumps(result,indent=2));print(result)

"""Offline CPU HOME rehearsal; never connects to the robot or changes hardware."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
preview=root/'scripts/preview_hd1910.py'
policy=root.parent/'microduck-observer/debug-server/models/hd1910-head-v5.onnx'
sys.argv=[str(preview),'--headless','--policy',str(policy)]
source=preview.read_text(encoding='utf-8').split('\nreset()\nprint(')[0]
source=source.replace('    global last_action,last_vel\n    rotation=', '    global last_action,last_vel,last_target,last_obs\n    rotation=')
source=source.replace('    last_vel=data.qvel[vadr].copy()', '    last_obs=obs.copy()\n    last_vel=data.qvel[vadr].copy()')
source=source.replace('    for _ in range(10):\n        history.append', '    last_target=target.copy()\n    for _ in range(10):\n        history.append')
ns={'__file__':str(preview),'__name__':'offline_preview'}
exec(compile(source,str(preview),'exec'),ns)
results=[]
for name,reference,clamp in [('fixed_HOME',True,False),('policy_training_targets',False,False),('policy_hardware_targets',False,True)]:
 ns['a'].reference_pose=reference;ns['a'].official_targets=clamp
 ns['reset']();trace=[]
 for tick in range(1000):
  tilt=ns['step']();data=ns['data'];q= data.qpos[ns['qadr']].copy()
  trace.append(dict(time=float(data.time),tiltDeg=tilt,height=float(data.qpos[2]),position=q.tolist(),target=ns['last_target'].tolist(),observation=ns['last_obs'].tolist()))
 tail=trace[-250:];positions=np.array([x['position'] for x in tail]);targets=np.array([x['target'] for x in tail])
 first=trace[0]
 summary=dict(name=name,seconds=20,finite=bool(np.isfinite(positions).all()),maxTiltDeg=max(x['tiltDeg'] for x in trace),tailMaxTiltDeg=max(x['tiltDeg'] for x in tail),minHeight=min(x['height'] for x in trace),endHeight=trace[-1]['height'],firstTargetDeltaDeg=np.degrees(np.array(first['target'])-ns['home']).tolist(),tailPositionSpanDeg=np.degrees(np.ptp(positions,axis=0)).tolist(),tailTargetSpanDeg=np.degrees(np.ptp(targets,axis=0)).tolist(),tailMeanHomeErrorDeg=np.degrees(np.mean(positions-ns['home'],axis=0)).tolist())
 results.append(summary)
 print(json.dumps(summary),flush=True)
 out=root.parent/'microduck-observer/docs/实测记录/附件/HAT/2026-10-04/home-simulation'
 out.mkdir(parents=True,exist_ok=True)
 (out/(name+'.json')).write_text(json.dumps(trace),encoding='utf-8')
report=dict(hardwareAccess=False,policySha256=hashlib.sha256(policy.read_bytes()).hexdigest(),jointNames=ns['meta']['jointOrder'],physicsHz=500,controlHz=50,bamKp=5,bamVin=7.4,targetDelayPhysicsSteps=4,targetDelayMs=8,bodyGroundMeshes=ns['body_contacts'],note='CPU BAM rehearsal, not full randomized CUDA training environment or physical acceptance',results=results)
(out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

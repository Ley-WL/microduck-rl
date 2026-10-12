"""Read-only CPU walk rehearsal: no network or hardware access."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
preview=root/'scripts/preview_hd1910.py'
model_path=root/'artifacts/hd1910-walk-v6-symmetry500/hd1910-walk-v6-symmetry500.onnx'
sys.path.insert(0,str(preview.parent))
sys.argv=[str(preview),'--headless','--policy',str(model_path),'--official-targets','--walk-speed','.2']
src=preview.read_text(encoding='utf-8').split('\nreset()\nprint(')[0]
src=src.replace('    global last_action,last_vel','    global last_action,last_vel,filtered_action,filtered_gyro,last_target')
src=src.replace('    obs=np.concatenate((data.qvel[3:6],gravity,data.qpos[qadr]-home,last_vel,last_action,command)).astype(np.float32)\n    last_vel=data.qvel[vadr].copy()', '    last_vel=.4**2*last_vel+(1-.4**2)*data.qvel[vadr]\n    filtered_gyro=.5**2*filtered_gyro+(1-.5**2)*data.qvel[3:6]\n    obs=np.concatenate((filtered_gyro,gravity,data.qpos[qadr]-home,last_vel,last_action,command)).astype(np.float32)')
src=src.replace('    target=home+action','    filtered_action=action_alpha*filtered_action+(1-action_alpha)*action\n    target=home+filtered_action')
src=src.replace('    for _ in range(10):\n        history.append','    last_target=target.copy()\n    for _ in range(10):\n        history.append')
ns={'__file__':str(preview),'__name__':'offline_walk'};exec(compile(src,str(preview),'exec'),ns)
out=root/'artifacts/hd1910-walk-v6-symmetry500/runtime-ema-rehearsal-20261004';out.mkdir(exist_ok=True)
results=[]
for gain,alpha in [(5,0.),(5,.45),(6,0.),(6,.45)]:
 ns['bam'].actuator.kp=gain;ns['action_alpha']=alpha;ns['filtered_action']=np.zeros(14,dtype=np.float32);ns['filtered_gyro']=np.zeros(3);ns['reset']();trace=[]
 for tick in range(1000):
  tilt=ns['step']();d=ns['data'];rot=d.xmat[ns['model'].body('trunk_base').id].reshape(3,3)
  trace.append(dict(time=float(d.time),tilt=tilt,height=float(d.qpos[2]),xy=d.qpos[:2].tolist(),yaw=float(np.arctan2(rot[1,0],rot[0,0])),q=d.qpos[ns['qadr']].tolist(),target=ns['last_target'].tolist(),speed=float(np.linalg.norm(d.qvel[:2]))))
 moving=trace[100:750];stop=trace[850:];q=np.array([v['q'] for v in moving]);yaw=np.unwrap([v['yaw'] for v in moving]);xy=np.array([v['xy'] for v in moving]);r=dict(bamKp=gain,actionAlpha=alpha,maxTiltDeg=max(v['tilt'] for v in trace),minHeight=min(v['height'] for v in trace),endHeight=trace[-1]['height'],movingTiltP95=float(np.percentile([v['tilt'] for v in moving],95)),headingDriftP95Deg=float(np.percentile(np.abs(np.degrees(yaw-yaw[0])),95)),forwardTravelM=float(xy[-1,0]-xy[0,0]),crossTrackMaxM=float(np.max(np.abs(xy[:,1]-xy[0,1]))),neckHomeMeanErrorDeg=float(np.mean(np.degrees(q[:,5]-ns['home'][5]))),stopSpeedP95=float(np.percentile([v['speed'] for v in stop],95)),finite=bool(np.isfinite(q).all()))
 results.append(r);print(json.dumps(r),flush=True);(out/f'trace-kp{gain}-alpha{alpha}.json').write_text(json.dumps(trace),encoding='utf-8')
(out/'summary.json').write_text(json.dumps(dict(hardwareAccess=False,policySha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),seconds=20,walkStart=2,walkEnd=15,command=.2,physicsHz=500,controlHz=50,results=results,note='Nominal CPU BAM, official target saturation; gyro/velocity EMA time-scaled at 50Hz using exact simulator velocities. No HD register quantization, firmware PID/D/current mapping, domain randomization or hardware validation.'),indent=2),encoding='utf-8')

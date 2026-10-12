"""CPU walking/pitch response on the local mesh; no physical PD equivalence."""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
preview=ROOT/'scripts/preview_xgoduck.py'
source=preview.read_text(encoding='utf-8').split('\nreset()\n')[0]
source=source.replace('a=p.parse_args()', 'a=p.parse_args(["--headless"])')
ns={'__file__':str(preview),'__name__':'pitch_probe'}
exec(compile(source,str(preview),'exec'),ns)
ns['a'].headless=False
mj=ns['mujoco'];model=ns['model'];data=ns['data'];trunk=model.body('trunk_base').id
output=ROOT/'artifacts/xgoduck-pitch-20261010';output.mkdir(exist_ok=True)
results=[]
cases=[(-0.4,0),(-0.4,10),(-0.4,-10)] if "--backward" in sys.argv else [(0.1,0),(0.2,0),(0.4,0),(0.2,10),(0.2,-10)]
for speed,perturb in cases:
    ns['reset']();trace=[]
    for tick in range(1000):
        t=tick*.02
        ns['velocity_command'][0]=speed if 2<=t<12 else 0
        if perturb and tick==300:
            q=np.array([math.cos(math.radians(perturb)/2),0,math.sin(math.radians(perturb)/2),0])
            quat=np.empty(4);mj.mju_mulQuat(quat,q,data.qpos[3:7].copy());data.qpos[3:7]=quat;mj.mj_forward(model,data)
        ns['step']()
        r=data.xmat[trunk].reshape(3,3)
        pitch=math.degrees(math.asin(np.clip(-r[2,0],-1,1)))
        velocity=r.T@data.qvel[:3]
        trace.append({'time':t,'pitchDeg':pitch,'tiltDeg':math.degrees(math.acos(np.clip(r[2,2],-1,1))),'vx':float(velocity[0]),'height':float(data.qpos[2]),'x':float(data.qpos[0]),'command':float(ns['velocity_command'][0]),'targetRadians':(ns['home']+ns['filtered']).tolist()})
    moving=[r for r in trace if 3<=r['time']<12];tail=trace[-100:]
    result={'speedCommand':speed,'pitchPerturbAt6sDeg':perturb,'maxWalkAbsPitchDeg':max(abs(r['pitchDeg']) for r in moving),'meanWalkVx':float(np.mean([r['vx'] for r in moving])),'maxTiltDeg':max(r['tiltDeg'] for r in trace),'finalMeanPitchDeg':float(np.mean([r['pitchDeg'] for r in tail])),'finalMeanSpeed':float(np.mean([abs(r['vx']) for r in tail])),'finite':bool(np.isfinite(data.qpos).all())}
    if perturb:
        after=[r for r in trace if 6<=r['time']<12]
        result['firstAbsPitchUnder3DegAfterPerturbSeconds']=next((r['time']-6 for r in after if abs(r['pitchDeg'])<3),None)
    name=f'walk-{speed}-pitch-{perturb}'
    (output/(name+'.json')).write_text(json.dumps({'result':result,'trace':trace}),encoding='utf-8')
    results.append(result);print(json.dumps(result),flush=True)
(output/('summary-backward.json' if '--backward' in sys.argv else 'summary.json')).write_text(json.dumps({'robot':'local replica mesh + HD1910 BAM kp5/7.4V','policy':str(ns['a'].policy),'note':'Perfect simulated sensing; abrupt quaternion perturbation is synthetic; physical P7/D15 cannot map directly to BAM kp5. Command is explicitly zeroed at 12s.','results':results},indent=2),encoding='utf-8')

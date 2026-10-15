"""Body-frame tracking and actual foot travel from frozen CPU rollout files."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
import kissqy_walk_physics as physics

p=argparse.ArgumentParser();p.add_argument('root',type=Path)
p.add_argument('--window',type=float,nargs=2,default=[5.,9.])
p.add_argument('--seconds',type=float,default=12.)
a=p.parse_args()
assert 1 <= a.window[0] < a.window[1] <= a.seconds-3
ROOT=Path(__file__).resolve().parents[1]
spec=mujoco.MjSpec.from_file(str(ROOT/'src/mjlab_microduck/robot/replica/scene_walk_backlash.xml'))
physics.apply(spec);model=spec.compile();data=mujoco.MjData(model)
feet=[model.site(n).id for n in ['left_foot','right_foot']];results=[]
for path in sorted(a.root.glob('cpu-*/seed*.json')):
    frames=json.loads(path.read_text())['frames'];t=np.array([f['time'] for f in frames])
    q=np.array([f['qpos'][3:7] for f in frames]);v=np.array([f['velocity'][:3] for f in frames]);w,x,y,z=q.T
    vx=(1-2*(y*y+z*z))*v[:,0]+2*(x*y+w*z)*v[:,1]+2*(x*z-w*y)*v[:,2]
    angle=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)));yaw=np.gradient(angle,t)
    command=np.array([f['command'] for f in frames]);moving=(t>=a.window[0])&(t<a.window[1]);stopped=t>=a.seconds-2.5
    actual=np.stack((vx,yaw),-1);desired=command[:,[0,2]]
    rolling=np.stack([np.convolve(actual[:,i]-desired[:,i],np.ones(50)/50,mode='valid') for i in range(2)],-1)
    window=(t[49:]>=a.window[0]+1)&(t[49:]<a.window[1])
    previous=[True,True];lift=[None,None];steps=[]
    for frame in frames:
        data.qpos[:]=frame['qpos'];mujoco.mj_forward(model,data)
        for i,name in enumerate(['left_foot_collision','right_foot_collision']):
            contact=any(name in pair and 'floor' in pair for pair in frame['contacts'])
            if previous[i] and not contact:
                w,x,y,z=frame['qpos'][3:7]
                direction=np.array([1-2*(y*y+z*z),2*(x*y+w*z)])*np.sign(frame['command'][0])
                lift[i]=(frame['time'],data.site_xpos[feet[i],:2].copy(),direction)
            if contact and not previous[i] and lift[i] is not None:
                start,xy,direction=lift[i]
                if a.window[0]<=start<a.window[1] and frame['time']<a.window[1] and frame['time']-start>=.04:
                    steps.append(float(np.dot(data.site_xpos[feet[i],:2]-xy,direction)))
                lift[i]=None
            previous[i]=contact
    results.append(dict(source=str(path.relative_to(a.root)),completed=len(frames)==round(a.seconds/.02),
        maxTiltDeg=max(f['tilt'] for f in frames),meanCommand=desired[moving].mean(0).tolist(),
        meanActual=actual[moving].mean(0).tolist(),rolling1sMAE=np.abs(rolling[window]).mean(0).tolist(),
        rolling1sP95=np.quantile(np.abs(rolling[window]),.95,axis=0).tolist(),
        stopMeanHorizontalSpeed=float(np.linalg.norm(v[stopped,:2],axis=1).mean()),
        stepDistance=float(np.mean(steps)) if steps else None,stepCount=len(steps)))
output=dict(simulationOnly=True,hardwareValidated=False,windowSeconds=a.window,seconds=a.seconds,rollingSeconds=1.,results=results)
(a.root/'cpu-body-tracking.json').write_text(json.dumps(output,indent=2))
for name in sorted({r['source'].split('\\')[0].split('/')[0] for r in results}):
    for sign in (-1, 0, 1):
        rows=[r for r in results if r['source'].startswith(name+'/') or r['source'].startswith(name+'\\')]
        rows=[r for r in rows if int(np.sign(r['meanCommand'][0])) == sign]
        if not rows: continue
        steps=[r['stepDistance'] for r in rows if r['stepDistance'] is not None]
        print(name,'direction',sign,'command',np.round(np.mean([r['meanCommand'] for r in rows],0),4),
            'actual vx/yaw',np.round(np.mean([r['meanActual'] for r in rows],0),4),
            'rolling errors',np.round(np.mean([r['rolling1sMAE'] for r in rows],0),4),
            'step',round(float(np.mean(steps)),4) if steps else None)

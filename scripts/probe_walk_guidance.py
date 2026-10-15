"""Offline teacher-command feasibility probes, never deployment or calibration."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import mujoco
import kissqy_walk_physics as physics

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--onnx',type=Path,required=True)
p.add_argument('--wide',action='store_true',help='Diagnostic only: includes extrapolated teacher yaw commands')
p.add_argument('--seeds',type=int,nargs='+',default=[191101])
a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
grid=([(yaw,[-.35,-.2,0.,.18,.3]) for yaw in [-1.2,-.8,-.4,0.,.4,.8,1.2,1.6]] if a.wide
      else [(.2,[.3,-.35]),(.4,[.3,-.35]),(.6,[.3,-.35]),(1.2,[0.]),(-1.2,[0.])])
for yaw,speeds in grid:
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/evaluate_walk_runtime_cpu.py'),
        '--onnx',str(a.onnx),'--physics-profile','kissqy','--output',str(a.output/f'yaw{yaw:g}'),
        '--speeds',*[str(v) for v in speeds],'--seeds',*[str(seed) for seed in a.seeds],'--trials','1','--seconds','12',
        '--turn-rate',str(yaw)],check=True)
spec=mujoco.MjSpec.from_file(str(ROOT/'src/mjlab_microduck/robot/replica/scene_walk_backlash.xml'))
physics.apply(spec);model=spec.compile();data=mujoco.MjData(model)
feet=[model.site(n).id for n in ['left_foot','right_foot']]
output=[]
for path in a.output.glob('yaw*/seed*.json'):
    frames=json.loads(path.read_text())['frames'];selected=[f for f in frames if 5<=f['time']<9]
    q=np.array([f['qpos'][3:7] for f in selected]);v=np.array([f['velocity'][:3] for f in selected])
    w,x,y,z=q.T;body_vx=(1-2*(y*y+z*z))*v[:,0]+2*(x*y+w*z)*v[:,1]+2*(x*z-w*y)*v[:,2]
    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    lift=[None,None];previous=[True,True];steps=[]
    for frame in frames:
        data.qpos[:]=frame['qpos'];mujoco.mj_forward(model,data)
        for i,name in enumerate(['left_foot_collision','right_foot_collision']):
            contact=any(name in pair and 'floor' in pair for pair in frame['contacts'])
            if previous[i] and not contact:
                quat=np.array(frame['qpos'][3:7]);w,x,y,z=quat
                direction=np.array([1-2*(y*y+z*z),2*(x*y+w*z)])*np.sign(frame['command'][0])
                lift[i]=(frame['time'],data.site_xpos[feet[i],:2].copy(),direction)
            if contact and not previous[i] and lift[i] is not None:
                t,xy,direction=lift[i]
                if 5<=t<9 and .04<=frame['time']-t and abs(frame['command'][0])>.03:
                    steps.append(float(np.dot(data.site_xpos[feet[i],:2]-xy,direction)))
                lift[i]=None
            previous[i]=contact
    output.append(dict(source=str(path.relative_to(a.output)),command=np.mean([f['command'] for f in selected],axis=0).tolist(),
        bodyForward=float(body_vx.mean()),yawRate=float(np.diff(yaw).mean()/.02),
        completed=len(frames)==600,maxTilt=max(f['tilt'] for f in frames),
        stepDistance=float(np.mean(steps)) if steps else None,stepCount=len(steps)))
(a.output/'guidance-feasibility.json').write_text(json.dumps(dict(simulationOnly=True,
    calibrationApplied=False,training=False,seeds=a.seeds,results=output),indent=2))
print(json.dumps(output,indent=2))

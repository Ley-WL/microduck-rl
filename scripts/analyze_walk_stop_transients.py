"""Resolve complete stopping events from optional GPU motion traces."""
import argparse,json
from pathlib import Path
import numpy as np

def analyze(values,dt):
    assert values.ndim==3 and values.shape[-1]==8 and np.isfinite(values).all()
    out=[]
    for world in range(values.shape[1]):
        v=values[:,world];moving=np.linalg.norm(v[:,3:6],axis=1)>.001
        transitions=np.flatnonzero(moving[:-1]&~moving[1:])+1
        for start in transitions:
            next_move=np.flatnonzero(moving[start:]);end=start+int(next_move[0]) if len(next_move) else len(v)
            stop=v[start:end];speed=np.linalg.norm(stop[:,:2],axis=1);yaw=np.abs(stop[:,2]);duration=len(stop)*dt
            if duration<1.:continue
            good=(speed<.01)&(yaw<.1)&(stop[:,7]>.5)
            width=round(.5/dt);windows=np.convolve(good.astype(int),np.ones(width,dtype=int),mode='valid')
            ids=np.flatnonzero(windows==width)
            out.append(dict(world=world,startSeconds=(start+1)*dt,duration=duration,
                survived=bool(np.all(stop[:,7]>.5)),settleSeconds=float(ids[0]*dt) if len(ids) else None,
                firstSecondTravel=float(speed[:round(1/dt)].sum()*dt),
                tailHorizontal=float(speed[round(1/dt):].mean()),tailYaw=float(yaw[round(1/dt):].mean()),maxTilt=float(stop[:,6].max())))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();results={}
    for path in sorted(a.root.glob('*-motion.npz')):
        d=np.load(path);assert d['columns'].tolist()==['vx','vy','yaw','commandVx','commandVy','commandYaw','tilt','live']
        results[path.name]=analyze(d['values'],float(d['dt']))
    assert results,'No motion traces found'
    (a.root/'stop-transients.json').write_text(json.dumps(dict(simulationOnly=True,speedThreshold=.01,yawThreshold=.1,holdSeconds=.5,results=results),indent=2))
    for name,rows in results.items():
        settled=[r['settleSeconds'] for r in rows if r['settleSeconds'] is not None]
        print(name,'events',len(rows),'settled',len(settled),'mean settle',float(np.mean(settled)) if settled else None,'tail speed',float(np.mean([r['tailHorizontal'] for r in rows])))

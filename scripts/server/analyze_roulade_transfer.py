"""CPU assessment of cached screens; stricter one-roll stable-exit gate.

This is independent postprocessing, never a training update or hardware action.
"""
import json,math,argparse,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
folders={'Baseline':'baseline-screen-v3','Control':'control-screen-v3','Repair':'repair-screen-v3'}
reports={};sources={}
for name,folder in folders.items():
    rows=[]
    for path in sorted((a.evidence/folder).glob('*.npz')):
        sources[str(path.relative_to(a.evidence))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as f:
            stable=(f['tilt']<20)&(f['height']>.095)&(f['accum']>=math.radians(330))&f['head']
            stable &= f['accum']<=math.radians(420)
            run=np.zeros(stable.shape[1],dtype=int);times=np.full(stable.shape[1],np.inf)
            for step in range(len(stable)):
                run=np.where(stable[step],run+1,0)
                new=(run>=10)&~np.isfinite(times);times[new]=(step+1)*.02
            # Reaching upright for a moment followed by falling/re-rolling fails.
            success=(np.isfinite(times)&(stable[-15:].mean(0)>=.95)&
                (f['accum'].max(0)<math.radians(450))&~f['done'].any(0))
            times[~success]=np.inf
            delta=np.degrees(abs(np.diff(f['targets'],axis=0)))
            acceleration=np.linalg.norm(np.diff(f['velocity'],axis=0)/.02,axis=2)
            stopped=(abs(f['gyro'][:,:,1])<.5)&(f['accum']>math.radians(30))&(f['accum']<math.radians(330))
            rows.append(dict(file=path.name,initial=f['initialQpos'],success=success,times=times,
                targetFrameMax=float(delta.max()),targetFrameP95=float(np.quantile(delta,.95)),
                accelerationMax=float(acceleration.max()),meanSlowTransitSeconds=float(stopped.sum(0).mean()*.02)))
    assert len(rows)==4, 'Incomplete screen; do not draw acceptance conclusions'
    reports[name]=rows

def summary(rows):
    times=np.concatenate([r['times'] for r in rows]);s=np.concatenate([r['success'] for r in rows])
    return dict(successes=int(s.sum()),trials=len(s),within1s=int((times<=1).sum()),within2s=int((times<=2).sum()),
        meanCompletionSeconds=float(times[s].mean()) if s.any() else None,
        targetFrameMaxDeg=max(r['targetFrameMax'] for r in rows),
        targetFrameP95Deg=max(r['targetFrameP95'] for r in rows),
        baseAccelerationMax=max(r['accelerationMax'] for r in rows),
        meanSlowTransitSeconds=float(np.mean([r['meanSlowTransitSeconds'] for r in rows])))

def compare(before,after):
    lost=0;btime=[];atime=[]
    for b,c in zip(before,after):
        assert b['file']==c['file'] and np.array_equal(b['initial'],c['initial'])
        lost+=int((b['success']&~c['success']).sum())
        pair=b['success']&c['success'];btime.extend(b['times'][pair]);atime.extend(c['times'][pair])
    gain=(1-np.mean(atime)/np.mean(btime))*100 if len(btime) else None
    bs=summary(before);cs=summary(after)
    return dict(lostSuccesses=lost,pairedCompletionImprovementPercent=float(gain) if gain is not None else None,
        frameMaxRatio=cs['targetFrameMaxDeg']/max(bs['targetFrameMaxDeg'],1e-8),
        accelerationMaxRatio=cs['baseAccelerationMax']/max(bs['baseAccelerationMax'],1e-8))

result={name:summary(rows) for name,rows in reports.items()}
result['RepairVsBaseline']=compare(reports['Baseline'],reports['Repair'])
result['RepairVsControl']=compare(reports['Control'],reports['Repair'])
checks=[result[k] for k in ('RepairVsBaseline','RepairVsControl')]
result['eligibleForFreshSeeds']=all(r['lostSuccesses']==0 and r['pairedCompletionImprovementPercent'] is not None
    and r['pairedCompletionImprovementPercent']>=5 and r['frameMaxRatio']<=1.05 and r['accelerationMaxRatio']<=1.05 for r in checks)
result.update(sourceHashes=sources,formalAccepted=False,deployment=False,simulationOnly=True,
    note='Strict single-roll completion plus >=95% final300ms stable; impact/target statistics are simulation proxies')
(a.evidence/'stable-exit-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

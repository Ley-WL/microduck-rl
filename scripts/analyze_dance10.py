"""Compare frozen rollout traces without treating stationary survival as dancing."""
import argparse
import json
from pathlib import Path
import numpy as np


def motion_metrics(folder):
    summary=json.loads((folder/'summary.json').read_text())
    trace=np.load(folder/'trace.npz')
    successful=np.array(summary['firstFailureSeconds'])<0
    # A failed trajectory stays failed even when the simulator automatically resets.
    active=(trace['time']%8>=1)&(trace['time']%8<=7)
    head=trace['head'][active]
    target=trace['headCommand'][active]
    amplitude=np.ptp(head[:,:,2],axis=0)
    nod=np.ptp(head[:,:,1],axis=0)
    error=np.sqrt(np.mean((head-target)**2,axis=(0,2)))
    x=head[:,:,2]-head[:,:,2].mean(axis=0)
    y=target[:,:,2]-target[:,:,2].mean(axis=0)
    correlation=np.sum(x*y,axis=0)/np.maximum(1e-8,np.sqrt(np.sum(x*x,axis=0)*np.sum(y*y,axis=0)))
    dance=successful & (amplitude>=.40) & (nod>=.15) & (error<=.08) & (correlation>=.85)
    return dict(worlds=len(successful),survived=int(successful.sum()),
        danceScreenPassed=int(dance.sum()),survival=successful.tolist(),
        headYawAmplitudeRad=amplitude.tolist(),nodAmplitudeRad=nod.tolist(),
        headRmseRad=error.tolist(),yawCommandCorrelation=correlation.tolist(),
        accepted=False,note='Diagnostic thresholds fixed before candidate results; not hardware acceptance.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();results=[]
    for summary in sorted(a.baseline.glob('*/summary.json')):
        name=summary.parent.name
        baseline=motion_metrics(summary.parent);candidate=motion_metrics(a.candidate/name)
        b=np.array(baseline['survival']);c=np.array(candidate['survival'])
        results.append(dict(profile=name,baseline=baseline,candidate=candidate,
            lostWorlds=np.flatnonzero(b&~c).tolist(),gainedWorlds=np.flatnonzero(~b&c).tolist()))
    a.output.write_text(json.dumps(results,indent=2))
    for r in results:
        print(r['profile'],r['baseline']['survived'],'->',r['candidate']['survived'],
              'dance',r['candidate']['danceScreenPassed'],'lost',r['lostWorlds'])


if __name__=='__main__':main()

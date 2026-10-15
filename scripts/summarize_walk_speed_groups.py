"""Report actual tracking, startup and foot travel by speed, retaining failures."""
import argparse
import json
from pathlib import Path
import numpy as np


def gpu_groups(data):
    speeds=np.repeat(np.abs(np.array(data['cases'])[:,0]),4)
    result={}
    for label in ('baseline','candidate'):
        rows=[r for r in data['results'] if r['policy']==label]
        groups={}
        for speed in np.unique(speeds):
            ids=np.isclose(speeds,speed)
            m=np.concatenate([np.array(r['metrics'])[ids] for r in rows])
            live=np.concatenate([np.array(r['survival'])[ids] for r in rows])
            count=np.concatenate([np.array(r['stepCount'])[ids].sum(-1) for r in rows])
            stop=np.concatenate([np.array(r['stop'])[ids] for r in rows])
            stride=np.concatenate([np.array(r['stepDistance'])[ids] for r in rows])
            air=np.concatenate([np.array(r['stepAirTime'])[ids] for r in rows])
            started=(m[:,:,0]*m[:,:,1]>0)&(np.abs(m[:,:,1])>.03)&(count[:,None]>=8)&live[:,None]
            groups[str(speed)]=dict(worlds=int(len(live)),survived=int(live.sum()),
                startedPhases=int(started.sum()),totalPhases=int(started.size),
                phaseBias=np.abs(m[:,:, [1,3]]-m[:,:,[0,2]]).mean((0,1)).tolist(),
                stopMean=stop.mean(0).tolist(),stopMax=stop.max(0).tolist(),
                footTravelMeanIncludingNoSteps=float(stride.mean()),airTimeMean=float(air.mean()),
                completedFootfalls=int(count.sum()))
        result[label]=groups
    result['lost']={str(c['seed']):c['lost'] for c in data['comparisons']}
    return result


def cpu_groups(data):
    result={}
    for speed in sorted(set(round(abs(r['meanCommand'][0]),4) for r in data['results'])):
        rows=[r for r in data['results'] if abs(abs(r['meanCommand'][0])-speed)<1e-5]
        result[str(speed)]=dict(trajectories=len(rows),survived=sum(r['completed'] for r in rows),
            started=sum(r['meanCommand'][0]*r['meanActual'][0]>0 and abs(r['meanActual'][0])>.03
                        and r['stepCount']>=4 and r['completed'] for r in rows),
            rollingError=np.mean([r['rolling1sMAE'] for r in rows],0).tolist(),
            stopMean=float(np.mean([r['stopMeanHorizontalSpeed'] for r in rows])),
            footTravelMeanIncludingNoSteps=float(np.mean([r['stepDistance'] or 0. for r in rows])),
            straight=[r for r in rows if abs(r['meanCommand'][1])<.001])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sources',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report={'diagnosticOnly':True,'accepted':False,'sources':{}}
    for path in a.sources:
        data=json.loads(path.read_text(encoding='utf-8'))
        report['sources'][str(path)]=gpu_groups(data) if 'cases' in data else cpu_groups(data)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

"""Compare recorded-input replay and independent closed-loop screens."""
import argparse,json
from pathlib import Path
import numpy as np
import onnxruntime as ort


def replay(path,trace):
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    meta=json.loads(path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
    home=np.array(meta['homeRadians']);bounds=np.array([meta['trainingTargetClipRadians'][n] for n in meta['jointOrder']])
    result=[]
    for kind in ('sitstand_sit','sitstand_stand'):
        rows=[r for r in trace['frames'] if r['type']=='inference' and r['kind']==kind]
        if not rows:continue
        filtered=np.array(rows[0]['emaPreviousRadians'])
        targets=[];raw=[]
        for row in rows:
            obs=np.array(row['observation'],dtype=np.float32)
            action=session.run(None,{session.get_inputs()[0].name:obs[None]})[0][0]
            filtered=.45*filtered+.55*action
            if row['mono']-rows[0]['mono']<2.:
                raw.append(np.degrees(home+action)[[3,12]])
                targets.append(np.degrees(np.clip(home+filtered,bounds[:,0],bounds[:,1]))[[3,12]])
        raw=np.array(raw);targets=np.array(targets)
        result.append({'kind':kind,'samples':len(targets),
                       'rawKneeTravelDeg':np.abs(np.diff(raw,axis=0)).sum(axis=0).tolist(),
                       'filteredKneeTravelDeg':np.abs(np.diff(targets,axis=0)).sum(axis=0).tolist(),
                       'filteredKneeMaxFrameChangeDeg':np.abs(np.diff(targets,axis=0)).max(axis=0).tolist()})
    return result


def main(a):
    trace=json.loads(a.trace.read_text(encoding='utf-8'))
    result={'hardwareValidated':False,'replayScope':'Fixed recorded physical inputs including original prev_action; in-sample for repaired actors. Not a closed-loop hardware result.',
            'candidates':[]}
    for path in a.policies:
        tag='baseline' if path.stem=='single-actor-seed' else path.stem
        summary=json.loads((a.evaluation/tag/'summary.json').read_text(encoding='utf-8'))
        rows=[r for g in summary['groups'] for r in g['results']]
        transitions=[t for r in rows for t in r.get('transitionSmoothness',[]) if t['time']>0]
        result['candidates'].append({'name':tag,'recordedReplay':replay(path,trace),
            'closedLoopSuccesses':sum(r['success'] for r in rows),'closedLoopTrials':len(rows),
            'closedLoopCycleSuccesses':sum(r['success'] for r in rows if r['scenario'].endswith('cycle')),
            'closedLoopCycleTrials':sum(r['scenario'].endswith('cycle') for r in rows),
            'closedLoopMeanTransitionKneeTravelDeg':np.mean([t['targetTravelDeg'] for t in transitions],axis=0).tolist(),
            'closedLoopMeanTransitionKneeExcessTravelDeg':np.mean([t['excessTravelDeg'] for t in transitions],axis=0).tolist(),
            'accepted':summary['accepted']})
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('trace','evaluation','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--policies',type=Path,nargs='+',required=True)
    main(p.parse_args())

"""Counterfactual bias: average measured motion, compare with current command."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_walk_full_cycle_rewards import relative_bias


def current_command_bias(inputs, dt):
    mean = np.zeros_like(inputs[0, :, :3], dtype=np.float64)
    rates = []
    for frame in inputs:
        actual, command = frame[:, :3], frame[:, 3:]
        speed = np.linalg.norm(command[:, :2], axis=-1)
        tau = np.where(speed > .001, np.clip(.1/np.maximum(speed,.025),.7,4.),.7)
        mean += (actual-mean)*dt/(tau[:,None]+dt)
        scale = np.tile([.1,.1,.4],(len(frame),1))
        scale[:,0] = np.where(abs(command[:,0])>.001,np.clip(abs(command[:,0]),.025,.1),.1)
        rates.append(-4*np.minimum((((mean-command)/scale)**2).sum(-1),25.))
    return np.asarray(rates)


def replay(root):
    data=json.loads((root/'reward-audit.json').read_text())
    loaded={}; checks=[]
    for row in data['results']:
        with np.load(root/row['biasInputTrace']) as z:inputs=z['values'].astype(np.float64)
        with np.load(root/row['rewardTrace']) as z:
            rates=z['weightedRates'].astype(np.float64); dt=float(z['dt']); terms=z['terms'].tolist()
        keep=np.asarray(row['survival'],dtype=bool)
        old=relative_bias(inputs[:,keep],dt)
        np.testing.assert_allclose(old,rates[:,keep,terms.index('tracking_bias')],atol=2e-5,rtol=1e-5)
        altered=rates.sum(-1).copy()
        altered[:,keep]+=current_command_bias(inputs[:,keep],dt)-old
        loaded[row['seed'],row['policy']]=(row,inputs,rates.sum(-1),altered)
        checks.append(dict(seed=row['seed'],policy=row['policy'],verifiedWorlds=int(keep.sum())))
    pairs=[]; failed=[]
    windows=[('startForward',50,150),('walkForward',150,550),('startReverse',800,900),('walkReverse',900,1300)]
    for seed in sorted({k[0] for k in loaded}):
        b,c=loaded[seed,'baseline'],loaded[seed,'candidate']
        assert b[0]['initialStateSha256']==c[0]['initialStateSha256']
        for world,(blive,clive) in enumerate(zip(b[0]['survival'],c[0]['survival'])):
            if not(blive and clive):
                failed.append(dict(seed=seed,world=world,baseline=blive,candidate=clive));continue
            np.testing.assert_array_equal(b[1][:,world,3:],c[1][:,world,3:])
            for name,lo,hi in windows:
                scores={}; motion={}
                for label,s in [('baseline',b),('candidate',c)]:
                    actual=s[1][lo:hi,world,:3].mean(0);command=s[1][lo:hi,world,3:].mean(0)
                    motion[label]=dict(actual=actual.tolist(),command=command.tolist(),
                        vxError=float(abs(actual[0]-command[0])),yawError=float(abs(actual[2]-command[2])))
                    scores[label]=[float(s[k][lo:hi,world].mean()) for k in (2,3)]
                bv,cv=motion['baseline'],motion['candidate']
                winner=None
                if bv['vxError']+.01<cv['vxError'] and bv['yawError']<=cv['yawError']+.05:winner='baseline'
                elif cv['vxError']+.01<bv['vxError'] and cv['yawError']<=bv['yawError']+.05:winner='candidate'
                yaw_winner=None
                if bv['yawError']+.05<cv['yawError'] and bv['vxError']<=cv['vxError']+.01:yaw_winner='baseline'
                elif cv['yawError']+.05<bv['yawError'] and cv['vxError']<=bv['vxError']+.01:yaw_winner='candidate'
                pairs.append(dict(seed=seed,world=world,phase=name,motion=motion,scores=scores,
                    speedWinner=winner,yawWinner=yaw_winner))
    summary=[]
    for seed in sorted({r['seed'] for r in pairs}):
        for category in ('speedWinner','yawWinner'):
            for phase_group in ('start','walk'):
                eligible=[r for r in pairs if r['seed']==seed and r[category] and r['phase'].startswith(phase_group)]
                correct=[sum(r['scores'][r[category]][k]>r['scores']['candidate' if r[category]=='baseline' else 'baseline'][k] for r in eligible) for k in (0,1)]
                summary.append(dict(seed=seed,criterion=category,phase=phase_group,eligible=len(eligible),correctOldNew=correct))
    return dict(accepted=False,training=False,hypothesis='EMA(actual)-currentCommand replaces EMA(actual-command) only for tracking_bias',
        limitation='Same-trajectory counterfactual, not independent gait validation; other rewards and physical trajectories unchanged',
        checks=checks,failed=failed,summary=summary,pairs=pairs)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=replay(a.evidence)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['summary'],indent=2))

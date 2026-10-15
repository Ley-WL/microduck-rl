"""Compare frozen low-speed rewards on actual stalled/accurate paired rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
p.add_argument('--legacy',action='store_true',help='Read original reward scores on the very same trajectories')
a=p.parse_args();data=json.loads(a.source.read_text(encoding='utf-8'))
assert len(data['comparisons'])==len({r['seed'] for r in data['results']})
phases=(1,4);records=[]
for seed in sorted({r['seed'] for r in data['results']}):
    b,c=[next(r for r in data['results'] if r['seed']==seed and r['policy']==label) for label in ('baseline','candidate')]
    assert b['initialStateSha256']==c['initialStateSha256']
    assert b['commandTraceSha256']==c['commandTraceSha256']
    terms=b['rewardAudit']['terms'];assert terms==c['rewardAudit']['terms']
    for world in range(len(b['survival'])):
        for phase in phases:
            bb,cc=b['rewardAudit'],c['rewardAudit']
            cmd=np.array(bb['meanCommand'][world][phase])
            if abs(cmd[2])>.001 or abs(cmd[0])<.04:continue
            ba=np.array(bb['meanActual'][world][phase]);ca=np.array(cc['meanActual'][world][phase])
            assert np.allclose(cmd,cc['meanCommand'][world][phase],atol=1e-7)
            reward_key='meanLegacyWeightedRate' if a.legacy else 'meanWeightedRate'
            br=np.array(bb[reward_key][world][phase]);cr=np.array(cc[reward_key][world][phase])
            records.append(dict(seed=seed,world=world,phase=bb['phases'][phase],command=cmd.tolist(),
                baselineActual=ba.tolist(),teacherActual=ca.tolist(),
                bothSurvived=b['survival'][world] and c['survival'][world],
                fullWindow=bb['samples'][world][phase]>=299 and cc['samples'][world][phase]>=299,
                stalledBaseline=bool(abs(ba[0])<.01),
                accurateTeacher=bool(abs(ca[0]-cmd[0])<.02 and abs(ca[2]-cmd[2])<.15),
                baselineRate=dict(zip(terms,br.tolist())),teacherRate=dict(zip(terms,cr.tolist())),
                baselineTotal=float(br.sum()),teacherTotal=float(cr.sum())))
eligible=[r for r in records if r['bothSurvived'] and r['fullWindow']]
stalled=[r for r in eligible if r['stalledBaseline'] and r['accurateTeacher']]
def aggregate(rows):
    if not rows:return dict(count=0)
    rates={label:{term:float(np.mean([r[label+'Rate'][term] for r in rows])) for term in terms}
           for label in ['baseline','teacher']}
    return dict(count=len(rows),distinctWorlds=len({(r['seed'],r['world']) for r in rows}),
        meanRate=rates,meanTotal={label:sum(values.values()) for label,values in rates.items()},
        teacherHigher=sum(r['teacherTotal']>r['baselineTotal'] for r in rows))
result=dict(simulationOnly=True,optimizationSteps=0,teacherIsNotStudent=True,
    legacySameTrajectoryScores=a.legacy,
    rewardUnit='Mean weighted reward per simulated second, not discounted return',
    allStraight=aggregate(eligible),stalledToAccurate=aggregate(stalled),pairs=records,
    limitations='Matched initial states; no counterfactual same-state causal claim, no gradient or critic audit, no final gait acceptance.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))

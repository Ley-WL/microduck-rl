"""Explain reward preferences when a fine-tuned policy develops yaw bias."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('source',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
data=json.loads(a.source.read_text(encoding='utf-8'))
seeds=sorted({r['seed'] for r in data['results']})
assert len(data['comparisons'])==len(seeds)
records=[]
for seed in seeds:
    b,c=[next(r for r in data['results'] if r['seed']==seed and r['policy']==label)
         for label in ('baseline','candidate')]
    assert b['initialStateSha256']==c['initialStateSha256']
    assert b['commandTraceSha256']==c['commandTraceSha256']
    bb,cc=b['rewardAudit'],c['rewardAudit']
    terms=bb['terms'];assert terms==cc['terms']
    for world in range(len(b['survival'])):
        for phase in (1,4):
            cmd=np.array(bb['meanCommand'][world][phase])
            assert np.allclose(cmd,cc['meanCommand'][world][phase],atol=1e-7)
            if not (b['survival'][world] and c['survival'][world]):continue
            if min(bb['samples'][world][phase],cc['samples'][world][phase])<299:continue
            actual=[np.array(row['meanActual'][world][phase]) for row in (bb,cc)]
            errors=[np.abs(v-cmd) for v in actual]
            rates=[np.array(row['meanWeightedRate'][world][phase]) for row in (bb,cc)]
            records.append(dict(seed=seed,world=world,phase=bb['phases'][phase],
                command=cmd.tolist(),baselineActual=actual[0].tolist(),candidateActual=actual[1].tolist(),
                straight=bool(abs(cmd[2])<.001),
                yawRegressed=bool(errors[0][0]<.03 and errors[0][2]<.15 and errors[1][2]>errors[0][2]+.05),
                baselineRate=dict(zip(terms,rates[0].tolist())),
                candidateRate=dict(zip(terms,rates[1].tolist())),
                baselineTotal=float(rates[0].sum()),candidateTotal=float(rates[1].sum())))

def summarize(rows):
    if not rows:return dict(count=0)
    return dict(count=len(rows),candidateHigher=sum(r['candidateTotal']>r['baselineTotal'] for r in rows),
        meanRate={label:{term:float(np.mean([r[label+'Rate'][term] for r in rows])) for term in terms}
                  for label in ('baseline','candidate')},
        meanTotal={label:float(np.mean([r[label+'Total'] for r in rows])) for label in ('baseline','candidate')})

result=dict(simulationOnly=True,optimizationSteps=0,accepted=False,
    allMoving=summarize(records),allStraight=summarize([r for r in records if r['straight']]),
    yawRegressed=summarize([r for r in records if r['yawRegressed']]),pairs=records,
    limitations='Matched reset states and command sequence, not same-state counterfactuals or proof of PPO gradient cause. Scores are weighted reward rates, not discounted returns.')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))

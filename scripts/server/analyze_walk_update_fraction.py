"""Check targeted frozen update-fraction cases against locked initial states."""
import json,argparse
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];E=R/'evidence'
p=argparse.ArgumentParser();p.add_argument('--labels',nargs='+',default=['Fraction0.25','Fraction0.5','Fraction0.75'])
a=p.parse_args()
references=json.loads((R/'inputs/reference.json').read_text());results={}
for label in a.labels:
    rows=[];lost=[];repair_failed=[]
    for folder in E.glob(label+'-*'):
        if not folder.is_dir():continue
        terrain=folder.name[len(label)+1:]
        summary=json.loads((folder/'summary.json').read_text())
        for row in summary['results']:
            seed,trial,speed=row['seed'],row['trial'],row['speed']
            key=f'{terrain}/{seed}/{trial}/{speed:g}';ref=references[key]
            trace=json.loads((folder/f'seed{seed}-trial{trial}-speed{speed:g}.json').read_text())
            np.testing.assert_array_equal(trace['initialQpos'],ref['Baseline']['initialQpos'])
            np.testing.assert_array_equal(trace['frames'][0]['velocity'],ref['Baseline']['initialVelocity'])
            def forward(r):return r['completed'] and r['movingMeanVx'] is not None and r['movingMeanVx']>=max(.02,.3*speed)
            base=ref['Baseline']['result'];surface=ref['Surface']['result']
            if (base['completed'] and not row['completed']) or (forward(base) and not forward(row)):lost.append(key)
            if not forward(base) and forward(surface) and not forward(row):repair_failed.append(key)
            rows.append(dict(key=key,result=row,effectiveForward=forward(row),baseline=base,surface=surface))
    assert len(rows)==6
    results[label]=dict(cases=rows,lostBaselineSuccess=lost,lostSurfaceImprovement=repair_failed,
        targetedGatePassed=not lost and not repair_failed,accepted=False)
(E/'comparison.json').write_text(json.dumps(dict(results=results,initialStatesMatched=True,
    evaluationOptimizerSteps=0,hardwareAccess=False,accepted=False,
    boundary='18 cases selected using prior failure/improvement; fitted diagnostic, not independent acceptance'),indent=2))
print(json.dumps({p:{k:v for k,v in data.items() if k!='cases'} for p,data in results.items()}),flush=True)

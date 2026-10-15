"""Paired frozen14ms/16ms causality diagnostic; neither is measured real latency."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];E=R/'evidence'
ref=json.loads((R/'inputs/delay-reference.json').read_text());rows=[]
for folder in E.glob('*14ms-*'):
    if not folder.is_dir():continue
    label,terrain=folder.name.split('-',1)
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['manifest']['effectiveCommandDelaySteps']==7
    for result in summary['results']:
        seed,trial,speed=result['seed'],result['trial'],result['speed']
        key=f'{label}/{terrain}/{seed}/{trial}/{speed:g}';old=ref[key]
        trace=json.loads((folder/f'seed{seed}-trial{trial}-speed{speed:g}.json').read_text())
        np.testing.assert_array_equal(trace['initialQpos'],old['initialQpos'])
        np.testing.assert_array_equal(trace['frames'][0]['velocity'],old['initialVelocity'])
        np.testing.assert_array_equal(trace['frames'][0]['raw'],old['initialRaw'])
        def forward(r):return r['completed'] and r['movingMeanVx'] is not None and r['movingMeanVx']>=max(.02,.3*speed)
        rows.append(dict(key=key,old16ms=old['result'],new14ms=result,
            gainedSurvival=not old['result']['completed'] and result['completed'],
            lostSurvival=old['result']['completed'] and not result['completed'],
            gainedForward=not forward(old['result']) and forward(result),
            lostForward=forward(old['result']) and not forward(result)))
assert len(rows)==18
report=dict(rows=rows,initialStatesAndFirstRawMatched=True,optimizerSteps=0,hardwareAccess=False,accepted=False,
    boundary='CPU effective delay16->14ms only; same policy, same state and controller; not measured real UART/PD lag or fullCPU/GPU equivalence')
(E/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps({label:{name:sum(x[name] for x in rows if x['key'].startswith(label+'/'))
    for name in ('gainedSurvival','lostSurvival','gainedForward','lostForward')}
    for label in ('Baseline14ms','Surface14ms','Encoder14ms')}),flush=True)

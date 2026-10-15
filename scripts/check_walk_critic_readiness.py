"""Predeclared value-fit gate, never a gait or deployment acceptance gate."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('source',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
data=json.loads(a.source.read_text(encoding='utf-8'))
assert data['optimizationSteps']==0 and data['actorAndCriticUnchanged']
assert len(data['results'])==2
assert {r['seed'] for r in data['results']}=={235151,235157}
rows=[]
for result in data['results']:
    assert result['unobservedTailDiscount']<1e-6
    assert {p['phase'] for p in result['phases']}=={'stand','forward','stopFirst','reverse','stopSecond','fullCycle'}
    for phase in result['phases']:
        if phase['phase']=='stand':continue  # Initial zero command hides future command choice.
        assert phase['samples']>0 and phase['meanReturn']>0
        full=phase['phase']=='fullCycle'
        bias=abs(phase['bias'])/phase['meanReturn']
        rmse=phase['rmse']/phase['meanReturn']
        passed=bias<(0.2 if full else 0.25) and rmse<(0.3 if full else 0.35)
        if full:passed=passed and phase['explainedVariance'] is not None and phase['explainedVariance']>.3
        rows.append(dict(seed=result['seed'],phase=phase['phase'],biasFraction=bias,
            rmseFraction=rmse,explainedVariance=phase['explainedVariance'],passed=passed))
report=dict(passed=all(r['passed'] for r in rows),gaitAccepted=False,hardwareAccepted=False,
    sourceSha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
    checkpointSha256=data['checkpointSha256'],checks=rows,
    scope='Readiness to investigate actor refinement with this value initializer; not evidence of gait improvement.')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

"""Preserve paired outcomes of short slope-regularized controls."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]/'artifacts/sit-transfer-train-20261010'
def load(path):
    report=json.loads(path.read_text(encoding='utf-8'))
    rows={(g['scenario'],g['seed'],r['trial']):r for g in report['groups'] for r in g['results']}
    edges=[t for r in rows.values() for t in r['transitionSmoothness'] if t['time']>0]
    return rows,{'successes':sum(r['success'] for r in rows.values()),'trials':len(rows),
                'meanKneeExcessTravelDeg':np.mean([r['excessTravelDeg'] for r in edges],axis=0).tolist(),
                'p95MaxKneeFrameChangeDeg':np.percentile([r['maxFrameChangeDeg'] for r in edges],95,axis=0).tolist()}
baselines={k:load(root/('training-baseline-'+k)/'summary.json') for k in ('nominal','stress')}
out={'hardwareValidated':False,'baseline':{k:v[1] for k,v in baselines.items()},'candidates':[]}
for weight in (3,10):
    item={'gyroConsistencyWeight':weight,'supervisedGradientSteps':300,'conditions':{},'selected':False}
    for condition in ('nominal','stress'):
        rows,stats=load(root/f'gyro{weight}-{condition}/summary.json')
        reference=baselines[condition][0];assert set(rows)==set(reference)
        for key in rows:np.testing.assert_allclose(rows[key]['initialQpos'],reference[key]['initialQpos'],atol=1e-12)
        stats['regressions']=[list(key) for key,r in reference.items() if r['success'] and not rows[key]['success']]
        item['conditions'][condition]=stats
    item['reason']='Lost successful baseline trials under transition stress; in-sample smoothness cannot justify selection.'
    out['candidates'].append(item)
(root/'gyro-consistency-comparison.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))

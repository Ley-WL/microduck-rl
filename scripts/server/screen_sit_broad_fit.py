"""Paired screening after broad successful-rollout retention repair."""
import argparse,concurrent.futures,hashlib,json
from pathlib import Path
import numpy as np
import finalize_sit_transfer as f
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--folder',default='broad-fit')
p.add_argument('--weights',type=float,nargs='+',default=[1.,3.,10.])
a=p.parse_args()
folder=root/'evidence'/a.folder
f.evidence=root/'evidence/state-reference'
f.extra_conditions={'gain-low':['--actuator-kp-scale','.8'],'gain-high':['--actuator-kp-scale','1.2']}
baseline=f.screen(root/'weight-1p0.onnx','baseline')
outcome={'hardwareValidated':False,'independentEvaluation':False,'selected':None,'candidates':[],
         'baseline':{k:f.summarize(v)[1] for k,v in baseline.items()},
         'selectionRule':'Zero lost paired successes in all four conditions; stress mean excess travel improves >=5%; each knee p95 jumps <=1.05x baseline.'}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures={w:pool.submit(f.screen,folder/('weight-'+str(w).replace('.','p')+'.onnx'),a.folder+'-'+str(w)) for w in a.weights}
    for weight,future in futures.items():
        reports=future.result();conditions={};lost=0
        for condition,report in reports.items():
            old,_=f.summarize(baseline[condition]);new,stats=f.summarize(report)
            assert old.keys()==new.keys()
            for key in old:np.testing.assert_allclose(old[key]['initialQpos'],new[key]['initialQpos'],atol=1e-12)
            regressions=[list(k) for k,v in old.items() if v['success'] and not new[k]['success']]
            stats['regressions']=regressions;lost+=len(regressions);conditions[condition]=stats
        before=outcome['baseline']['stress'];after=conditions['stress']
        ratio=float(np.mean(after['meanKneeExcessTravelDeg'])/np.mean(before['meanKneeExcessTravelDeg']))
        eligible=lost==0 and ratio<=.95 and all(v<=b*1.05+1e-6 for v,b in zip(after['p95MaxKneeFrameChangeDeg'],before['p95MaxKneeFrameChangeDeg']))
        policy=folder/('weight-'+str(weight).replace('.','p')+'.onnx')
        item={'weight':weight,'sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),'conditions':conditions,
              'stressExcessTravelRatio':ratio,'eligible':eligible}
        outcome['candidates'].append(item)
        if eligible and (outcome['selected'] is None or ratio<outcome['selected']['stressExcessTravelRatio']):outcome['selected']=item
        (folder/'comparison.json').write_text(json.dumps(outcome,indent=2));print(json.dumps(item),flush=True)
(folder/'screening-completed.txt').write_text('complete\n')

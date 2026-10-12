"""Read only completed matched CPU screens, without disturbing GPU jobs."""
import json
import numpy as np
import finalize_sit_transfer as f
folder=f.root/'evidence/ablation'
conditions=('nominal','stress','gain-low','gain-high')
def load(tag):
    paths=[folder/'screening'/tag/condition/'summary.json' for condition in conditions]
    if not all(path.exists() for path in paths):return None
    reports={condition:json.loads(path.read_text()) for condition,path in zip(conditions,paths)}
    if any(len(report['groups'])!=(8 if condition=='nominal' else 4) for condition,report in reports.items()):return None
    return reports
reference=load('baseline');output={'progress':json.loads((folder/'progress.json').read_text()),'cases':{}}
if reference:
    output['baseline']={condition:f.summarize(report)[1] for condition,report in reference.items()}
    for case in ('Coverage','WeakProcess','Temporal','Sensor','getup-combo'):
        reports=load(case)
        if reports is None:continue
        values={}
        for condition,report in reports.items():
            old,_=f.summarize(reference[condition]);new,item=f.summarize(report)
            assert set(old)==set(new)
            for key in old:np.testing.assert_allclose(old[key]['initialQpos'],new[key]['initialQpos'],atol=1e-12)
            item['regressions']=[list(key) for key,row in old.items() if row['success'] and not new[key]['success']]
            values[condition]=item
        output['cases'][case]=values
print(json.dumps(output))

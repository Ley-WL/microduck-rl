"""Frozen paired CPU screening; survival alone is not walking success."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'evidence'
terrains=('flat','pile3-soft.02','pile6-soft.04')
policies=('Baseline','Control','Surface')
results={}
for terrain in terrains:
    reports={p:json.loads((E/(p+'-'+terrain)/'summary.json').read_text()) for p in policies}
    for p,r in reports.items():
        assert r['manifest']['frictionConstraintRevision']=='dof-id-friction-force-v1'
        assert r['manifest']['jointVelocityObservation']=='previous native50Hz feedback sample'
        assert r['manifest']['terrainHash']==reports['Baseline']['manifest']['terrainHash']
    bykey={p:{(r['seed'],r['trial'],r['speed']):r for r in data['results']} for p,data in reports.items()}
    assert all(set(bykey[p])==set(bykey['Baseline']) for p in policies)
    items=[]
    for key,base in bykey['Baseline'].items():
        seed,trial,speed=key;name=f'seed{seed}-trial{trial}-speed{speed:g}.json'
        traces={p:json.loads((E/(p+'-'+terrain)/name).read_text()) for p in policies}
        metrics={}
        for p,t in traces.items():
            np.testing.assert_array_equal(t['initialQpos'],traces['Baseline']['initialQpos'])
            np.testing.assert_array_equal(t['frames'][0]['velocity'],traces['Baseline']['frames'][0]['velocity'])
            frames=t['frames'];target=np.asarray([f['target'] for f in frames])
            frame=np.abs(np.diff(target,axis=0))*180/np.pi
            r=bykey[p][key]
            # Threshold is an explicit diagnostic gate, not field acceptance.
            effective=r['completed'] and r['movingMeanVx'] is not None and r['movingMeanVx']>=max(.02,.3*speed)
            metrics[p]=dict(**r,effectiveForward=effective,
                targetTravelDeg=np.abs(np.diff(target,axis=0)).sum(axis=0).tolist(),
                frameP95Deg=np.percentile(frame,95,axis=0).tolist(),frameMaxDeg=frame.max(axis=0).tolist(),
                headContactFrames=sum(any('head' in str(pair) or 'neck' in str(pair) for pair in f['contacts']) for f in frames))
        items.append(dict(seed=seed,trial=trial,speed=speed,policies=metrics))
    results[terrain]=items

comparisons={}
for candidate in ('Control','Surface'):
    for reference in ('Baseline','Control'):
        if candidate==reference:continue
        lost_survival=[];lost_forward=[];gained_forward=[]
        for terrain,items in results.items():
            for i in items:
                r=i['policies'][reference];c=i['policies'][candidate]
                key=[terrain,i['seed'],i['trial'],i['speed']]
                if r['completed'] and not c['completed']:lost_survival.append(key)
                if r['effectiveForward'] and not c['effectiveForward']:lost_forward.append(key)
                if not r['effectiveForward'] and c['effectiveForward']:gained_forward.append(key)
        comparisons[candidate+'-vs-'+reference]=dict(lostSurvival=lost_survival,
            lostEffectiveForward=lost_forward,gainedEffectiveForward=gained_forward)
(E/'comparison.json').write_text(json.dumps(dict(initialStatesMatched=True,results=results,
    comparisons=comparisons,accepted=False,simulationOnly=True,
    gate='No lost survival or effective-forward cases; inspect speed/tilt/contact/target frames before independent holdout',
    limitations=['synthetic carpet proxy, not measured deformable carpet','BAM gain not identified real PD',
        'training terrain differs from CPU diagnostic terrain','screen is not hardware acceptance']),indent=2))
print(json.dumps(comparisons),flush=True)

"""Re-rank already exported actors after fixing CPU friction constraints.

No new PPO or repeated training. Preserve old-physics evidence separately.
"""
import concurrent.futures,hashlib,json,os
from pathlib import Path
import numpy as np
import finalize_sit_transfer as f
root=Path(__file__).resolve().parents[2];folder=root/'evidence/corrected-archive-screen'
folder.mkdir(parents=True,exist_ok=True)
CASES=('Coverage','WeakProcess','Temporal','Sensor','StateReference','Upright','UprightCapped','CriticWarm','ProcessV2')
f.evidence=folder;f.environment=os.environ.copy();f.environment['OMP_NUM_THREADS']='2'
f.extra_conditions={'gain-low':['--actuator-kp-scale','.8'],'gain-high':['--actuator-kp-scale','1.2']}

if __name__=='__main__':
    try:
        assert not (folder/'comparison.json').exists(),'Inspect current partial/completed job'
        baseline={k:json.loads((root/'evidence/cpu-friction-control/screening/baseline'/k/'summary.json').read_text())
                  for k in ('nominal','stress','gain-low','gain-high')}
        assert all(r['configuration']['frictionConstraintRevision']=='dof-id-friction-force-v1' for r in baseline.values())
        result={'hardwareAccess':False,'training':False,'formalAccepted':False,'frictionConstraintRevision':'dof-id-friction-force-v1',
            'baseline':{k:f.summarize(v)[1] for k,v in baseline.items()},'cases':{}}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs={k:pool.submit(f.screen,root/'policies'/f'{k}.onnx',k) for k in CASES}
            for label in CASES:
                reports=jobs[label].result();stats={};lost=0
                for condition,report in reports.items():
                    assert report['configuration']['frictionConstraintRevision']=='dof-id-friction-force-v1'
                    old,_=f.summarize(baseline[condition]);new,summary=f.summarize(report);assert set(old)==set(new)
                    for k in old:np.testing.assert_allclose(old[k]['initialQpos'],new[k]['initialQpos'],atol=1e-12,rtol=0)
                    summary['regressions']=[list(k) for k in old if old[k]['success'] and not new[k]['success']]
                    lost+=len(summary['regressions']);stats[condition]=summary
                bs=result['baseline']['stress'];cs=stats['stress']
                ratio=float(np.mean(cs['meanKneeExcessTravelDeg'])/np.mean(bs['meanKneeExcessTravelDeg']))
                frame_ok=all(c<=b*1.05+1e-6 for c,b in zip(cs['p95MaxKneeFrameChangeDeg'],bs['p95MaxKneeFrameChangeDeg']))
                result['cases'][label]={'conditions':stats,'stressExcessTravelRatio':ratio,
                    'eligibleForHoldout':lost==0 and ratio<=.95 and frame_ok}
                (folder/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({label:result['cases'][label]}),flush=True)
        manifest=json.loads((folder/'launch.json').read_text())
        assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        (folder/'source-check.json').write_text(json.dumps({'sourceHashesExactlyPreserved':True}))
        (folder/'completed.txt').write_text('complete\n');(folder/'job.exit').write_text('0')
    except BaseException:
        (folder/'job.exit').write_text('1');raise

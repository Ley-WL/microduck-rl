"""Re-screen frozen actors after the DOF-friction CPU physics correction.

New experiment namespace; never reuse scores from the old friction controller.
No training, GPU, or hardware. Pair actors within corrected physics only.
"""
import concurrent.futures,hashlib,json,os
from pathlib import Path
import numpy as np
import finalize_sit_transfer as f
root=Path(__file__).resolve().parents[2];folder=root/'evidence/cpu-friction-control'
folder.mkdir(parents=True,exist_ok=True)
f.evidence=folder;f.environment=os.environ.copy();f.environment['OMP_NUM_THREADS']='2'
f.extra_conditions={'gain-low':['--actuator-kp-scale','.8'],'gain-high':['--actuator-kp-scale','1.2']}

if __name__=='__main__':
    try:
        assert not (folder/'comparison.json').exists(),'Preserve existing results'
        policies={k:root/'policies'/f'{k}.onnx' for k in ('baseline','BoundaryGain','ArrivalGate','LowExplore')}
        result={'hardwareAccess':False,'gpuAccess':False,'training':False,'formalAccepted':False,
            'frictionConstraintRevision':'dof-id-friction-force-v1',
            'scope':'Pair within corrected CPU physics; old-physics counts and successes are not paired with these.',
            'baseline':{},'cases':{}}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs={k:pool.submit(f.screen,p,k) for k,p in policies.items()}
            baseline=jobs['baseline'].result()
            assert all(r['configuration']['frictionConstraintRevision']=='dof-id-friction-force-v1' for r in baseline.values())
            result['baseline']={k:f.summarize(v)[1] for k,v in baseline.items()}
            for label in ('BoundaryGain','ArrivalGate','LowExplore'):
                reports=jobs[label].result();stats={}
                for condition,report in reports.items():
                    assert report['configuration']['frictionConstraintRevision']=='dof-id-friction-force-v1'
                    old,_=f.summarize(baseline[condition]);new,summary=f.summarize(report)
                    assert set(old)==set(new)
                    for key in old:np.testing.assert_allclose(old[key]['initialQpos'],new[key]['initialQpos'],atol=1e-12,rtol=0)
                    summary['regressions']=[list(k) for k in old if old[k]['success'] and not new[k]['success']]
                    stats[condition]=summary
                result['cases'][label]=stats
                (folder/'comparison.json').write_text(json.dumps(result,indent=2))
                print(json.dumps({'case':label,'conditions':stats}),flush=True)
        manifest=json.loads((folder/'launch.json').read_text())
        assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        (folder/'source-check.json').write_text(json.dumps({'sourceHashesExactlyPreserved':True}))
        (folder/'completed.txt').write_text('complete\n');(folder/'job.exit').write_text('0')
    except BaseException:
        (folder/'job.exit').write_text('1');raise

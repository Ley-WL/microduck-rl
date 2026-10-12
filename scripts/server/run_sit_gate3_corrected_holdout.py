"""Fresh-seed regression for Gate3 after the corrected-physics screen passed."""
import hashlib,json
from pathlib import Path
import numpy as np
import run_sit_supervised_corrected_holdout as h
root=Path(__file__).resolve().parents[2];h.folder=root/'evidence/gate3-corrected-holdout'
h.folder.mkdir(parents=True,exist_ok=True)

if __name__=='__main__':
    try:
        assert not (h.folder/'comparison.json').exists(),'Inspect existing run'
        baseline={name:json.loads((root/'evidence/supervised-corrected-holdout/baseline'/name/'summary.json').read_text()) for name in h.conditions}
        assert all(r['configuration']['frictionConstraintRevision']=='dof-id-friction-force-v1' for r in baseline.values())
        after=h.evaluate('Gate3');result={'hardwareAccess':False,'formalAccepted':False,'training':False,'seeds':h.seeds,'trialsPerGroup':4,'baseline':{},'candidate':{}}
        result['policyLabel']='Gate3'
        lost=0
        for name in h.conditions:
            old,bs=h.f.summarize(baseline[name]);new,stats=h.f.summarize(after[name]);assert set(old)==set(new)
            for k in old:np.testing.assert_allclose(old[k]['initialQpos'],new[k]['initialQpos'],atol=1e-12,rtol=0)
            stats['regressions']=[list(k) for k in old if old[k]['success'] and not new[k]['success']]
            lost+=len(stats['regressions']);result['baseline'][name]=bs;result['candidate'][name]=stats
        bs=result['baseline']['stress'];cs=result['candidate']['stress']
        ratio=float(np.mean(cs['meanKneeExcessTravelDeg'])/np.mean(bs['meanKneeExcessTravelDeg']))
        frame_ok=all(c<=b*1.05+1e-6 for c,b in zip(cs['p95MaxKneeFrameChangeDeg'],bs['p95MaxKneeFrameChangeDeg']))
        result['lostPairedSuccesses']=lost;result['frameGatePassed']=frame_ok
        result['stressExcessTravelRatio']=ratio;result['eligibleForFormalRegression']=lost==0 and ratio<=.95 and frame_ok
        manifest=json.loads((h.folder/'launch.json').read_text())
        assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        result['sourceHashesExactlyPreserved']=True;(h.folder/'comparison.json').write_text(json.dumps(result,indent=2))
        (h.folder/'completed.txt').write_text('complete\n');(h.folder/'job.exit').write_text('0');print(json.dumps(result),flush=True)
    except BaseException:
        (h.folder/'job.exit').write_text('1');raise

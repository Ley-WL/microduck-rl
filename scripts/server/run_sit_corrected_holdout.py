"""Fresh seeds after corrected-physics screen; frozen policies only."""
import concurrent.futures,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import finalize_sit_transfer as f
root=Path(__file__).resolve().parents[2];folder=root/'evidence/corrected-holdout'
folder.mkdir(parents=True,exist_ok=True);python=root.parent/'microduck-rl/.venv/bin/python'
seeds=(12017,24029)
conditions={'nominal':[],'stress':['--scenario','stand_cycle','sit_cycle','--transition-stress','--joint-velocity-lag-steps','1'],
    'gain-low':['--scenario','stand_cycle','sit_cycle','--actuator-kp-scale','.8'],
    'gain-high':['--scenario','stand_cycle','sit_cycle','--actuator-kp-scale','1.2']}

def evaluate(tag):
    result={};environment=os.environ.copy();environment['OMP_NUM_THREADS']='2'
    for name,extra in conditions.items():
        path=folder/tag/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_name(name+'.log').open('w') as log:
            subprocess.run([str(python),'scripts/evaluate_skills_cpu.py','--onnx',str(root/'policies'/f'{tag}.onnx'),
                '--output',str(path),'--trials','4','--seeds',*map(str,seeds),*extra],cwd=root,env=environment,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        result[name]=json.loads((path/'summary.json').read_text())
        assert result[name]['configuration']['frictionConstraintRevision']=='dof-id-friction-force-v1'
    return result

if __name__=='__main__':
    try:
        assert not (folder/'comparison.json').exists(),'Inspect completed/partial existing job'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            b=pool.submit(evaluate,'baseline');a=pool.submit(evaluate,'LowExplore');before,after=b.result(),a.result()
        result={'hardwareAccess':False,'formalAccepted':False,'training':False,'seeds':seeds,'trialsPerGroup':4,'baseline':{},'candidate':{}}
        lost=0
        for name in conditions:
            old,bs=f.summarize(before[name]);new,stats=f.summarize(after[name]);assert set(old)==set(new)
            for k in old:np.testing.assert_allclose(old[k]['initialQpos'],new[k]['initialQpos'],atol=1e-12,rtol=0)
            stats['regressions']=[list(k) for k in old if old[k]['success'] and not new[k]['success']]
            lost+=len(stats['regressions']);result['baseline'][name]=bs;result['candidate'][name]=stats
        bs=result['baseline']['stress'];cs=result['candidate']['stress']
        ratio=float(np.mean(cs['meanKneeExcessTravelDeg'])/np.mean(bs['meanKneeExcessTravelDeg']))
        frame_ok=all(c<=b*1.05+1e-6 for c,b in zip(cs['p95MaxKneeFrameChangeDeg'],bs['p95MaxKneeFrameChangeDeg']))
        result['stressExcessTravelRatio']=ratio;result['eligibleForFormalRegression']=lost==0 and ratio<=.95 and frame_ok
        manifest=json.loads((folder/'launch.json').read_text())
        assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        result['sourceHashesExactlyPreserved']=True
        (folder/'comparison.json').write_text(json.dumps(result,indent=2))
        (folder/'completed.txt').write_text('complete\n');(folder/'job.exit').write_text('0');print(json.dumps(result),flush=True)
    except BaseException:
        (folder/'job.exit').write_text('1');raise

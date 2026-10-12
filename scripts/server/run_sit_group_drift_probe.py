"""Isolate head vs leg actor drift after the paired all-joint cap probe failed."""
import concurrent.futures,json,os,subprocess,hashlib
from pathlib import Path
import numpy as np
import finalize_sit_transfer as f
root=Path(__file__).resolve().parents[2];folder=root/'evidence/group-drift-probe'
folder.mkdir(parents=True,exist_ok=True)
python=root.parent/'microduck-rl/.venv/bin/python';original=root/'evidence/boundary-gain-control'
conditions={'stress':['--transition-stress','--joint-velocity-lag-steps','1'],'gain-low':['--actuator-kp-scale','.8']}

def evaluate(group):
    results={};environment=os.environ.copy();environment['OMP_NUM_THREADS']='2'
    for name,extra in conditions.items():
        path=folder/group/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_name(name+'.log').open('w') as log:
            subprocess.run([str(python),'scripts/server/evaluate_sit_residual_cap.py',
                '--anchor',str(original/'baseline.onnx'),'--residual-cap-deg','.25','--residual-group',group,
                '--onnx',str(original/'BoundaryGain/train.onnx'),'--output',str(path),'--trials','4',
                '--seeds','8111','9001','--scenario','stand_cycle','sit_cycle','--capture-policy-frames',*extra],
                cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((path/'summary.json').read_text())
        old,_=f.summarize(json.loads((original/'screening/baseline'/name/'summary.json').read_text()))
        new,stats=f.summarize(report);assert set(old)==set(new) and len(new)==16
        for key in old:np.testing.assert_allclose(old[key]['initialQpos'],new[key]['initialQpos'],atol=1e-12,rtol=0)
        stats['regressions']=[list(k) for k in old if old[k]['success'] and not new[k]['success']]
        stats['diagnosticComposite']=report['diagnosticComposite'];results[name]=stats
    return results

if __name__=='__main__':
    try:
        assert not (folder/'comparison.json').exists(),'Inspect existing result'
        result={'hardwareAccess':False,'training':False,'gpuAccess':False,'deployable':False,'cases':{}}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={group:pool.submit(evaluate,group) for group in ('legs','head')}
            for group,future in futures.items():
                result['cases'][group]=future.result();(folder/'comparison.json').write_text(json.dumps(result,indent=2))
                print(json.dumps({'group':group,'conditions':result['cases'][group]}),flush=True)
        manifest=json.loads((folder/'launch.json').read_text())
        assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        (folder/'source-check.json').write_text(json.dumps({'sourceHashesExactlyPreserved':True}))
        (folder/'completed.txt').write_text('complete\n');(folder/'job.exit').write_text('0')
    except BaseException:
        (folder/'job.exit').write_text('1');raise

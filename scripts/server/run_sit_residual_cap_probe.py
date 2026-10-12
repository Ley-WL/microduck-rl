"""Two bounded frozen-policy probes, CPU only; no additional PPO run."""
import concurrent.futures,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import finalize_sit_transfer as f
root=Path(__file__).resolve().parents[2];folder=root/'evidence/residual-cap-probe-v2'
folder.mkdir(parents=True,exist_ok=True)
python=root.parent/'microduck-rl/.venv/bin/python'
original=root/'evidence/boundary-gain-control'
conditions={'nominal':[], 'stress':['--scenario','stand_cycle','sit_cycle','--transition-stress','--joint-velocity-lag-steps','1'],
    'gain-low':['--scenario','stand_cycle','sit_cycle','--actuator-kp-scale','.8'],
    'gain-high':['--scenario','stand_cycle','sit_cycle','--actuator-kp-scale','1.2']}

def evaluate(cap):
    results={};environment=os.environ.copy();environment['OMP_NUM_THREADS']='2'
    for name,extra in conditions.items():
        path=folder/str(cap)/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_name(name+'.log').open('w') as log:
            subprocess.run([str(python),'scripts/server/evaluate_sit_residual_cap.py',
                '--anchor',str(original/'baseline.onnx'),'--residual-cap-deg',str(cap),
                '--onnx',str(original/'BoundaryGain/train.onnx'),'--output',str(path),
                '--trials','4','--seeds','8111','9001',*extra],cwd=root,env=environment,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((path/'summary.json').read_text())
        old,_=f.summarize(json.loads((original/'screening/baseline'/name/'summary.json').read_text()))
        new,stats=f.summarize(report);assert set(old)==set(new)
        for key in old:np.testing.assert_allclose(old[key]['initialQpos'],new[key]['initialQpos'],atol=1e-12,rtol=0)
        stats['regressions']=[list(k) for k in old if old[k]['success'] and not new[k]['success']]
        stats['diagnosticComposite']=report['diagnosticComposite'];results[name]=stats
    return results

if __name__=='__main__':
    try:
        assert not (folder/'comparison.json').exists(),'Inspect existing experiment'
        result={'hardwareAccess':False,'training':False,'gpuAccess':False,'deployable':False,'cases':{}}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={cap:pool.submit(evaluate,cap) for cap in (.25,.5)}
            for cap,future in futures.items():
                result['cases'][str(cap)]=future.result()
                (folder/'comparison.json').write_text(json.dumps(result,indent=2))
                print(json.dumps({'capDeg':cap,'conditions':result['cases'][str(cap)]}),flush=True)
        (folder/'completed.txt').write_text('complete\n');(folder/'job.exit').write_text('0')
    except BaseException:
        (folder/'job.exit').write_text('1');raise

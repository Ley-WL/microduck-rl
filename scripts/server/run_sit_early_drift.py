"""Read-only low-gain screen of an already exported five-iteration actor.

Different smoke batch size prevents treating this as a same-run learning curve.
No training, GPU, or hardware access. Preserve existing results on restart.
"""
import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[2]
folder=root/'evidence/early-drift';folder.mkdir(parents=True,exist_ok=True)
python=root.parent/'microduck-rl/.venv/bin/python'

def main():
    assert not (folder/'completed.txt').exists(),'Already completed; inspect evidence'
    original=root/'evidence/boundary-gain-control'
    policy=original/'BoundaryGain/smoke.onnx'
    output=folder/'smoke-gain-low'
    assert not (output/'summary.json').exists(),'Inspect partial/completed existing screen'
    environment=os.environ.copy();environment['OMP_NUM_THREADS']='2'
    with (folder/'screen.log').open('w') as log:
        subprocess.run([str(python),'scripts/evaluate_skills_cpu.py','--onnx',str(policy),
            '--output',str(output),'--trials','4','--seeds','8111','9001',
            '--scenario','stand_cycle','sit_cycle','--actuator-kp-scale','.8',
            '--capture-policy-frames'],cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    before=json.loads((original/'screening/baseline/gain-low/summary.json').read_text())
    after=json.loads((output/'summary.json').read_text())
    def rows(report):return {(g['scenario'],g['seed'],r['trial']):r for g in report['groups'] for r in g['results']}
    old,new=rows(before),rows(after);assert set(old)==set(new) and len(new)==16
    for key in old:np.testing.assert_allclose(old[key]['initialQpos'],new[key]['initialQpos'],atol=1e-12,rtol=0)
    result={'hardwareAccess':False,'newTraining':False,'gpuAccess':False,
        'scope':'64-world five-iteration smoke vs source, not an intermediate of the 1024-world final run',
        'baselineSuccesses':sum(r['success'] for r in old.values()),
        'smokeSuccesses':sum(r['success'] for r in new.values()),'trials':16,
        'regressions':[list(k) for k in old if old[k]['success'] and not new[k]['success']],
        'policySha256':hashlib.sha256(policy.read_bytes()).hexdigest()}
    (folder/'comparison.json').write_text(json.dumps(result,indent=2))
    (folder/'completed.txt').write_text('complete\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (folder/'job.exit').write_text('1');raise
    else:(folder/'job.exit').write_text('0')

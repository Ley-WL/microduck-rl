"""Frozen candidate, fresh seeds, per-pose 200-trial simulation gate. CPU only."""
import concurrent.futures,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'evidence/mechanical-formal'
PYTHON=ROOT.parent/'microduck-rl/.venv/bin/python'
POLICY=ROOT/'evidence/mechanical-training/train.onnx'
SHA='4b9c82e5bbd003a465b9bf861d4e7020014e7781d311f3780001e5257a588838'

def main():
    manifest=json.loads((FOLDER/'launch.json').read_text())
    assert hashlib.sha256(POLICY.read_bytes()).hexdigest()==SHA
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',LIBGL_ALWAYS_SOFTWARE='1')
    def run(kind):
        folder=FOLDER/kind;folder.mkdir()
        args=[str(PYTHON),'scripts/evaluate_getup_mechanical_cpu.py','--onnx',str(POLICY),
            '--profile','runtime','--seeds','517031','629017','--trials','200','--output',str(folder)]
        if kind=='stress':args+=['--transition-stress','--joint-velocity-lag-steps','1']
        with (FOLDER/(kind+'.log')).open('w') as stream:
            subprocess.run(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        report=json.loads((folder/'summary.json').read_text())
        assert report['scopeComplete'] and len(report['groups'])==12
        assert report['configuration']['mechanicalExecutionContract']=='kissqy-mechanical-ema045-rawhistory-encoder4096-v1'
        groups=[dict(pose=g['scenario'],seed=g['seed'],trials=len(g['results']),successes=sum(r['success'] for r in g['results']),successRate=g['successRate']) for g in report['groups']]
        passed=all(g['trials']==200 and g['successRate']>=.95 for g in groups)
        return dict(groups=groups,perPosePerSeedGate=passed)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        jobs={kind:executor.submit(run,kind) for kind in ('normal','stress')}
        result={k:v.result() for k,v in jobs.items()}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest['sourceHashes'].items())
    (FOLDER/'result.json').write_text(json.dumps(dict(policySha256=SHA,conditions=result,
        simulationGatePass=all(v['perPosePerSeedGate'] for v in result.values()),
        hardwareValidated=False,deployment=False,sourceHashesPreserved=True),indent=2))

if __name__=='__main__':
    try:main();(FOLDER/'job.exit').write_text('0')
    except BaseException:(FOLDER/'job.exit').write_text('1');raise

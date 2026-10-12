"""CPU replay with full policy frames and strict comparison to completed screen."""
import concurrent.futures,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'evidence/full-batch-failure-capture'
SCREEN=ROOT/'evidence/full-batch-control'
PYTHON=ROOT.parent/'microduck-rl/.venv/bin/python'

def capture(label,condition):
    policy=SCREEN/'baseline.onnx' if label=='baseline' else SCREEN/label/'train.onnx'
    output=FOLDER/label/condition;output.mkdir(parents=True,exist_ok=True)
    extra=['--actuator-kp-scale','.8'] if condition=='gain-low' else ['--transition-stress','--joint-velocity-lag-steps','1']
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2')
    with output.with_suffix('.log').open('wb') as log:
        subprocess.run([str(PYTHON),'scripts/evaluate_skills_cpu.py','--onnx',str(policy),'--output',str(output),'--trials','4','--seeds','8111','9001','--scenario','stand_cycle','sit_cycle','--capture-policy-frames',*extra],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    original=SCREEN/'screening'/label/condition
    report=json.loads((output/'summary.json').read_text())
    assert report['configuration']['frictionConstraintRevision']=='dof-id-friction-force-v1'
    rows=[]
    for npz in sorted(original.glob('*.npz')):
        old=np.load(npz)['qpos'];new=np.load(output/npz.name)['qpos']
        np.testing.assert_allclose(old,new,atol=1e-12,rtol=0)
        before=json.loads(npz.with_suffix('.json').read_text())['result'];after=json.loads((output/npz.with_suffix('.json').name).read_text())['result']
        assert before==after
        rows.append({'trial':npz.stem,'maxQposDifference':float(np.max(np.abs(old-new))),'resultExactlyPreserved':True})
    assert len(rows)==16
    return {'label':label,'condition':condition,'replayMatchesOriginal':True,'trials':rows}

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert (SCREEN/'job.exit').read_text()=='0'
    manifest=json.loads((FOLDER/'launch.json').read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
    result={'training':False,'gpuAccess':False,'hardwareAccess':False,'scope':'New frame capture of same exact original 96 CPU trials. Preserve original scenario order/seeds/trials and compare full qpos/result, not new independent validation.','runs':[]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(capture,label,condition) for label in ('baseline','WeakProcess','FullBatch') for condition in ('gain-low','stress')]
        for future in futures:result['runs'].append(future.result());(FOLDER/'summary.json').write_text(json.dumps(result,indent=2))
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
    result['sourceHashesExactlyPreserved']=True;(FOLDER/'summary.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    try:main();(FOLDER/'job.exit').write_text('0')
    except BaseException:(FOLDER/'job.exit').write_text('1');raise

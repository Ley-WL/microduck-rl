"""Separate self-collision from target-clipping in the newly found getup fail."""
import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];folder=root/'evidence/contact-isolation-v2';folder.mkdir(parents=True,exist_ok=True)
python=root.parent/'microduck-rl/.venv/bin/python'
cases={'original':[],'self-only':['--self-contact-only'],'clip-only':['--target-clip-only'],'combined':['--contact-repair']}

if __name__=='__main__':
    try:
        assert not (folder/'comparison.json').exists(),'Inspect existing run'
        environment=os.environ.copy();environment['OMP_NUM_THREADS']='1'
        result={'hardwareAccess':False,'training':False,'formalAccepted':False,
            'scope':'Full six-scenario RNG sequence at seed9001, four trials per scenario. Subset-only run changed the failing initial state.', 'cases':{}}
        reference=None
        for name,extra in cases.items():
            path=folder/name
            with (folder/(name+'.log')).open('w') as log:
                subprocess.run([str(python),'scripts/evaluate_skills_cpu.py','--onnx','policies/getup.onnx','--output',str(path),
                    '--trials','4','--seeds','9001','--transition-stress',
                    '--joint-velocity-lag-steps','1','--capture-policy-frames',*extra],cwd=root,env=environment,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads((path/'summary.json').read_text());rows=[r for g in report['groups'] for r in g['results']];assert len(rows)==24
            if reference is None:reference=rows
            for old,new in zip(reference,rows):np.testing.assert_allclose(old['initialQpos'],new['initialQpos'],atol=1e-12,rtol=0)
            if name=='combined':
                expected=json.loads((root/'failure-reference.json').read_text())['result']['initialQpos']
                actual=json.loads((path/'right_side-9001-0.json').read_text())['result']['initialQpos']
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
            result['cases'][name]={'successes':sum(r['success'] for r in rows),'trials':24,'successByTrial':[r['success'] for r in rows],
                'finalTiltMeanByTrial':[float(np.mean([r['tiltDeg'] for r in json.loads((path/f'right_side-9001-{i}.json').read_text())['trace'] if r['time']>=12])) for i in range(4)],
                'configuration':report['configuration']}
            (folder/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({name:result['cases'][name]}),flush=True)
        manifest=json.loads((folder/'launch.json').read_text())
        assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        result['sourceHashesExactlyPreserved']=True;(folder/'comparison.json').write_text(json.dumps(result,indent=2))
        (folder/'completed.txt').write_text('complete\n');(folder/'job.exit').write_text('0')
    except BaseException:
        (folder/'job.exit').write_text('1');raise

"""Read-only simulation probe from current v3 sitting joints, original actor."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=root/'evidence/getup-v3-seated';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'completed.txt').exists(),'Preserve prior probe'
source=root/'evidence/xgoduck_getup.onnx'
policy=folder/'diagnostic.onnx';policy.write_bytes(source.read_bytes())
metadata=json.loads(source.with_suffix('.metadata.json').read_text())
original=metadata['sitJointOverrides'].copy()
metadata['sitJointOverrides'].update({'2':-.4579,'11':.4579,'3':1.25,'12':-1.25})
metadata['diagnosticInitializationProvenance']={'kind':'current-v3-seated-pose-only','originalSitJointOverrides':original,
    'sourcePolicySha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sameActor':True,'hardwareValidated':False,'accepted':False}
policy.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2))
with (folder/'evaluation.log').open('w') as log:
    result=subprocess.run([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/evaluate_skills_cpu.py',
        '--onnx',str(policy),'--output',str(folder/'nominal'),'--getup-seated-probe','--scenario','sit','stand',
        '--trials','4','--seeds','8111','9001','--contact-repair','--capture-policy-frames'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
(folder/'job.exit').write_text(str(result.returncode))
assert result.returncode==0
(folder/'completed.txt').write_text('complete\n')

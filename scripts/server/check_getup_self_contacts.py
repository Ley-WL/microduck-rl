"""Keep old getup qualification separate from a stricter contact hypothesis."""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
python=root.parent/'microduck-rl/.venv/bin/python'
folder=root/'evidence/getup-self-contacts';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'completed.txt').exists()
for condition in ('nominal','stress'):
    command=[str(python),'scripts/evaluate_skills_cpu.py','--onnx','evidence/xgoduck_getup.onnx',
             '--output',str(folder/condition),'--trials','4','--seeds','8111','9001','--contact-repair','--capture-policy-frames']
    if condition=='stress':command+=['--transition-stress','--joint-velocity-lag-steps','1']
    with (folder/(condition+'.log')).open('w') as output:
        result=subprocess.run(command,cwd=root,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:
        (folder/'job.exit').write_text(str(result.returncode));raise RuntimeError('Getup stricter contact diagnostic failed')
    report=json.loads((folder/condition/'summary.json').read_text())
    print(json.dumps({'condition':condition,'successes':sum(r['success'] for g in report['groups'] for r in g['results'])}),flush=True)
(folder/'job.exit').write_text('0');(folder/'completed.txt').write_text('complete\n')

"""Bounded CPU-only filtering ablation; do not alter the robot or train run."""
import json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
python=root.parent/'microduck-rl/.venv/bin/python'
evidence=root/'evidence/filter-ablation';evidence.mkdir(parents=True,exist_ok=True)
assert not (evidence/'completed.txt').exists(),'Read completed evidence instead of repeating'
for alpha in (.55,.65,.75):
    tag=str(alpha).replace('.','p')
    for condition in ('nominal','stress'):
        folder=evidence/tag/condition;folder.mkdir(parents=True,exist_ok=True)
        command=[str(python),'scripts/evaluate_skills_cpu.py','--onnx','weight-1p0.onnx','--output',str(folder),
                 '--trials','4','--seeds','8111','9001','--action-ema-old-weight',str(alpha)]
        if condition=='stress':command+=['--scenario','stand_cycle','sit_cycle','--transition-stress','--joint-velocity-lag-steps','1']
        with (folder.parent/(condition+'.log')).open('w') as stream:
            result=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            (evidence/'job.exit').write_text(str(result.returncode));raise RuntimeError('CPU filtering diagnostic failed')
        report=json.loads((folder/'summary.json').read_text())
        print(json.dumps({'alpha':alpha,'condition':condition,'successes':sum(r['success'] for g in report['groups'] for r in g['results'])}),flush=True)
(evidence/'job.exit').write_text('0');(evidence/'completed.txt').write_text('complete\n')

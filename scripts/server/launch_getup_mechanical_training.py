"""Launch one guarded experiment on user-authorized GPUs0/1."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence/mechanical-training'
OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'launch.json').exists(),'Track existing launch/PID; do not duplicate'
assert (OUT/'config-tests.exit').read_text().strip()=='0'
files=[p for p in (ROOT/'src').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
files += [ROOT/p for p in ('seed.pt','scripts/server/run_getup_mechanical_training.py',
    'scripts/server/launch_getup_mechanical_training.py','scripts/server/audit_getup_mechanical_chain.py',
    'scripts/server/write_getup_mechanical_metadata.py','scripts/evaluate_getup_mechanical_cpu.py',
    'scripts/bam_cpu_controller.py','scripts/export.py','tests/test_getup_mechanical_contract.py')]
manifest=dict(training=True,hardwareAccess=False,gpus=[0,1],seed=61217,
    sourceHashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    stages=['capacity-check-GPU0/1','baseline-export-and-current-contract-screen','64-per-GPU-x5-smoke','live-action-audit','1024-per-GPU-x300','paired-screen','conditional-fresh-seed-screen'],
    autoDeployment=False)
(OUT/'launch.json').write_text(json.dumps(manifest,indent=2))
with (OUT/'job.log').open('wb') as stream:
    process=subprocess.Popen([str(ROOT.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_getup_mechanical_training.py'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(OUT/'launch.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(pid=process.pid,gpus=[0,1],stages=manifest['stages'],hardwareAccess=False)))

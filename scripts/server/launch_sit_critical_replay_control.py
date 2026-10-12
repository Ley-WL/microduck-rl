import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/critical-replay';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Follow existing launch'
usage=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert usage<200
files=['seed.pt','transfer-data.npz','original-transfer-data.npz','critical-observations.npz','critical-replay-dataset-audit.json','scripts/server/run_sit_critical_replay_control.py','scripts/server/launch_sit_critical_replay_control.py','scripts/server/finalize_sit_transfer.py','scripts/server/write_policy_metadata.py','scripts/export.py','scripts/evaluate_skills_cpu.py','scripts/bam_cpu_controller.py','tests/test_sit_critical_replay.py']
files += [str(p.relative_to(root)) for base in ('src','references') for p in sorted((root/base).rglob('*')) if p.is_file() and '__pycache__' not in str(p)]
manifest={'training':True,'hardwareAccess':False,'gpu':6,'gpuMemoryBeforeMiB':usage,'trainSeed':57121,'iterations':20,'sourceHashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
(folder/'launch.json').write_text(json.dumps(manifest,indent=2))
with (folder/'job.log').open('wb') as log:process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_critical_replay_control.py'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'pid':process.pid,'gpuMemoryBeforeMiB':usage}))

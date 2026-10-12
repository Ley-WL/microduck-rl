import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/full-batch-control';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Follow existing launch'
usage=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert usage<200
files=['seed.pt','transfer-data.npz','scripts/server/run_sit_full_batch_control.py','scripts/server/launch_sit_full_batch_control.py','scripts/server/finalize_sit_transfer.py','scripts/server/write_policy_metadata.py','scripts/export.py','scripts/evaluate_skills_cpu.py','scripts/bam_cpu_controller.py','tests/test_sit_full_batch.py','tests/test_sit_reward_gradients.py','tests/test_cpu_bam_friction_constraint.py']
files += [str(p.relative_to(root)) for p in sorted((root/'src').rglob('*')) if p.is_file() and '__pycache__' not in str(p)]
manifest={'training':True,'hardwareAccess':False,'gpu':6,'gpuMemoryBeforeMiB':usage,'trainSeed':57121,'iterationsPerCase':20,'sourceHashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
(folder/'launch.json').write_text(json.dumps(manifest,indent=2))
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_full_batch_control.py'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'pid':process.pid,'gpuMemoryBeforeMiB':usage,'trainSeed':57121}))

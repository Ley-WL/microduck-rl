import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/cached24';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Follow existing launch'
usage=None
files=['checkpoints/Main0.pt','checkpoints/Main99.pt','seed.pt','transfer-data.npz','scripts/server/audit_sit_gradient_contributions.py','scripts/server/launch_sit_cached24_gradients.py','tests/test_sit_reward_gradients.py','scripts/server/audit_sit_reward_gradients.py','scripts/server/audit_sit_cached24_gradients.py']
files += [str(p.relative_to(root)) for p in sorted((root/'src').rglob('*')) if p.is_file() and '__pycache__' not in str(p)]
files += [str(p.relative_to(root)) for p in sorted((root/'inputs').rglob('*')) if p.is_file()]
manifest={'gpuAccess':False,'hardwareAccess':False,'training':False,'gpu':None,'gpuMemoryBeforeMiB':usage,'sourceHashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
(folder/'launch.json').write_text(json.dumps(manifest,indent=2))
environment=os.environ.copy();environment.update(PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MICRODUCK_WARM_START='1',WANDB_MODE='offline',MUJOCO_GL='egl')
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/audit_sit_cached24_gradients.py'],cwd=root,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'launch.tmp').write_text(json.dumps(manifest,indent=2));(folder/'launch.tmp').replace(folder/'launch.json')
print(json.dumps({'pid':process.pid,'gpuMemoryBeforeMiB':usage,'training':False}))

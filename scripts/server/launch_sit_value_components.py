import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/formal24'
assert (folder/'job.exit').read_text()=='0'
assert not (folder/'value-components-launch.json').exists(),'Follow existing launch'
environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',WANDB_MODE='offline')
manifest={'training':False,'gpuAccess':False,'hardwareAccess':False,'sourceHash':hashlib.sha256((root/'scripts/server/audit_sit_value_components.py').read_bytes()).hexdigest()}
(folder/'value-components-launch.json').write_text(json.dumps(manifest,indent=2))
with (folder/'value-components.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/audit_sit_value_components.py'],cwd=root,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'value-components-launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

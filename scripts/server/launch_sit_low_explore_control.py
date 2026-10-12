"""Start only after task tests; isolated bounded GPU6 diagnostic controls."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=root/'evidence/low-explore-control';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Inspect the existing job'
usage=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert usage<200,'GPU6 occupied'
with (folder/'job.log').open('wb') as stream:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_low_explore_control.py'],cwd=root,
        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
manifest={'pid':process.pid,'gpu':6,'cases':1,'iterationsPerCase':100,'worlds':1024,'smokeIterations':5,
          'explorationStdRange':[.02,.03],'parentExperiment':'WeakProcess100','hardwareAccess':False,'sourceHashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
          ('seed.pt','transfer-data.npz','src/mjlab_microduck/tasks/sit_settled_tilt.py','src/mjlab_microduck/tasks/sit_process_ablation_env_cfg.py',
           'src/mjlab_microduck/tasks/sit_process_cost.py','src/mjlab_microduck/tasks/sit_transfer_ppo.py','scripts/server/run_sit_low_explore_control.py')}}
(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

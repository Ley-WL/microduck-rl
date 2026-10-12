"""One detached CPU holdout, no new training."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/corrected-holdout';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Follow existing PID'
files=['scripts/evaluate_skills_cpu.py','scripts/bam_cpu_controller.py','scripts/server/run_sit_corrected_holdout.py',
       'policies/baseline.onnx','policies/LowExplore.onnx']
manifest={'hardwareAccess':False,'gpuAccess':False,'training':False,'seeds':[12017,24029],
    'sourceHashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_corrected_holdout.py'],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

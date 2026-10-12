"""Detached CPU getup regression with one recorded launch."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/getup-corrected';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Track existing launch'
files=['scripts/evaluate_skills_cpu.py','scripts/bam_cpu_controller.py','scripts/server/run_getup_corrected_friction.py']
files+=['policies/'+p+s for p in ('getup','getup-v3') for s in ('.onnx','.metadata.json')]
manifest={'hardwareAccess':False,'gpuAccess':False,'training':False,'sourceHashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_getup_corrected_friction.py'],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

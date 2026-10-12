"""One detached two-worker CPU re-screen of frozen historical exports."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/corrected-archive-screen';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Follow recorded launch'
cases=('Coverage','WeakProcess','Temporal','Sensor','StateReference','Upright','UprightCapped','CriticWarm','ProcessV2')
files=['scripts/evaluate_skills_cpu.py','scripts/bam_cpu_controller.py','scripts/server/finalize_sit_transfer.py','scripts/server/run_sit_corrected_archive_screen.py']
files+=['policies/'+p+s for p in (*cases,'baseline') for s in ('.onnx','.metadata.json')]
manifest={'hardwareAccess':False,'gpuAccess':False,'training':False,'cases':cases,'workers':2,
    'sourceHashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_corrected_archive_screen.py'],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

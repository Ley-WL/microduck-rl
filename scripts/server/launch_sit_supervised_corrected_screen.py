"""One detached two-worker CPU re-screen of frozen historical exports."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/supervised-corrected-screen';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Follow recorded launch'
cases=('Broad1','Broad3','Broad10','Knee1','Knee2','Knee3','Knee10','Knee30','Gate1','Gate2','Gate3')
files=['scripts/evaluate_skills_cpu.py','scripts/bam_cpu_controller.py','scripts/server/finalize_sit_transfer.py','scripts/server/run_sit_supervised_corrected_screen.py']
files+=['policies/'+p+s for p in (*cases,'baseline') for s in ('.onnx','.metadata.json')]
files += [str(p.relative_to(root)) for p in sorted((root/'src').rglob('*')) if p.is_file() and '__pycache__' not in str(p)]
files += [str(p.relative_to(root)) for p in sorted((root/'evidence/cpu-friction-control/screening/baseline').rglob('summary.json'))]
manifest={'hardwareAccess':False,'gpuAccess':False,'training':False,'cases':cases,'workers':2,
    'sourceHashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
(folder/'launch.json').write_text(json.dumps(manifest,indent=2))
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_supervised_corrected_screen.py'],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid
(folder/'launch.tmp').write_text(json.dumps(manifest,indent=2));(folder/'launch.tmp').replace(folder/'launch.json')
print(json.dumps({'pid':process.pid,'cases':cases,'sourceCount':len(files)}))

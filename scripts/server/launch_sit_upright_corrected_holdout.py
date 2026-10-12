"""One CPU holdout of frozen Upright; corrected baseline evidence is reused."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/upright-corrected-holdout';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Track existing launch'
files=['scripts/evaluate_skills_cpu.py','scripts/bam_cpu_controller.py','scripts/server/run_sit_corrected_holdout.py',
       'scripts/server/run_sit_upright_corrected_holdout.py','policies/Upright.onnx','policies/Upright.metadata.json']
files+=['evidence/corrected-holdout/baseline/'+k+'/summary.json' for k in ('nominal','stress','gain-low','gain-high')]
manifest={'hardwareAccess':False,'gpuAccess':False,'training':False,'seeds':[12017,24029],'sourceHashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_upright_corrected_holdout.py'],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

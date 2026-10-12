"""One CPU holdout of frozen Gate3; corrected baseline evidence is reused."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/gate3-corrected-holdout';folder.mkdir(parents=True,exist_ok=True)
assert (root/'evidence/supervised-corrected-holdout/job.exit').read_text().strip()=='0'
assert json.loads((root/'evidence/supervised-corrected-screen/comparison.json').read_text())['cases']['Gate3']['eligibleForHoldout']
assert not (folder/'launch.json').exists(),'Track existing launch'
files=['scripts/evaluate_skills_cpu.py','scripts/bam_cpu_controller.py','scripts/server/run_sit_supervised_corrected_holdout.py',
       'scripts/server/run_sit_gate3_corrected_holdout.py','policies/Gate3.onnx','policies/Gate3.metadata.json']
files+=['evidence/supervised-corrected-holdout/baseline/'+k+'/summary.json' for k in ('nominal','stress','gain-low','gain-high')]
manifest={'hardwareAccess':False,'gpuAccess':False,'training':False,'seeds':[33013,44017],'sourceHashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
(folder/'launch.json').write_text(json.dumps(manifest,indent=2))
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_gate3_corrected_holdout.py'],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'launch.tmp').write_text(json.dumps(manifest,indent=2));(folder/'launch.tmp').replace(folder/'launch.json');print(json.dumps({'pid':process.pid,'candidate':'Gate3','seeds':manifest['seeds']}))

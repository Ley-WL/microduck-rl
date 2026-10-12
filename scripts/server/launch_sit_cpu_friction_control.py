"""Start one frozen corrected-physics CPU diagnostic in an isolated snapshot."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=root/'evidence/cpu-friction-control';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Track existing job'
with (folder/'tests.log').open('w') as log:
    subprocess.run([str(root.parent/'microduck-rl/.venv/bin/python'),'-m','pytest',
        'tests/test_cpu_bam_friction_constraint.py','tests/test_cpu_skill_rehearsal.py','-q'],
        cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
files=['scripts/evaluate_skills_cpu.py','scripts/bam_cpu_controller.py','scripts/server/run_sit_cpu_friction_control.py']
files+=['policies/'+k+suffix for k in ('baseline','BoundaryGain','ArrivalGate','LowExplore') for suffix in ('.onnx','.metadata.json')]
manifest={'gpuAccess':False,'hardwareAccess':False,'training':False,'cases':3,'worldsPerPolicy':80,
    'sourceHashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}}
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_cpu_friction_control.py'],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

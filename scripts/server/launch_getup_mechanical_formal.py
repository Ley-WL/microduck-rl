import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'evidence/mechanical-formal'
FOLDER.mkdir(parents=True,exist_ok=True)
assert not (FOLDER/'launch.json').exists(),'Track existing formal validation'
files=['scripts/server/validate_getup_mechanical_formal.py','scripts/server/launch_getup_mechanical_formal.py',
    'scripts/evaluate_getup_mechanical_cpu.py','scripts/bam_cpu_controller.py',
    'src/mjlab_microduck/tasks/getup_mechanical_contract.py',
    'evidence/mechanical-training/train.onnx','evidence/mechanical-training/train.metadata.json']
manifest=dict(training=False,gpuAccess=False,hardwareAccess=False,seeds=[517031,629017],
    trialsPerPosePerSeed=200,totalTrials=4800,conditions=['normal','stress'],
    sourceHashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files})
(FOLDER/'launch.json').write_text(json.dumps(manifest,indent=2))
with (FOLDER/'job.log').open('wb') as stream:
    process=subprocess.Popen([str(ROOT.parent/'microduck-rl/.venv/bin/python'),'scripts/server/validate_getup_mechanical_formal.py'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(FOLDER/'launch.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(pid=process.pid,totalTrials=4800,gpuAccess=False,hardwareAccess=False)))

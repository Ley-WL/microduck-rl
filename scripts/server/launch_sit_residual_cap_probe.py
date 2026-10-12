"""Detached CPU causal probe; preserves PID and immutable driver hashes."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=root/'evidence/residual-cap-probe-v2';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Follow the existing launch'
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),
        'scripts/server/run_sit_residual_cap_probe.py'],cwd=root,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest={'pid':process.pid,'gpuAccess':False,'hardwareAccess':False,'training':False,
    'capsDeg':[.25,.5],'sourceHashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
        ('scripts/server/evaluate_sit_residual_cap.py','scripts/server/run_sit_residual_cap_probe.py',
         'evidence/boundary-gain-control/baseline.onnx','evidence/boundary-gain-control/BoundaryGain/train.onnx')}}
(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

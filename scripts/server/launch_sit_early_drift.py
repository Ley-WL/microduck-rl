"""Detached read-only diagnostic; never duplicate a recorded launch."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=root/'evidence/early-drift';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Track the existing launch'
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),
        'scripts/server/run_sit_early_drift.py'],cwd=root,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(folder/'launch.json').write_text(json.dumps({'pid':process.pid,'hardwareAccess':False,
    'gpuAccess':False,'training':False,'driverSha256':hashlib.sha256(
        (root/'scripts/server/run_sit_early_drift.py').read_bytes()).hexdigest()},indent=2))
print(process.pid)

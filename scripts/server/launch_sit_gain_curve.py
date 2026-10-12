import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=root/'evidence/gain-curve';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Inspect current gain curve job'
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_gain_curve.py'],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest={'pid':process.pid,'hardwareAccess':False,'gpuAccess':False,'newTraining':False,'trials':64,
    'sourceSha256':hashlib.sha256((root/'scripts/evaluate_skills_cpu.py').read_bytes()).hexdigest(),
    'driverSha256':hashlib.sha256((root/'scripts/server/run_sit_gain_curve.py').read_bytes()).hexdigest()}
(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

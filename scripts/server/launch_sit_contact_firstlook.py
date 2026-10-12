"""Launch exactly one finite training/evaluation job in this isolated snapshot."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2]
evidence=root/'evidence/firstlook'
evidence.mkdir(parents=True,exist_ok=True)
report=evidence/'launch.json'
if report.exists():
    raise SystemExit('Existing first-look launch; inspect it instead of duplicating')
environment=os.environ.copy()
environment['MICRODUCK_GPU']='6'
with (evidence/'job.log').open('wb') as output:
    process=subprocess.Popen(['bash','scripts/server/run_sit_contact_firstlook.sh'],cwd=root,
        env=environment,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
payload={'pid':process.pid,'maxIterations':1000,'numEnvs':4096,'gpu':6,'hardwareAccess':False,
         'runName':environment.get('MICRODUCK_RUN_NAME','contact-firstlook1000'),
         'seedRun':environment.get('MICRODUCK_SEED_RUN'),
         'objectiveConfigSha256':hashlib.sha256((root/'src/mjlab_microduck/tasks/microduck_hd1910_sit_contact_env_cfg.py').read_bytes()).hexdigest(),
         'rewardImplementationSha256':hashlib.sha256((root/'src/mjlab_microduck/tasks/mdp.py').read_bytes()).hexdigest(),
         'scriptSha256':hashlib.sha256((root/'scripts/server/run_sit_contact_firstlook.sh').read_bytes()).hexdigest()}
report.write_text(json.dumps(payload,indent=2))
print(json.dumps(payload))

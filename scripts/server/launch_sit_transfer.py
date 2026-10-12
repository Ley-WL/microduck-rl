"""One finite 1000-iteration launch; verify smoke exit/export first."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
evidence=root/'evidence/train';evidence.mkdir(parents=True,exist_ok=True)
assert (root/'evidence/smoke/job.exit').read_text().strip()=='0'
assert (root/'evidence/smoke/completed.txt').is_file()
assert (root/'evidence/smoke/policy.onnx').is_file()
smoke=json.loads((root/'evidence/smoke/contract-check.json').read_text())
assert smoke['finite'] and smoke['transitionKickCount']>0 and smoke['normalizerExactlyPreservedInSmoke']
assert not (evidence/'launch.json').exists(),'Existing job: inspect instead of duplicate'
environment=os.environ.copy();environment['MICRODUCK_GPU']='6'
with (evidence/'job.log').open('wb') as output:
    process=subprocess.Popen(['bash','scripts/server/run_sit_transfer.sh','train'],cwd=root,
        env=environment,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
payload={'pid':process.pid,'iterations':1000,'numEnvs':4096,'gpu':6,'hardwareAccess':False,
         'sourceHashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in [root/'seed.pt',root/'transfer-data.npz',root/'src/mjlab_microduck/tasks/mdp.py',
                     root/'src/mjlab_microduck/tasks/sit_transfer_ppo.py',root/'src/mjlab_microduck/tasks/microduck_hd1910_sit_transfer_env_cfg.py',
                     root/'scripts/server/run_sit_transfer.sh',root/'scripts/server/finalize_sit_transfer.py']}}
(evidence/'launch.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload))

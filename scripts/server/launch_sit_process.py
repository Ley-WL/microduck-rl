"""Finite isolated process candidate; smoke guard and duplicate launch guard."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train'],required=True);a=p.parse_args()
folder=root/'evidence'/a.mode
assert not (folder/'launch.json').exists(),'Read the existing job instead of relaunching'
if a.mode=='train':
    smoke=root/'evidence/smoke'
    assert (smoke/'job.exit').read_text().strip()=='0' and (smoke/'completed.txt').is_file()
    proof=json.loads((smoke/'contract-check.json').read_text())
    assert proof['finite'] and proof['transitionKickCount']>0 and proof['normalizerExactlyPreservedInSmoke']
    smoke_launch=json.loads((smoke/'launch.json').read_text())
    assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in smoke_launch['sourceHashes'].items()),'Training sources changed after smoke'
usage=subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
assert int(usage)<200,'GPU6 is no longer free'
seed=root/'logs/rsl_rl/hd1910_sitstand_process_v2/transfer_seed/seed.pt'
seed.parent.mkdir(parents=True,exist_ok=True)
if not seed.exists():seed.write_bytes((root/'seed.pt').read_bytes())
folder.mkdir(parents=True,exist_ok=True)
environment=os.environ.copy();environment['MICRODUCK_GPU']='6'
with (folder/'job.log').open('wb') as output:
    process=subprocess.Popen(['bash','scripts/server/run_sit_process.sh',a.mode],cwd=root,env=environment,
        stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
sources=['seed.pt','transfer-data.npz','src/mjlab_microduck/tasks/mdp.py',
         'src/mjlab_microduck/tasks/microduck_hd1910_sit_process_env_cfg.py',
         'src/mjlab_microduck/tasks/sit_process_cost.py','src/mjlab_microduck/tasks/sit_transfer_ppo.py',
         'scripts/server/run_sit_process.sh']
manifest={'pid':process.pid,'iterations':5 if a.mode=='smoke' else 1000,'numEnvs':64 if a.mode=='smoke' else 4096,
          'gpu':6,'hardwareAccess':False,'sourceHashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}}
(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

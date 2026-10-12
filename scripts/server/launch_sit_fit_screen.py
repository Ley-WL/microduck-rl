"""Detach CPU screening from SSH transport; preserve bounded source manifest."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--weights',type=float,nargs='+',required=True)
a=p.parse_args();folder=root/'evidence'/a.folder
assert not (folder/'screen-launch.json').exists(),'Inspect existing screen before repeating'
environment=os.environ.copy();environment['OMP_NUM_THREADS']='1'
with (folder/'screen.log').open('w') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/screen_sit_broad_fit.py',
        '--folder',a.folder,'--weights',*map(str,a.weights)],cwd=root,env=environment,
        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
manifest={'pid':process.pid,'weights':a.weights,'hardwareAccess':False,
    'sourceHashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
        ['scripts/server/screen_sit_broad_fit.py','scripts/server/finalize_sit_transfer.py','scripts/evaluate_skills_cpu.py']}}
(folder/'screen-launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

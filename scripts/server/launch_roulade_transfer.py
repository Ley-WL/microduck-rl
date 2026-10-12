import json,hashlib,subprocess,os,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence/roulade-transfer'
p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,default=1);args=p.parse_args()
name='launch.json' if args.attempt==1 else 'launch-v'+str(args.attempt)+'.json'
assert not (OUT/name).exists(),'Follow existing PID; never duplicate'
if args.attempt>1:
    assert (OUT/'job.exit').read_text().strip()=='1','Only explicitly failed preflight may resume'
    previous=json.loads((OUT/'launch.json').read_text())
    assert subprocess.run(['kill','-0',str(previous['pid'])],stderr=subprocess.DEVNULL).returncode!=0,'Previous launcher still alive'
    (OUT/'job.exit').unlink()
assert (OUT/'config-tests.exit').read_text().strip()=='0'
files=[p for folder in ('src','inputs') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
files += [ROOT/p for p in ('scripts/server/prepare_roulade_transfer.py','scripts/server/run_roulade_transfer.py',
    'scripts/server/screen_roulade_transfer.py','scripts/server/launch_roulade_transfer.py','scripts/export.py',
    'scripts/bam_cpu_controller.py','tests/test_roulade_transfer.py')]
manifest=dict(training=True,hardwareAccess=False,allowedGpus=[0,1],minimumFreeMiB=12288,maxUtilization=30,
    trainingSeed=62173,branches=['Control','Repair'],mainIterations=20,worlds=1024,
    sourceHashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},autoDeployment=False)
env=os.environ.copy();env['ROULADE_LAUNCH_MANIFEST']=str(OUT/name)
with (OUT/('job.log' if args.attempt==1 else 'job-v'+str(args.attempt)+'.log')).open('wb') as stream:
    p=subprocess.Popen([str(ROOT.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_roulade_transfer.py'],
        cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
manifest['pid']=p.pid;(OUT/name).write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest | {'sourceHashes':'saved'}))

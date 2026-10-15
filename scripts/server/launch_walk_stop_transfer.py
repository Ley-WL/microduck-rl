"""Finite isolated training job with smoke and frozen paired screen."""
import os,sys,json,subprocess,hashlib,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
if '--worker' not in sys.argv:
 assert not (OUT/'launch.json').exists(),'Inspect existing job; do not restart'
 manifest=dict(hardwareAccess=False,autoDeployment=False,smoke=[64,5],main=[1024,100],gpu=1,sourceHashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','scripts','inputs'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)})
 (OUT/'launch.json').write_text(json.dumps(manifest,indent=2))
 with (OUT/'job.log').open('w') as log:
  process=subprocess.Popen([sys.executable,__file__,'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 (OUT/'pid').write_text(str(process.pid));print(json.dumps({'pid':process.pid}));sys.exit()
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'),MICRODUCK_WALK_SURFACE_INPUTS=str(ROOT/'inputs'),CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline')
def run(stage,script,*args):
 (OUT/'progress.json').write_text(json.dumps({'stage':stage,'updated':time.time()}))
 with (OUT/(stage+'.log')).open('w') as log:subprocess.run([sys.executable,'scripts/server/'+script,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 text=(OUT/(stage+'.log')).read_text();assert 'contact match overflow' not in text and 'collision overflow' not in text
status=1
try:
 memory,usage=map(int,subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
 assert memory>12288 and usage<30,(memory,usage)
 run('tests','test_walk_stop_transfer.py')
 run('prepare','prepare_walk_stop_transfer.py')
 shutil.copy2(OUT/'seed.pt',ROOT/'inputs/seed.pt')
 run('smoke','train_walk_stop_transfer.py','--envs','64','--iterations','5','--output','evidence/smoke')
 run('main','train_walk_stop_transfer.py','--envs','1024','--iterations','100','--output','evidence/main')
 run('screen','evaluate_walk_stop_transfer.py')
 manifest=json.loads((OUT/'launch.json').read_text());assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest['sourceHashes'].items())
 (OUT/'source-check.json').write_text(json.dumps({'passed':True}))
 (OUT/'progress.json').write_text(json.dumps({'stage':'complete','accepted':json.loads((OUT/'paired-screen.json').read_text())['accepted'],'hardwareValidated':False}))
 status=0
finally:(OUT/'job.exit').write_text(str(status))

"""Bounded baseline, smoke, transfer and candidate evaluation pipeline."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
GPU=os.environ.get('FLAMINGO_GPU','2')
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
def pipeline():
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES=GPU,MUJOCO_GL='egl',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline',PYTHONUNBUFFERED='1')
    env.pop('MICRODUCK_WARM_START',None)
    stages=[('baseline','eval',ROOT/'inputs/policy.onnx',16,0),('smoke','train',ROOT/'inputs/policy.onnx',64,5),('refine100','train',ROOT/'inputs/policy.onnx',1024,100),('candidate','eval',OUT/'refine100/policy.onnx',16,0)]
    try:
        for name,mode,policy,worlds,iters in stages:
            save('progress.json',dict(stage=name,startedAt=time.time(),hardware=False))
            with (OUT/(name+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/server/flamingo_hd1910_trial.py'),mode,'--policy',str(policy),'--output',str(OUT/name),'--envs',str(worlds),'--iterations',str(iters)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        b=json.loads((OUT/'baseline/results.json').read_text());c=json.loads((OUT/'candidate/results.json').read_text())
        comparison={}
        for key in b:
            assert b[key]['initialQposQvelSha256']==c[key]['initialQposQvelSha256'],key
            comparison[key]={field:{'baseline':b[key][field],'candidate':c[key][field]} for field in ['survived','holdSuccess','lowerSuccess','targetDeltaP95Deg']}
        save('comparison.json',comparison);save('progress.json',dict(stage='complete',hardware=False,accepted=False,needsVideoReview=True));(OUT/'job.exit').write_text('0')
    except BaseException as error:
        save('progress.json',dict(stage='failed',error=repr(error),hardware=False));(OUT/'job.exit').write_text('1');raise
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='pipeline':pipeline()
    else:
        assert not (OUT/'launch.json').exists(),'Already launched'
        free,util=map(int,subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        assert free>16000 and util<15,(free,util)
        hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','scripts','inputs'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        save('source-manifest.json',hashes)
        with (OUT/'main.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'pipeline'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        report=dict(pid=child.pid,gpu=GPU,root=str(ROOT),startedAt=time.time(),hardware=False);save('launch.json',report);print(json.dumps(report))

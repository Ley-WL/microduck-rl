"""Smoke, finite export, bounded refinement, matched rollout screen. No deployment."""
import json, os, subprocess, sys, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence'
ENV=os.environ.copy()
ENV.update(PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES='1',MUJOCO_GL='egl',
           OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline',MICRODUCK_WARM_START='0')
TASK=os.environ.get('DANCE_TASK','Mjlab-Dance10Stable-Flat-MicroDuck-HD1910')


def status(stage,**kw):
    (OUT/'progress.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))


def run(args,name):
    status(name)
    with (OUT/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)


def gate():
    s=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    free,util=map(int,s.strip().split(','));assert free>12288 and util<30,(free,util)


def main():
    hashes=json.loads((ROOT/'source-manifest.json').read_text())
    for path,sha in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    for name,n,iters in [('smoke',64,5),('refine400',1024,400)]:
        gate()
        run(['scripts/server/train_dance10_stable.py','--checkpoint',ROOT/'seed.pt',
             '--output',OUT/name,'--envs',n,'--iterations',iters,'--task',TASK,
             '--exploration',os.environ.get('DANCE_EXPLORATION','.10')],name)
        run(['scripts/export.py',TASK,'--checkpoint-file',OUT/name/'final.pt',
             '--onnx-file',OUT/(name+'.onnx'),'--num-envs',1,'--device','cuda:0'],name+'-export')
        import onnxruntime as ort
        import onnx
        import numpy as np
        model=onnx.load(str(OUT/(name+'.onnx')))
        assert any('normalizer' in v.name for v in model.graph.initializer)
        options=ort.SessionOptions();options.intra_op_num_threads=1
        session=ort.InferenceSession(str(OUT/(name+'.onnx')),sess_options=options,providers=['CPUExecutionProvider'])
        for row in np.random.default_rng(97211).normal(0,.1,(16,61)).astype('float32'):
            value=session.run(None,{session.get_inputs()[0].name:row[None]})[0]
            assert value.shape==(1,14) and np.isfinite(value).all()
    reports=[]
    for profile,seed,n in [('nominal',81371,32),('nominal',91373,32),('robust',81371,32),('robust',91373,32),('stress',81371,64)]:
        name=profile+'-'+str(seed)
        args=['scripts/server/evaluate_dance10.py','--checkpoint',OUT/'refine400/final.pt','--output',OUT/name,'--seed',seed,'--envs',n]
        if profile!='nominal':args+=['--robust']
        if profile=='stress':args+=['--stress']
        if profile=='nominal' and seed==81371:args+=['--video']
        run(args,name)
        candidate=json.loads((OUT/name/'summary.json').read_text())
        baseline=json.loads((ROOT/'baseline'/name/'summary.json').read_text())
        before=np.array(baseline['firstFailureSeconds'])<0
        after=np.array(candidate['firstFailureSeconds'])<0
        reports.append(dict(profile=name,baseline=int(before.sum()),candidate=int(after.sum()),
            lost=int((before&~after).sum()),gained=int((~before&after).sum()),
            meanHeadRmseBefore=float(np.mean(baseline['headRmseRad'])),meanHeadRmseAfter=float(np.mean(candidate['headRmseRad'])),
            minHeadYawAmplitudeBefore=float(min(baseline['headYawPeakToPeakRad'])),minHeadYawAmplitudeAfter=float(min(candidate['headYawPeakToPeakRad']))))
    (OUT/'comparison.json').write_text(json.dumps(reports,indent=2))
    fresh=[]
    if os.environ.get('DANCE_FRESH_SCREEN')=='1':
        # Held-out seeds were not used to select v2 or tune this recovery recipe.
        for seed in (100003,100019):
            for label, checkpoint in [('baseline',ROOT/'original-v1.pt'),('candidate',OUT/'refine400/final.pt')]:
                name=f'fresh-{seed}-{label}'
                run(['scripts/server/evaluate_dance10.py','--checkpoint',checkpoint,'--output',OUT/name,
                     '--seed',seed,'--envs',64,'--robust','--stress'],name)
            b=json.loads((OUT/f'fresh-{seed}-baseline/summary.json').read_text())
            c=json.loads((OUT/f'fresh-{seed}-candidate/summary.json').read_text())
            for field in ('initialQposSha256','initialQvelSha256','initialActorObservationSha256'):
                assert b[field]==c[field], (seed,field)
            fresh.append(dict(seed=seed,baseline=b['survived'],candidate=c['survived'],
                initialQposQvelObservationMatched=True,
                lost=sum(x<0 and y>=0 for x,y in zip(b['firstFailureSeconds'],c['firstFailureSeconds']))))
        (OUT/'fresh-comparison.json').write_text(json.dumps(fresh,indent=2))
    status('screen-complete',comparison=reports,freshComparison=fresh,accepted=False,needsVideoReview=True,deployment=False)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    try:main()
    except BaseException as e:
        status('failed',error=repr(e));(OUT/'job.exit').write_text('1');raise
    (OUT/'job.exit').write_text('0')

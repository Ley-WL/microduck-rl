"""Larger ten-beat dance: frozen reference, smoke, ramped training, full-size review."""
import json, os, subprocess, sys, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence'
TASK='Mjlab-Dance10Expressive-Flat-MicroDuck-HD1910'
ENV=os.environ.copy()
ENV.update(PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES='1',MUJOCO_GL='egl',
           OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline',MICRODUCK_WARM_START='0')


def status(stage,**kw):
    (OUT/'progress.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))


def run(args,name):
    status(name)
    with (OUT/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)


def gate():
    v=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    free,util=map(int,v.strip().split(','));assert free>12288 and util<30,(free,util)


def evaluate(checkpoint,name,seed=110017,robust=False,stress=False,small=False,video=False):
    args=['scripts/server/evaluate_dance10.py','--checkpoint',checkpoint,'--output',OUT/name,
          '--seed',seed,'--envs',32,'--task',('Mjlab-Dance10Recovery-Flat-MicroDuck-HD1910' if small else TASK)]
    if robust:args+=['--robust']
    if stress:args+=['--stress']
    if video:args+=['--video']
    run(args,name)


def main():
    manifest=json.loads((ROOT/'source-manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    gate()
    evaluate(ROOT/'seed.pt','reference-small',small=True)
    evaluate(ROOT/'seed.pt','reference-full',video=True)
    for name,worlds,iterations in [('smoke',64,5),('expressive800',1024,800)]:
        gate()
        run(['scripts/server/train_dance10_stable.py','--checkpoint',ROOT/'seed.pt',
             '--output',OUT/name,'--envs',worlds,'--iterations',iterations,
             '--task',TASK,'--exploration','.10'],name)
        run(['scripts/export.py',TASK,'--checkpoint-file',OUT/name/'final.pt',
             '--onnx-file',OUT/(name+'.onnx'),'--num-envs',1,'--device','cuda:0'],name+'-export')
        import onnxruntime as ort
        import onnx
        graph=onnx.load(str(OUT/(name+'.onnx')))
        assert any('normalizer' in v.name for v in graph.graph.initializer)
        options=ort.SessionOptions();options.intra_op_num_threads=1
        session=ort.InferenceSession(str(OUT/(name+'.onnx')),sess_options=options,providers=['CPUExecutionProvider'])
        for row in np.random.default_rng(110029).normal(0,.1,(16,61)).astype('float32'):
            values=session.run(None,{session.get_inputs()[0].name:row[None]})[0]
            assert values.shape==(1,14) and np.isfinite(values).all()
    checkpoint=OUT/'expressive800/final.pt'
    for seed in (110017,110023):
        evaluate(checkpoint,f'nominal-{seed}',seed,video=seed==110017)
        evaluate(checkpoint,f'robust-{seed}',seed,robust=True)
        evaluate(checkpoint,f'stress-{seed}',seed,robust=True,stress=True)
    reports=[]
    for folder in [OUT/'reference-small',OUT/'reference-full',*sorted(OUT.glob('nominal-*')), *sorted(OUT.glob('robust-*')),*sorted(OUT.glob('stress-*'))]:
        if not folder.is_dir():continue
        summary=json.loads((folder/'summary.json').read_text())
        d=np.load(folder/'trace.npz');alive=np.array(summary['firstFailureSeconds'])<0
        active=(d['time']%8>=1)&(d['time']%8<=7)
        # Report each surviving world's real motion, not target commands or reset jumps.
        def span(x):
            values=np.ptp(x[active][:,alive],axis=0)
            return dict(mean=float(values.mean()),min=float(values.min())) if len(values) else None
        reports.append(dict(profile=folder.name,survived=int(alive.sum()),worlds=len(alive),
            headYawSpanRad=span(d['head'][:,:,2]),headRollSpanRad=span(d['head'][:,:,3]),
            headPitchSpanRad=span(d['head'][:,:,1]),bodyRollSpanRad=span(d['roll']),
            bodyHeightSpanM=span(d['position'][:,:,2]),lateralSpanM=span(d['position'][:,:,1])))
    (OUT/'amplitude-comparison.json').write_text(json.dumps(reports,indent=2))
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    status('screen-complete-needs-video-review',amplitudeComparison=reports,accepted=False,deployment=False)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    try:main()
    except BaseException as e:
        status('failed',error=repr(e));(OUT/'job.exit').write_text('1');raise
    (OUT/'job.exit').write_text('0')

"""Full-amplitude recovery refinement with preserved v4 and fresh-seed comparisons."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
import run_dance10_expressive as ops

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence'
TASK='Mjlab-Dance10Resilient-Flat-MicroDuck-HD1910'
EVAL_TASK='Mjlab-Dance10Expressive-Flat-MicroDuck-HD1910'


def export(checkpoint,name):
    ops.run(['scripts/export.py',TASK,'--checkpoint-file',checkpoint,'--onnx-file',OUT/(name+'.onnx'),
             '--num-envs',1,'--device','cuda:0'],name+'-export')
    import onnx,onnxruntime as ort
    graph=onnx.load(str(OUT/(name+'.onnx')))
    assert any('normalizer' in v.name for v in graph.graph.initializer)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(OUT/(name+'.onnx')),sess_options=options,providers=['CPUExecutionProvider'])
    for row in np.random.default_rng(121039).normal(0,.1,(16,61)).astype('float32'):
        values=session.run(None,{session.get_inputs()[0].name:row[None]})[0]
        assert values.shape==(1,14) and np.isfinite(values).all()


def evaluate(checkpoint,name,seed,profile,worlds=32,video=False):
    args=['scripts/server/evaluate_dance10.py','--checkpoint',checkpoint,'--output',OUT/name,
          '--task',EVAL_TASK,'--seed',seed,'--envs',worlds]
    if profile!='nominal':args+=['--robust']
    if profile=='stress':args+=['--stress']
    if video:args+=['--video']
    ops.run(args,name)
    return json.loads((OUT/name/'summary.json').read_text())


def compare(b,c,name):
    for field in ('initialQposSha256','initialQvelSha256','initialActorObservationSha256'):
        assert b[field]==c[field],(name,field)
    before=np.array(b['firstFailureSeconds'])<0;after=np.array(c['firstFailureSeconds'])<0
    d=np.load(OUT/name/'trace.npz');active=(d['time']%8>=1)&(d['time']%8<=7)
    yaw=np.ptp(d['head'][active,:,2],axis=0)
    nod=np.ptp(d['head'][active,:,1],axis=0)
    body=np.ptp(d['roll'][active],axis=0)
    height=np.ptp(d['position'][active,:,2],axis=0)
    # Fix the minimum real motion before seeing new results; never count resets as dancing.
    motion=after&(yaw>=.90)&(nod>=.40)&(body>=.18)&(height>=.014)
    return dict(profile=name,baseline=int(before.sum()),candidate=int(after.sum()),worlds=len(after),
        lostWorlds=np.flatnonzero(before&~after).tolist(),gainedWorlds=np.flatnonzero(~before&after).tolist(),
        fullMotionPassed=int(motion.sum()),initialQposQvelObservationMatched=True,
        survivingHeadYawMeanRad=float(yaw[after].mean()) if after.any() else None,
        survivingBodyRollMeanRad=float(body[after].mean()) if after.any() else None,
        survivingHeightSpanMeanM=float(height[after].mean()) if after.any() else None)


def main():
    manifest=json.loads((ROOT/'source-manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    for name,worlds,iterations in [('smoke',64,5),('resilient1000',1024,1000)]:
        ops.gate()
        ops.run(['scripts/server/train_dance10_stable.py','--checkpoint',ROOT/'seed.pt',
                 '--output',OUT/name,'--envs',worlds,'--iterations',iterations,
                 '--task',TASK,'--exploration','.08','--seed',121031],name)
        export(OUT/name/'final.pt',name)
    checkpoint=OUT/'resilient1000/final.pt';reports=[]
    for seed in (110017,110023):
        for profile in ('nominal','robust','stress'):
            name=f'{profile}-{seed}'
            c=evaluate(checkpoint,name,seed,profile,video=profile=='nominal' and seed==110017)
            b=json.loads((ROOT/'baseline'/name/'summary.json').read_text())
            reports.append(compare(b,c,name))
            (OUT/'comparison.json').write_text(json.dumps(reports,indent=2))
    for seed in (120011,120017):
        b=evaluate(ROOT/'seed.pt',f'fresh-baseline-{seed}',seed,'stress',64)
        name=f'fresh-candidate-{seed}'
        c=evaluate(checkpoint,name,seed,'stress',64)
        reports.append(compare(b,c,name))
        (OUT/'comparison.json').write_text(json.dumps(reports,indent=2))
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    ops.status('screen-complete-needs-video-review',comparison=reports,accepted=False,deployment=False)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    try:main()
    except BaseException as e:
        ops.status('failed',error=repr(e));(OUT/'job.exit').write_text('1');raise
    (OUT/'job.exit').write_text('0')

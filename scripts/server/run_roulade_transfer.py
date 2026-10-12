"""Smoke -> two matched 20-iteration roots -> independent frozen roll screen."""
import json,subprocess,os,time,hashlib,copy
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence/roulade-transfer'
PY=ROOT.parent/'microduck-rl/.venv/bin/python';SEED=62173
ENV=os.environ.copy();ENV.update(PYTHONPATH=str(ROOT/'src'),MICRODUCK_ROULADE_INPUTS=str(ROOT/'inputs'),
    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline',MUJOCO_GL='egl',MICRODUCK_WARM_START='1')

def status(stage,**kw):
    (OUT/'progress.json').write_text(json.dumps(dict(stage=stage,time=time.time(),**({'trainingSeed':SEED}|kw)),indent=2))

def capacity(stage):
    while True:
        rows=[list(map(int,line.split(','))) for line in subprocess.check_output(['nvidia-smi','--id=0,1',
            '--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
        ready=[r for r in rows if r[1]>12288 and r[2]<30]
        if ready:
            chosen=max(ready,key=lambda r:r[1]);ENV['CUDA_VISIBLE_DEVICES']=str(chosen[0]);status(stage,gpu=chosen[0],gpuCapacity=rows);return
        status('waiting-capacity',nextStage=stage,gpuCapacity=rows);time.sleep(20)

def command(args,label):
    with (OUT/(label+'.log')).open('w') as stream:
        subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert 'contact match overflow' not in (OUT/(label+'.log')).read_text(), 'Contact capacity invalidates this stage'

def export(checkpoint,label):
    task='Mjlab-RouladeTransferControl-Flat-MicroDuck'
    capacity('export-'+label)
    command([PY,'scripts/export.py',task,'--checkpoint-file',checkpoint,'--onnx-file',OUT/(label+'.onnx'),
        '--num-envs','1','--device','cuda:0'],label+'-export')
    session=ort.InferenceSession(str(OUT/(label+'.onnx')),providers=['CPUExecutionProvider'])
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert all(bool(torch.isfinite(v).all()) for key in ('actor_state_dict','critic_state_dict') for v in state[key].values())
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from dataclasses import asdict
    from mjlab.tasks.registry import load_rl_cfg
    import mjlab_microduck.tasks
    cfg={k:v for k,v in asdict(load_rl_cfg(task).actor).items() if k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
    rows=np.random.default_rng(62173).normal(0,.5,(128,61)).astype(np.float32)
    obs=TensorDict({'actor':torch.tensor(rows)},batch_size=[128]);model=MLPModel(obs,{'actor':['actor']},'actor',14,**cfg)
    model.load_state_dict(state['actor_state_dict']);model.eval()
    with torch.no_grad():expected=model(obs).numpy()
    actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:r[None]})[0] for r in rows])
    error=float(abs(actual-expected).max());assert error<2e-5,error
    steps={int(v['step']) for v in state['optimizer_state_dict']['state'].values()}
    (OUT/(label+'-export-check.json')).write_text(json.dumps(dict(finite=True,normalizerIncluded=True,
        onnxTorchMaxError=error,adamSteps=sorted(steps),checkpoint=str(checkpoint),hardwareAccess=False),indent=2))
    return steps

def screen(checkpoint,label,seeds=None):
    capacity('screen-'+label)
    args=[PY,'scripts/server/screen_roulade_transfer.py','--checkpoint',checkpoint,'--output',OUT/label]
    if seeds:args+=['--seeds',*seeds]
    command(args,label)
    return json.loads((OUT/label/'summary.json').read_text())

def compare(base,candidate):
    lost=0;times=[];before=[]
    for b,c in zip(base['results'],candidate['results']):
        assert b['seed']==c['seed'] and b['voltage']==c['voltage']
        assert np.array_equal(b['initialQpos'],c['initialQpos'])
        lost+=sum(bs and not cs for bs,cs in zip(b['success'],c['success']))
        for bs,cs,bt,ct in zip(b['success'],c['success'],b['completionSeconds'],c['completionSeconds']):
            if bs and cs:before.append(bt);times.append(ct)
    return dict(lostBaselineSuccesses=lost,pairedMeanSecondsBefore=float(np.mean(before)) if before else None,
        pairedMeanSecondsAfter=float(np.mean(times)) if times else None,
        successes=sum(sum(r['success']) for r in candidate['results']),
        targetFrameMaxDeg=max(r['targetFrameMaxDeg'] for r in candidate['results']),
        baseAccelerationMax=max(r['baseAccelerationMax'] for r in candidate['results']))

def main():
    seed=ROOT/'logs/rsl_rl/roulade_mechanical_transfer/roll_seed/seed.pt'
    capacity('prepare-seed-and-live-chain-v3')
    command([PY,'scripts/server/prepare_roulade_transfer.py'],'prepare-seed-v3')
    if not (OUT/'baseline-export-check.json').exists():export(seed,'baseline')
    base=screen(seed,'baseline-screen-v3')
    checkpoints={}
    for branch in ('Control','Repair'):
        task='Mjlab-RouladeTransfer'+branch+'-Flat-MicroDuck'
        for stage,worlds,iters in [('smoke',64,5),('main',1024,20)]:
            capacity(branch+'-'+stage)
            run=branch.lower()+'-'+stage+'-v3-seed62173'
            logs=ROOT/'logs/rsl_rl/roulade_mechanical_transfer'
            assert not list(logs.glob('*'+run)),'Do not repeat an existing training run'
            command([PY.parent/'train',task,'--env.scene.num-envs',worlds,'--gpu-ids','all',
                '--agent.max-iterations',iters,'--agent.seed',SEED,'--agent.resume','True',
                '--agent.load-run','roll_seed','--agent.load-checkpoint','seed.pt',
                '--agent.run-name',run,'--agent.logger','tensorboard'],run)
            matches=list(logs.glob('*'+run+'/model_'+str(iters-1)+'.pt'));assert len(matches)==1
            steps=export(matches[0],branch.lower()+'-'+stage+'-v3')
            assert steps=={iters*20},steps
            if stage=='main':checkpoints[branch]=matches[0]
    reports={b:screen(p,b.lower()+'-screen-v3') for b,p in checkpoints.items()}
    result={b:dict(vsBaseline=compare(base,r),vsControl=compare(reports['Control'],r)) for b,r in reports.items()}
    result.update(deployment=False,formalAccepted=False,optimizerStepsPerMain=400,trainingSeed=SEED,
        note='Shared seed does not imply identical training trajectories; fresh critic and synthesized actor count are transfer assumptions')
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    manifest=json.loads(Path(os.environ.get('ROULADE_LAUNCH_MANIFEST',OUT/'launch.json')).read_text())
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in manifest['sourceHashes'].items())
    status('small-budget-complete',**result)

if __name__=='__main__':
    try:main();(OUT/'job.exit').write_text('0')
    except BaseException:
        status('failed');(OUT/'job.exit').write_text('1');raise

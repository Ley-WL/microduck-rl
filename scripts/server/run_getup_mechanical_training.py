"""Bounded warm-start experiment on user-authorized GPUs0/1; no hardware access."""
import hashlib,json,os,subprocess,time,sys
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'evidence/mechanical-training'
PYTHON=ROOT.parent/'microduck-rl/.venv/bin/python'
TASK='Mjlab-GetupMechanical-Flat-MicroDuck-HD1910'
ENV=os.environ.copy();ENV.update(PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES='0,1',
    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',WANDB_MODE='offline')
TRAIN_SEED=61217

def status(stage,**extra):
    (OUT/'progress.json').write_text(json.dumps(dict(stage=stage,trainingSeed=TRAIN_SEED,time=time.time(),**extra),indent=2))

def wait_gpu(stage):
    # The user explicitly allows GPUs0/1 with resident low-utilization jobs.
    # Keep at least 12GiB headroom on each; never terminate/evict another job.
    consecutive=0
    while consecutive<2:
        lines=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
        usage=[list(map(int,line.split(','))) for line in lines]
        available=len(usage)==2 and all(free>12288 and util<30 for _,free,util in usage)
        consecutive=consecutive+1 if available else 0
        status('waiting-gpu',nextStage=stage,gpuFreeMemoryAndUtilization=usage)
        if consecutive<2:time.sleep(10)
    status(stage,gpuBefore=usage)

def command(args,log,cpu=False):
    env=ENV.copy()
    if cpu:env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MUJOCO_GL='egl',LIBGL_ALWAYS_SOFTWARE='1')
    with log.open('w') as stream:subprocess.run(list(map(str,args)),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def export(checkpoint,name):
    wait_gpu('export-'+name)
    policy=OUT/(name+'.onnx')
    command([PYTHON,'scripts/export.py',TASK,'--checkpoint-file',checkpoint,'--onnx-file',policy,'--num-envs','1','--device','cuda:0'],OUT/(name+'-export.log'))
    command([PYTHON,'scripts/server/write_getup_mechanical_metadata.py',policy],OUT/(name+'-metadata.log'),cpu=True)
    session=ort.InferenceSession(str(policy),providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape[-1]==61
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert all(bool(torch.isfinite(v).all()) for name in ('actor_state_dict','critic_state_dict') for v in state[name].values())
    rows=np.random.default_rng(90217).normal(0,.2,(64,61)).astype(np.float32)
    values=np.concatenate([session.run(None,{session.get_inputs()[0].name:r[None]})[0] for r in rows])
    assert values.shape==(64,14) and np.isfinite(values).all()
    # Exporter uses actor.as_onnx including the updated normalizer. Reconstruct
    # CPU MLP to verify the exported mapping, not just shape or finite values.
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from mjlab.tasks.registry import load_rl_cfg
    import mjlab_microduck.tasks
    from dataclasses import asdict
    cfg={k:v for k,v in asdict(load_rl_cfg(TASK).actor).items() if k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
    obs=TensorDict({'actor':torch.tensor(rows)},batch_size=[len(rows)])
    model=MLPModel(obs,{'actor':['actor']},'actor',14,**cfg)
    model.load_state_dict(state['actor_state_dict']);model.eval()
    with torch.no_grad():expected=model(obs).numpy()
    error=float(np.max(np.abs(values-expected)));assert error<2e-5,error
    (OUT/(name+'-export-check.json')).write_text(json.dumps(dict(finite=True,observationDim=61,actionDim=14,normalizerIncluded=True,onnxTorchMaxAbsError=error,hardwareAccess=False)))
    return policy

def evaluate(policy,name,seeds='15601',trials=2,stress=False):
    folder=OUT/name;folder.mkdir()
    args=[PYTHON,'scripts/evaluate_getup_mechanical_cpu.py','--onnx',policy,'--profile','runtime','--seeds',*seeds.split(','),'--trials',trials,'--output',folder]
    if stress:args+=['--transition-stress','--joint-velocity-lag-steps','1']
    command(args,OUT/(name+'.log'),cpu=True)
    report=json.loads((folder/'summary.json').read_text())
    assert report['configuration']['frictionConstraintRevision']=='dof-id-friction-force-v1'
    assert report['configuration']['mechanicalExecutionContract']=='kissqy-mechanical-ema045-rawhistory-encoder4096-v1'
    return report

def count(report):
    rows=[r for g in report['groups'] for r in g['results']]
    return dict(successes=sum(r['success'] for r in rows),trials=len(rows))

def main():
    assert (OUT/'config-tests.exit').read_text().strip()=='0'
    baseline=OUT/'baseline.onnx'
    if not (OUT/'baseline-export-check.json').exists():baseline=export(ROOT/'seed.pt','baseline')
    summary=OUT/'baseline-screen/summary.json'
    baseline_report=json.loads(summary.read_text()) if summary.exists() else evaluate(baseline,'baseline-screen')
    for name,worlds,iterations in [('smoke',64,5),('train',1024,300)]:
        wait_gpu(name)
        run='mechanical-'+name+'-f64v2-seed61217'
        logs=ROOT/'logs/rsl_rl/hd1910_getup_mechanical'
        previous=list(logs.glob('*'+run))
        if name=='smoke' and previous:
            assert len(previous)==1 and (OUT/'smoke-export-check.json').exists(), 'Incomplete smoke requires inspection'
        else:
            assert not previous, 'Do not repeat a run'
            command([PYTHON.parent/'train',TASK,'--env.scene.num-envs',worlds,
                '--gpu-ids','all',
                '--agent.max-iterations',iterations,'--agent.seed',TRAIN_SEED,'--agent.resume','True',
                '--agent.load-run','mechanical_seed','--agent.load-checkpoint','seed.pt',
                '--agent.run-name',run,'--agent.logger','tensorboard'],OUT/(name+'.log'))
        checkpoints=list(logs.glob('*'+run+'/model_'+str(iterations-1)+'.pt'));assert len(checkpoints)==1
        policy=OUT/(name+'.onnx') if previous else export(checkpoints[0],name)
        if name=='smoke':
            wait_gpu('live-action-audit')
            command([PYTHON,'scripts/server/audit_getup_mechanical_chain.py','--output',OUT/'live-chain.json'],OUT/'live-chain.log')
    # Baseline and trained screen share initial physics, seed, scenario order.
    trained=evaluate(policy,'trained-screen')
    result=dict(baseline=count(baseline_report),trained=count(trained),deployment=False,formalAccepted=False)
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    if result['trained']['successes']>=11:
        normal=evaluate(policy,'fresh-normal','137031,249017',4)
        stressed=evaluate(policy,'fresh-stress','137031,249017',4,True)
        result.update(freshNormal=count(normal),freshStress=count(stressed))
        (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    # This first bounded experiment never silently continues another 1000
    # iterations or deploys a candidate based on a small-screen pass.
    manifest=json.loads((OUT/'launch.json').read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in manifest['sourceHashes'].items())
    status('experiment-complete',**result)

if __name__=='__main__':
    try:main();(OUT/'job.exit').write_text('0')
    except BaseException:
        status('failed');(OUT/'job.exit').write_text('1');raise

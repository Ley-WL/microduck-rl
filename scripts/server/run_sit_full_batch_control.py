"""Matched twenty-iteration minibatch factor; coupled optimizer/aux steps disclosed."""
import concurrent.futures,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
import finalize_sit_transfer as f
root=Path(__file__).resolve().parents[2];python=root.parent/'microduck-rl/.venv/bin/python'
folder=root/'evidence/full-batch-control';folder.mkdir(parents=True,exist_ok=True)
environment=os.environ.copy();environment.update(PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='4',WANDB_MODE='offline',MUJOCO_GL='egl',MICRODUCK_WARM_START='1')
f.evidence=folder;f.environment=environment
f.extra_conditions={'gain-low':['--actuator-kp-scale','.8'],'gain-high':['--actuator-kp-scale','1.2']}
seed=root/'logs/rsl_rl/hd1910_sit_process_ablation/transfer_seed/seed.pt'
seed.parent.mkdir(parents=True,exist_ok=True);seed.write_bytes((root/'seed.pt').read_bytes())
TRAIN_SEED=57121

def command(args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:subprocess.run(list(map(str,args)),cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)

def gpu_idle():
    usage=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert usage<200,'GPU6 occupied; leave others alone'
    return usage

def export(task,checkpoint,output):
    gpu_idle()
    command([python,'scripts/export.py',task,'--checkpoint-file',checkpoint,'--onnx-file',output,'--num-envs','1','--device','cuda:0'],output.with_suffix('.export.log'))
    command([python,'scripts/server/write_policy_metadata.py',output,task],output.with_suffix('.metadata.log'))

def steps(state):
    optimizer=state.get('optimizer_state_dict',{})
    return sorted(set(float(v['step']) for v in optimizer.get('state',{}).values() if 'step' in v))

def comparison(reference,reports):
    stats={};lost=0
    for condition,report in reports.items():
        assert report['configuration']['frictionConstraintRevision']=='dof-id-friction-force-v1'
        before,_=f.summarize(reference[condition]);after,item=f.summarize(report)
        assert set(before)==set(after)
        for key in before:np.testing.assert_allclose(before[key]['initialQpos'],after[key]['initialQpos'],atol=1e-12,rtol=0)
        item['regressions']=[list(key) for key,r in before.items() if r['success'] and not after[key]['success']]
        lost+=len(item['regressions']);stats[condition]=item
    original=f.summarize(reference['stress'])[1];current=stats['stress']
    ratio=float(np.mean(current['meanKneeExcessTravelDeg'])/max(np.mean(original['meanKneeExcessTravelDeg']),1e-8))
    frame_ok=all(v<=b*1.05+1e-6 for v,b in zip(current['p95MaxKneeFrameChangeDeg'],original['p95MaxKneeFrameChangeDeg']))
    return {'conditions':stats,'lostSuccessfulTrials':lost,'stressExcessTravelRatio':ratio,'frameGate':frame_ok,'eligible':lost==0 and ratio<=.95 and frame_ok}

def main():
    base_task='Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910'
    export(base_task,root/'seed.pt',folder/'baseline.onnx')
    futures={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        baseline=pool.submit(f.screen,folder/'baseline.onnx','baseline')
        for case,minibatches in (('WeakProcess',4),('FullBatch',1)):
            task='Mjlab-SitStandProcess'+case+'-Flat-MicroDuck-HD1910'
            case_folder=folder/case;case_folder.mkdir(parents=True,exist_ok=True)
            for mode,iterations,worlds in (('smoke',5,64),('train',20,1024)):
                usage=gpu_idle();run=case.lower()+'-matched20-'+mode
                assert not list((root/'logs/rsl_rl/hd1910_sit_process_ablation').glob('*'+run)),'Do not repeat a run'
                (folder/'progress.json').write_text(json.dumps({'case':case,'stage':mode,'gpuMemoryBeforeMiB':usage,'trainSeed':TRAIN_SEED}))
                command([python.parent/'train',task,'--env.scene.num-envs',worlds,'--agent.max-iterations',iterations,'--agent.seed',TRAIN_SEED,'--agent.resume','True','--agent.load-run','transfer_seed','--agent.load-checkpoint','seed.pt','--agent.run-name',run,'--agent.logger','tensorboard'],case_folder/(mode+'.log'))
                checkpoints=list((root/'logs/rsl_rl/hd1910_sit_process_ablation').glob('*'+run+'/model_'+str(iterations-1)+'.pt'));assert len(checkpoints)==1
                loaded=torch.load(checkpoints[0],map_location='cpu',weights_only=False);state=loaded['actor_state_dict']
                original=torch.load(root/'seed.pt',map_location='cpu',weights_only=False)
                assert steps(original)==[],'This experiment expects the common seed with empty Adam state'
                assert steps(loaded)==[float(iterations*2*minibatches)],'Unexpected actual optimizer/aux-hook call count'
                assert all(bool(torch.isfinite(v).all()) for v in state.values())
                for key,v in original['actor_state_dict'].items():
                    if key.startswith('obs_normalizer.'):torch.testing.assert_close(state[key],v,atol=0,rtol=0)
                (case_folder/(mode+'-contract-check.json')).write_text(json.dumps({'normalizerExactlyPreserved':True,'finiteParameters':True,'trainSeed':TRAIN_SEED,'iterations':iterations,'worlds':worlds,'rolloutSteps':24,'learningEpochs':2,'miniBatches':minibatches,'samplesPerOptimizerStep':worlds*24//minibatches,'expectedOptimizerAndAuxHookSteps':iterations*2*minibatches,'optimizerStepsBefore':steps(original),'optimizerStepsAfter':steps(loaded),'meanExplorationStd':float(state['distribution.std_param'].mean()),'checkpointSha256':hashlib.sha256(checkpoints[0].read_bytes()).hexdigest()}))
                export(task,checkpoints[0],case_folder/(mode+'.onnx'))
            futures[case]=pool.submit(f.screen,case_folder/'train.onnx',case)
        reference=baseline.result();reports={k:v.result() for k,v in futures.items()}
        outcome={'trainingSeed':TRAIN_SEED,'formalAccepted':False,'hardwareAccess':False,'deployment':False,'scope':'Only PPO minibatch count changes. Optimizer and auxiliary hook steps 8->2 per iteration, sample count per step 6144->24576; coupled factor, not pure variance isolation. Same seed initial RNG; training trajectories diverge due to optimizer and sampling, evaluation initial states are paired.','cases':{case:comparison(reference,report) for case,report in reports.items()},'fullBatchVsMatchedControl':comparison(reports['WeakProcess'],reports['FullBatch'])}
        outcome['screenEligible']=outcome['cases']['FullBatch']['eligible'] and outcome['fullBatchVsMatchedControl']['eligible']
        (folder/'comparison.json').write_text(json.dumps(outcome,indent=2))
    manifest=json.loads((folder/'launch.json').read_text())
    assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
    (folder/'source-check.json').write_text(json.dumps({'sourceHashesExactlyPreserved':True}))

if __name__=='__main__':
    try:main();(folder/'job.exit').write_text('0')
    except BaseException:(folder/'job.exit').write_text('1');raise

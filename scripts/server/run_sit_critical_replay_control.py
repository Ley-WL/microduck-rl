"""Twenty iterations, only offline teacher-observation distribution changes."""
import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
import finalize_sit_transfer as f
ROOT=Path(__file__).resolve().parents[2];PYTHON=ROOT.parent/'microduck-rl/.venv/bin/python'
FOLDER=ROOT/'evidence/critical-replay';TASK='Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910'
ENV=os.environ.copy();ENV.update(PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='4',WANDB_MODE='offline',MUJOCO_GL='egl',MICRODUCK_WARM_START='1')
f.evidence=FOLDER;f.environment=ENV;f.extra_conditions={'gain-low':['--actuator-kp-scale','.8'],'gain-high':['--actuator-kp-scale','1.2']}

def command(args,log):
    with log.open('wb') as output:subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,stdout=output,stderr=subprocess.STDOUT,check=True)

def gpu_idle():
    usage=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert usage<200
    return usage

def compare(reference,candidate):
    conditions={};lost=0
    for condition,report in candidate.items():
        assert report['configuration']['frictionConstraintRevision']=='dof-id-friction-force-v1'
        old,_=f.summarize(reference[condition]);new,item=f.summarize(report)
        assert set(old)==set(new)
        for key in old:np.testing.assert_allclose(old[key]['initialQpos'],new[key]['initialQpos'],atol=1e-12,rtol=0)
        item['regressions']=[list(key) for key,r in old.items() if r['success'] and not new[key]['success']]
        lost+=len(item['regressions']);conditions[condition]=item
    previous=f.summarize(reference['stress'])[1];current=conditions['stress']
    ratio=float(np.mean(current['meanKneeExcessTravelDeg'])/max(np.mean(previous['meanKneeExcessTravelDeg']),1e-8))
    frame_ok=all(v<=b*1.05+1e-6 for v,b in zip(current['p95MaxKneeFrameChangeDeg'],previous['p95MaxKneeFrameChangeDeg']))
    return {'conditions':conditions,'lostSuccessfulTrials':lost,'stressExcessTravelRatio':ratio,'frameGate':frame_ok,'eligible':lost==0 and ratio<=.95 and frame_ok}

def main():
    manifest=json.loads((FOLDER/'launch.json').read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
    seed=ROOT/'logs/rsl_rl/hd1910_sit_process_ablation/transfer_seed/seed.pt';seed.parent.mkdir(parents=True,exist_ok=True);seed.write_bytes((ROOT/'seed.pt').read_bytes())
    original=torch.load(ROOT/'seed.pt',map_location='cpu',weights_only=False)
    for stage,iterations,worlds in [('smoke',5,64),('train',20,1024)]:
        usage=gpu_idle();run='criticalreplay-matched20-'+stage
        assert not list((ROOT/'logs/rsl_rl/hd1910_sit_process_ablation').glob('*'+run)),'Do not repeat run'
        (FOLDER/'progress.json').write_text(json.dumps({'stage':stage,'gpuMemoryBeforeMiB':usage,'trainSeed':57121}))
        command([PYTHON.parent/'train',TASK,'--env.scene.num-envs',worlds,'--agent.max-iterations',iterations,'--agent.seed','57121','--agent.resume','True','--agent.load-run','transfer_seed','--agent.load-checkpoint','seed.pt','--agent.run-name',run,'--agent.logger','tensorboard'],FOLDER/(stage+'.log'))
        checkpoints=list((ROOT/'logs/rsl_rl/hd1910_sit_process_ablation').glob('*'+run+'/model_'+str(iterations-1)+'.pt'));assert len(checkpoints)==1
        loaded=torch.load(checkpoints[0],map_location='cpu',weights_only=False);state=loaded['actor_state_dict']
        assert all(bool(torch.isfinite(v).all()) for v in state.values())
        for key,v in original['actor_state_dict'].items():
            if key.startswith('obs_normalizer.'):torch.testing.assert_close(state[key],v,atol=0,rtol=0)
        steps=sorted(set(float(v['step']) for v in loaded['optimizer_state_dict']['state'].values() if 'step' in v));assert steps==[float(iterations*8)]
        (FOLDER/(stage+'-contract-check.json')).write_text(json.dumps({'normalizerExactlyPreserved':True,'finiteParameters':True,'actualAdamSteps':steps,'meanExplorationStd':float(state['distribution.std_param'].mean()),'checkpointSha256':hashlib.sha256(checkpoints[0].read_bytes()).hexdigest(),'trainSeed':57121}))
        gpu_idle();command([PYTHON,'scripts/export.py',TASK,'--checkpoint-file',checkpoints[0],'--onnx-file',FOLDER/(stage+'.onnx'),'--num-envs','1','--device','cuda:0'],FOLDER/(stage+'.export.log'))
        command([PYTHON,'scripts/server/write_policy_metadata.py',FOLDER/(stage+'.onnx'),TASK],FOLDER/(stage+'.metadata.log'))
    reports=f.screen(FOLDER/'train.onnx','CriticalReplay')
    references={label:{condition:json.loads((ROOT/'references/screening'/label/condition/'summary.json').read_text()) for condition in reports} for label in ('baseline','WeakProcess')}
    output={'formalAccepted':False,'hardwareAccess':False,'deployment':False,'trainingSeed':57121,'evaluationSeedsNowUsedInFitting':[8111,9001],'scope':'Only offline retention distribution changes, same 20 iterations and 160 Adam/aux steps as prior control. Same seed but subsequent trajectories differ. Initial screen is in-fit, not independent validation. Fresh holdout required.','versusBaseline':compare(references['baseline'],reports),'versusMatchedControl':compare(references['WeakProcess'],reports)}
    output['inFitEligible']=output['versusBaseline']['eligible'] and output['versusMatchedControl']['eligible']
    (FOLDER/'comparison.json').write_text(json.dumps(output,indent=2))
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
    (FOLDER/'source-check.json').write_text(json.dumps({'sourceHashesExactlyPreserved':True}))

if __name__=='__main__':
    try:main();(FOLDER/'job.exit').write_text('0')
    except BaseException:(FOLDER/'job.exit').write_text('1');raise

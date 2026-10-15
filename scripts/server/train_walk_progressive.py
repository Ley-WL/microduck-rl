"""150-step staged PPO with frozen paired screens; continue only on measured progress."""
import argparse, copy, hashlib, json, os, subprocess, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from walk_transfer_task import make_cfg, rl_cfg, TASK

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert TASK in ('walk_tracking_transfer','walk_stride_transfer','walk_sustained_transfer')
sustained = TASK == 'walk_sustained_transfer'
ROOT=Path(__file__).resolve().parents[2];a.output.mkdir(parents=True,exist_ok=False)
torch.manual_seed(68131);cfg=make_cfg();cfg.scene.num_envs=1024
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    agent=asdict(rl_cfg());agent.update(upload_model=False,logger='tensorboard')
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=str(a.output/'logs'),device=env.device)
    runner.load(str(ROOT/'inputs/seed.pt'),load_cfg={'actor':True} if sustained else None,map_location=env.device)
    assert not runner.alg.optimizer.state
    if sustained:
        assert all(group['lr']==1e-5 for group in runner.alg.optimizer.param_groups)
    runner.current_learning_iteration=0;env.common_step_counter=0
    actor=runner.alg.actor;normalizer={k:v.clone() for k,v in actor.state_dict().items() if k.startswith('obs_normalizer.')}
    actor.update_normalization=lambda *args,**kwargs:None
    anchor=copy.deepcopy(actor).eval()
    for parameter in anchor.parameters():parameter.requires_grad_(False)
    retention=[]
    def retain(optimizer,args,kwargs):
        rows=runner.alg.storage.observations['actor'].flatten(0,1)
        sample=rows[torch.randint(len(rows),(512,),device=env.device)]
        td=TensorDict({'actor':sample},batch_size=[len(sample)])
        with torch.no_grad():target=anchor(td)
        loss=(actor(td)-target).square().mean();((5. if sustained else 10.)*loss).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),runner.alg.max_grad_norm)
        retention.append(float(loss.detach()))
    hook=runner.alg.optimizer.register_step_pre_hook(retain)
    stages=[];previous_score=0.
    for total in (150,300,450,600):
        (a.output/'progress.json').write_text(json.dumps(dict(stage='training',targetTotalIterations=total,task=TASK)))
        runner.learn(num_learning_iterations=150,init_at_random_ep_len=False)
        runner.current_learning_iteration=total
        assert all(torch.equal(actor.state_dict()[k],v) for k,v in normalizer.items())
        assert all(bool(torch.isfinite(v).all()) for model in (actor,runner.alg.critic) for v in model.state_dict().values())
        steps={int(v['step']) for v in runner.alg.optimizer.state.values()};assert steps=={total*8}
        assert len(retention)==total*8
        stage=a.output/f'iter{total}';stage.mkdir()
        runner.save(str(stage/'checkpoint.pt'));runner.export_policy_to_onnx(str(stage),'policy.onnx')
        rows=np.random.default_rng(191091).normal(0,.4,(128,61)).astype(np.float32)
        actor.eval()
        with torch.no_grad():expected=actor(TensorDict({'actor':torch.tensor(rows,device=env.device)},batch_size=[128])).cpu().numpy()
        session=ort.InferenceSession(str(stage/'policy.onnx'),providers=['CPUExecutionProvider'])
        actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:x[None]})[0] for x in rows])
        error=float(np.max(abs(actual-expected)));assert error<2e-5
        sha=hashlib.sha256((stage/'policy.onnx').read_bytes()).hexdigest()
        meta=json.loads((ROOT/'inputs/walk.metadata.json').read_text());meta.update(policySha256=sha,hardwareValidated=False,artifactType='progressive-walk-transfer-candidate')
        meta['walkExecution']['policySha256']=sha
        meta['progressiveTransfer']=dict(task=TASK,totalIterations=total,accepted=False,actorNormalizerFrozen=True)
        (stage/'policy.metadata.json').write_text(json.dumps(meta,indent=2))
        audit=dict(task=TASK,iterations=total,adamSteps=total*8,policySha256=sha,onnxTorchMaxError=error,
            optimizerLearningRates=[g['lr'] for g in runner.alg.optimizer.param_groups],
            anchorCoefficient=5. if sustained else 10.,
            normalizerUnchanged=True,finite=True,retentionMean=float(np.mean(retention)),hardwareAccess=False)
        (stage/'audit.json').write_text(json.dumps(audit,indent=2))
        # Save/restore all RNG states: screening must not alter future PPO samples.
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all();np_rng=np.random.get_state()
        (a.output/'progress.json').write_text(json.dumps(dict(stage='screen',totalIterations=total,task=TASK)))
        result_name=f'{a.output.name}-iter{total}-screen.json'
        args=[sys.executable,'scripts/server/evaluate_walk_tracking_transfer.py','--candidate',str(stage/'checkpoint.pt'),'--output',result_name]
        if sustained: args+=['--long-hold','--seeds','205151','205157']
        cache=ROOT/'evidence/progressive-baseline.json'
        if cache.exists() and not sustained:args+=['--baseline-cache',str(cache)]
        with (stage/'screen.log').open('w') as log:subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng);np.random.set_state(np_rng)
        result=json.loads((ROOT/'evidence'/result_name).read_text())
        if not cache.exists():cache.write_text(json.dumps(result,indent=2))
        improvements=[]
        for comparison in result['comparisons']:
            b= comparison['aggregate']['baseline'];c=comparison['aggregate']['candidate']
            if sustained:
                # The target is sustained tracking, not suppression of gait yaw.
                rr=[r for r in result['results'] if r['seed']==comparison['seed']]
                keep=np.array(rr[0]['survival']) & np.array(rr[1]['survival'])
                b,c=[dict(vxMAE=float(np.abs(np.array(r['metrics'])[keep,:,1]-np.array(r['metrics'])[keep,:,0]).mean()),
                    yawMAE=float(np.abs(np.array(r['metrics'])[keep,:,3]-np.array(r['metrics'])[keep,:,2]).mean()),
                    stepDistance=comparison['aggregate'][r['policy']]['stepDistance']) for r in rr]
            improvements.append([(b['vxMAE']-c['vxMAE'])/b['vxMAE'],(b['yawMAE']-c['yawMAE'])/b['yawMAE'],(c['stepDistance']-b['stepDistance'])/b['stepDistance']])
        improvement=np.mean(improvements,axis=0)
        score=float(improvement.mean());lost=sum(len(c['lost']) for c in result['comparisons'])
        # Continuation is exploration, not acceptance: allow at most two new
        # failures while metrics improve; final candidate still requires zero.
        continuing=(lost<=2 and min(improvement)>-.03 and score>previous_score+.01)
        if sustained:
            stop_ok=all(bool(np.all(np.array(c['candidate']['stop']) <= np.array(c['baseline']['stop'])*1.05+np.array([.001,.01])))
                for comparison in result['comparisons'] for c in comparison['perCase'] if c['paired'])
            continuing=continuing and stop_ok and lost==0
        record=dict(improvement=improvement.tolist(),score=score,lost=lost,
            screenAccepted=result['accepted'],continueTraining=bool(continuing),**audit)
        stages.append(record);(a.output/'stages.json').write_text(json.dumps(stages,indent=2))
        print('STAGE',json.dumps(record),flush=True)
        if result['accepted'] or not continuing:break
        previous_score=score
    hook.remove()
    (a.output/'progress.json').write_text(json.dumps(dict(stage='complete',stages=stages,deploymentAccepted=False)))
finally:
    env.close()

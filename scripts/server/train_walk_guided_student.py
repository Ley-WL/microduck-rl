"""Learn validated large-step references into one original-shape actor network.

Teacher queries exist only during data generation. Export has 61 inputs and 14
outputs with the frozen original normalizer, no runtime command adapter.
"""
import copy,hashlib,json,os,subprocess,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import onnxruntime as ort
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab_microduck.tasks.walk_tracking_transfer import make_cfg,rl_cfg
from walk_guidance_map import GuidanceMap

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
torch.manual_seed(201131);cfg=make_cfg();cfg.scene.num_envs=252;cfg.episode_length_s=26.
cfg.observations['actor'].enable_corruption=False
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    agent=asdict(rl_cfg());agent['upload_model']=False
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=None,device=env.device)
    runner.load(str(ROOT/'inputs/seed.pt'),map_location=env.device)
    actor=runner.alg.actor;actor.eval()
    normalizer={k:v.clone() for k,v in actor.state_dict().items() if k.startswith('obs_normalizer.')}
    guidance=GuidanceMap(ROOT/'inputs/guidance-map.json')
    obs,_=env.reset(seed=201131);robot=env.scene['robot']
    cases=[(v,y) for v in (-.2,-.15,-.1,-.05,0.,.05,.1,.15,.2) for y in (-.6,-.4,-.2,0.,.2,.4,.6)]
    grid=torch.tensor(cases,device=env.device).repeat_interleave(4,dim=0)
    command=env.command_manager.get_term('twist');command.stop_speed[:]=grid[:,0];command.stop_yaw[:]=grid[:,1]
    failed=torch.zeros(252,dtype=torch.bool,device=env.device);inputs=[];targets=[]
    initial=torch.cat([robot.data.root_link_pos_w,robot.data.root_link_quat_w,robot.data.joint_pos,robot.data.joint_vel],-1).cpu()
    (OUT/'student-progress.json').write_text(json.dumps(dict(stage='collect-teacher',worlds=252,steps=1200,trainingSeed=201131)))
    for tick in range(1200):
        with torch.inference_mode():action=actor(guidance.query(obs))
        inputs.append(obs['actor'].detach().cpu());targets.append(action.detach().cpu())
        obs,_,done,timeout,_=env.step(action)
        tilt=torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1.,1.)))
        failed|=done|timeout|(tilt>=45)
    survived=(~failed).cpu();assert int(survived.sum())>=240,'Teacher reference insufficiently stable for learning'
    x=torch.stack(inputs);y=torch.stack(targets);del inputs,targets
    # Entire worlds, not adjacent frames, form the regression-validation split.
    validation=survived & (torch.arange(252)%4==3);training=survived & ~validation
    assert int(validation.sum())>=56
    train_x=x[:,training].flatten(0,1).to(env.device);train_y=y[:,training].flatten(0,1).to(env.device)
    validation_x=x[:,validation].flatten(0,1).to(env.device);validation_y=y[:,validation].flatten(0,1).to(env.device)
    torch.save(dict(observation=x,target=y,survived=survived,validationWorlds=validation,cases=cases),OUT/'teacher-data.pt')
    data_audit=dict(seed=201131,worlds=252,steps=1200,teacherSurvived=int(survived.sum()),
        excludedFailedWorlds=failed.nonzero().flatten().cpu().tolist(),trainingWorlds=int(training.sum()),
        validationWorlds=int(validation.sum()),inputDim=61,outputDim=14,commandSlots=[48,49,50],
        originalDesiredCommandsRetained=True,teacherMapSha256=hashlib.sha256((ROOT/'inputs/guidance-map.json').read_bytes()).hexdigest(),
        initialStateSha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest())
    (OUT/'teacher-data-audit.json').write_text(json.dumps(data_audit,indent=2))
    del x,y
    actor.update_normalization=lambda *args,**kwargs:None
    for parameter in actor.distribution.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=1e-4)
    generator=torch.Generator(device=env.device);generator.manual_seed(201139)
    ids=torch.randperm(len(validation_x),generator=generator,device=env.device)[:8192]
    probe_x=validation_x[ids];probe_y=validation_y[ids]
    def predict(rows):return actor(TensorDict({'actor':rows},batch_size=[len(rows)]))
    with torch.no_grad():initial_rms=float((predict(probe_x)-probe_y).square().mean().sqrt())
    previous=0;previous_rms=initial_rms;records=[];losses=[]
    for total in (500,1500,3000):
        (OUT/'student-progress.json').write_text(json.dumps(dict(stage='supervised-fit',targetSteps=total)))
        actor.eval()  # Fixed normalizer; autograd remains enabled.
        for step in range(previous,total):
            ids=torch.randint(len(train_x),(1024,),device=env.device,generator=generator)
            loss=(predict(train_x[ids])-train_y[ids]).square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        assert all(torch.equal(actor.state_dict()[k],v) for k,v in normalizer.items())
        assert all(bool(torch.isfinite(v).all()) for v in actor.state_dict().values())
        with torch.no_grad():error=(predict(probe_x)-probe_y).abs()
        rms=float(error.square().mean().sqrt());p99=float(torch.quantile(error.flatten(),.99))
        stage=OUT/f'step{total}';stage.mkdir(exist_ok=False)
        runner.save(str(stage/'checkpoint.pt'))
        torch.save(dict(optimizer=optimizer.state_dict(),steps=total,losses=losses),stage/'supervised-state.pt')
        runner.export_policy_to_onnx(str(stage),'policy.onnx')
        session=ort.InferenceSession(str(stage/'policy.onnx'),providers=['CPUExecutionProvider'])
        assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
        probe=probe_x[:128].cpu().numpy()
        with torch.no_grad():expected=predict(probe_x[:128]).cpu().numpy()
        actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:r[None]})[0] for r in probe])
        export_error=float(np.max(abs(actual-expected)));assert export_error<2e-5
        sha=hashlib.sha256((stage/'policy.onnx').read_bytes()).hexdigest()
        meta=json.loads((ROOT/'inputs/walk.metadata.json').read_text());meta.update(policySha256=sha,hardwareValidated=False,artifactType='guided-walk-student-candidate')
        meta['walkExecution']['policySha256']=sha
        meta['guidedTraining']=dict(seed=201131,supervisedSteps=total,teacherMapUsedAtRuntime=False,
            actorNormalizerFrozen=True,accepted=False,teacherReference=data_audit['teacherMapSha256'])
        (stage/'policy.metadata.json').write_text(json.dumps(meta,indent=2))
        audit=dict(supervisedSteps=total,initialValidationRms=initial_rms,validationRms=rms,validationP99=p99,
            normalizerUnchanged=True,finite=True,onnxTorchMaxError=export_error,policySha256=sha,
            teacherMapUsedAtRuntime=False,hardwareAccess=False)
        (stage/'audit.json').write_text(json.dumps(audit,indent=2))
        (OUT/'student-progress.json').write_text(json.dumps(dict(stage='closed-loop-screen',supervisedSteps=total)))
        args=[sys.executable,'scripts/server/evaluate_walk_tracking_transfer.py','--candidate',str(stage/'checkpoint.pt'),
            '--seeds','201151','201157','--output',f'student-step{total}-screen.json']
        cache=OUT/'student-baseline.json'
        if cache.exists():args+=['--baseline-cache',str(cache)]
        with (stage/'screen.log').open('w') as log:subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((OUT/f'student-step{total}-screen.json').read_text())
        if not cache.exists():cache.write_text(json.dumps(result,indent=2))
        lost=sum(len(r['lost']) for r in result['comparisons'])
        records.append(dict(**audit,lostOriginalSuccesses=lost,originalScreenAccepted=result['accepted']))
        (OUT/'student-stages.json').write_text(json.dumps(records,indent=2));print('STUDENT',records[-1],flush=True)
        if rms>=previous_rms*.9 or lost>6:break
        previous=total;previous_rms=rms
    (OUT/'student-progress.json').write_text(json.dumps(dict(stage='complete',records=records,deploymentAccepted=False)))
finally:env.close()

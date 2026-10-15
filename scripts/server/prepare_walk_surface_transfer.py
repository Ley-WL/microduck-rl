"""64-world contract audit and ONNX actor recovery, no optimizer update."""
import json,hashlib,os
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
import mjlab_microduck.tasks
from mjlab_microduck.tasks.walk_surface_transfer import make_cfg,rl_cfg
from mjlab_microduck.tasks.getup_mechanical_contract import encode_target

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence'
torch.manual_seed(68131)
targets=[];histories=[];velocities=[];positions=[];shapes=[]
for surface in (False,True):
    cfg=make_cfg(surface=surface);cfg.scene.num_envs=64
    cfg.observations['actor'].enable_corruption=False
    cfg.observations['actor'].terms['joint_pos'].params['biased']=False
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        obs,_=env.reset(seed=68131);assert obs['actor'].shape==(64,61)
        shapes.append(int(obs['critic'].shape[-1]))
        term=env.action_manager.get_term('joint_pos');robot=env.scene['robot']
        active,_=robot.find_joints(list(term._target_names),preserve_order=True)
        passive,_=robot.find_joints(['passive_'+n+'_backlash' for n in term._target_names],preserve_order=True)
        filtered=np.zeros((64,14),dtype=np.float32);initialized=np.zeros((64,1),dtype=bool)
        alpha=np.asarray([.7]*5+[.5]*4+[.7]*5,dtype=np.float32)
        home=np.asarray(json.loads((ROOT/'inputs/walk.metadata.json').read_text())['homeRadians'],dtype=np.float32)
        for tick in range(40):
            action=np.random.default_rng(68200+tick).normal(0,.025,(64,14)).astype(np.float32)
            scaled=np.float32(.9)*action
            filtered=np.where(initialized,(1-alpha)*filtered+alpha*scaled,scaled)
            expected=encode_target(filtered.astype(np.float64)+home.astype(np.float64))
            previous_velocity=(robot.data.joint_vel[:,active]+robot.data.joint_vel[:,passive]).cpu().numpy().copy()
            obs,_,done,timeout,_=env.step(torch.tensor(action,device=env.device))
            keep=~(done|timeout).cpu().numpy();assert int(keep.sum())>=16,'Insufficient surviving audit worlds'
            targets.append(float(abs(term.position_target.cpu().numpy()[keep]-expected[keep]).max()))
            histories.append(float(abs(obs['actor'][keep,34:48].cpu().numpy()-action[keep]).max()))
            velocities.append(float(abs(obs['actor'][keep,20:34].cpu().numpy()-previous_velocity[keep]).max()))
            current=(robot.data.joint_pos[:,active]+robot.data.joint_pos[:,passive]).cpu().numpy().astype(np.float64)
            from mjlab_microduck.tasks.getup_mechanical_contract import REFERENCES,DIRECTIONS,ENCODER_STEP
            refs=np.asarray(REFERENCES);direction=np.asarray(DIRECTIONS)
            encoded=np.floor(refs+current/ENCODER_STEP*direction+.5)
            expected_position=((encoded-refs)*ENCODER_STEP*direction-home.astype(np.float64)).astype(np.float32)
            positions.append(float(abs(obs['actor'][keep,6:20].cpu().numpy()-expected_position[keep]).max()))
            initialized[:]=True;initialized[~keep]=False;filtered[~keep]=0
        partial=dict(surface=surface,targetError=max(targets),historyError=max(histories),velocityError=max(velocities),positionError=max(positions))
        (OUT/'prepare-partial-audit.json').write_text(json.dumps(partial,indent=2));print(partial,flush=True)
        assert max(targets)<2e-6 and max(histories)<2e-6 and max(velocities)<2e-4 and max(positions)<2e-6,partial
        if not surface:
            rl=asdict(rl_cfg());rl['upload_model']=False
            runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),rl,log_dir=None,device='cuda:0')
            actor=runner.alg.actor;state=actor.state_dict()
            tensors={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(ROOT/'inputs/walk.onnx').graph.initializer}
            for name,value in tensors.items():
                if name.startswith('mlp.') or name=='obs_normalizer._mean':state[name]=torch.tensor(value,device=env.device)
            std=torch.tensor(tensors['onnx::Div_24'],device=env.device)-.01
            assert bool((std>=-1e-7).all());std=std.clamp_min(0)
            state['obs_normalizer._std']=std;state['obs_normalizer._var']=std.square()
            state['obs_normalizer.count'].fill_(1_000_000);state['distribution.std_param'].fill_(.1)
            actor.load_state_dict(state);actor.eval()
            rows=np.random.default_rng(68131).normal(0,.4,(128,61)).astype(np.float32)
            session=ort.InferenceSession(str(ROOT/'inputs/walk.onnx'),providers=['CPUExecutionProvider'])
            expected=np.concatenate([session.run(None,{session.get_inputs()[0].name:x[None]})[0] for x in rows])
            with torch.no_grad():actual=actor(TensorDict({'actor':torch.tensor(rows,device=env.device)},batch_size=[128])).cpu().numpy()
            error=float(abs(actual-expected).max());assert error<2e-5,error
            assert not runner.alg.optimizer.state
            env.common_step_counter=0;runner.current_learning_iteration=0
            assert not (OUT/'seed.pt').exists();runner.save(str(OUT/'seed.pt'))
    finally:env.close()
assert len(set(shapes))==1
(OUT/'prepare-audit.json').write_text(json.dumps(dict(worlds=64,stepsPerRoot=40,roots=2,
    targetError=max(targets),rawHistoryError=max(histories),previousVelocityError=max(velocities),
    encoderPositionError=max(positions),actorPositionQuantized4096=True,
    actorOnnxMaxError=error,criticDim=shapes[0],actorNormalizerMeanStdRestored=True,
    normalizerCountSynthesized=1000000,critic='fresh-random-for-new-transfer-task',optimizer='fresh-empty-Adam',
    actorNormalizerWillBeFrozen=True,trainingSeed=68131,learningPerformed=False,hardwareAccess=False,
    seedSha256=hashlib.sha256((OUT/'seed.pt').read_bytes()).hexdigest()),indent=2))

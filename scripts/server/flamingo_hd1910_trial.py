"""Isolated ONNX actor transfer and paired HD1910 cycle simulation; no hardware."""
import argparse
from dataclasses import asdict
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from tensordict import TensorDict
import mjlab_microduck.tasks
from mjlab_microduck.tasks.flamingo_hd1910 import make_cfg,rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper

ROOT=Path(__file__).resolve().parents[2]
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2),encoding='utf-8')

def restore(actor,path,device):
    tensors={x.name:numpy_helper.to_array(x).copy() for x in onnx.load(path).graph.initializer}
    state=actor.state_dict()
    for key in state:
        if key.startswith('mlp.') or key=='obs_normalizer._mean':
            state[key]=torch.tensor(tensors[key],device=device)
    div=[v for k,v in tensors.items() if k.startswith('onnx::Div')]
    assert len(div)==1, list(tensors)
    std=torch.tensor(div[0],device=device)-.01
    assert (std>=-1e-7).all()
    state['obs_normalizer._std']=std.clamp_min(0)
    state['obs_normalizer._var']=std.clamp_min(0).square()
    state['obs_normalizer.count'].fill_(1000000)
    state['distribution.std_param'].fill_(.08)
    actor.load_state_dict(state);actor.eval()
    rows=np.random.default_rng(211021).normal(0,.3,(128,61)).astype('float32')
    with torch.no_grad():actual=actor(TensorDict({'actor':torch.tensor(rows,device=device)},batch_size=[128])).cpu().numpy()
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    ses=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    expected=np.concatenate([ses.run(None,{ses.get_inputs()[0].name:r[None]})[0] for r in rows])
    error=float(abs(expected-actual).max());assert error<2e-5,error
    actor.update_normalization=lambda obs:None
    return error

def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(211021);np.random.seed(211021)
    cfg=make_cfg(play=a.mode=='eval');cfg.seed=211021;cfg.scene.num_envs=a.envs
    if a.mode=='eval':
        cfg.episode_length_s=17
        cfg.sim.mujoco.timestep=.002;cfg.decimation=10
        cfg.commands['head_pose'].ranges=tuple((0.,0.) for _ in cfg.commands['head_pose'].ranges)
        cfg.curriculum.clear()
        for key in ['push_robot','randomize_com','randomize_head_com','randomize_mass_inertia','randomize_armature','randomize_joint_friction','encoder_bias','imu_orientation']:
            cfg.events.pop(key,None)
        cfg.events['set_flamingo_cycle_state'].params.update(in_pose_prob=0.,tilt_noise=0.,joint_noise_std=0.,standing_z_min=.115,standing_z_max=.115)
        cfg.observations['actor'].enable_corruption=False
        cfg.observations['actor'].terms['joint_pos'].params['biased']=False
        for act in cfg.scene.entities['robot'].articulation.actuators:
            act.vin_range=(7.4,7.4);act.vin_drop_gain_range=(0.,0.);act.delay_min_lag=7;act.delay_max_lag=7
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        rl=asdict(rl_cfg())
        runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),rl,log_dir=str(out/'logs'),device=env.device)
        actor=runner.alg.actor
        error=restore(actor,Path(a.policy),env.device)
        norm={k:v.clone() for k,v in actor.obs_normalizer.state_dict().items()}
        save(out/'contract.json',dict(actorOnnxMaxError=error,sourceSha256=hashlib.sha256(Path(a.policy).read_bytes()).hexdigest(),worlds=a.envs,
            critic='fresh random',optimizer='fresh Adam',normalizer='restored mean/std, synthetic count1000000, frozen',
            lr=1e-5,exploration=.08,hardware=False,task='official FlamingoCycleHard + HD1910/observer limits'))
        if a.mode=='train':
            # Precalibrate the fresh critic; the imported actor must remain exactly fixed.
            before={k:v.clone() for k,v in actor.state_dict().items()}
            for p in actor.parameters():p.requires_grad_(False)
            runner.learn(10,init_at_random_ep_len=True)
            assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
            for p in actor.parameters():p.requires_grad_(True)
            runner.current_learning_iteration=0
            runner.learn(a.iterations,init_at_random_ep_len=True)
            assert all(torch.equal(v,actor.obs_normalizer.state_dict()[k]) for k,v in norm.items())
            runner.save(str(out/'final.pt'))
            runner.export_policy_to_onnx(str(out),'policy.onnx')
            save(out/'complete.json',dict(iterations=a.iterations,criticWarmup=10,actorNormalizerUnchanged=True))
            return
        actor.eval();robot=env.scene['robot'];sites,_=robot.find_sites(['left_foot','right_foot'],preserve_order=True)
        reports={}
        for side in (1,-1):
            for push in ('none','backward','outward'):
                torch.manual_seed(211031);obs,_=env.reset(seed=211031)
                initial=hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()+env.sim.data.qvel.cpu().numpy().tobytes()).hexdigest()
                failed=torch.zeros(a.envs,dtype=torch.bool,device=env.device)
                frames=[];targets=[];qposes=[];term=env.command_manager.get_term('twist')
                for tick in range(800):
                    flag=1. if 100<=tick<550 else 0.
                    term.vel_command_b[:]=torch.tensor([flag,side,0.],device=env.device)
                    term._stance_side.fill_(side);term.time_left.fill_(100.)
                    # Raw command slots, no command interpolation; alpha belongs to rewards only.
                    obs['actor'][:,48:51]=term.vel_command_b
                    if tick==300 and push!='none':
                        delta=torch.zeros((a.envs,6),device=env.device)
                        delta[:,0 if push=='backward' else 1]=-.10 if push=='backward' else side*.10
                        q=robot.data.root_link_quat_w
                        yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
                        x=delta[:,0].clone();y=delta[:,1].clone()
                        delta[:,0]=torch.cos(yaw)*x-torch.sin(yaw)*y
                        delta[:,1]=torch.sin(yaw)*x+torch.cos(yaw)*y
                        robot.write_root_link_velocity_to_sim(robot.data.root_link_vel_w+delta)
                    with torch.no_grad():action=actor(obs);obs,_,done,timeout,_=env.step(action)
                    tilt=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))
                    failed|=done|timeout|(tilt>torch.deg2rad(torch.tensor(55.,device=env.device)))
                    feet=robot.data.site_pos_w[:,sites,2]-env.scene.env_origins[:,2:3]
                    contacts=env.scene['feet_ground_contact'].data.found>0
                    swing=0 if side==1 else 1;stance=1-swing
                    single=contacts[:,stance]&~contacts[:,swing]&(feet[:,swing]>.03)&~failed
                    lowered=contacts.all(-1)&(tilt<.25)&~failed
                    frames.append(torch.stack([single,lowered,failed,feet[:,swing],tilt],-1).cpu().numpy())
                    targets.append(env.action_manager.get_term('joint_pos')._processed_actions.cpu().numpy().copy())
                    if push=='none' and tick%2==0:qposes.append(env.sim.data.qpos[0].cpu().numpy().copy())
                f=np.asarray(frames);t=np.asarray(targets);hold=f[200:500,:,0].mean(0);lower=f[700:800,:,1].mean(0)
                key=f'{"left_lift" if side==1 else "right_lift"}_{push}'
                reports[key]=dict(survived=int((~failed).sum()),worlds=a.envs,holdSuccess=int((hold>=.8).sum()),lowerSuccess=int((lower>=.8).sum()),
                    holdFraction=hold.tolist(),lowerFraction=lower.tolist(),swingHeightMean=float(f[200:500,:,3][f[200:500,:,2]==0].mean()) if (f[200:500,:,2]==0).any() else None,
                    targetDeltaP95Deg=float(np.rad2deg(np.percentile(abs(np.diff(t,axis=0))[(f[1:,:,2]==0)&(f[:-1,:,2]==0)],95))) if (f[1:,:,2]==0).any() else None,initialQposQvelSha256=initial)
                save(out/'results.json',reports);np.savez_compressed(out/(key+'.npz'),frames=f,targets=t,qpos=np.asarray(qposes))
                print(key,reports[key],flush=True)
        save(out/'complete.json',dict(cases=len(reports),worlds=a.envs,hardware=False))
        # Render recorded first-world nominal cycles; simulation metrics above remain separate.
        import mujoco,imageio.v2 as imageio
        renderer=mujoco.Renderer(env.sim.mj_model,height=480,width=640);data=mujoco.MjData(env.sim.mj_model)
        camera=mujoco.MjvCamera();camera.azimuth=135;camera.elevation=-18;camera.distance=.65
        try:
            for key in ('left_lift_none','right_lift_none'):
                with imageio.get_writer(str(out/(key+'.mp4')),fps=25) as writer:
                    for qpos in np.load(out/(key+'.npz'))['qpos']:
                        data.qpos[:]=qpos;mujoco.mj_forward(env.sim.mj_model,data)
                        camera.lookat[:]=data.xpos[env.sim.mj_model.body('robot/trunk_base').id]
                        renderer.update_scene(data,camera=camera);writer.append_data(renderer.render())
        finally:renderer.close()
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','eval']);p.add_argument('--policy',required=True);p.add_argument('--output',required=True)
    p.add_argument('--envs',type=int,default=16);p.add_argument('--iterations',type=int,default=100);run(p.parse_args())

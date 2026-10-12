"""Replay deployed getup -> walk on replica STL and HD1910 BAM. No hardware."""
import argparse, hashlib, importlib.util, json, math
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort
from bam.model import load_model
from bam.mujoco import MujocoController

ROOT=Path(__file__).resolve().parents[1]
POSES={"supine":(0.,-math.pi/2),"prone":(0.,math.pi/2),"left_side":(-math.pi/2,0.),"right_side":(math.pi/2,0.)}

def quat(roll,pitch,yaw):
    c=np.cos(np.array([roll,pitch,yaw])/2);s=np.sin(np.array([roll,pitch,yaw])/2)
    return np.array([c[0]*c[1]*c[2]+s[0]*s[1]*s[2],s[0]*c[1]*c[2]-c[0]*s[1]*s[2],c[0]*s[1]*c[2]+s[0]*c[1]*s[2],c[0]*c[1]*s[2]-s[0]*s[1]*c[2]])

def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    policies=ROOT.parent/'microduck-observer/models'
    meta=json.loads((policies/'xgoduck_getup.metadata.json').read_text())
    home=np.array(meta['homeRadians'],dtype=np.float32);order=meta['jointOrder']
    source=ROOT/'src/mjlab_microduck/robot'
    modspec=importlib.util.spec_from_file_location('getup_ground',source/'ground_contact.py')
    ground=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(ground)
    spec=mujoco.MjSpec.from_file(str(source/'replica/scene_walk.xml'))
    body_contacts=ground.add_body_ground_contacts(spec)
    params=source/'hd1910/1910_m6.json';bam=load_model(str(params))
    bam.actuator.kp=5.;bam.actuator.vin=7.4;bam.actuator.max_current=None
    limit=7.4*bam.kt.value/bam.R.value
    for a in spec.actuators:
        a.set_to_motor();a.forcelimited=True;a.forcerange=(-limit,limit);a.ctrllimited=False;a.gear=[1.,0,0,0,0,0]
    donor_ranges={}
    if args.joint_limits:
        donor_ranges={j.attrib['name']:tuple(map(float,j.attrib['range'].split())) for j in ET.parse(args.joint_limits).iter('joint') if j.attrib.get('name') in order and 'range' in j.attrib}
        assert set(donor_ranges)==set(order), 'Expected all 14 donor joint limits'
    for j in spec.joints:
        if j.name in donor_ranges:j.range=donor_ranges[j.name];j.limited=True
        if j.name in order:j.damping=np.zeros((3,1));j.frictionloss=0;j.solref_friction=(-5e4,-2e2);j.solimp_friction=(.99,.9999,.001,.5,2)
    for g in spec.geoms:
        if g.name in ('left_foot_collision','right_foot_collision') or g.name.startswith('body_ground_'):
            g.condim=3;g.friction=[1.,.005,.0001]
            if g.name.endswith('foot_collision'):g.priority=1
    model=spec.compile();model.opt.timestep=.002;model.opt.iterations=10;model.opt.ls_iterations=20
    model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    data=mujoco.MjData(model);controller=MujocoController(bam,order,model,data,vin_drop_gain=0.,vin_min=4.)
    qadr=controller.qpos_indexes;vadr=controller.dof_indexes;trunk=model.body('trunk_base').id
    assert [model.joint(int(model.dof_jntid[i])).name for i in vadr]==order
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    sessions={name:ort.InferenceSession(str(policies/('xgoduck_'+name+'.onnx')),sess_options=opts,providers=['CPUExecutionProvider']) for name in ('getup','walk')}
    for session in sessions.values():assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
    # Derive stand reference from this exact mesh and HOME, rather than a donor height.
    data.qpos[:7]=[0,0,.12,1,0,0,0];data.qpos[qadr]=home;mujoco.mj_forward(model,data)
    nominal=float(data.xpos[trunk,2]-ground.lowest_mesh_z(model,data))
    manifest={'hardwareAccess':False,'robot':'replica scene_walk + body_ground contacts','bodyGroundMeshes':body_contacts,'massKg':float(model.body_mass.sum()),'nominalHeightM':nominal,'actuatorSha256':hashlib.sha256(params.read_bytes()).hexdigest(),'kpBam':5.,'vin':7.4,'actionEmaOld':.45,'rawPreviousAction':True,'physicsHz':500,'policyHz':50,'extraTargetDelayMs':8,'bamCommandDelayMs':1000*float(json.loads(params.read_text())['command_delay']),'runtimeTiltHandoffDeg':15,'runtimeHandoffSeconds':1,'successCriterion':'upright handoff + final 3s >=95% samples tilt<20deg, trunk>85%HOME, XYspeed<0.2m/s, angular speed<2rad/s','noAutoReset':True,'handoffToWalk':not args.no_handoff,'policyHashes':{name:hashlib.sha256((policies/('xgoduck_'+name+'.onnx')).read_bytes()).hexdigest() for name in sessions},'jointLimitsSource':str(args.joint_limits) if args.joint_limits else 'existing replica XML','jointLimitsRadians':{n:model.jnt_range[model.joint(n).id].tolist() for n in order},'robotFiles':{str(f.relative_to(source/'replica')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (source/'replica').rglob('*') if f.is_file() and f.suffix.lower() in ('.xml','.stl')}}
    results=[];rng=np.random.default_rng(args.seed)
    for pose,(roll,pitch) in POSES.items():
        for trial in range(args.trials):
            mujoco.mj_resetData(model,data);model.dof_damping[vadr]=0;model.dof_frictionloss[vadr]=0
            perturb=np.zeros(3) if trial==0 else rng.uniform(-math.radians(5),math.radians(5),3)
            yaw=0. if trial==0 else float(rng.uniform(-math.pi,math.pi))
            data.qpos[:7]=[0,0,.2,*quat(roll+perturb[0],pitch+perturb[1],yaw)]
            data.qpos[qadr]=home+(0 if trial==0 else rng.uniform(-.03,.03,14));mujoco.mj_forward(model,data)
            data.qpos[2]+=.003-ground.lowest_mesh_z(model,data);mujoco.mj_forward(model,data)
            controller.reset(data.qpos);controller.last_ts=data.time;bam.actuator.q_target_smooth=data.qpos[qadr].copy()
            initial=data.qpos[qadr].copy();history=deque([initial.copy() for _ in range(8)],maxlen=8)
            for _ in range(200):controller.q_target[:]=initial;controller.update();mujoco.mj_step(model,data)
            start=float(data.time);initial_qpos=data.qpos.copy();last=np.zeros(14,dtype=np.float32);filtered=last.copy();velocity=data.qvel[vadr].copy();gyro=data.qvel[3:6].copy();upright=0.;handoff=None;trace=[];qpositions=[]
            renderer=writer=None
            if args.video and trial==0:
                renderer=mujoco.Renderer(model,height=480,width=640);writer=imageio.get_writer(str(args.output/(pose+'.mp4')),fps=25)
                camera=mujoco.MjvCamera();camera.azimuth=135;camera.elevation=-18;camera.distance=.65
            try:
                for tick in range(round(args.seconds/.02)):
                    rotation=data.xmat[trunk].reshape(3,3);gravity=rotation.T@np.array([0.,0.,-1.]);bodyvel=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,trunk,bodyvel,1)
                    velocity=.16*velocity+.84*data.qvel[vadr];gyro=.25*gyro+.75*bodyvel[:3]
                    obs=np.concatenate((gyro,gravity,data.qpos[qadr]-home,velocity,last,np.zeros(13))).astype(np.float32)
                    assert obs.shape==(61,) and np.isfinite(obs).all()
                    kind='getup' if handoff is None or args.no_handoff else 'walk';session=sessions[kind];action=session.run(None,{session.get_inputs()[0].name:obs[None]})[0][0]
                    assert np.isfinite(action).all();filtered=.45*filtered+.55*action;last=action.copy();target=home+filtered
                    # Platform donor targets are not clipped to training joint angle bounds.
                    for _ in range(10):
                        history.append(target.copy());controller.q_target[:]=history[-5];controller.update();mujoco.mj_step(model,data)
                    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                    rot=data.xmat[trunk].reshape(3,3);angle=float(np.degrees(np.arccos(np.clip(rot[2,2],-1,1))));height=float(data.xpos[trunk,2]);xy=float(np.linalg.norm(data.qvel[:2]));angular=float(np.linalg.norm(data.qvel[3:6]));t=float(data.time-start)
                    upright=upright+.02 if angle<15. else 0.
                    if handoff is None and upright>=1.:
                        handoff=t
                        if not args.no_handoff:last[:]=0.;filtered[:]=0.
                    trace.append({'time':t,'tiltDeg':angle,'heightM':height,'speedXY':xy,'angularSpeed':angular,'policy':kind,'contactCount':int(data.ncon)})
                    qpositions.append(data.qpos.copy())
                    if writer and tick%2==0:
                        camera.lookat[:]=data.xpos[trunk];renderer.update_scene(data,camera=camera);writer.append_data(renderer.render())
            finally:
                if writer:writer.close();renderer.close()
            tail=trace[-min(len(trace),150):];standing=[r['tiltDeg']<20 and r['heightM']>.85*nominal and r['speedXY']<.2 and r['angularSpeed']<2 for r in tail]
            result={'pose':pose,'trial':trial,'initialQpos':initial_qpos.tolist(),'seconds':args.seconds,'handoffSeconds':handoff,'success':bool(handoff is not None and np.mean(standing)>=.95),'finalStableFraction':float(np.mean(standing)),'endTiltDeg':trace[-1]['tiltDeg'],'endHeightM':trace[-1]['heightM'],'minTiltDeg':min(r['tiltDeg'] for r in trace),'maxHeightM':max(r['heightM'] for r in trace),'finite':True}
            results.append(result);(args.output/(pose+'-'+str(trial)+'.json')).write_text(json.dumps({'result':result,'trace':trace},indent=2));np.savez_compressed(args.output/(pose+'-'+str(trial)+'.npz'),qpos=np.array(qpositions))
            print(json.dumps(result),flush=True)
            (args.output/'summary.json').write_text(json.dumps({'configuration':manifest,'results':results,'completed':len(results),'expected':4*args.trials},indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=4);p.add_argument('--seconds',type=float,default=15.);p.add_argument('--seed',type=int,default=42);p.add_argument('--video',action='store_true');p.add_argument('--joint-limits',type=Path);p.add_argument('--no-handoff',action='store_true');p.add_argument('--output',type=Path,required=True);run(p.parse_args())

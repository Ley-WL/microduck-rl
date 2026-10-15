"""Frozen official kicks on current replica/HD1910 mechanical simulation. No hardware."""
import json, hashlib, importlib.util
from pathlib import Path
from collections import deque
import numpy as np
import mujoco
import onnxruntime as ort
from bam.model import load_model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/official-kick-audit-20261014'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
mechanical=module('mechanical',ROOT/'src/mjlab_microduck/tasks/getup_mechanical_contract.py')
imu=module('imu',ROOT/'scripts/evaluate_skills_cpu.py')
cpu=module('cpu',ROOT/'scripts/bam_cpu_controller.py')
ground=module('ground',ROOT/'src/mjlab_microduck/robot/ground_contact.py')
robot=ROOT/'src/mjlab_microduck/robot'
inputs=ROOT/'artifacts/walk-fast-audit-20261013/update-fraction/inputs'
meta=json.loads((inputs/'walk.metadata.json').read_text());home=np.asarray(meta['homeRadians'])
spec=mujoco.MjSpec.from_file(str(robot/'replica/scene_walk.xml'))
ground.add_body_ground_contacts(spec)
donor=json.loads((inputs/'body-contract.json').read_text())
bodies={b.name:b for b in spec.bodies}
for row in donor['model']['bodies']:
    if row['name']=='world':continue
    b=bodies[row['name']];b.mass=row['mass_kg'];b.fullinertia=np.asarray([np.nan,0,0,0,0,0])
    for field in ('inertia','ipos','iquat','pos','quat'):setattr(b,field,np.asarray(row[field]))
    b.explicitinertial=True
mechanical.edit_mechanical_spec(spec)
bam=load_model(str(robot/'hd1910/1910_m6.json'));bam.actuator.kp=5.;bam.actuator.vin=7.4;bam.actuator.max_current=None
maximum=7.4*bam.kt.value/bam.R.value
for j in spec.joints:
    if j.name in mechanical.NAMES:
        j.armature=float(bam.actuator.get_extra_inertia());j.damping=np.zeros((3,1));j.frictionloss=0
        j.solref_friction=(-5e4,-2e2);j.solimp_friction=(.99,.9999,.001,.5,2)
for a in spec.actuators:
    a.set_to_motor();a.forcelimited=True;a.forcerange=(-maximum,maximum);a.gear=[1.,0,0,0,0,0]
for g in spec.geoms:
    if g.name in ('left_foot_collision','right_foot_collision') or g.name.startswith('body_ground_'):
        g.contype=1;g.conaffinity=0;g.condim=3;g.friction=[1.,.005,.0001]
        if g.name.endswith('foot_collision'):g.priority=1
    elif g.type!=mujoco.mjtGeom.mjGEOM_PLANE and g.meshname!='leg':g.contype=0;g.conaffinity=0
ball=spec.worldbody.add_body(name='ball',pos=[.3,0,.035],mass=.015,inertia=[1.225e-5]*3,explicitinertial=True)
ball.add_freejoint(name='ball_free')
ball.add_geom(name='ball_geom',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.035,0,0],rgba=[1,.55,0,1],contype=1,conaffinity=1,friction=[.5,.005,.0001])
model=spec.compile();model.opt.timestep=.002;model.opt.iterations=10;model.opt.ls_iterations=20
model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST;model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
data=mujoco.MjData(model)
ctrl=cpu.MujocoController(bam,list(mechanical.NAMES),model,data,vin_drop_gain=0.,vin_min=4.)
qa=ctrl.qpos_indexes;va=ctrl.dof_indexes;trunk=model.body('trunk_base').id
bq=model.jnt_qposadr[model.joint('ball_free').id];bv=model.jnt_dofadr[model.joint('ball_free').id];bg=model.geom('ball_geom').id
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
sessions={n:ort.InferenceSession(str(OUT/(n+'.onnx')),sess_options=opts,providers=['CPUExecutionProvider']) for n in ('ball_kick_left','ball_kick_right','alpha_stand')}
delay=int(np.ceil(float(json.loads((robot/'hd1910/1910_m6.json').read_text())['command_delay'])/.002))
results=[]
for side in ('left','right'):
 for duration in (.5,3.):
  for trial in range(5):
    mujoco.mj_resetData(model,data);model.dof_damping[va]=0;model.dof_frictionloss[va]=0
    data.qpos[:7]=[0,0,.12,1,0,0,0];data.qpos[qa]=home;data.qpos[bq:bq+3]=[.6,0,.035]
    mujoco.mj_forward(model,data);data.qpos[2]+=.002-ground.lowest_mesh_z(model,data);mujoco.mj_forward(model,data)
    ctrl.reset(data.qpos);ctrl.last_ts=data.time;bam.actuator.q_target_smooth=home.copy()
    for _ in range(200):ctrl.q_target[:]=home;ctrl.update();mujoco.mj_step(model,data)
    rng=np.random.default_rng(77131+trial);offset=np.zeros(2) if trial==0 else rng.uniform(-.015,.015,2)
    mat=data.xmat[trunk].reshape(3,3);yaw=np.arctan2(mat[1,0],mat[0,0]);rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    data.qpos[bq:bq+3]=[* (data.qpos[:2]+rotation@([.09,(.042 if side=='left' else -.042)]+offset)),.035]
    data.qvel[bv:bv+6]=0;mujoco.mj_forward(model,data)
    initial_penetration=[float(c.dist) for c in data.contact if bg in (c.geom1,c.geom2) and min(c.geom1,c.geom2)!=model.geom('floor').id]
    start=data.time;ball_start=data.qpos[bq:bq+3].copy();history=deque([home.copy()]*(delay+1),maxlen=delay+1)
    previous=np.zeros(14,dtype=np.float32);rows=[];qposes=[];contacts=set();contact_first=None;clipped=0
    for tick in range(250):
      gyro,gravity=imu.root_body_imu_observation(model,data,trunk)
      # Unified 61D actor; baked normalization. Original raw previous action.
      pos=np.round(data.qpos[qa]/mechanical.ENCODER_STEP)*mechanical.ENCODER_STEP
      obs=np.concatenate([gyro,gravity,pos-home,data.qvel[va],previous,np.zeros(13)]).astype(np.float32)
      session=sessions['ball_kick_'+side if tick*.02<duration else 'alpha_stand']
      raw=session.run(None,{session.get_inputs()[0].name:obs[None]})[0][0];assert np.isfinite(raw).all()
      desired=home+raw;target=mechanical.encode_target(desired);previous=raw.copy()
      clipped+=int(np.any(np.abs(desired-np.clip(desired,np.asarray(list(mechanical.LIMITS.values()))[:,0],np.asarray(list(mechanical.LIMITS.values()))[:,1]))>1e-6))
      tilt=float(np.degrees(np.arccos(np.clip(-gravity[2],-1,1))))
      rows.append([tick*.02,tilt,*data.qpos[bq:bq+3],*data.qvel[bv:bv+3],float(data.qpos[2])]);qposes.append(data.qpos.copy())
      for _ in range(10):
        history.append(target.copy());ctrl.q_target[:]=history[0];ctrl.update();mujoco.mj_step(model,data)
        for c in data.contact:
          if bg in (c.geom1,c.geom2):
            other=c.geom2 if c.geom1==bg else c.geom1;name=model.geom(other).name or model.body(model.geom_bodyid[other]).name
            if name!='floor':contacts.add(name);contact_first=float(data.time-start) if contact_first is None else contact_first
    arr=np.asarray(rows);travel=(arr[:,2:4]-ball_start[:2])@rotation[:,0];speed=arr[:,5:7]@rotation[:,0]
    key=f'{side}-{duration:g}-trial{trial}';np.savez_compressed(OUT/(key+'.npz'),trajectory=arr,qpos=qposes,initialBall=ball_start,offset=offset)
    result=dict(side=side,duration=duration,trial=trial,offset=offset.tolist(),initialBallPenetration=initial_penetration,contacts=sorted(contacts),firstContact=contact_first,maxForwardSpeed=float(speed.max()),forwardDisplacement=float(travel[-1]),maxTilt=float(arr[:,1].max()),finalTilt=float(arr[-1,1]),finalHeight=float(arr[-1,-1]),clippedFrames=clipped,standingFinalFraction=float(np.mean((arr[-50:,1]<15)&(arr[-50:,-1]>.09))))
    result['complete']=bool(contacts and speed.max()>=1. and travel[-1]>.1 and arr[:,1].max()<60 and result['standingFinalFraction']>=.95 and not any(x<-.001 for x in initial_penetration))
    result['kickAndRecover']=bool(contacts and speed.max()>=1. and travel[-1]>.03 and arr[:,1].max()<60 and result['standingFinalFraction']>=.95 and not any(x<-.001 for x in initial_penetration))
    # complete is an explicitly stricter diagnostic distance gate, not official acceptance.
    results.append(result);print(json.dumps(result),flush=True)
sources=[Path(__file__),ROOT/'scripts/bam_cpu_controller.py',ROOT/'src/mjlab_microduck/tasks/getup_mechanical_contract.py',inputs/'body-contract.json',robot/'hd1910/1910_m6.json',*OUT.glob('*.onnx')]
(OUT/'summary.json').write_text(json.dumps(dict(simulationOnly=True,training=False,hardwareAccess=False,frictionRevision=cpu.FRICTION_CONSTRAINT_REVISION,bamKp=5,voltage=7.4,actionScale=1,ema=False,delaySteps=delay,limitations=['BAM gain is not identified firmware PD','synthetic HOME start, ideal IMU and simplified contact','single seed ball offsets, diagnostic not formal acceptance','no backlash in this first screen'],hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},results=results),indent=2))
import imageio.v2 as imageio
renderer=mujoco.Renderer(model,height=480,width=640)
camera=mujoco.MjvCamera();camera.lookat[:]=[.10,0,.10];camera.distance=.65;camera.azimuth=110;camera.elevation=-22
for side in ('left','right'):
    capture=np.load(OUT/f'{side}-0.5-trial0.npz')
    with imageio.get_writer(OUT/f'{side}.mp4',fps=25) as video:
      for qpos in capture['qpos'][::2]:
        data.qpos[:]=qpos;mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera);video.append_data(renderer.render())
renderer.close()

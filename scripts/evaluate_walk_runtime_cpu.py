"""Frozen walking diagnostic with current Rust arithmetic; never accesses hardware.

Explicit local or kissqy nominal physics; both remain simulation assumptions.
BAM kp is a diagnostic parameter, not an identified firmware P conversion.
"""
import argparse
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import mujoco
import onnxruntime as ort
from bam.model import load_model

ROOT = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

mechanical = module('walk_mechanical', ROOT/'src/mjlab_microduck/tasks/getup_mechanical_contract.py')
bam_cpu = module('walk_bam', ROOT/'scripts/bam_cpu_controller.py')
imu = module('walk_imu', ROOT/'scripts/evaluate_getup_mechanical_cpu.py')

def command_step(previous, target):
    return (previous + np.float32(.2)*(target-previous)).astype(np.float32)

def action_step(previous, raw, initialized, scale=.9):
    target = np.float32(scale)*raw
    alpha = np.full(14, .7, dtype=np.float32); alpha[5:9] = .5
    return ((1-alpha)*previous+alpha*target if initialized else target).astype(np.float32)

def observed_angles(qpos):
    # Encoder feedback is rounded, but never clipped to software target limits.
    refs=np.asarray(mechanical.REFERENCES);directions=np.asarray(mechanical.DIRECTIONS)
    steps=np.floor(refs+qpos/mechanical.ENCODER_STEP*directions+.5)
    return (steps-refs)*mechanical.ENCODER_STEP*directions

class BacklashController(bam_cpu.MujocoController):
    """Position loop reads output angle; motor velocity drives back-EMF/friction."""
    def update(self):
        data=self.mujoco_data;ids=self.dof_indexes
        q=data.qpos[self.qpos_indexes]+data.qpos[self.passive_qadr]
        dq=data.qvel[ids];act=self.model.actuator
        dt=data.time-self.last_ts;self.last_ts=data.time
        control=act.compute_control(self.q_target,q,dq,dt)
        torque=act.compute_torque(control,True,q,dq)
        data.ctrl[self.act_indexes]=torque
        external=-data.qfrc_bias[ids]+data.qfrc_constraint[ids]-bam_cpu.dof_friction_force(data,ids)
        friction,damping=self.model.compute_frictions(data.qfrc_actuator[ids],external,dq)
        self.mujoco_model.dof_frictionloss[ids]=friction
        self.mujoco_model.dof_damping[ids]=damping

def run(args):
    donor_physics=module('kissqy_physics', ROOT/'scripts/kissqy_walk_physics.py')
    aligned=args.physics_profile=='kissqy' or args.body_contract is not None
    if aligned:
        args.backlash=True
        if args.terrain_height_mm or args.contact_timeconst is not None or args.friction!=1.:
            raise ValueError('kissqy nominal profile requires default flat contact settings')
    args.output.mkdir(parents=True, exist_ok=False)
    meta=json.loads(args.onnx.with_suffix('.metadata.json').read_text())
    assert hashlib.sha256(args.onnx.read_bytes()).hexdigest()==meta['policySha256']
    assert tuple(meta['jointOrder'])==mechanical.NAMES
    action_scale=float(meta['walkExecution']['actionScale'])
    assert 0<action_scale<=1
    assert meta['walkExecution']['headNewWeight']==.5 and meta['walkExecution']['legsNewWeight']==.7
    home=np.asarray(meta['homeRadians'],dtype=np.float32)
    robot=ROOT/'src/mjlab_microduck/robot'
    ground=module('walk_ground',robot/'ground_contact.py')
    spec=mujoco.MjSpec.from_file(str(robot/'replica'/('scene_walk_backlash.xml' if args.backlash else 'scene_walk.xml')))
    contacts=0 if aligned else ground.add_body_ground_contacts(spec)
    # The donor profile is explicit and isolated from training defaults.
    mechanical.edit_mechanical_spec(spec)
    params=robot/'hd1910/1910_m6.json';bam=load_model(str(params))
    assert hashlib.sha256(params.read_bytes()).hexdigest()==meta['actuatorParamsSha256']
    bam.actuator.kp=args.bam_kp;bam.actuator.vin=args.voltage;bam.actuator.max_current=None
    maximum=args.voltage*bam.kt.value/bam.R.value
    for j in spec.joints:
        if j.name in mechanical.NAMES:
            j.armature=float(bam.actuator.get_extra_inertia());j.damping=np.zeros((3,1));j.frictionloss=0
            j.solref_friction=(-5e4,-2e2);j.solimp_friction=(.99,.9999,.001,.5,2)
    for a in spec.actuators:
        a.set_to_motor();a.forcelimited=True;a.forcerange=(-maximum,maximum);a.gear=[1.,0,0,0,0,0]
    for g in ([] if aligned else spec.geoms):
        if g.name in ('left_foot_collision','right_foot_collision') or g.name.startswith('body_ground_'):
            g.contype=1;g.conaffinity=0;g.condim=3;g.friction=[args.friction,.005,.0001]
            if args.contact_timeconst is not None:g.solref=[args.contact_timeconst,1.]
            if g.name.endswith('foot_collision'):g.priority=1
        elif g.type!=mujoco.mjtGeom.mjGEOM_PLANE and g.meshname!='leg':g.contype=0;g.conaffinity=0
    contract=donor_physics.apply(spec,args.body_contract or donor_physics.DEFAULT_CONTRACT) if aligned else None
    terrain_hash=None
    if args.terrain_height_mm:
        amplitude=args.terrain_height_mm/1000
        y,x=np.meshgrid(np.linspace(-1,1,257),np.linspace(-3,3,769),indexing='ij')
        height=np.sin(2*np.pi*x/.071)*np.cos(2*np.pi*y/.093)+.4*np.sin(2*np.pi*(x+y)/.037)
        height=(height-height.min())/(height.max()-height.min())
        terrain_hash=hashlib.sha256(height.astype(np.float64).tobytes()).hexdigest()
        spec.add_hfield(name='diagnostic_pile',nrow=257,ncol=769,size=[3,1,amplitude,.02],userdata=height.ravel().tolist())
        for g in spec.geoms:
            if g.type==mujoco.mjtGeom.mjGEOM_PLANE:g.pos[2]=-.03
        spec.worldbody.add_geom(name='diagnostic_pile_ground',type=mujoco.mjtGeom.mjGEOM_HFIELD,hfieldname='diagnostic_pile',
            pos=[0,0,-amplitude/2],contype=1,conaffinity=1,condim=3,friction=[args.friction,.005,.0001])
    model=spec.compile();model.opt.timestep=.002;model.opt.iterations=10;model.opt.ls_iterations=20
    model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    physics_audit=donor_physics.configure_and_validate(model,contract) if aligned else None
    dt=float(model.opt.timestep);decimation=round(.02/dt)
    data=mujoco.MjData(model)
    controller=bam_cpu.MujocoController(bam,list(mechanical.NAMES),model,data,vin_drop_gain=0.,vin_min=7. if aligned else 4.)
    qadr=controller.qpos_indexes;vadr=controller.dof_indexes;trunk=model.body('trunk_base').id
    passive_qadr=np.asarray([model.jnt_qposadr[model.joint('passive_'+name+'_backlash').id] for name in mechanical.NAMES]) if args.backlash else None
    passive_vadr=np.asarray([model.jnt_dofadr[model.joint('passive_'+name+'_backlash').id] for name in mechanical.NAMES]) if args.backlash else None
    if args.backlash:
        controller.__class__=BacklashController;controller.passive_qadr=passive_qadr
        if not aligned:
            for name in mechanical.NAMES:
                np.testing.assert_allclose(model.joint('passive_'+name+'_backlash').range,[-np.pi/180,np.pi/180],atol=1e-12)
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.onnx),sess_options=opts,providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
    stop_path=getattr(args,'stop_teacher_onnx',None)
    assert not stop_path or not getattr(args,'feedback_teacher',False), 'Select one simulation teacher'
    stop_session=ort.InferenceSession(str(stop_path),sess_options=opts,providers=['CPUExecutionProvider']) if stop_path else None
    if stop_session:
        assert stop_session.get_inputs()[0].shape==[1,61] and stop_session.get_outputs()[0].shape==[1,14]
    manifest=dict(simulationOnly=True,hardwareAccess=False,training=False,policySha256=meta['policySha256'],
        policyForwardDiagnostic=getattr(args,'policy_forward',False),
        privilegedFeedbackTeacher=getattr(args,'feedback_teacher',False),
        stopTeacherSha256=hashlib.sha256(stop_path.read_bytes()).hexdigest() if stop_path else None,
        feedbackVxCorrectionLimit=getattr(args,'feedback_vx_limit',.12),
        feedbackQueryVxLimit=getattr(args,'feedback_query_vx_limit',.3),
        hardwareProfile="microduck-replica-hd1910-v1",bodyContractApplied=aligned,physicsProfile='kissqy' if aligned else 'local',physicsAudit=physics_audit,bodyGroundMeshes=contacts,
        massKg=float(model.body_mass.sum()),bamKp=args.bam_kp,voltage=args.voltage,friction=args.friction,
        contactTimeconst=args.contact_timeconst,compliantContactIsNotMeasuredCarpet=True,
        terrainHeightMm=args.terrain_height_mm,terrainHash=terrain_hash,terrainBoundary='synthetic smooth heightfield, no measured pile geometry or deformable shear',
        frictionConstraintRevision=bam_cpu.FRICTION_CONSTRAINT_REVISION,
        commandAlpha=.2,actionScale=action_scale,headNewWeight=.5,legsNewWeight=.7,
        jointVelocityObservation='previous native50Hz feedback sample',gyroFilter=False,backlash=args.backlash,
        encoderThroughBacklash=args.backlash,motorVelocityForFriction=True,
        physicsDt=dt,policyDt=.02,extraTransportDelayMs=args.transport_ms,
        limitations=['BAM kp not identified from firmware P5/D0','nominal flat terrain only; training terrain curriculum and domain randomization not reproduced',
            'ideal IMU, quantized encoder position, unquantized simulated velocity','fixed transport delay, no UART faults',
            'initial standing state is synthetic, not the actual getup handoff'])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    reports=[]
    real_start=json.loads(args.start_observation.read_text()) if args.start_observation else None
    cadence=json.loads(args.timing_recording.read_text()) if args.timing_recording else None
    if cadence:
        manifest['cadenceRecordingSha256']=hashlib.sha256(args.timing_recording.read_bytes()).hexdigest()
        manifest['cadenceBoundary']='repeat earlier600 observed inference intervals; quantized to physicsDt; not missing failure-tail timing'
    manifest['voltageCompensation']=args.voltage_compensation
    turn_rate=getattr(args,'turn_rate',0.)
    turn_at=getattr(args,'turn_at',4.)
    manifest['turn']={'yawRateRadPerSecond':turn_rate,'startSeconds':turn_at,'endSeconds':args.seconds-3}
    push_speed=getattr(args,'push_speed',0.)
    push_at=getattr(args,'push_at',4.)
    push_direction=getattr(args,'push_direction','backward')
    manifest['push']={'deltaVelocityMps':push_speed,'atSeconds':push_at,'direction':push_direction,
        'frame':'horizontal body heading at impulse; base translation only'}
    manifest['realStartBoundary']='joint angles, gravity, gyro and observed joint speed; root linear velocity and passive gear state unknown' if real_start else None
    if real_start:manifest['startObservationSha256']=hashlib.sha256(args.start_observation.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    delay=int(np.ceil((float(json.loads(params.read_text())['command_delay'])+args.transport_ms/1000)/dt))
    if aligned:
        delay=4  # nominal 20ms inside donor training's 3..6 physics-step lag range
        manifest['delayBoundary']='fixed 4 ticks; donor randomized 3..6 ticks, no extra transport added'
        manifest['extraTransportDelayMs']=0
        manifest['contractSha256']=hashlib.sha256((args.body_contract or donor_physics.DEFAULT_CONTRACT).read_bytes()).hexdigest()
    if args.command_delay_steps is not None:
        assert 0<=args.command_delay_steps<=20
        delay=args.command_delay_steps
    manifest['effectiveCommandDelaySteps']=delay
    manifest['effectiveCommandDelayMs']=delay*dt*1000
    manifest['commandDelayOverride']=args.command_delay_steps is not None
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for seed in args.seeds:
        for trial in range(args.trials):
            rng=np.random.default_rng(seed+trial)
            perturb=rng.uniform(-.015,.015,14)
            for speed in args.speeds:
                mujoco.mj_resetData(model,data);model.dof_damping[vadr]=0;model.dof_frictionloss[vadr]=0
                data.qpos[:7]=[0,0,.12,1,0,0,0];data.qpos[qadr]=home+perturb;mujoco.mj_forward(model,data)
                data.qpos[2]+=.002-ground.lowest_mesh_z(model,data);mujoco.mj_forward(model,data)
                controller.reset(data.qpos);controller.last_ts=data.time;bam.actuator.q_target_smooth=data.qpos[qadr].copy()
                initial=data.qpos[qadr].copy()
                for _ in range(round(.4/dt)):controller.q_target[:]=initial;controller.update();mujoco.mj_step(model,data)
                if real_start:
                    state=np.asarray(real_start['observation']);gravity=state[3:6]
                    roll=np.arctan2(-gravity[1],-gravity[2]);pitch=np.arcsin(np.clip(gravity[0],-1,1))
                    data.qpos[:7]=[0,0,.12,*imu.quaternion(roll,pitch,0)]
                    data.qpos[qadr]=home+state[6:20]
                    if args.backlash:data.qpos[passive_qadr]=0
                    data.qvel[:]=0;data.qvel[3:6]=state[:3];data.qvel[vadr]=state[20:34]
                    mujoco.mj_forward(model,data);data.qpos[2]+=.002-ground.lowest_mesh_z(model,data);mujoco.mj_forward(model,data)
                    controller.reset(data.qpos);controller.last_ts=data.time;bam.actuator.q_target_smooth=data.qpos[qadr].copy()
                    initial=mechanical.encode_target(np.asarray(real_start['homeRadians'])+np.asarray(real_start['filteredActionRadians']))
                start=data.time;start_qpos=data.qpos.copy();previous=np.zeros(14,dtype=np.float32)
                filtered=previous.copy();command=np.zeros(3,dtype=np.float32)
                history=deque([initial.copy() for _ in range(delay+1)],maxlen=delay+1)
                velocity_previous=data.qvel[vadr].copy()+(data.qvel[passive_vadr] if args.backlash else 0);rows=[];fallen=None
                if real_start:
                    previous=np.asarray(real_start['observation'][34:48],dtype=np.float32)
                    filtered=np.asarray(real_start['emaPreviousRadians'],dtype=np.float32)
                    command=np.asarray(real_start['commandTwist'],dtype=np.float32)
                pushed=False
                has_moved=False
                teacher=None
                if getattr(args,'feedback_teacher',False):
                    from walk_feedback_teacher import FeedbackTeacher
                    teacher=FeedbackTeacher(vx_limit=getattr(args,"feedback_vx_limit",.12),query_vx_limit=getattr(args,"feedback_query_vx_limit",.3))
                for tick in range(round(args.seconds/.02)):
                    t=float(data.time-start)
                    if t>=args.seconds:break
                    if push_speed and not pushed and t>=push_at-1e-8:
                        heading=data.xmat[trunk].reshape(3,3)[:2,0].copy()
                        heading/=np.linalg.norm(heading)
                        left=np.array([-heading[1],heading[0]])
                        direction={'forward':heading,'backward':-heading,'left':left,'right':-left}[push_direction]
                        data.qvel[:2]+=push_speed*direction
                        pushed=True
                    target_command=np.asarray([speed if 2<=t<args.seconds-3 else 0.,0,0],dtype=np.float32)
                    if turn_at<=t<args.seconds-3:target_command[2]=turn_rate
                    command=command_step(command,target_command)
                    if getattr(args, 'policy_forward', False):
                        # Diagnostic parity with mjlab's once-per-policy forward.
                        # This also refreshes constraint loads read by BAM.
                        mujoco.mj_forward(model, data)
                    gyro,gravity=imu.root_body_imu_observation(model,data,trunk)
                    encoder_q=data.qpos[qadr]+(data.qpos[passive_qadr] if args.backlash else 0)
                    obs=np.concatenate([gyro,gravity,observed_angles(encoder_q)-home,velocity_previous,previous,command,np.zeros(10)]).astype(np.float32)
                    query=obs.copy()
                    if teacher is not None:
                        body_velocity=data.xmat[trunk].reshape(3,3).T@data.qvel[:3]
                        query[48:51]=teacher.query(command,[body_velocity[0],gyro[2]],
                            enabled=np.linalg.norm(target_command)>.001)
                    moving_now=np.linalg.norm(target_command)>.001
                    has_moved=has_moved or moving_now
                    active_session=stop_session if stop_session and has_moved and not moving_now else session
                    raw=active_session.run(None,{active_session.get_inputs()[0].name:query[None]})[0][0]
                    if not np.isfinite(raw).all():raise ValueError('Non-finite frozen policy output')
                    voltage_scale=np.float32(7.4/np.clip(args.voltage,6.,9.5)) if args.voltage_compensation else np.float32(1.)
                    filtered=action_step(filtered,raw*voltage_scale,tick>0,action_scale);desired=home.astype(np.float64)+filtered
                    target=mechanical.encode_target(desired);previous=raw.copy()
                    tilt=float(np.degrees(np.arccos(np.clip(-gravity[2],-1,1))))
                    contact_pairs=[[model.geom(c.geom1).name,model.geom(c.geom2).name] for c in data.contact]
                    rows.append(dict(time=t,tilt=tilt,command=command.tolist(),raw=raw.tolist(),target=target.tolist(),contacts=contact_pairs,
                        qpos=data.qpos.tolist(),velocity=data.qvel.tolist(),obs=obs.tolist(),teacherCommand=query[48:51].tolist() if teacher is not None else None,clipped=(np.abs(desired-np.clip(desired,np.asarray(list(mechanical.LIMITS.values()))[:,0],np.asarray(list(mechanical.LIMITS.values()))[:,1]))>1e-7).tolist()))
                    if tilt>=60:fallen=t;break
                    velocity_previous=data.qvel[vadr].copy()+(data.qvel[passive_vadr] if args.backlash else 0)
                    steps=max(1,round(cadence[tick%len(cadence)]/dt)) if cadence else decimation
                    for physics_tick in range(steps):
                        history.append(target.copy());controller.q_target[:]=history[0]
                        controller.update();mujoco.mj_step(model,data)
                key=f'seed{seed}-trial{trial}-speed{speed:g}'
                (args.output/(key+'.json')).write_text(json.dumps(dict(initialQpos=start_qpos.tolist(),frames=rows)))
                end=np.asarray(rows[-1]['qpos'])
                moving=[r for r in rows if 4<=r['time']<args.seconds-3]
                row=dict(seed=seed,trial=trial,speed=speed,fallSeconds=fallen,completed=fallen is None,
                    maxTiltDeg=max(r['tilt'] for r in rows),distanceM=float(end[0]-start_qpos[0]),
                    movingMeanVx=float(np.mean([r['velocity'][0] for r in moving])) if moving else None,
                    clippedFrames=sum(any(r['clipped']) for r in rows),pushApplied=pushed)
                reports.append(row);print(json.dumps(row),flush=True)
                (args.output/'summary.json').write_text(json.dumps(dict(manifest=manifest,results=reports),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True)
    p.add_argument('--physics-profile',choices=['local','kissqy'],default='local');p.add_argument('--body-contract',type=Path,help='Explicit donor physics contract; enables backlash and kissqy profile');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--speeds',type=float,nargs='+',default=[.1,.15,.2,.25]);p.add_argument('--seeds',type=int,nargs='+',default=[130031,240017])
    p.add_argument('--trials',type=int,default=2);p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--voltage',type=float,default=7.4);p.add_argument('--friction',type=float,default=1.)
    p.add_argument('--bam-kp',type=float,default=5.);p.add_argument('--transport-ms',type=float,default=8.)
    p.add_argument('--backlash',action='store_true')
    p.add_argument('--start-observation',type=Path)
    p.add_argument('--voltage-compensation',action='store_true')
    p.add_argument('--timing-recording',type=Path)
    p.add_argument('--contact-timeconst',type=float,help='Diagnostic normal-contact compliance, not an identified carpet model')
    p.add_argument('--terrain-height-mm',type=float,default=0.)
    p.add_argument('--command-delay-steps',type=int,help='Frozen diagnostic override in physics ticks (local 2ms, kissqy 5ms); not measured hardware latency')
    p.add_argument('--push-speed',type=float,default=0.,help='Instant horizontal base velocity change in m/s; not force')
    p.add_argument('--push-at',type=float,default=4.)
    p.add_argument('--push-direction',choices=['forward','backward','left','right'],default='backward')
    p.add_argument('--turn-rate',type=float,default=0.,help='Signed yaw command in rad/s during walking')
    p.add_argument('--turn-at',type=float,default=4.,help='Start turning after the initial straight walk')
    p.add_argument('--policy-forward',action='store_true',help='Diagnostic: refresh constraints before each policy observation, as mjlab does')
    p.add_argument('--stop-teacher-onnx',type=Path,help='Simulation-only stop controller diagnostic')
    p.add_argument('--feedback-query-vx-limit',type=float,default=.3,help='Simulation teacher command query bound; runtime unchanged')
    p.add_argument('--feedback-vx-limit',type=float,default=.12,help='Simulation teacher correction bound; runtime unchanged')
    p.add_argument('--feedback-teacher',action='store_true',help='Privileged simulation-only feedback demonstration; not a deployed policy')
    run(p.parse_args())

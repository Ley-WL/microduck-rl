"""Write the bounded training contract; no hardware qualification claims."""
import hashlib,json,re,sys
from pathlib import Path
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks.microduck_hd1910_stand_env_cfg import PARAMS
from mjlab_microduck.tasks.microduck_hd1910_skills_env_cfg import STAND_Z,SIT_Z,POSE_NAMES,POSE_ANGLES,SITTING_TARGET_OVERRIDES
from mjlab_microduck.tasks.getup_mechanical_contract import NAMES,LIMITS,REFERENCES,DIRECTIONS,IDS,REVISION

policy=Path(sys.argv[1]);task='Mjlab-GetupMechanical-Flat-MicroDuck-HD1910'
cfg=load_env_cfg(task,play=True);robot=cfg.scene.entities['robot'];spec=robot.spec_fn()
for collision in robot.collisions:collision.edit_spec(spec)
model=spec.compile();names=[model.actuator(i).name for i in range(model.nu)]
assert tuple(names)==NAMES
home=[next(v for k,v in HOME_FRAME.joint_pos.items() if re.fullmatch(k,n)) for n in names]
sim=cfg.sim.mujoco;replica=PARAMS.parent.parent/'replica'
metadata=dict(artifactType='training-policy',task=task,skillKind='getup',runtimeKind='hd1910_getup',
    policySha256=hashlib.sha256(policy.read_bytes()).hexdigest(),actorObservationDim=61,actionDim=14,controlHz=50,
    observationNormalization='baked into ONNX by upstream exporter',
    actionSemantics='joint position delta in radian from HOME, scale=1.0; runtime must add HOME',
    jointOrder=names,homeRadians=home,trainingTargetClipRadians=LIMITS,
    mechanicalPhysicalJointLimitsRadians=LIMITS,mechanicalExecutionContract=REVISION,
    targetEncoderReferences=dict(zip(IDS,REFERENCES)),targetEncoderDirections=dict(zip(IDS,DIRECTIONS)),
    actionEmaOldWeight=.45,actionEmaNewWeight=.55,runtimeActionAlpha=.45,
    lastActionObservation='raw policy output before EMA, clipping and encoding',
    nominalTrunkHeightM=STAND_Z,nominalSeatedTrunkHeightM=SIT_Z,sitJointOverrides=SITTING_TARGET_OVERRIDES,
    spawnPoseAnglesDeg=dict(zip(POSE_NAMES,POSE_ANGLES)),
    skillPhysicsRevision='getup-kissqy-mechanical-v1',geometry='replica STL, body-floor contacts, current kissqy mechanical stops',
    simulation=dict(physicsTimestepSeconds=sim.timestep,controlTimestepSeconds=sim.timestep*cfg.decimation,
        integrator=sim.integrator,solverIterations=sim.iterations,lineSearchIterations=sim.ls_iterations,
        disableflags=list(sim.disableflags),enableflags=list(sim.enableflags)),
    bodyGroundContactMeshes=sum(model.geom(i).name.startswith('body_ground_') for i in range(model.ngeom)),
    assembledBodyGroundContactMeshes=sum(model.geom(i).name.startswith('body_ground_') for i in range(model.ngeom)),
    actuatorParamsSha256=hashlib.sha256(PARAMS.read_bytes()).hexdigest(),
    identifiedCommandDelaySeconds=json.loads(PARAMS.read_text())['command_delay'],
    robotSourceHashes={p.relative_to(replica).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in replica.rglob('*') if p.is_file() and p.suffix in ('.xml','.stl')},
    simulationMassKg=float(model.body_mass.sum()),gainBaseline=5,getupMinSeconds=3,uprightHoldSeconds=1,
    hardwareValidated=False,formalAcceptance=False,
    unverified='Physical inertia/contact and HD1910 firmware P/D correspondence are not measured by this training run')
policy.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps({'policySha256':metadata['policySha256'],'executionContract':REVISION,'physicalStopsAndTargetBounds':True}))

"""Write training provenance; this is not a hardware deployment manifest."""
import hashlib,json,re,sys
from pathlib import Path
import mjlab_microduck.tasks
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks.microduck_hd1910_stand_env_cfg import PARAMS,STAND_HEIGHT,get_replica_spec
policy=Path(sys.argv[1])
task=sys.argv[2] if len(sys.argv)>2 else "Mjlab-StandHold-Flat-MicroDuck-HD1910"
from mjlab.tasks.registry import load_env_cfg
env_cfg=load_env_cfg(task,play=True)
robot_cfg=env_cfg.scene.entities["robot"]
spec=robot_cfg.spec_fn()
for collision in robot_cfg.collisions:
    collision.edit_spec(spec)
model=spec.compile()
names=[model.actuator(i).name for i in range(model.nu)]
home=[next(v for k,v in HOME_FRAME.joint_pos.items() if re.fullmatch(k,n)) for n in names]
metadata={"artifactType":"training-policy","task":task,"policySha256":hashlib.sha256(policy.read_bytes()).hexdigest(),"actuatorParamsSha256":hashlib.sha256(PARAMS.read_bytes()).hexdigest(),"geometry":"microduck-replica robot_walk.xml snapshot in robot/replica","controlHz":50,"actorObservationDim":61,"actionDim":14,"observationNormalization":"baked into ONNX by upstream exporter","actionSemantics":"joint position delta in radian from HOME, scale=1.0; runtime must add HOME","jointOrder":names,"homeRadians":home,"nominalTrunkHeightM":STAND_HEIGHT,"simulationMassKg":float(model.body_mass.sum()),"hardwareValidated":False,"intendedStart":"upright near HOME; not arbitrary fallen-pose recovery","gainBaseline":5,"unverified":"HD1910 firmware gain mapping, physical mass/inertia, original identification conditions"}
metadata["bodyGroundContactMeshes"]=sum("body_ground_" in model.geom(i).name for i in range(model.ngeom))
action_cfg=env_cfg.actions.get('joint_pos')
metadata['trainingTargetClipRadians']=getattr(action_cfg,'clip',None)
metadata['actionEmaOldWeight']=getattr(action_cfg,'alpha',0.)
metadata['actionEmaNewWeight']=1.-metadata['actionEmaOldWeight']
metadata['lastActionObservation']='raw policy output before final target clipping'
if task in ('Mjlab-Getup-Flat-MicroDuck-HD1910', 'Mjlab-SitStand-Flat-MicroDuck-HD1910', 'Mjlab-SitStandContact-Flat-MicroDuck-HD1910', 'Mjlab-SitStandTransfer-Flat-MicroDuck-HD1910') or task.startswith('Mjlab-SitStandProcess'):
    from mjlab_microduck.tasks.microduck_hd1910_skills_env_cfg import STAND_Z, SIT_Z, POSE_NAMES, POSE_ANGLES, SITTING_TARGET_OVERRIDES
    if 'SitStandContact' in task or 'SitStandTransfer' in task or 'SitStandProcess' in task:
        from mjlab_microduck.tasks.microduck_hd1910_sit_contact_env_cfg import STAND_Z, SIT_Z, SIT_OVERRIDES as SITTING_TARGET_OVERRIDES
    metadata['nominalTrunkHeightM'] = STAND_Z
    metadata['nominalSeatedTrunkHeightM'] = SIT_Z
    metadata['geometry'] = 'updated replica STL with floor-only body contacts and XgoDuck joint limits'
    metadata['intendedStart'] = ('fallen poses: '+', '.join(POSE_NAMES[2:])) if 'Getup' in task else 'upright or seated; commanded bidirectional transitions and holds'
    metadata['commandSemantics'] = 'all 13 command slots zero' if 'Getup' in task else 'twist.vx = 1 for sit, 0 for stand; remaining slots zero'
    metadata['skillKind'] = 'getup' if 'Getup' in task else 'sitstand'
    metadata['sitJointOverrides'] = SITTING_TARGET_OVERRIDES
    metadata['spawnPoseAnglesDeg'] = dict(zip(POSE_NAMES, POSE_ANGLES))
    metadata['skillPhysicsRevision'] = 'body-ground-collision-selection-v2'
    sim = env_cfg.sim.mujoco
    metadata['simulation'] = {
        'physicsTimestepSeconds': sim.timestep, 'controlTimestepSeconds': sim.timestep*env_cfg.decimation,
        'integrator': sim.integrator, 'solverIterations': sim.iterations,
        'lineSearchIterations': sim.ls_iterations, 'disableflags': list(sim.disableflags),
        'enableflags': list(sim.enableflags),
    }
    metadata['identifiedCommandDelaySeconds'] = json.loads(PARAMS.read_text())['command_delay']
    metadata['skillEnvironmentConfigSha256'] = hashlib.sha256(
        (Path(mjlab_microduck.tasks.__file__).parent/'microduck_hd1910_skills_env_cfg.py').read_bytes()).hexdigest()
    contacts = [i for i in range(model.ngeom) if model.geom(i).name.startswith('body_ground_')]
    is_contact_candidate = 'SitStandContact' in task or 'SitStandTransfer' in task or 'SitStandProcess' in task
    masks = (3, 2) if is_contact_candidate else (1, 0)
    assert len(contacts) == 68 and all((model.geom_contype[i],model.geom_conaffinity[i]) == masks for i in contacts)
    if is_contact_candidate:
        metadata['skillPhysicsRevision'] = 'sit-self-contact-bounded-target-v3'
        metadata['geometry'] = 'replica STL, body-floor + self contact; explicit neck/jaw assembly exclusion'
        metadata['runtimeKind'] = 'hd1910_sitstand'
        metadata['selfCollisionAssemblyExcludes'] = [['neck_pitch', 'jaw_soft']]
        metadata.update(sitSettleSeconds=4, standSeconds=4, uprightHoldSeconds=1,
                        commandEncoding={'slot':'twist.vx','sit':1,'stand':0})
        metadata['skillEnvironmentConfigSha256'] = hashlib.sha256(
            (Path(mjlab_microduck.tasks.__file__).parent/('microduck_hd1910_sit_process_env_cfg.py' if 'SitStandProcess' in task else 'microduck_hd1910_sit_transfer_env_cfg.py' if 'SitStandTransfer' in task else 'microduck_hd1910_sit_contact_env_cfg.py')).read_bytes()).hexdigest()
        metadata['contactContractSourceSha256'] = hashlib.sha256(
            (PARAMS.parent.parent/'sit_contact_contract.py').read_bytes()).hexdigest()
        if 'SitStandProcess' in task and task!='Mjlab-SitStandProcess-Flat-MicroDuck-HD1910':
            metadata['skillEnvironmentConfigSha256']=hashlib.sha256((Path(mjlab_microduck.tasks.__file__).parent/'sit_process_ablation_env_cfg.py').read_bytes()).hexdigest()
    metadata['assembledBodyGroundContactMeshes'] = len(contacts)
    replica_dir = PARAMS.parent.parent/'replica'
    metadata['robotSourceHashes'] = {p.relative_to(replica_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in replica_dir.rglob('*') if p.is_file() and p.suffix in ('.xml', '.stl')}
policy.with_suffix(".metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
print(json.dumps(metadata))

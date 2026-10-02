"""Write training provenance; this is not a hardware deployment manifest."""
import hashlib,json,re,sys
from pathlib import Path
import mjlab_microduck.tasks
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks.microduck_hd1910_stand_env_cfg import PARAMS,STAND_HEIGHT,get_replica_spec
policy=Path(sys.argv[1])
task=sys.argv[2] if len(sys.argv)>2 else "Mjlab-StandHold-Flat-MicroDuck-HD1910"
from mjlab.tasks.registry import load_env_cfg
model=load_env_cfg(task,play=True).scene.entities["robot"].spec_fn().compile()
names=[model.actuator(i).name for i in range(model.nu)]
home=[next(v for k,v in HOME_FRAME.joint_pos.items() if re.fullmatch(k,n)) for n in names]
metadata={"artifactType":"training-policy","task":task,"policySha256":hashlib.sha256(policy.read_bytes()).hexdigest(),"actuatorParamsSha256":hashlib.sha256(PARAMS.read_bytes()).hexdigest(),"geometry":"microduck-replica robot_walk.xml snapshot in robot/replica","controlHz":50,"actorObservationDim":61,"actionDim":14,"observationNormalization":"baked into ONNX by upstream exporter","actionSemantics":"joint position delta in radian from HOME, scale=1.0; runtime must add HOME","jointOrder":names,"homeRadians":home,"nominalTrunkHeightM":STAND_HEIGHT,"simulationMassKg":float(model.body_mass.sum()),"hardwareValidated":False,"intendedStart":"upright near HOME; not arbitrary fallen-pose recovery","gainBaseline":5,"unverified":"HD1910 firmware gain mapping, physical mass/inertia, original identification conditions"}
metadata["bodyGroundContactMeshes"]=sum("body_ground_" in model.geom(i).name for i in range(model.ngeom))
policy.with_suffix(".metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
print(json.dumps(metadata))

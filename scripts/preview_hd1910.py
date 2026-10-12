"""Windows-compatible CPU MuJoCo + Feetech BAM live policy preview."""
import argparse,json,time,queue
from collections import deque
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
from bam.model import load_model
from bam.mujoco import MujocoController

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument("--headless",action="store_true")
p.add_argument("--official-targets",action="store_true",help="Rehearse hardware official-angle target saturation")
p.add_argument("--reference-pose",action="store_true",help="Hold HOME without policy, for equilibrium diagnostics")
p.add_argument("--initial-joints",type=Path,help="Saved policy failure evidence: start from measured joint angles")
p.add_argument("--home-transition",type=float,default=0,help="Reference-pose transition duration in seconds")
p.add_argument("--foot-only",action="store_true",help="Reproduce the old feet-only ground collision for diagnosis")
p.add_argument("--seconds",type=float,default=20)
p.add_argument("--walk-speed",type=float,default=0.,help="Body-frame forward command in m/s (zero keeps the standing preview)")
p.add_argument("--walk-start",type=float,default=2.,help="Seconds of standing before walking")
p.add_argument("--walk-duration",type=float,default=13.,help="Walking duration before a zero-command stop; R restarts")
p.add_argument("--push-strength",type=float,default=.30,help="Instant velocity change in m/s")
p.add_argument("--push-direction",choices=["forward","backward","left","right","up","down"])
p.add_argument("--push-at",type=float,default=4.0)
p.add_argument("--policy",type=Path,default=root/"artifacts/hd1910-stand-v2/hd1910-stand-v2-250.onnx")
a=p.parse_args()
meta=json.loads(a.policy.with_suffix(".metadata.json").read_text(encoding="utf-8"))
validation_note=""
proof_path=a.policy.parent/"verification.json"
if proof_path.exists():
    proof=json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("acceptedInSimulation") is False:
        validation_note="Validation FAILED: diagnostic preview"
home=np.array(meta["homeRadians"],dtype=np.float32)
initial=home.copy()
if a.initial_joints:
    if not a.reference_pose or a.home_transition<=0:
        p.error("initial-joints requires reference-pose and positive home-transition")
    evidence=json.loads(a.initial_joints.read_text(encoding="utf-8"))
    ids=[20,21,22,23,24,30,31,32,33,10,11,12,13,14]
    calibration=evidence["jointCalibration"]
    initial=np.array([(evidence["feedback"][str(sid)]["position"]-calibration["references"][str(sid)])*2*np.pi/4096/calibration["directions"].get(str(sid),-1) for sid in ids],dtype=np.float32)
    assert np.isfinite(initial).all()
bam=load_model(str(root/"src/mjlab_microduck/robot/hd1910/1910_m6.json"))
bam.actuator.kp=5.0; bam.actuator.vin=7.4; bam.actuator.max_current=None
spec=mujoco.MjSpec.from_file(str(root/"src/mjlab_microduck/robot/replica/scene_walk.xml"))
from preview_ground import add_body_ground_contacts, lowest_mesh_z
body_contacts=0 if a.foot_only else add_body_ground_contacts(spec)
limit=7.4*bam.kt.value/bam.R.value
for actuator in spec.actuators:
    actuator.set_to_motor();actuator.forcelimited=True;actuator.forcerange=(-limit,limit)
    actuator.ctrllimited=False;actuator.gear=[1.0,0,0,0,0,0]
for joint in spec.joints:
    if joint.name in meta["jointOrder"]:
        joint.damping=np.zeros((3,1));joint.frictionloss=0
        joint.solref_friction=(-5e4,-2e2);joint.solimp_friction=(.99,.9999,.001,.5,2)
for geom in spec.geoms:
    if geom.name in ("left_foot_collision","right_foot_collision"):
        geom.condim=3;geom.priority=1;geom.friction=[1.0,.005,.0001]
model=spec.compile();model.opt.timestep=.002
model.opt.iterations=10;model.opt.ls_iterations=20
model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
data=mujoco.MjData(model)
controller=MujocoController(bam,meta["jointOrder"],model,data,vin_drop_gain=0.0,vin_min=4.0)
qadr=controller.qpos_indexes;vadr=controller.dof_indexes
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
policy=ort.InferenceSession(str(a.policy),sess_options=opts,providers=["CPUExecutionProvider"])
push_strength=a.push_strength
last_push="None"
last_push_delta=None
keys=queue.Queue();history=deque(maxlen=8);last_action=np.zeros(14,dtype=np.float32);last_vel=np.zeros(14,dtype=np.float32)

def reset():
    global last_action,last_vel
    mujoco.mj_resetData(model,data)
    data.qpos[:7]=[0,0,meta["nominalTrunkHeightM"]+.002,1,0,0,0]
    data.qpos[qadr]=initial
    mujoco.mj_forward(model,data)
    controller.reset(data.qpos);controller.last_ts=data.time
    bam.actuator.q_target_smooth=data.qpos[qadr].copy()
    last_action=np.zeros(14,dtype=np.float32);last_vel=np.zeros(14,dtype=np.float32)
    history.clear()
    for _ in range(8):history.append(initial.copy())

def push(direction):
    global last_push,last_push_delta
    vectors={"forward":(1,0,0),"backward":(-1,0,0),"left":(0,1,0),"right":(0,-1,0),"up":(0,0,1),"down":(0,0,-1)}
    vector=np.array(vectors[direction],dtype=float)
    if direction not in ("up","down"):
        rotation=data.xmat[model.body("trunk_base").id].reshape(3,3)
        # Horizontal body-heading frame, independent of the camera or trunk tilt.
        heading=rotation[:2,0].copy();heading/=np.linalg.norm(heading)
        left=np.array([-heading[1],heading[0]])
        vector[:2]=heading*vector[0]+left*vector[1]
    delta=vector*push_strength
    data.qvel[:3]+=delta
    last_push=f"{direction} {push_strength:.2f} m/s"
    last_push_delta=delta.tolist()
    print(f"Push: {last_push}; world velocity delta={last_push_delta}",flush=True)

def step():
    global last_action,last_vel
    rotation=data.xmat[model.body("trunk_base").id].reshape(3,3)
    gravity=rotation.T@np.array([0,0,-1.0])
    command=np.zeros(13,dtype=np.float32)
    if a.walk_start<=data.time<a.walk_start+a.walk_duration:
        command[0]=a.walk_speed
    obs=np.concatenate((data.qvel[3:6],gravity,data.qpos[qadr]-home,last_vel,last_action,command)).astype(np.float32)
    last_vel=data.qvel[vadr].copy()
    action=np.zeros(14,dtype=np.float32) if a.reference_pose else policy.run(None,{policy.get_inputs()[0].name:obs[None]})[0][0]
    assert np.isfinite(action).all()
    last_action=action.copy()
    target=home+action
    if a.reference_pose and a.home_transition>0:
        ratio=min(data.time/a.home_transition,1.0)
        blend=ratio*ratio*(3-2*ratio)
        target=initial+(home-initial)*blend
    if meta.get('trainingTargetClipRadians'):
        bounds=np.array([meta['trainingTargetClipRadians'][name] for name in meta['jointOrder']])
        target=np.clip(target,bounds[:,0],bounds[:,1])
    elif a.official_targets:
        ranges=np.array([[-25,30],[-22,22],[-90,90],[-90,90],[-90,90],
                         [-90,60],[-90,90],[-170,170],[-25,25],
                         [-30,25],[-22,22],[-90,90],[-90,90],[-90,90]])
        target=np.clip(target,np.radians(ranges[:,0]),np.radians(ranges[:,1]))
    for _ in range(10):
        history.append(target.copy());controller.q_target[:]=history[-5]
        controller.update();mujoco.mj_step(model,data)
    tilt=float(np.degrees(np.arccos(np.clip(-gravity[2],-1,1))))
    return tilt

reset()
print("HD1910 LIVE CPU PREVIEW | ONNX 61 -> 14 | 50Hz | 7.4V",flush=True)
print(f"Policy: {a.policy.name}",flush=True)
if a.walk_speed:
    print(f"Walk: {a.walk_speed:.2f} m/s, zero turn command, starts at {a.walk_start}s for {a.walk_duration}s; R restarts",flush=True)
print(f"Ground contacts: soles + {body_contacts} body meshes; training body contacts={meta.get('bodyGroundContactMeshes',0)}",flush=True)
if validation_note:print(validation_note,flush=True)
print("Arrows: body forward/back/left/right | PgUp/PgDn: up/down | F9 / F10: strength -/+0.05 m/s | P: forward | R: reset",flush=True)
if a.headless:
    tilts=[];root_heights=[]
    for tick in range(round(a.seconds/.02)):
        if a.push_direction and tick==round(a.push_at/.02):push(a.push_direction)
        tilts.append(step())
        root_heights.append(float(data.qpos[2]))
    print(json.dumps({"seconds":a.seconds,"maxTiltDeg":max(tilts),"endHeight":float(data.qpos[2]),"finite":bool(np.isfinite(data.qpos).all()),"pushDirection":a.push_direction,"pushStrength":push_strength,"pushDeltaWorld":last_push_delta}),flush=True)
    print(json.dumps({"jointDegrees":dict(zip(meta["jointOrder"],np.degrees(data.qpos[qadr]).tolist())),"homeErrorDegrees":dict(zip(meta["jointOrder"],np.degrees(data.qpos[qadr]-home).tolist()))}),flush=True)
    print(json.dumps({"bodyGroundMeshes":body_contacts,"minRootHeight":min(root_heights),"endLowestMeshZ":lowest_mesh_z(model,data)}),flush=True)
else:
    from preview_keyboard import ViewerKeyboard
    with mujoco.viewer.launch_passive(model,data) as viewer, ViewerKeyboard(keys) as keyboard:
        print("Native MuJoCo keyboard shortcuts disabled; custom controls only",flush=True)
        viewer.cam.azimuth=130;viewer.cam.elevation=-15;viewer.cam.distance=.60
        viewer.cam.lookat[:]=[0,0,.14]
        tick=0
        while viewer.is_running():
            begin=time.perf_counter()
            while not keys.empty():
                key=keys.get()
                if key in (ord("R"),ord("r")):reset()
                elif key in (ord("P"),ord("p"),265):push("forward")
                elif key==264:push("backward")
                elif key==263:push("left")
                elif key==262:push("right")
                elif key==266:push("up")
                elif key==267:push("down")
                elif key==299:push_strength+=.05  # GLFW F10; no native camera binding
                elif key==298:push_strength=max(0.0,push_strength-.05)  # GLFW F9
            tilt=step()
            if a.walk_speed:
                viewer.cam.lookat[:]=data.xpos[model.body("trunk_base").id]
            walk_note=(f"Walk {a.walk_speed:.2f}m/s / turn 0 | R: replay" if a.walk_speed else "")
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                f"Push strength: {push_strength:.2f} m/s\nArrows: F/B/L/R | PgUp/Dn: vertical\nF9 / F10: strength | P: forward | R: reset",
                f"{'HOME transition '+str(a.home_transition)+'s (no policy)' if a.reference_pose else 'Policy: '+a.policy.stem}\n{validation_note}\n{walk_note}\nLast push: {last_push}\nTilt: {tilt:.1f} deg"))
            viewer.sync();tick+=1
            if tick%50==0:print(f"time={data.time:.1f}s tilt={tilt:.2f}deg height={data.qpos[2]:.4f}m",flush=True)
            time.sleep(max(0,.02-(time.perf_counter()-begin)))

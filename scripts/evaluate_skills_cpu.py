"""CPU ONNX deployment rehearsal on replica STL and HD1910 BAM.

Metadata supplies HOME, measured heights, sitting targets and pose definitions.
No automatic resets or policy swaps. Metadata supplies the actual training
integrator and physics step. Runtime mode adds observation filters and 8ms
transport delay to the explicitly queued identified actuator delay.
Success is measured for every pose/cycle/hold, using the GPU evaluator's gates.
This does not emulate UART failures or unverified hardware gain calibration.
"""
import argparse
from collections import deque
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from bam.model import load_model

# Load by file path so standalone runs and importlib-based tests use one fix.
_controller_spec=importlib.util.spec_from_file_location('skill_cpu_bam_controller',Path(__file__).with_name('bam_cpu_controller.py'))
_controller_module=importlib.util.module_from_spec(_controller_spec)
_controller_spec.loader.exec_module(_controller_module)
MujocoController=_controller_module.MujocoController

ROOT = Path(__file__).resolve().parents[1]


def quaternion(roll, pitch, yaw):
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def segments_for(kind, scenario):
    if kind == 'getup':
        return [(15., False)]
    sit = scenario.startswith('sit_')
    return [(2., sit), (8., not sit), (14., sit), (20., not sit)] if scenario.endswith('cycle') else [(15., sit)]


def trial_success(trace, segments, kind, finite):
    if not finite or not trace:
        return False, []
    fractions = []
    for end, _ in (segments[1:] if len(segments) > 1 else segments):
        window = 3. if kind == 'getup' else 2.
        rows = [row for row in trace if end-window-1e-8 <= row['time'] < end-1e-8]
        fractions.append(float(np.mean([row['stable'] for row in rows])) if len(rows) == round(window/.02) else 0.)
    success = all(f >= .95 for f in fractions)
    if kind == 'sitstand':
        success &= max(row['tiltDeg'] for row in trace) < 45.
    return bool(success), fractions


def transition_smoothness(frames, joint_names, window_seconds=2.):
    """Report motion quality separately from endpoint success; no invented gate.

    Include startup and every command edge. Excess travel catches repeated
    reversals even when the robot eventually reaches the requested pose.
    """
    indexes = [i for i, name in enumerate(joint_names) if name.endswith('knee')]
    starts = [i for i, row in enumerate(frames)
              if i == 0 or row['sit'] != frames[i-1]['sit']]
    results = []
    for start in starts:
        begin = frames[start]['time']
        following = next((i for i in starts if i > start), len(frames))
        rows = [r for r in frames[start:following] if r['time'] < begin+window_seconds-1e-8]
        if len(rows) < 2:
            continue
        target = np.degrees(np.array([r['target'] for r in rows]))[:, indexes]
        delta = np.diff(target, axis=0)
        travel = np.abs(delta).sum(axis=0)
        net = np.abs(target[-1]-target[0])
        results.append({'time':begin, 'sit':bool(rows[0]['sit']), 'samples':len(rows),
                        'jointNames':[joint_names[i] for i in indexes],
                        'targetTravelDeg':travel.tolist(), 'netTravelDeg':net.tolist(),
                        'excessTravelDeg':np.maximum(0, travel-net).tolist(),
                        'maxFrameChangeDeg':np.abs(delta).max(axis=0).tolist()})
    return results


def recovery_smoothness(frames, joint_names):
    """Separate legitimate recovery travel from final standing oscillation."""
    result=[]
    for begin,end in ((0.,3.),(3.,6.),(6.,9.),(9.,12.),(12.,15.)):
        rows=[r for r in frames if begin-1e-8<=r['time']<end-1e-8]
        if len(rows)<2:continue
        targets=np.degrees(np.asarray([r['target'] for r in rows]))
        differences=np.diff(targets,axis=0)
        travel=np.abs(differences).sum(axis=0)
        net=np.abs(targets[-1]-targets[0])
        result.append({'begin':begin,'end':end,'samples':len(rows),'jointNames':joint_names,
                       'targetTravelDeg':travel.tolist(),'excessTravelDeg':(travel-net).clip(0).tolist(),
                       'maxFrameChangeDeg':np.abs(differences).max(axis=0).tolist()})
    return result


def apply_simulation_contract(model, metadata):
    simulation = metadata['simulation']
    dt = float(simulation['physicsTimestepSeconds'])
    control_dt = float(simulation['controlTimestepSeconds'])
    if dt <= 0 or not math.isclose(control_dt, .02) or not math.isclose(control_dt/dt, round(control_dt/dt)):
        raise ValueError('Expected integral physics steps at 50Hz control')
    model.opt.timestep = dt
    model.opt.integrator = {'euler': 0, 'rk4': 1, 'implicit': 2, 'implicitfast': 3}[simulation['integrator']]
    model.opt.iterations = simulation['solverIterations']
    model.opt.ls_iterations = simulation['lineSearchIterations']
    model.opt.disableflags = sum(int(getattr(mujoco.mjtDisableBit, 'mjDSBL_'+flag.upper())) for flag in simulation['disableflags'])
    model.opt.enableflags = sum(int(getattr(mujoco.mjtEnableBit, 'mjENBL_'+flag.upper())) for flag in simulation['enableflags'])


def command_delay_steps(metadata, profile):
    # MujocoController does not implement the identified command_delay itself.
    delay = float(metadata['identifiedCommandDelaySeconds']) + (.008 if profile == 'runtime' else 0.)
    if delay < 0:
        raise ValueError('Negative command delay')
    return math.ceil(delay/float(metadata['simulation']['physicsTimestepSeconds']))


class VelocityObservationDelay:
    """Diagnostic policy-frame lag; reset buffers start at zero like mjlab."""
    def __init__(self, steps):
        if steps < 0:
            raise ValueError('Negative observation delay')
        self.history = deque([np.zeros(14) for _ in range(steps+1)], maxlen=steps+1)

    def update(self, velocity):
        self.history.append(velocity.copy())
        return self.history[0].copy()


def root_body_imu_observation(model, data, body):
    """Free-joint angular qvel is body-frame, unlike BODY local object velocity.

    mj_objectVelocity(..., local=1) rotates into the inertial principal axes.
    CAD body_iquat need not be identity. Refresh kinematics after integration
    without recomputing the previous constraint loads used by BAM.
    """
    joint = int(model.body_jntadr[body])
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Expected free-joint root body for IMU observation')
    mujoco.mj_kinematics(model, data)
    dof = int(model.jnt_dofadr[joint])
    gyro = data.qvel[dof+3:dof+6].copy()
    gravity = data.xmat[body].reshape(3, 3).T@np.array([0., 0., -1.])
    return gyro, gravity


def observation_filter(previous, current, old_weight):
    if not 0 <= old_weight < 1:
        raise ValueError('Observation filter old weight must be in [0,1)')
    return old_weight*previous+(1-old_weight)*current


def build_model(metadata, contact_repair=False):
    source = ROOT/'src/mjlab_microduck/robot'
    helper = importlib.util.spec_from_file_location('cpu_skill_ground', source/'ground_contact.py')
    ground = importlib.util.module_from_spec(helper)
    helper.loader.exec_module(ground)
    spec = mujoco.MjSpec.from_file(str(source/'replica/scene_walk.xml'))
    names = metadata['jointOrder']
    params = source/'hd1910/1910_m6.json'
    assert hashlib.sha256(params.read_bytes()).hexdigest() == metadata['actuatorParamsSha256']
    for relative, expected in metadata['robotSourceHashes'].items():
        assert hashlib.sha256((source/'replica'/relative).read_bytes()).hexdigest() == expected, relative
    bam = load_model(str(params))
    bam.actuator.kp = 5.
    bam.actuator.vin = 7.4
    bam.actuator.max_current = None
    maximum = 7.4*bam.kt.value/bam.R.value
    count = 0
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname not in ('sole_left', 'sole_right') and geom.contype == 0 and geom.conaffinity == 0:
            geom.name = f'body_ground_{count}'
            geom.contype, geom.conaffinity = 1, 0
            geom.solref = np.array([.004, 1.])
            count += 1
        if geom.name in ('left_foot_collision', 'right_foot_collision') or geom.name.startswith('body_ground_'):
            geom.contype, geom.conaffinity = 1, 0
            geom.condim = 3
            geom.friction = [1., .005, .0001]
            if geom.name.endswith('foot_collision'):
                geom.priority = 1
        elif geom.type != mujoco.mjtGeom.mjGEOM_PLANE and geom.meshname != 'leg':
            # Actual v2 assembled model retains two unnamed leg meshes at 2/2
            # through their inherited default class, but disables power_support.
            # See checkpoint-2000/cpu-model-audit.json; intended selection alone
            # is not proof of the compiled unnamed-geom masks.
            geom.contype, geom.conaffinity = 0, 0
    if contact_repair or metadata.get('skillPhysicsRevision') == 'sit-self-contact-bounded-target-v3':
        helper = importlib.util.spec_from_file_location('sit_contact_contract', source/'sit_contact_contract.py')
        contract = importlib.util.module_from_spec(helper)
        helper.loader.exec_module(contract)
        contract.enable_sit_self_contacts(spec)
    for joint in spec.joints:
        if joint.name in names:
            joint.armature = float(bam.actuator.get_extra_inertia())
            joint.damping = np.zeros((3, 1))
            joint.frictionloss = 0
            joint.solref_friction = (-5e4, -2e2)
            joint.solimp_friction = (.99, .9999, .001, .5, 2)
    for actuator in spec.actuators:
        actuator.set_to_motor()
        actuator.forcelimited = True
        actuator.forcerange = (-maximum, maximum)
        actuator.gear = [1., 0, 0, 0, 0, 0]
    model = spec.compile()
    apply_simulation_contract(model, metadata)
    assert model.nu == 14 and model.nq == 21
    assert count == metadata['bodyGroundContactMeshes']
    assert [model.actuator(i).name for i in range(14)] == names
    if metadata.get('skillPhysicsRevision') == 'sit-self-contact-bounded-target-v3':
        bounds = metadata.get('trainingTargetClipRadians', {})
        if set(bounds) != set(names):
            raise ValueError('Candidate metadata is missing joint target bounds')
        for name in names:
            if not np.allclose(bounds[name], model.jnt_range[model.joint(name).id], atol=1e-12, rtol=0):
                raise ValueError('Candidate target bounds differ from training XML: '+name)
        if metadata.get('selfCollisionAssemblyExcludes') != [['neck_pitch', 'jaw_soft']]:
            raise ValueError('Candidate collision exclusions differ from training')
    return model, bam, ground


def evaluate(args):
    history_mode = getattr(args, 'action_history_mode', 'preserve')
    metadata = json.loads(args.onnx.with_suffix('.metadata.json').read_text())
    assert hashlib.sha256(args.onnx.read_bytes()).hexdigest() == metadata['policySha256']
    kind = metadata['skillKind']
    action_ema=getattr(args,'action_ema_old_weight',None)
    if action_ema is None:action_ema=float(metadata['actionEmaOldWeight'])
    if not 0<=action_ema<1:raise ValueError('Action EMA old weight must be in [0,1)')
    names = metadata['jointOrder']
    home = np.array(metadata['homeRadians'])
    seated = home.copy()
    for index, value in metadata['sitJointOverrides'].items():
        seated[int(index)] = value
    legs = [i for i, name in enumerate(names) if name.startswith(('left_', 'right_'))]
    heights = (metadata['nominalTrunkHeightM'], metadata['nominalSeatedTrunkHeightM'])
    scenarios = [pose for pose in metadata['spawnPoseAnglesDeg'] if pose not in ('stand', 'sit')] if kind == 'getup' else ['stand_cycle', 'sit_cycle', 'stand_hold', 'sit_hold']
    seated_probe=getattr(args,'getup_seated_probe',False)
    if seated_probe:
        if kind!='getup':raise ValueError('Seated getup probe requires a getup model')
        scenarios+=['sit','stand']
    if len(set(args.seeds)) != len(args.seeds) or (args.scenario and len(set(args.scenario)) != len(args.scenario)):
        raise ValueError('Seeds and scenarios must be unique')
    selected = args.scenario or scenarios
    assert set(selected).issubset(scenarios) and len(set(selected)) == len(selected)
    assert args.trials > 0 and len(set(args.seeds)) == len(args.seeds)
    self_contact_only=getattr(args,'self_contact_only',False)
    target_clip_only=getattr(args,'target_clip_only',False)
    if self_contact_only and target_clip_only:
        raise ValueError('Use --contact-repair for the combined diagnostic')
    model, bam, ground = build_model(metadata, args.contact_repair or self_contact_only)
    kp_scale=getattr(args,'actuator_kp_scale',1.)
    if kp_scale<=0:raise ValueError('Actuator kp scale must be positive')
    bam.actuator.kp*=kp_scale
    # BAM updates these mutable MuJoCo fields every physics tick. Restore
    # compile-time values before each independent trial; otherwise its initial
    # settling depends on the preceding trajectory despite a fresh MjData.
    default_damping=model.dof_damping.copy()
    default_friction=model.dof_frictionloss.copy()
    physics_dt = model.opt.timestep
    substeps = round(.02/physics_dt)
    delay_steps = command_delay_steps(metadata, args.profile)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape == [1, 61] and session.get_outputs()[0].shape == [1, 14]
    args.output.mkdir(parents=True, exist_ok=True)
    groups = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        for scenario in selected:
            results = []
            segments = segments_for(kind, scenario)
            initial_sit = (kind == 'sitstand' and scenario.startswith('sit_')) or (kind=='getup' and scenario=='sit')
            for trial in range(args.trials):
                stress_rng=np.random.default_rng(np.random.SeedSequence([seed,trial,scenarios.index(scenario),4242]))
                model.dof_damping[:]=default_damping
                model.dof_frictionloss[:]=default_friction
                data = mujoco.MjData(model)
                controller = MujocoController(bam, names, model, data, vin_drop_gain=0., vin_min=4.)
                qadr, vadr = controller.qpos_indexes, controller.dof_indexes
                trunk = model.body('trunk_base').id
                pose = scenario if kind == 'getup' else ('sit' if initial_sit else 'stand')
                roll, pitch = np.radians(metadata['spawnPoseAnglesDeg'][pose])
                roll, pitch = np.array([roll, pitch])+rng.uniform(-1., 1., 2)*math.radians(12. if kind == 'getup' else 5.)
                data.qpos[:7] = [0, 0, .12, *quaternion(roll, pitch, rng.uniform(-math.pi, math.pi))]
                data.qpos[qadr] = (seated if initial_sit else home)+rng.uniform(-.08 if kind == 'getup' else -.06, .08 if kind == 'getup' else .06, 14)
                ranges = np.array([model.jnt_range[model.joint(name).id] for name in names])
                data.qpos[qadr] = np.clip(data.qpos[qadr], ranges[:, 0], ranges[:, 1])
                mujoco.mj_forward(model, data)
                data.qpos[2] += .003-ground.lowest_mesh_z(model, data)
                mujoco.mj_forward(model, data)
                controller.q_target[:] = data.qpos[qadr]
                bam.actuator.q_target_smooth = np.array(data.qpos[qadr])
                # Let a real fallen contact state settle without running the policy.
                if args.settle_seconds < 0:
                    raise ValueError('Negative settling duration')
                for _ in range(round(args.settle_seconds/physics_dt)):
                    controller.update()
                    mujoco.mj_step(model, data)
                history = deque([data.qpos[qadr].copy() for _ in range(delay_steps+1)], maxlen=delay_steps+1)
                initial_qpos=data.qpos.copy()
                previous = filtered = np.zeros(14)
                previous_sit = None
                transition_begin = None
                stress_impulses = []
                next_impulse_tick = None
                policy_frames = []
                gyro = np.zeros(3)
                velocity = np.zeros(14)
                velocity_delay = VelocityObservationDelay(args.joint_velocity_lag_steps)
                finite = True
                trace, qpositions = [], []
                renderer = writer = None
                if args.video and trial == 0:
                    import imageio.v2 as imageio
                    renderer = mujoco.Renderer(model, height=480, width=640)
                    camera = mujoco.MjvCamera()
                    camera.azimuth, camera.elevation, camera.distance = 150, -18, .7
                    writer = imageio.get_writer(str(args.output/f'{scenario}-{seed}.mp4'), fps=25)
                try:
                    for tick in range(round(segments[-1][0]/.02)):
                        time = tick*.02
                        sit = next(flag for end, flag in segments if time < end-1e-6)
                        if previous_sit is not None and sit != previous_sit:
                            transition_begin = tick
                            next_impulse_tick = tick
                            if history_mode == 'reset':
                                previous, filtered = np.zeros(14), np.zeros(14)
                            elif history_mode == 'pose':
                                previous, filtered = np.zeros(14), data.qpos[qadr].copy()-home
                        previous_sit = sit
                        if kind=='getup' and tick==50:
                            # An independent recovery stress window, 1.0-1.8s;
                            # all baseline initial states and random draws match.
                            transition_begin=next_impulse_tick=tick
                        if getattr(args,'transition_stress',False) and transition_begin is not None and tick-transition_begin<40 and tick==next_impulse_tick:
                            # Physical pitch disturbance; gravity evolves through
                            # MuJoCo, rather than feeding invented gyro values.
                            impulse=float(stress_rng.uniform(.7,1.6))*float(stress_rng.choice([-1,1]))
                            data.qvel[4]+=impulse
                            stress_impulses.append({'time':time,'pitchVelocityImpulseRadS':impulse})
                            next_impulse_tick=tick+int(stress_rng.integers(6,11))
                        body_gyro, gravity = root_body_imu_observation(model, data, trunk)
                        if args.profile == 'runtime':
                            velocity = observation_filter(velocity, data.qvel[vadr], args.joint_velocity_old_weight)
                            gyro = observation_filter(gyro, body_gyro, args.gyro_old_weight)
                        else:
                            velocity, gyro = data.qvel[vadr].copy(), body_gyro
                        commands = np.zeros(13)
                        if kind == 'sitstand':
                            commands[0] = float(sit)
                        observed_velocity = velocity_delay.update(velocity)
                        observation = np.concatenate([gyro, gravity, data.qpos[qadr]-home, observed_velocity, previous, commands]).astype(np.float32)
                        if not np.isfinite(observation).all():
                            finite = False
                            break
                        action = session.run(None, {session.get_inputs()[0].name: observation[None]})[0][0]
                        if not np.isfinite(action).all():
                            finite = False
                            break
                        filtered = action_ema*filtered+(1-action_ema)*action
                        previous = action.copy()
                        target = home+filtered
                        if args.contact_repair or target_clip_only or metadata.get('skillPhysicsRevision') == 'sit-self-contact-bounded-target-v3':
                            hard = model.jnt_range[[model.joint(name).id for name in names]]
                            target = np.clip(target, hard[:, 0], hard[:, 1])
                        if getattr(args, 'capture_policy_frames', False) or kind in ('sitstand','getup'):
                            policy_frames.append({'time':time, 'sit':bool(sit), 'observation':observation.tolist(),
                                                  'rawAction':action.tolist(), 'filteredAction':filtered.tolist(),
                                                  'target':target.tolist(), 'position':data.qpos[qadr].tolist()})
                        for _ in range(substeps):
                            history.append(target.copy())
                            controller.q_target[:] = history[0]
                            controller.update()
                            mujoco.mj_step(model, data)
                        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                            finite = False
                            break
                        rotation = data.xmat[trunk].reshape(3, 3)
                        tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1., 1.))))
                        height = float(data.xpos[trunk, 2])
                        xy_speed, angular_speed = float(np.linalg.norm(data.qvel[:2])), float(np.linalg.norm(data.qvel[3:6]))
                        leg_error = float(np.mean(np.abs(data.qpos[qadr][legs]-(seated if sit else home)[legs])))
                        stable = tilt < 20. and abs(height-heights[int(sit)]) < .012 and xy_speed < .2 and angular_speed < 2. and leg_error < .25
                        trace.append({'time': time, 'sit': sit, 'tiltDeg': tilt, 'heightM': height, 'xySpeed': xy_speed, 'angularSpeed': angular_speed, 'legError': leg_error, 'stable': stable, 'contacts': int(data.ncon)})
                        qpositions.append(data.qpos.copy())
                        if writer and tick % 2 == 0:
                            camera.lookat[:] = data.xpos[trunk]
                            renderer.update_scene(data, camera=camera)
                            writer.append_data(renderer.render())
                finally:
                    if writer:
                        writer.close()
                        renderer.close()
                success, fractions = trial_success(trace, segments, kind, finite)
                result = {'scenario': scenario, 'seed': seed, 'trial': trial, 'success': success, 'finite': finite, 'stableFractions': fractions, 'resetCount': 0}
                result['initialQpos']=initial_qpos.tolist()
                if kind == 'sitstand':
                    result['transitionSmoothness'] = transition_smoothness(policy_frames, names)
                    result['transitionStressImpulses'] = stress_impulses
                if kind=='getup':
                    result['recoverySmoothness']=recovery_smoothness(policy_frames,names)
                    result['recoveryStressImpulses']=stress_impulses
                results.append(result)
                path = args.output/f'{scenario}-{seed}-{trial}'
                path.with_suffix('.json').write_text(json.dumps({'result': result, 'trace': trace}, indent=2))
                if getattr(args, 'capture_policy_frames', False):
                    path.with_suffix('.policy.json').write_text(json.dumps(policy_frames))
                np.savez_compressed(path.with_suffix('.npz'), qpos=np.array(qpositions))
            group = {'scenario': scenario, 'seed': seed, 'trials': args.trials, 'successRate': float(np.mean([r['success'] for r in results])), 'results': results}
            groups.append(group)
            summary = {'kind': kind, 'profile': args.profile, 'policySha256': metadata['policySha256'], 'hardwareValidated': False,
                       'noAutoReset': True, 'configuration': {'physicsHz': round(1/physics_dt), 'policyHz': 50,
                           'integrator': metadata['simulation']['integrator'], 'queuedTargetDelayMs': delay_steps*physics_dt*1000,
                           'identifiedCommandDelayMs': metadata['identifiedCommandDelaySeconds']*1000,
                           'extraTransportDelayMs': 8 if args.profile == 'runtime' else 0,
                           'diagnosticJointVelocityLagSteps': args.joint_velocity_lag_steps,
                           'diagnosticActionHistoryMode': history_mode,
                           'diagnosticTransitionStress':getattr(args,'transition_stress',False),
                           'diagnosticActuatorKpScale':kp_scale,
                           'actionEmaOldWeight':action_ema,
                           'diagnosticGetupSeatedProbe':seated_probe,
                           'independentTrialFrictionReset':True,
                           'frictionConstraintRevision':_controller_module.FRICTION_CONSTRAINT_REVISION,
                           'initialSettleSeconds': args.settle_seconds,
                           'jointVelocityFilterOldWeight': args.joint_velocity_old_weight if args.profile == 'runtime' else 0.,
                           'gyroFilterOldWeight': args.gyro_old_weight if args.profile == 'runtime' else 0.,
                           'imuAngularFrame': 'root-body-frame-freejoint-qvel',
                           'diagnosticContactRepair': args.contact_repair,
                           'diagnosticSelfContactOnly':self_contact_only,
                           'diagnosticTargetClipOnly':target_clip_only,
                           'contactSelectionRevision': 'v2-assembled-with-two-unnamed-leg-meshes',
                           'activeContactGeomCount': int(((model.geom_contype != 0) | (model.geom_conaffinity != 0)).sum()),
                           'actuatorParamsSha256': metadata['actuatorParamsSha256']},
                       'groups': groups, 'scopeComplete': len(groups) == len(scenarios)*len(args.seeds),
                       'accepted': len(groups) == len(scenarios)*len(args.seeds) and args.trials >= 200 and len(set(args.seeds)) >= 2
                                   and not args.contact_repair
                                   and not self_contact_only and not target_clip_only
                                   and metadata.get('skillPhysicsRevision') in ('body-ground-collision-selection-v2','sit-self-contact-bounded-target-v3')
                                   and metadata.get('assembledBodyGroundContactMeshes') == 68
                                   and args.profile == 'runtime' and args.joint_velocity_lag_steps == 0
                                   and history_mode == 'preserve'
                                   and not getattr(args,'transition_stress',False)
                                   and not seated_probe
                                   and math.isclose(kp_scale,1.)
                                   and math.isclose(action_ema,float(metadata['actionEmaOldWeight']))
                                   and math.isclose(args.settle_seconds, .4)
                                   and math.isclose(args.joint_velocity_old_weight, .16)
                                   and math.isclose(args.gyro_old_weight, .25)
                                   and all(g['successRate'] >= .95 for g in groups)}
            (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps({k: group[k] for k in ('scenario', 'seed', 'successRate')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trials', type=int, default=8)
    p.add_argument('--seeds', type=int, nargs='+', default=[1009, 2027])
    p.add_argument('--scenario', nargs='+')
    p.add_argument('--profile', choices=['runtime', 'direct'], default='runtime')
    p.add_argument('--action-history-mode', choices=['preserve','reset','pose'], default='preserve',
                   help='Diagnostic command-switch history; non-preserve cannot count as acceptance')
    p.add_argument('--capture-policy-frames', action='store_true', help='Save raw actions and final targets for offline diagnostics')
    p.add_argument('--transition-stress',action='store_true',help='Physical pitch impulses for 0.8s after each sitstand command edge or at getup t=1s; diagnostic hypothesis')
    p.add_argument('--joint-velocity-lag-steps', type=int, default=0,
                   help='Diagnostic observation lag; nonzero runs cannot count as runtime acceptance')
    p.add_argument('--actuator-kp-scale',type=float,default=1.,help='Diagnostic BAM proportional multiplier, not a measured HD1910 register conversion')
    p.add_argument('--action-ema-old-weight',type=float,default=None,help='Diagnostic model output filtering override; differs from metadata means no acceptance')
    p.add_argument('--getup-seated-probe',action='store_true',help='Add seated/upright initial poses to getup diagnostic scope; cannot count as original six-fallen-cohort acceptance')
    p.add_argument('--settle-seconds', type=float, default=.4,
                   help='Diagnostic startup settling duration; altered durations cannot count as runtime acceptance')
    p.add_argument('--joint-velocity-old-weight', type=float, default=.16,
                   help='Diagnostic velocity filter; altered weights cannot count as runtime acceptance')
    p.add_argument('--gyro-old-weight', type=float, default=.25,
                   help='Diagnostic gyro filter; altered weights cannot count as runtime acceptance')
    p.add_argument('--video', action='store_true')
    p.add_argument('--contact-repair', action='store_true',
                   help='Diagnostic candidate self contact + XML target clamp; cannot count as old-model acceptance')
    p.add_argument('--self-contact-only',action='store_true',help='Diagnostic self collisions without changing v2 policy target clipping')
    p.add_argument('--target-clip-only',action='store_true',help='Diagnostic target clipping without changing v2 collision selection')
    evaluate(p.parse_args())

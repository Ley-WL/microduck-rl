import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import mujoco
import pytest

spec = importlib.util.spec_from_file_location('skill_rehearsal', Path(__file__).parents[1]/'scripts/evaluate_skills_cpu.py')
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)


def rows(seconds, stable=True, tilt=0.):
    return [{'time': tick*.02, 'stable': stable, 'tiltDeg': tilt} for tick in range(round(seconds/.02))]


def test_recovery_process_separates_motion_from_standing_oscillation():
    frames=[{'time':tick*.02,'target':[min(tick,100)*.005, (-1)**tick*.1 if tick>=600 else 0.]} for tick in range(750)]
    metrics=cpu.recovery_smoothness(frames,['left_knee','right_knee'])
    assert len(metrics)==5
    assert metrics[0]['excessTravelDeg'][0]<1e-10
    assert metrics[-1]['excessTravelDeg'][1]>1000
    assert all(row['samples']==150 for row in metrics)


def test_getup_rejects_partial_or_nonfinite_rollouts():
    segments = cpu.segments_for('getup', 'supine')
    assert cpu.trial_success(rows(15.), segments, 'getup', True)[0]
    assert not cpu.trial_success(rows(14.), segments, 'getup', True)[0]
    assert not cpu.trial_success(rows(15.), segments, 'getup', False)[0]
    assert not cpu.trial_success([], segments, 'getup', True)[0]


def test_getup_final_window_requires_95_percent_stable_samples():
    trace = rows(15.)
    for row in trace[-8:]:
        row['stable'] = False
    assert not cpu.trial_success(trace, [(15., False)], 'getup', True)[0]
    trace[-1]['stable'] = True
    assert cpu.trial_success(trace, [(15., False)], 'getup', True)[0]


def test_cycle_requires_every_endpoint_and_rejects_tipping_between_them():
    segments = cpu.segments_for('sitstand', 'stand_cycle')
    assert [flag for _, flag in segments] == [False, True, False, True]
    assert [flag for _, flag in cpu.segments_for('sitstand', 'sit_cycle')] == [True, False, True, False]
    trace = rows(20.)
    assert cpu.trial_success(trace, segments, 'sitstand', True)[0]
    trace[250]['tiltDeg'] = 50.
    assert not cpu.trial_success(trace, segments, 'sitstand', True)[0]
    trace[250]['tiltDeg'] = 0.
    for row in trace[300:400]:
        row['stable'] = False
    assert not cpu.trial_success(trace, segments, 'sitstand', True)[0]


def test_orientation_quaternion_covers_inverted_and_oblique_starts():
    for degrees in ((0, 90, 45), (180, 0, -70), (70, 45, 130)):
        quat = cpu.quaternion(*np.radians(degrees))
        np.testing.assert_allclose(np.linalg.norm(quat), 1., atol=1e-12)


def test_cpu_reads_actual_training_integrator_step_and_explicit_delay():
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    metadata = {'simulation': {'physicsTimestepSeconds': .005, 'controlTimestepSeconds': .02,
                              'integrator': 'implicitfast', 'solverIterations': 30,
                              'lineSearchIterations': 50, 'disableflags': [], 'enableflags': []},
                'identifiedCommandDelaySeconds': .006005}
    cpu.apply_simulation_contract(model, metadata)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.timestep == .005 and model.opt.disableflags == 0
    assert cpu.command_delay_steps(metadata, 'runtime') == 3
    assert cpu.command_delay_steps(metadata, 'direct') == 2
    metadata['simulation']['physicsTimestepSeconds'] = .003
    with pytest.raises(ValueError, match='integral physics'):
        cpu.apply_simulation_contract(model, metadata)


def test_velocity_observation_lag_is_one_policy_frame_and_resets_to_zero():
    delayed = cpu.VelocityObservationDelay(1)
    current = np.arange(14, dtype=float)
    np.testing.assert_array_equal(delayed.update(current), np.zeros(14))
    current[:] = 50
    np.testing.assert_array_equal(delayed.update(current), np.arange(14, dtype=float))
    np.testing.assert_array_equal(cpu.VelocityObservationDelay(0).update(current), current)
    with pytest.raises(ValueError, match='Negative observation'):
        cpu.VelocityObservationDelay(-1)


def test_cpu_contact_selection_matches_actual_v2_unnamed_leg_masks():
    source = cpu.ROOT/'src/mjlab_microduck/robot'
    names = list(json.loads((source/'xgoduck_joint_limits.json').read_text())['joints'])
    metadata = {'jointOrder': names, 'bodyGroundContactMeshes': 68,
                'actuatorParamsSha256': hashlib.sha256((source/'hd1910/1910_m6.json').read_bytes()).hexdigest(),
                'robotSourceHashes': {},
                'simulation': {'physicsTimestepSeconds': .005, 'controlTimestepSeconds': .02,
                               'integrator': 'implicitfast', 'solverIterations': 30,
                               'lineSearchIterations': 50, 'disableflags': [], 'enableflags': []}}
    model, _, _ = cpu.build_model(metadata)
    active = []
    for i in range(model.ngeom):
        if model.geom_contype[i] or model.geom_conaffinity[i]:
            active.append(model.geom(i).name)
            assert model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE or model.geom(i).name in (
                'left_foot_collision', 'right_foot_collision') or model.geom(i).name.startswith('body_ground_') or (
                    model.geom(i).name == '' and model.mesh(model.geom_dataid[i]).name == 'leg'
                    and model.geom_contype[i] == 2 and model.geom_conaffinity[i] == 2)
    assert len(active) == 73  # 68 body meshes + 2 soles + 2 legacy legs + floor
    assert all(model.actuator_ctrllimited)


def test_imu_uses_body_axes_not_rotated_cad_principal_inertia_axes():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="base">
      <freejoint/><inertial pos="0 0 0" quat="0.707106781 0 0 0.707106781"
      mass="1" diaginertia="1 2 3"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[3:6] = [1., 2., 3.]
    mujoco.mj_forward(model, data)
    body = model.body('base').id
    principal = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, principal, 1)
    np.testing.assert_allclose(principal[:3], [2., -1., 3.], atol=1e-8)
    gyro, gravity = cpu.root_body_imu_observation(model, data, body)
    np.testing.assert_array_equal(gyro, [1., 2., 3.])
    np.testing.assert_allclose(gravity, [0., 0., -1.], atol=1e-8)
    # Kinematics must reflect the newly integrated pose, not the last solve.
    data.qpos[3:7] = [np.cos(np.pi/4), np.sin(np.pi/4), 0., 0.]
    _, gravity = cpu.root_body_imu_observation(model, data, body)
    np.testing.assert_allclose(gravity, [0., -1., 0.], atol=1e-8)


def test_observation_filter_response_and_invalid_weights():
    np.testing.assert_allclose(cpu.observation_filter(np.zeros(3), np.ones(3), .25), [.75]*3)
    np.testing.assert_array_equal(cpu.observation_filter(np.ones(3), np.zeros(3), 0), np.zeros(3))
    for weight in (-.1, 1., float('nan')):
        with pytest.raises(ValueError, match='filter old weight'):
            cpu.observation_filter(np.zeros(3), np.ones(3), weight)


def test_smoothness_distinguishes_same_endpoint_with_repeated_reversals():
    def frames(values):
        return [{'time':i*.02,'sit':True,'target':np.radians([v]).tolist()}
                for i,v in enumerate(values)]
    direct=cpu.transition_smoothness(frames([0,10,20,30]), ['left_knee'])[0]
    shaking=cpu.transition_smoothness(frames([0,30,10,30]), ['left_knee'])[0]
    assert direct['netTravelDeg'] == shaking['netTravelDeg']
    np.testing.assert_allclose(direct['excessTravelDeg'], [0])
    np.testing.assert_allclose(shaking['excessTravelDeg'], [40])


def test_smoothness_never_counts_a_new_command_inside_old_transition():
    frames=[{'time':i*.02,'sit':i<3,'target':np.radians([i*10.]).tolist()} for i in range(6)]
    result=cpu.transition_smoothness(frames,['left_knee'])
    assert [r['samples'] for r in result] == [3,3]
    assert [r['sit'] for r in result] == [True,False]
    np.testing.assert_allclose([r['targetTravelDeg'] for r in result], [[20],[20]])

"""Explicit donor nominal physics for CPU comparisons, never a training default.

Source: kissqy/Microduck 7a52d297a9adfe6d2315481953dc64060252d142,
microduck-hd1910/firmware/policies/walk/deployment-contract.json.
Keeps local STL meshes; validates their compiled collision geometry against donor.
Terrain curriculum and domain randomization are not reproduced by this profile.
"""
import json
from pathlib import Path
import mujoco
import numpy as np

DEFAULT_CONTRACT = Path(__file__).with_name('contracts') / 'kissqy-walk-7a52d29.json'


def apply(spec, path=DEFAULT_CONTRACT):
    contract = json.loads(Path(path).read_text())
    model = contract['model']
    counts = {}
    for geom in list(spec.geoms):
        if geom.meshname != 'xl330':
            continue
        body = geom.parent.name
        index = counts.get(body, 0)
        counts[body] = index + 1
        name = f'{body}_{index}_servo_collision'
        if geom.contype == 0 and geom.conaffinity == 0:
            geom.parent.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                                 meshname=geom.meshname, pos=geom.pos, quat=geom.quat,
                                 mass=0, group=3)
        else:
            geom.name = name
    for row in model['bodies']:
        body = spec.body(row['name'])
        body.mass = row['mass_kg']
        body.fullinertia = [np.nan, 0, 0, 0, 0, 0]
        for key in ('pos', 'quat', 'ipos', 'iquat', 'inertia'):
            setattr(body, key, row[key])
        body.explicitinertial = True
    for row in model['joints']:
        joint = spec.joint(row['name'])
        joint.pos = row['pos']
        joint.axis = row['axis']
        joint.range = row['range']
        joint.limited = row['limited']
    for row in model['entity_compiled']['joints']:
        joint = spec.joint(row['joint'])
        joint.damping = np.full((3, 1), row['damping'])
        joint.armature = row['armature']
        joint.frictionloss = row['frictionloss']
    collisions = {g['name'] for g in model['collision_geoms']}
    settings = {g['name']: g for g in model['entity_compiled']['geoms'] if g['name'] in collisions}
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if geom.name in settings:
            for key in ('friction', 'condim', 'contype', 'conaffinity', 'solref', 'solimp'):
                setattr(geom, key, settings[geom.name][key])
            geom.priority = int(geom.name.endswith('foot_collision'))
        else:
            geom.contype = geom.conaffinity = 0
    return contract


def configure_and_validate(model, contract):
    opt = contract['training_physics']['simulation']['mujoco']
    for key in ('timestep', 'impratio', 'iterations', 'tolerance', 'ls_iterations',
                'ls_tolerance', 'ccd_iterations', 'gravity'):
        setattr(model.opt, key, opt[key])
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.disableflags = model.opt.enableflags = 0
    donor = contract['model']
    np.testing.assert_allclose(model.body_mass.sum(), donor['total_mass_kg'], atol=1e-10)
    assert model.nu == 14 and model.njnt == 29
    for row in donor['bodies']:
        body = model.body(row['name'])
        np.testing.assert_allclose(body.mass, row['mass_kg'], atol=1e-12)
        np.testing.assert_allclose(body.inertia, row['inertia'], atol=1e-12)
    for name, bounds in donor['gear_backlash']['ranges_rad'].items():
        np.testing.assert_allclose(model.joint(name).range, bounds, atol=1e-12)
    for row in donor['entity_compiled']['joints']:
        joint = model.joint(row['joint'])
        address = int(joint.dofadr[0])
        for key in ('damping', 'armature', 'frictionloss'):
            np.testing.assert_allclose(getattr(model, 'dof_' + key)[address], row[key], atol=1e-12)
    for row in donor['entity_compiled']['geoms']:
        if not row['name']:
            continue
        geom = model.geom(row['name'])
        for key in ('friction', 'condim', 'contype', 'conaffinity', 'solref', 'solimp'):
            np.testing.assert_allclose(getattr(geom, key), row[key], atol=1e-12)
    # Compiled mesh transforms include principal-frame corrections: compare,
    # never copy them back into MjSpec and apply the correction twice.
    for row in donor['collision_geoms']:
        geom = model.geom(row['name'])
        for key in ('pos', 'size'):
            np.testing.assert_allclose(getattr(geom, key), row[key], atol=2e-7, err_msg=row['name'])
        assert abs(np.dot(geom.quat, row['quat'])) > 1 - 1e-7, row['name']
    return {'massKg': float(model.body_mass.sum()), 'collisionGeoms': len(donor['collision_geoms']),
            'physicsDt': model.opt.timestep, 'policyDecimation': 4,
            'compiledCollisionGeometryVerified': True}

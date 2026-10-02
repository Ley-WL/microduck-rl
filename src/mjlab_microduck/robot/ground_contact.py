"""Add floor-only contact to replica geometry without robot self-collision pairs."""
import mujoco
import numpy as np

BODY_GROUND_BIT = 4


def add_body_ground_contacts(spec):
    floor = next((g for g in spec.geoms if g.name == "floor"), None)
    if floor is not None:
        floor.conaffinity |= BODY_GROUND_BIT
    count = 0
    for geom in spec.geoms:
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        # Soles already have ground collision proxies. Do not duplicate contact.
        if geom.meshname in ("sole_left", "sole_right"):
            continue
        if geom.contype == 0 and geom.conaffinity == 0:
            geom.name = f"body_ground_{count}"
            geom.contype = BODY_GROUND_BIT
            geom.conaffinity = 0
            geom.solref = np.array([.004, 1.])
            count += 1
    return count


def lowest_mesh_z(model, data):
    bottom = float("inf")
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        z_axis = data.geom_xmat[g].reshape(3, 3)[2]
        z = data.geom_xpos[g, 2] + (model.mesh_vert[start:start + count] @ z_axis).min()
        bottom = min(bottom, float(z))
    return bottom

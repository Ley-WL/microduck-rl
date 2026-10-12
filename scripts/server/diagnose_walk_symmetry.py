"""Verify replica leg reflection before using the upstream mirror loss."""
import json
import re
from pathlib import Path
import numpy as np
import mujoco
from mjlab_microduck.tasks.microduck_hd1910_stand_env_cfg import get_replica_spec
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks.symmetry import _JOINT_PERM, _JOINT_SIGN


def verify_replica_mirror():
    model=get_replica_spec().compile()
    data=mujoco.MjData(model)
    ids=[i for i in range(model.njnt) if model.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE]
    assert len(ids)==14
    adr=model.jnt_qposadr[ids]
    home=np.zeros(14)
    for k,j in enumerate(ids):
        name=model.joint(j).name
        home[k]=next(value for pattern,value in HOME_FRAME.joint_pos.items() if re.fullmatch(pattern,name))
    reflect=np.array([1.,-1.,1.])
    rng=np.random.default_rng(42)
    errors=[]
    # Sites are attached to the same mirrored sole locations in the replica.
    for _ in range(32):
        delta=rng.uniform(-.15,.15,14)
        data.qpos[adr]=home+delta
        mujoco.mj_forward(model,data)
        feet=data.site_xpos[[model.site('left_foot').id,model.site('right_foot').id]].copy()
        data.qpos[adr]=home+delta[_JOINT_PERM]*_JOINT_SIGN
        mujoco.mj_forward(model,data)
        mirrored=data.site_xpos[[model.site('right_foot').id,model.site('left_foot').id]].copy()
        errors.append(np.max(np.abs(mirrored-feet*reflect)))
    return {'samples':32,'maxFootReflectionErrorM':float(max(errors)),
            'mirrorVerified':bool(max(errors)<1e-5)}


if __name__=='__main__':
    result=verify_replica_mirror()
    print(json.dumps(result),flush=True)
    (Path(__file__).resolve().parents[3]/'setup/hd1910-walk-mirror-physics.json').write_text(json.dumps(result,indent=2))
    assert result['mirrorVerified'],result

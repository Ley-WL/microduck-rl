"""BAM CPU controller with MuJoCo DOF-friction constraint indexing corrected.

Upstream MujocoController matches efc_id against joint IDs. Friction DOF IDs
are velocity-coordinate IDs, which differ on robots with a floating base.
Keep upstream control/voltage/target handling and replace only its resulting
friction budget before the next physics integration. No extra control update.
"""
import numpy as np
import mujoco
from bam.mujoco import MujocoController as UpstreamMujocoController

FRICTION_CONSTRAINT_REVISION='dof-id-friction-force-v1'

def dof_friction_force(data,dof_ids):
    ids=np.asarray(dof_ids,dtype=int)
    rows=data.efc_type==int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
    return (data.efc_force[rows][None,:]*(data.efc_id[rows][None,:]==ids[:,None])).sum(axis=1)

class MujocoController(UpstreamMujocoController):
    def update(self):
        super().update()
        data=self.mujoco_data;ids=self.dof_indexes
        external=-data.qfrc_bias[ids]+data.qfrc_constraint[ids]-dof_friction_force(data,ids)
        friction,damping=self.model.compute_frictions(data.qfrc_actuator[ids],external,data.qvel[ids])
        self.mujoco_model.dof_frictionloss[ids]=friction
        self.mujoco_model.dof_damping[ids]=damping

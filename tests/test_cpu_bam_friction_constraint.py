"""Exercise real MuJoCo constraints with joint IDs different from DOF IDs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import mujoco
import numpy as np

spec=importlib.util.spec_from_file_location('cpu_bam_fix',Path(__file__).parents[1]/'scripts/bam_cpu_controller.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def model():
    xml='''<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".05" mass="1"/>
    <body pos="0 0 .1"><joint name="hinge" frictionloss=".2"/><geom type="capsule" size=".02 .1" mass=".1"/></body>
    </body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    d.qvel[6]=1.;mujoco.mj_forward(m,d)
    return m,d

def test_floating_base_friction_force_uses_dof_not_joint_id():
    m,d=model();j=m.joint('hinge').id;v=int(m.jnt_dofadr[j])
    assert j==1 and v==6
    force=module.dof_friction_force(d,[v])[0]
    assert abs(force)>1e-6
    np.testing.assert_allclose(force,d.qfrc_constraint[v],atol=1e-10)
    assert module.dof_friction_force(d,[j])[0]==0.

def test_budget_replaces_only_friction_after_single_upstream_control_update():
    m,d=model();v=int(m.jnt_dofadr[m.joint('hinge').id]);calls=[]
    class Budget:
        def compute_frictions(self,motor,external,velocity):
            calls.append((motor.copy(),external.copy(),velocity.copy()))
            return np.array([.123]),.456
    controller=object.__new__(module.MujocoController)
    controller.mujoco_model=m;controller.mujoco_data=d;controller.dof_indexes=np.array([v]);controller.model=Budget()
    ctrl=d.ctrl.copy()
    with patch.object(module.UpstreamMujocoController,'update') as upstream:
        controller.update();upstream.assert_called_once_with()
    expected=-d.qfrc_bias[v]+d.qfrc_constraint[v]-module.dof_friction_force(d,[v])[0]
    np.testing.assert_allclose(calls[0][1],[expected])
    assert m.dof_frictionloss[v]==.123 and m.dof_damping[v]==.456
    np.testing.assert_array_equal(d.ctrl,ctrl)

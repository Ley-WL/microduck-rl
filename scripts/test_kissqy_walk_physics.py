"""Run with .venv-preview/Scripts/python.exe scripts/test_kissqy_walk_physics.py."""
import unittest
from pathlib import Path
import mujoco
import kissqy_walk_physics as physics


class PhysicsContractTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'src/mjlab_microduck/robot/replica/scene_walk_backlash.xml'
        self.spec = mujoco.MjSpec.from_file(str(path))
        self.contract = physics.apply(self.spec)

    def test_compiled_contract_and_contact_selection(self):
        model = self.spec.compile()
        audit = physics.configure_and_validate(model, self.contract)
        self.assertTrue(audit['compiledCollisionGeometryVerified'])
        active = [i for i in range(model.ngeom) if model.geom_bodyid[i] and model.geom_contype[i]]
        self.assertEqual(len(active), 17)

    def test_geometry_drift_rejected(self):
        self.spec.geom('left_foot_collision').pos[0] += .001
        with self.assertRaises(AssertionError):
            physics.configure_and_validate(self.spec.compile(), self.contract)

    def test_backlash_drift_rejected(self):
        self.spec.joint('passive_left_hip_yaw_backlash').range = [-.02, .02]
        with self.assertRaises(AssertionError):
            physics.configure_and_validate(self.spec.compile(), self.contract)


if __name__ == '__main__':
    unittest.main()

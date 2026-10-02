"""CPU ground-contact regression; run with the preview Python environment."""
from pathlib import Path
import sys
import unittest
import mujoco
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root / "scripts"))
from preview_ground import add_body_ground_contacts, lowest_mesh_z


class GroundContactTest(unittest.TestCase):
    def test_only_floor_contact_pairs_added_and_mass_unchanged(self):
        spec = mujoco.MjSpec.from_file(str(root / "src/mjlab_microduck/robot/replica/scene_walk.xml"))
        original = spec.compile()
        self.assertGreater(add_body_ground_contacts(spec),50)
        updated = spec.compile()
        self.assertTrue(np.array_equal(original.body_mass,updated.body_mass))
        self.assertTrue(np.array_equal(original.body_inertia,updated.body_inertia))
        self.assertEqual(original.nu,updated.nu)
        floor = updated.geom("floor").id
        for i in range(original.ngeom):
            for j in range(i):
                if floor in (i,j):
                    continue
                def allowed(m):
                    return bool((m.geom_contype[i] & m.geom_conaffinity[j]) or (m.geom_contype[j] & m.geom_conaffinity[i]))
                self.assertEqual(allowed(original),allowed(updated))

    def test_fallen_body_supported_by_floor(self):
        spec = mujoco.MjSpec.from_file(str(root / "src/mjlab_microduck/robot/replica/scene_walk.xml"))
        add_body_ground_contacts(spec)
        for actuator in spec.actuators:
            actuator.set_to_motor()
        model = spec.compile()
        model.opt.timestep=.002
        for axis,angle in (([1.,0,0],1.57),([0,1.,0],1.57),([0,1.,0],3.14)):
            with self.subTest(axis=axis,angle=angle):
                data=mujoco.MjData(model)
                data.qpos[2]=.3
                mujoco.mju_axisAngle2Quat(data.qpos[3:7],np.array(axis),angle)
                for _ in range(2000):
                    mujoco.mj_step(model,data)
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertGreater(data.qpos[2],0)
                self.assertGreater(lowest_mesh_z(model,data),-.003)


if __name__ == "__main__":
    unittest.main()

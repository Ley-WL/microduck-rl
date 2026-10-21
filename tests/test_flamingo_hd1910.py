import unittest
from mjlab_microduck.tasks.flamingo_hd1910 import make_cfg
from mjlab_microduck.tasks.getup_mechanical_contract import LIMITS
from mjlab_microduck.tasks.microduck_flamingo_cycle_env_cfg import mirror_pose,FLAMINGO_POSE_RIGHT
import numpy as np

class FlamingoContract(unittest.TestCase):
    def test_hardware_and_semantics(self):
        cfg=make_cfg()
        self.assertEqual(cfg.actions['joint_pos'].scale,1.)
        self.assertEqual(cfg.actions['joint_pos'].clip,LIMITS)
        self.assertEqual(cfg.commands['twist'].ramp_s,1.5)
        self.assertAlmostEqual(cfg.sim.mujoco.timestep*cfg.decimation,.02)
        self.assertEqual(type(cfg.actions['joint_pos']).__name__,'JointPositionActionCfg')
        model=cfg.scene.entities['robot'].spec_fn().compile()
        for name,bounds in LIMITS.items():np.testing.assert_allclose(model.joint(name).range,bounds,atol=1e-12)
        for actuator in cfg.scene.entities['robot'].articulation.actuators:
            self.assertEqual(actuator.kp_fw,5.)
            self.assertTrue(str(actuator.json_path).endswith('hd1910/1910_m6.json'))
        self.assertEqual(cfg.rewards['com_target'].weight,3.)
        self.assertEqual(cfg.rewards['swing_foot_clear'].weight,1.5)
    def test_mirror_round_trip(self):
        np.testing.assert_allclose(mirror_pose(mirror_pose(FLAMINGO_POSE_RIGHT)),FLAMINGO_POSE_RIGHT)

if __name__=='__main__':unittest.main()

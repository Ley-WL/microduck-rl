"""Contract checks run with the existing server training environment."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab_microduck.tasks.getup_mechanical_contract import LIMITS, NAMES
from mjlab_microduck.tasks.walk_observer_contract import apply_observer_contract


class ObserverWalkContractTests(unittest.TestCase):
    def test_only_hardware_fields_change(self):
        cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
        names = ("rewards", "curriculum", "commands", "observations", "events", "terminations")
        before = {name: repr(getattr(cfg, name)) for name in names}
        apply_observer_contract(cfg)
        self.assertEqual(before, {name: repr(getattr(cfg, name)) for name in names})
        self.assertEqual(cfg.actions["joint_pos"].clip, LIMITS)
        self.assertEqual(cfg.actions["joint_pos"].scale, 1.)
        model = cfg.scene.entities["robot"].spec_fn().compile()
        for name, bounds in LIMITS.items():
            np.testing.assert_allclose(model.joint(name).range, bounds, atol=1e-12)
        for actuator in cfg.scene.entities["robot"].articulation.actuators:
            self.assertEqual(actuator.kp_fw, 5.)
            self.assertTrue(str(actuator.json_path).endswith("hd1910/1910_m6.json"))

    def test_saturation_after_offset_preserves_raw_history(self):
        limits = torch.tensor([LIMITS[name] for name in NAMES])[None]
        obj = SimpleNamespace(
            cfg=SimpleNamespace(clip=LIMITS), _raw_actions=torch.zeros(2, 14),
            _scale=1., _offset=torch.full((2, 14), .1), _clip=limits,
        )
        raw = torch.stack((torch.full((14,), 10.), torch.full((14,), -10.)))
        JointPositionAction.process_actions(obj, raw)
        torch.testing.assert_close(obj._raw_actions, raw)
        torch.testing.assert_close(obj._processed_actions[0], limits[0, :, 1])
        torch.testing.assert_close(obj._processed_actions[1], limits[0, :, 0])
        JointPositionAction.process_actions(obj, torch.zeros_like(raw))
        torch.testing.assert_close(obj._processed_actions, obj._offset)


if __name__ == "__main__":
    unittest.main()

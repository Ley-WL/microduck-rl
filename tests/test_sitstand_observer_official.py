"""Keep the public training recipe intact while adapting physical limits."""
import unittest
from dataclasses import asdict
from mjlab_microduck.tasks.sitstand_observer_official import make_cfg, rl_cfg, spec
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg, MicroduckSitStandRlCfg
from mjlab_microduck.tasks.getup_mechanical_contract import LIMITS


class Recipe(unittest.TestCase):
    def test_rewards_curriculum_commands(self):
        actual, original = make_cfg(), make_microduck_sitstand_env_cfg()
        for name in ('rewards', 'curriculum', 'commands', 'terminations'):
            self.assertEqual(repr(getattr(actual, name)), repr(getattr(original, name)))
        self.assertEqual(actual.episode_length_s, original.episode_length_s)
        self.assertEqual(actual.actions['joint_pos'].clip, LIMITS)

    def test_ppo(self):
        actual, original = asdict(rl_cfg()), asdict(MicroduckSitStandRlCfg)
        for name in ('actor', 'critic', 'algorithm', 'num_steps_per_env', 'max_iterations', 'save_interval'):
            self.assertEqual(actual[name], original[name])
        self.assertEqual(actual['max_iterations'], 15000)

    def test_physical_limits(self):
        robot = spec()
        for joint in robot.joints:
            if joint.name in LIMITS:
                self.assertEqual(tuple(joint.range), LIMITS[joint.name])


if __name__ == '__main__':
    unittest.main()

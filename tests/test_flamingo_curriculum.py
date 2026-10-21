"""Protect published curriculum and PPO settings against transfer overrides."""
import unittest
from dataclasses import asdict

from mjlab_microduck.tasks.flamingo_hd1910_curriculum import make_cfg, rl_cfg
from mjlab_microduck.tasks.microduck_flamingo_cycle_env_cfg import (
    make_microduck_flamingo_cycle_env_cfg, MicroduckFlamingoCycleRlCfg,
)
from mjlab_microduck.tasks.getup_mechanical_contract import LIMITS


class PublishedRecipe(unittest.TestCase):
    def test_curriculum_and_rewards_preserved(self):
        expected = make_microduck_flamingo_cycle_env_cfg(hard=True)
        actual = make_cfg()
        self.assertEqual(actual.curriculum.keys(), expected.curriculum.keys())
        for key in expected.curriculum:
            self.assertEqual(actual.curriculum[key].params, expected.curriculum[key].params)
        for key in expected.rewards:
            self.assertEqual(actual.rewards[key].weight, expected.rewards[key].weight)
            self.assertEqual(actual.rewards[key].params, expected.rewards[key].params)
        stages = actual.curriculum['push_magnitude'].params['push_stages']
        self.assertEqual([s['step']//24 for s in stages], [0, 500, 1000, 1400])
        self.assertEqual(actual.actions['joint_pos'].clip, LIMITS)

    def test_ppo_and_exploration_preserved(self):
        actual = asdict(rl_cfg())
        expected = asdict(MicroduckFlamingoCycleRlCfg)
        for key in ['actor', 'critic', 'algorithm', 'num_steps_per_env', 'max_iterations']:
            self.assertEqual(actual[key], expected[key])
        self.assertEqual(actual['algorithm']['learning_rate'], 1e-3)
        self.assertEqual(actual['algorithm']['schedule'], 'adaptive')
        self.assertEqual(actual['actor']['distribution_cfg']['init_std'], 1.)


if __name__ == '__main__':
    unittest.main()

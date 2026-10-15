"""Check continuous coverage and isolation from PD/reward/observation changes."""
from unittest.mock import patch

import torch

from mjlab_microduck.tasks import walk_continuous_coverage as new
from mjlab_microduck.tasks import walk_motion_critic_context as old

a, b = old.make_cfg(), new.make_cfg()
for field in ('sim', 'decimation', 'actions', 'observations', 'events', 'terminations', 'rewards'):
    assert getattr(a, field) == getattr(b, field), field
assert a.scene.entities['robot'].articulation == b.scene.entities['robot'].articulation
assert vars(a.commands['twist']) == vars(b.commands['twist'])
torch.manual_seed(251011)
speed, yaw = new.sample_commands(100000, 'cpu')
rest = (speed == 0) & (yaw == 0)
turn = (speed == 0) & (yaw != 0)
low = (speed.abs() >= .025) & (speed.abs() < .12)
high = speed.abs() >= .12
for mask, expected in [(rest, .1), (turn, .1), (low, .6), (high, .2)]:
    assert abs(float(mask.float().mean()) - expected) < .01
assert speed.abs().max() <= .4 and yaw.abs().max() <= .6
assert yaw[turn].abs().min() >= .15
assert abs(float((speed[low] > 0).float().mean()) - .5) < .01
assert abs(float((yaw[low] == 0).float().mean()) - .5) < .01
for value in (.035, .05, .075, .1):
    assert ((speed - value).abs() < .002).sum() > 900
class ResetProbe(new.ContinuousCommand):
    def __init__(self):
        self.stop_time = torch.zeros(8)
        self.stop_speed = torch.ones(8)
        self.stop_yaw = torch.ones(8)

    @property
    def device(self):
        return 'cpu'


state = ResetProbe()
with patch.object(new.SustainedCommand, 'reset', return_value={}):
    new.ContinuousCommand.reset(state, torch.tensor([1, 3]))
assert torch.equal(state.stop_speed[[0, 2, 4, 5, 6, 7]], torch.ones(6))
assert torch.equal(state.stop_yaw[[0, 2, 4, 5, 6, 7]], torch.ones(6))
print('Continuous buckets, signs, straight/pure-yaw/rest, partial reset and physical/reward/actor parity passed')

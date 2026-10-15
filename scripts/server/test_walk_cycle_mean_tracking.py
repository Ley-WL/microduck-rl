"""Gait-period scores, legacy replay, stop recovery and partial-reset checks."""
import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import walk_mean_motion_tracking as old
from mjlab_microduck.tasks import walk_cycle_mean_tracking as new
from mjlab_microduck.tasks.walk_sustained_transfer import SustainedBias

a, b = old.make_cfg(), new.make_cfg()
for field in ('sim', 'decimation', 'actions', 'observations', 'commands', 'events', 'terminations'):
    assert getattr(a, field) == getattr(b, field), field
assert a.scene.entities['robot'].articulation == b.scene.entities['robot'].articulation
changed = {'track_linear_velocity', 'track_angular_velocity', 'tracking_bias'}
for name in a.rewards:
    if name not in changed:
        assert a.rewards[name] == b.rewards[name], name
    else:
        assert a.rewards[name].weight == b.rewards[name].weight
        assert a.rewards[name].params == b.rewards[name].params
command = torch.tensor([[.025, 0., 0.], [.025, 0., 0.], [.2, 0., .6], [0., 0., 0.], [0., 0., .6]])
data = SimpleNamespace(root_link_lin_vel_b=torch.zeros(5, 3), root_link_ang_vel_b=torch.zeros(5, 3))
env = SimpleNamespace(num_envs=5, device='cpu', step_dt=.02,
    scene={'robot': SimpleNamespace(data=data)},
    command_manager=SimpleNamespace(get_command=lambda _: command))
torch.testing.assert_close(new.tracking_tau(command).flatten(), torch.tensor([4., 4., .7, .7, .7]))
linear = new.CycleTracking(b.rewards['track_linear_velocity'], env)
angular = new.CycleTracking(b.rewards['track_angular_velocity'], env)
bias = new.CycleBias(b.rewards['tracking_bias'], env)
reference = [old.MeanMotionTracking(a.rewards['track_linear_velocity'], env),
             old.MeanMotionTracking(a.rewards['track_angular_velocity'], env),
             SustainedBias(a.rewards['tracking_bias'], env)]
scores, legacy = [], []
for tick in range(3000):
    data.root_link_lin_vel_b.zero_(); data.root_link_ang_vel_b.zero_()
    if tick < 2000:
        data.root_link_lin_vel_b[0, 0] = .025 + .15 * math.sin(2 * math.pi * tick * .02 / 4.)
        data.root_link_lin_vel_b[2, 0] = .2
        data.root_link_ang_vel_b[2, 2] = .6
        data.root_link_ang_vel_b[4, 2] = .6
    else:
        command.zero_()
    values = []
    for index, (name, term) in enumerate(zip(
            ('track_linear_velocity', 'track_angular_velocity', 'tracking_bias'), (linear, angular, bias))):
        params = b.rewards[name].params
        value = term(env, **params) if index < 2 else term(env)
        original = reference[index](env, **params) if index < 2 else reference[index](env)
        assert torch.equal(term.legacy_reward, original)
        values.append(value * b.rewards[name].weight)
        if tick < 2000:
            torch.testing.assert_close(value[2:], original[2:], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(bias.error[:, :2], linear.error[:, :2], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(bias.error[:, 2], angular.error[:, 2], atol=1e-6, rtol=1e-6)
    if 1000 <= tick < 2000:
        scores.append(sum(values))
        legacy.append(sum(term.legacy_reward * b.rewards[name].weight for name, term in zip(
            ('track_linear_velocity', 'track_angular_velocity', 'tracking_bias'), (linear, angular, bias))))
assert torch.stack(legacy).mean(0)[0] < torch.stack(legacy).mean(0)[1]
assert torch.stack(scores).mean(0)[0] > torch.stack(scores).mean(0)[1]
assert bias.error.abs().max() < 1e-6, 'Stop memory must decay; no stale movement cost'
before = linear.error.clone(); linear.reset(torch.tensor([1]))
assert not linear.error[1].any() and torch.equal(linear.error[[0, 2, 3, 4]], before[[0, 2, 3, 4]])
print('Period-averaged low-speed preference; exact legacy replay; faster/rest/pure-yaw parity; stop decay and critic-memory consistency passed')

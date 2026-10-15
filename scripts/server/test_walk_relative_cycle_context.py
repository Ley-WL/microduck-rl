"""Relative scaling must match the independently replayed cost and critic contract."""
from types import SimpleNamespace
import runpy
from pathlib import Path

import numpy as np
import torch

from mjlab_microduck.tasks import walk_continuous_coverage as old
from mjlab_microduck.tasks import walk_relative_cycle_context as new

runpy.run_path(str(Path(__file__).with_name('test_walk_cycle_mean_tracking.py')))
a, b = old.make_cfg(), new.make_cfg()
for field in ('sim', 'decimation', 'actions', 'observations', 'commands', 'events', 'terminations'):
    assert getattr(a, field) == getattr(b, field), field
assert a.scene.entities['robot'].articulation == b.scene.entities['robot'].articulation
for name in a.rewards:
    if name not in ('track_linear_velocity', 'track_angular_velocity', 'tracking_bias'):
        assert a.rewards[name] == b.rewards[name], name
    else:
        assert a.rewards[name].weight == b.rewards[name].weight
        assert a.rewards[name].params == b.rewards[name].params
command = torch.tensor([[.025,0.,0.],[-.035,0.,-.6],[.05,0.,.6],[.075,0.,0.],[.2,0.,0.],[0.,0.,0.],[0.,0.,.6]])
torch.testing.assert_close(new.longitudinal_scale(command), torch.tensor([.025,.035,.05,.075,.1,.1,.1]))
data = SimpleNamespace(root_link_lin_vel_b=torch.zeros(7,3),root_link_ang_vel_b=torch.zeros(7,3))
env = SimpleNamespace(num_envs=7,device='cpu',step_dt=.02,scene={'robot':SimpleNamespace(data=data)},
                      command_manager=SimpleNamespace(get_command=lambda _:command))
term = new.RelativeCycleBias(b.rewards['tracking_bias'], env)
for tick in range(300):
    value = term(env)
    scale = torch.tensor([.1,.1,.4]).repeat(7,1);scale[:,0]=new.longitudinal_scale(command)
    expected=(term.error/scale).square().sum(-1).clamp(max=25.)
    assert torch.equal(value,expected)
    assert torch.equal(value[4:],term.cycle_reward[4:])
    assert (value>=0).all() and (value<=25).all()
saved=term.error.clone();term.reset(torch.tensor([1,3]));assert not term.error[[1,3]].any()
assert torch.equal(term.error[[0,2,4,5,6]],saved[[0,2,4,5,6]])
print('Relative cost, sign/cap, zero/pure-yaw/faster scale parity, critic contract and partial reset passed')

# Validate against real frozen reward-time sensor inputs, independently of the
# torch implementation. No physics rollout or actor update is performed here.
replay_path = Path(__file__).resolve().parents[2] / 'inputs/reward-replay.npz'
assert replay_path.exists()
with np.load(replay_path) as archive:
    frames = archive['values'].astype(np.float64)
n = frames.shape[1]
command = torch.zeros(n, 3)
data = SimpleNamespace(root_link_lin_vel_b=torch.zeros(n, 3), root_link_ang_vel_b=torch.zeros(n, 3))
env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02, scene={'robot':SimpleNamespace(data=data)},
    command_manager=SimpleNamespace(get_command=lambda _: command))
term = new.RelativeCycleBias(b.rewards['tracking_bias'], env)
error = np.zeros((n, 3)); max_error = 0.
for frame in frames:
    data.root_link_lin_vel_b[:, :2] = torch.tensor(frame[:, :2], dtype=torch.float32)
    data.root_link_ang_vel_b[:, 2] = torch.tensor(frame[:, 2], dtype=torch.float32)
    command[:] = torch.tensor(frame[:, 3:], dtype=torch.float32)
    target = frame[:, 3:]; speed = np.linalg.norm(target[:, :2], axis=-1)
    tau = np.where(speed > .001, np.clip(.1 / np.maximum(speed, .025), .7, 4.), .7)
    error += (frame[:, :3] - target - error) * .02 / (tau[:, None] + .02)
    scale = np.tile([.1, .1, .4], (n, 1)); vx = abs(target[:, 0])
    scale[:, 0] = np.where(vx > .001, np.clip(vx, .025, .1), .1)
    expected = np.minimum(((error / scale) ** 2).sum(-1), 25.)
    actual = term(env).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)
    max_error = max(max_error, float(abs(actual-expected).max()))
print('Real reward-time trajectory independent relative-cost replay max error', max_error)

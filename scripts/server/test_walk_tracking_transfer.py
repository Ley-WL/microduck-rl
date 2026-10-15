"""Validate command coverage, stride preference and bounded transfer settings."""
import torch
from mjlab_microduck.tasks.walk_tracking_transfer import make_cfg, rl_cfg, TrackingCommand
from mjlab_microduck.tasks import mdp

cfg = make_cfg()
assert cfg.decimation * cfg.sim.mujoco.timestep == .02
assert cfg.scene.entities['robot'].articulation.actuators[0].kp_fw == 5.
assert cfg.rewards['long_stride'].params['require_alternation']
assert cfg.rewards['long_stride'].params['complete_swing']
assert rl_cfg().algorithm.learning_rate == 3e-6
assert rl_cfg().algorithm.schedule == 'fixed'
# Test inherited reset and command schedule without creating a GPU simulation.
class CpuCommand(TrackingCommand):
    @property
    def device(self): return 'cpu'
command = object.__new__(CpuCommand)
command.stop_time = torch.zeros(1200)
command.stop_speed = torch.zeros(1200)
command.stop_yaw = torch.zeros(1200)
command.vel_command_b = torch.zeros(1200, 3)
torch.manual_seed(191043)
command.reset()
assert len(torch.unique(torch.stack((command.stop_speed, command.stop_yaw), -1), dim=0)) == 15
assert bool(((command.stop_speed == 0) & (command.stop_yaw != 0)).any())
speed = torch.full((3,), .2)
quality = mdp.hd1910_stride_component_quality(torch.tensor([.01, .05, .11]),
    torch.zeros(3), torch.full((3,), .018), torch.full((3,), .18), speed,
    cycle_time=.55, target_air=.18, target_height=.018)
assert quality[0] < quality[1] < quality[2]
print('tracking/turn-only/stride/config gates passed')

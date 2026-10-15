"""Experimental low-speed relative tracking with continuous commands and critic memory."""
import torch

from . import walk_continuous_coverage as base
from .walk_cycle_mean_tracking import CycleBias, CycleTracking


def longitudinal_scale(command):
    vx = command[:, 0].abs()
    return torch.where(vx > .001, vx.clamp(.025, .1), .1)


class RelativeCycleBias(CycleBias):
    def __call__(self, env):
        # Update the cycle error once and preserve the old .7s reference score.
        self.cycle_reward = super().__call__(env)
        scale = torch.tensor([.1, .1, .4], device=env.device).repeat(env.num_envs, 1)
        scale[:, 0] = longitudinal_scale(env.command_manager.get_command('twist'))
        return (self.error / scale).square().sum(-1).clamp(max=25.)


def make_cfg(surface=False, play=False):
    cfg = base.make_cfg(surface, play)
    for name in ('track_linear_velocity', 'track_angular_velocity'):
        cfg.rewards[name].func = CycleTracking
    cfg.rewards['tracking_bias'].func = RelativeCycleBias
    return cfg


def rl_cfg():
    cfg = base.rl_cfg()
    cfg.experiment_name = 'walk_relative_cycle_context'
    return cfg

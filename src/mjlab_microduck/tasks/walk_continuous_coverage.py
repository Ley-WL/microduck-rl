"""Continuous speed/yaw coverage, retaining the existing physical/reward contract."""
from dataclasses import dataclass

import torch

from . import walk_motion_critic_context as base
from .walk_sustained_transfer import SustainedCommand, SustainedCommandCfg


def sample_commands(count, device):
    """60% low speed, 20% higher speed, 10% pure yaw, 10% exact rest."""
    bucket = torch.rand(count, device=device)
    sign = torch.where(torch.rand(count, device=device) < .5, -1., 1.)
    speed = sign * (.025 + .095 * torch.rand(count, device=device))
    high = sign * (.12 + .28 * torch.rand(count, device=device))
    speed = torch.where(bucket >= .6, high, speed)
    speed = torch.where(bucket >= .8, 0., speed)
    yaw = (torch.rand(count, device=device) * 2 - 1) * .6
    # Half of moving commands are exactly straight. Pure yaw must not collapse
    # into a near-zero command and rest is an explicit bucket.
    yaw = torch.where((bucket < .8) & (torch.rand(count, device=device) < .5), 0., yaw)
    turn = torch.where(yaw < 0, -1., 1.) * (.15 + .45 * torch.rand(count, device=device))
    yaw = torch.where((bucket >= .8) & (bucket < .9), turn, yaw)
    yaw = torch.where(bucket >= .9, 0., yaw)
    return speed, yaw


class ContinuousCommand(SustainedCommand):
    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        speed, yaw = sample_commands(self.stop_time[ids].numel(), self.device)
        self.stop_speed[ids] = speed
        self.stop_yaw[ids] = yaw
        return result


@dataclass(kw_only=True)
class ContinuousCommandCfg(SustainedCommandCfg):
    def build(self, env):
        return ContinuousCommand(self, env)


def make_cfg(surface=False, play=False):
    cfg = base.make_cfg(surface, play)
    cfg.commands['twist'] = ContinuousCommandCfg(**vars(cfg.commands['twist']))
    return cfg


def rl_cfg():
    cfg = base.rl_cfg()
    cfg.experiment_name = 'walk_continuous_coverage'
    return cfg

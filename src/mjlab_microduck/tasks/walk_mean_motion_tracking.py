"""Experimental mean-error tracking for all movement; zero-command rewards unchanged."""
import torch
from . import walk_low_speed_mean_tracking as base


class MeanMotionTracking(base.MeanTracking):
    def __call__(self,env,original,std,command_name,angular=False,tau=.7):
        previous=super().__call__(env,original,std,command_name,angular,tau)
        # Preserve the previous low-speed EMA / high-speed instantaneous score
        # on this exact trajectory for paired reward-order diagnostics.
        self.legacy_reward=previous
        command=env.command_manager.get_command(command_name)
        moving=command.norm(dim=-1)>.001
        averaged=torch.exp(-self.error.square().sum(-1)/std**2)
        return torch.where(moving,averaged,previous)


def make_cfg(surface=False,play=False):
    cfg=base.make_cfg(surface,play)
    for name in ('track_linear_velocity','track_angular_velocity'):
        cfg.rewards[name].func=MeanMotionTracking
    return cfg


def rl_cfg():
    cfg=base.rl_cfg();cfg.experiment_name='walk_mean_motion_tracking'
    return cfg

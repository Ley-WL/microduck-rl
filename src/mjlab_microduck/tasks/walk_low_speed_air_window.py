"""Experimental low-speed swing duration; preserve original non-low-speed reward."""
import torch
from . import walk_low_speed_focused_control as base


def air_time_limit(command,threshold_max=.3,min_stride=.06):
    speed=command[:,:2].norm(dim=-1)
    # For symmetric walking, foot travel is approximately 2*v*swing_duration.
    # This is a reward heuristic, not a kinematic guarantee or runtime control.
    duration=(min_stride/(2*speed.clamp(min=.03))).clamp(min=threshold_max,max=.8)
    return torch.where((speed>.03)&(speed<.1),duration,threshold_max)


def adaptive_air_time(env,original,threshold_max=.3,**params):
    command=env.command_manager.get_command(params['command_name'])
    limit=air_time_limit(command,threshold_max)
    # Preserve the existing command gate, logging and reward calculation.
    return original(env,threshold_max=limit[:,None],**params)


def make_cfg(surface=False,play=False):
    cfg=base.make_cfg(surface,play)
    term=cfg.rewards['air_time']
    term.params['original']=term.func
    term.func=adaptive_air_time
    return cfg


def rl_cfg():
    cfg=base.rl_cfg();cfg.experiment_name='walk_low_speed_air_window'
    return cfg

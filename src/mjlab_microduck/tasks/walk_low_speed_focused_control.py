"""Matched low-speed coverage control for the stronger-stride experiment."""
from . import walk_low_speed_focused as base

def make_cfg(surface=False,play=False):
    return base.make_cfg(surface,play,stronger_stride=False)

def rl_cfg():
    cfg=base.rl_cfg();cfg.experiment_name='walk_low_speed_focused_control'
    return cfg

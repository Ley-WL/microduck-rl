"""Matched command-coverage control for the minimum-stride experiment."""
from . import walk_low_speed_stride as base

def make_cfg(surface=False,play=False):
    return base.make_cfg(surface,play,larger_steps=False)

def rl_cfg():
    cfg=base.rl_cfg();cfg.experiment_name='walk_low_speed_control'
    return cfg

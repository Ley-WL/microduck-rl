"""Controlled second swing probe: transfer reward mass from speed to swing."""
from copy import deepcopy
from dataclasses import replace
from .microduck_hd1910_walk_swing_env_cfg import make_hd1910_walk_swing_env_cfg, HD1910SwingWalkRlCfg


def make_hd1910_walk_swing_focus_env_cfg(play=False):
    cfg=make_hd1910_walk_swing_env_cfg(play=play)
    cfg.rewards['air_time'].weight=8.
    cfg.rewards['track_linear_velocity'].weight=5.
    cfg.rewards['command_progress'].weight=1.
    return cfg


HD1910SwingFocusWalkRlCfg=replace(deepcopy(HD1910SwingWalkRlCfg),
    experiment_name='microduck_hd1910_walk_swing_focus',run_name='swing_focus_v16',max_iterations=20000)

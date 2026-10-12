"""Relearn the head/leg coordination from a measured feasible HOME head pose."""
from .microduck_hd1910_walk_env_cfg import make_hd1910_walk_env_cfg,HD1910WalkRlCfg


def make_hd1910_walk_headstart_env_cfg(play=False):
    cfg=make_hd1910_walk_env_cfg(play=play)
    # The four-joint average diluted a 70-degree neck bias to ~.18 reward/sec.
    # Retain the slow EMA (not instantaneous gait oscillation) and price its DC
    # bias at the summed-joint scale. Failure cost from v9 stays enabled.
    cfg.rewards['head_pose_bias'].weight=2.4
    return cfg

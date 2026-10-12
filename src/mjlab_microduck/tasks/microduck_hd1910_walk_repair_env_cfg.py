"""Preserve learned gait while teaching .1m/s speed, DC yaw and swing quality."""
from copy import deepcopy
from dataclasses import replace
from . import mdp
from .microduck_hd1910_walk_slow_env_cfg import make_hd1910_walk_slow_env_cfg, HD1910SlowWalkRlCfg


def make_hd1910_walk_repair_env_cfg(play=False):
    cfg=make_hd1910_walk_slow_env_cfg(play=play)
    for term in cfg.rewards.values():
        if term.func==mdp.hd1910_walk_supported_reward:
            term.params['straight_quality']=False
    # Price sustained yaw through the existing .5s EMA, not every gait sway.
    cfg.rewards['yaw_error_cost'].weight=-.3
    cfg.rewards['track_angular_velocity'].weight=1.
    cfg.rewards['track_linear_velocity'].params['base_params']['std']=.08
    cfg.rewards['stride_quality'].params['motion_quality']=False
    cfg.rewards['stride_quality'].params['base_params'].update(
        complete_swing=False,component_shaping=True,cycle_time=.4,
        target_air=.12,target_height=.012)
    cfg.rewards['stride_quality'].weight=3.
    cfg.curriculum['stride_quality_weight'].params['weight_stages']=[
        {'step':0,'weight':3.},{'step':300*24,'weight':5.},{'step':700*24,'weight':7.}]
    cfg.rewards['straight_path_cost'].weight=-.02
    cfg.curriculum['straight_path_weight'].params['weight_stages']=[
        {'step':0,'weight':-.02},{'step':300*24,'weight':-.05},{'step':700*24,'weight':-.1}]
    return cfg


HD1910RepairWalkRlCfg=replace(deepcopy(HD1910SlowWalkRlCfg),
    experiment_name='microduck_hd1910_walk_repair',run_name='repair_v14')
HD1910RepairWalkRlCfg.save_interval=100
HD1910RepairWalkRlCfg.algorithm.learning_rate=5e-5

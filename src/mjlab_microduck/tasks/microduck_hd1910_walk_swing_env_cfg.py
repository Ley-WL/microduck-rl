"""Teach swing height/time continuously rather than only at sparse touchdowns."""
from copy import deepcopy
from dataclasses import replace
from mjlab.managers import RewardTermCfg
from . import mdp
from .microduck_hd1910_walk_repair_env_cfg import make_hd1910_walk_repair_env_cfg, HD1910RepairWalkRlCfg


def make_hd1910_walk_swing_env_cfg(play=False):
    cfg=make_hd1910_walk_repair_env_cfg(play=play)
    # Existing air_time only paid after 125ms, while the seed lands after 65ms.
    # Replace this dead discovery signal with a bounded single-support signal.
    term=cfg.rewards['air_time']
    term.params['base_func']=mdp.hd1910_walk_swing_progress
    term.params['base_params']={'target_height':.01,'target_air':.1}
    term.weight=3.
    cfg.rewards['command_progress'].weight=2.
    cfg.rewards['stride_quality'].weight=3.
    cfg.curriculum['stride_quality_weight'].params['weight_stages']=[{'step':0,'weight':3.}]
    # Isolate swing learning first; keep path conditions equal to the seed stage.
    cfg.curriculum['straight_path_weight'].params['weight_stages']=[{'step':0,'weight':-.02}]
    cfg.rewards['foot_swing_height'].params['target_height']=.01
    cfg.rewards['foot_clearance'].params['target_height']=.01
    return cfg


HD1910SwingWalkRlCfg=replace(deepcopy(HD1910RepairWalkRlCfg),
    experiment_name='microduck_hd1910_walk_swing',run_name='swing_v15')

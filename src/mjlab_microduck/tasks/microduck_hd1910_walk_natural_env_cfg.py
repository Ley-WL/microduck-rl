"""Fresh gait discovery with matched runtime EMA and no low-head reward floor."""
from copy import deepcopy
from dataclasses import replace
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from .microduck_hd1910_stand_env_cfg import STAND_HEIGHT
from . import mdp
from .microduck_hd1910_walk_env_cfg import make_hd1910_walk_env_cfg, HD1910WalkRlCfg
from .xgoduck_runtime_actions import XgoDuckPositionActionCfg


def make_hd1910_walk_natural_env_cfg(play=False):
    cfg = make_hd1910_walk_env_cfg(play=play)
    cfg.actions['joint_pos'] = XgoDuckPositionActionCfg(
        **vars(cfg.actions['joint_pos']), alpha=.45)
    cfg.events['foot_friction'].params['ranges'] = (.4, 1.5)
    for name, magnitude in (('base_ang_vel', .06), ('projected_gravity', .03), ('joint_pos', .01)):
        noise = cfg.observations['actor'].terms[name].noise
        noise.n_min, noise.n_max = -magnitude, magnitude
    # Penalize persistent head bias, while allowing the head to oscillate in gait.
    # All positive movement terms must lose payout for sustained low-head motion.
    for term in cfg.rewards.values():
        if term.func == mdp.hd1910_walk_supported_reward:
            term.params.update(head_quality=True, head_std=.25, head_floor=0., motion_quality=True)
    cfg.rewards['track_linear_velocity'].weight = 10.
    cfg.rewards['air_time'].weight = 2.
    cfg.rewards['stride_quality'].params['base_params']['require_alternation'] = True
    cfg.rewards['command_progress'] = RewardTermCfg(
        func=mdp.hd1910_walk_supported_reward, weight=8., params={
        'base_func':mdp.hd1910_walk_command_progress, 'base_params':{},
        'nominal_height':STAND_HEIGHT, 'head_quality':True,
        'head_std':.25, 'head_floor':0.})
    cfg.rewards['head_pose_bias'].weight = 1.2
    # Discover gait before imposing the full stride target and regularization.
    cfg.curriculum['stride_quality_weight'].params['weight_stages'] = [
        {'step':0,'weight':1.}, {'step':500*24,'weight':3.},
        {'step':1500*24,'weight':5.}]
    cfg.rewards['stride_quality'].weight = 1.
    cfg.curriculum['natural_head_weight'] = CurriculumTermCfg(
        func=mdp.reward_weight, params={'reward_name':'head_pose_bias',
        'weight_stages':[{'step':0,'weight':.3}, {'step':500*24,'weight':.6},
                         {'step':1500*24,'weight':1.2}]})
    return cfg


HD1910NaturalWalkRlCfg = replace(deepcopy(HD1910WalkRlCfg),
    experiment_name='microduck_hd1910_walk_natural',
    run_name='natural_v11', max_iterations=1000)
HD1910NaturalWalkRlCfg.algorithm.learning_rate = 3e-4

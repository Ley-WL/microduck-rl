"""0.1 m/s forward specialization, with complete swings and straight paths."""
from copy import deepcopy
from dataclasses import replace
from mjlab.managers import RewardTermCfg, CurriculumTermCfg
from . import mdp
from .microduck_hd1910_walk_natural_env_cfg import make_hd1910_walk_natural_env_cfg, HD1910NaturalWalkRlCfg


def make_hd1910_walk_slow_env_cfg(play=False):
    cfg=make_hd1910_walk_natural_env_cfg(play=play)
    twist=cfg.commands['twist']
    twist.rel_straight_envs=.95
    twist.straight_forward_prob=1.
    twist.straight_speed_range=(.07,.13)
    twist.exact_speed_prob=.8
    twist.exact_speed=.1
    twist.ranges.lin_vel_x=(.07,.13)
    twist.ranges.lin_vel_y=(-.02,.02)
    twist.ranges.ang_vel_z=(-.15,.15)
    twist.rel_turn_in_place_envs=0.
    twist.resampling_time_range=(4.,8.)
    for term in cfg.rewards.values():
        if term.func==mdp.hd1910_walk_supported_reward:
            term.params['straight_quality']=True
    cfg.rewards['track_linear_velocity'].params['base_params']['std']=.04
    cfg.rewards['command_progress'].weight=4.
    cfg.rewards['air_time'].weight=1.
    cfg.rewards['stride_quality'].params['base_params'].update(
        cycle_time=.6,target_air=.16,target_height=.015,complete_swing=True)
    cfg.rewards['stride_quality'].weight=5.
    cfg.curriculum['stride_quality_weight'].params['weight_stages']=[
        {'step':0,'weight':5.},{'step':300*24,'weight':8.},{'step':700*24,'weight':10.}]
    cfg.rewards['straight_path_cost']=RewardTermCfg(func=mdp.hd1910_walk_straight_path_cost,weight=-.1)
    cfg.curriculum['straight_path_weight']=CurriculumTermCfg(func=mdp.reward_weight,params={
        'reward_name':'straight_path_cost','weight_stages':[
            {'step':0,'weight':-.1},{'step':300*24,'weight':-.3},{'step':700*24,'weight':-.6}]})
    # Continue learned gait without restarting inherited exploration/smoothing.
    for name in ('action_rate_weight','standing_envs','natural_head_weight'):
        key='standing_stages' if name=='standing_envs' else 'weight_stages'
        stages=cfg.curriculum[name].params[key]
        # Use the v12 conditions at 1000 iterations, rather than its future stages.
        current=deepcopy([s for s in stages if s['step']<=1000*24][-1])
        current['step']=0
        cfg.curriculum[name].params[key]=[current]
    return cfg


HD1910SlowWalkRlCfg=replace(deepcopy(HD1910NaturalWalkRlCfg),
    experiment_name='microduck_hd1910_walk_slow',run_name='slow_v13',max_iterations=1000)
HD1910SlowWalkRlCfg.algorithm.learning_rate=1e-4

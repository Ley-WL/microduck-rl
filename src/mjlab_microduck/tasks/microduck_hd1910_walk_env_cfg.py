"""Replica/HD1910 flat-ground walking; train with a fresh command normalizer."""
from copy import deepcopy
from dataclasses import replace
from mjlab.managers import RewardTermCfg
from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg, MicroduckRlCfg
from .microduck_hd1910_recovery_env_cfg import make_hd1910_recovery_env_cfg
from .microduck_hd1910_stand_env_cfg import STAND_HEIGHT
from . import mdp


def make_hd1910_walk_env_cfg(play=False):
    cfg=deepcopy(make_microduck_velocity_env_cfg(play=play))
    physics=make_hd1910_recovery_env_cfg(play=play)
    cfg.scene.entities['robot']=deepcopy(physics.scene.entities['robot'])
    cfg.scene.terrain.collisions=deepcopy(physics.scene.terrain.collisions)
    cfg.events['reset_base']=deepcopy(physics.events['reset_base'])
    cfg.events['reset_robot_joints']=deepcopy(physics.events['reset_robot_joints'])
    cfg.events.pop('base_com',None)
    cfg.curriculum.pop('com_range',None)
    cfg.curriculum.pop('head_pose_range',None)
    cfg.curriculum.pop('body_pose_range',None)
    twist=cfg.commands['twist']
    twist.ranges.lin_vel_x=(-.2,.3)
    twist.ranges.lin_vel_y=(-.15,.15)
    twist.ranges.ang_vel_z=(-1.,1.)
    twist.rel_turn_in_place_envs=.15
    # Explicit zero commands are taught by the inherited standing curriculum.
    for name in ('head_pose','body_pose'):
        cfg.commands[name].ranges=tuple((-.005,.005) for _ in cfg.commands[name].ranges)
    cfg.rewards['idle_home']=RewardTermCfg(func=mdp.hd1910_walk_idle_home,weight=2.,
                                         params={'nominal_height':STAND_HEIGHT})
    # Avoid inheriting the stationary-only motion/leg taxes of stand training.
    cfg.rewards['head_target_cost']=RewardTermCfg(func=mdp.hd1910_settle_head_target_cost,weight=-.03)
    cfg.episode_length_s=20.
    if play:
        for name in ('head_pose','body_pose'):
            cfg.commands[name].ranges=tuple((0.,0.) for _ in cfg.commands[name].ranges)
        cfg.events.pop('push_robot',None)
    return cfg


HD1910WalkRlCfg=replace(deepcopy(MicroduckRlCfg),experiment_name='microduck_hd1910_walk',
                       run_name='replica_m6_walk',logger='tensorboard',max_iterations=6000)

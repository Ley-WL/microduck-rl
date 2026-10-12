"""Replica/HD1910 flat-ground walking; train with a fresh command normalizer."""
from copy import deepcopy
from dataclasses import replace
import math
from mjlab.managers import RewardTermCfg, TerminationTermCfg, CurriculumTermCfg
from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg, MicroduckRlCfg
from .microduck_hd1910_recovery_env_cfg import make_hd1910_recovery_env_cfg
from .microduck_hd1910_stand_env_cfg import STAND_HEIGHT
from . import mdp
from .symmetry import SYMMETRY_CFG


def make_hd1910_walk_env_cfg(play=False):
    cfg=deepcopy(make_microduck_velocity_env_cfg(play=play))
    physics=make_hd1910_recovery_env_cfg(play=play)
    cfg.scene.entities['robot']=deepcopy(physics.scene.entities['robot'])
    # Match the existing hardware runtime: clamp final HOME+delta targets,
    # not encoder feedback or the actor's raw last-action observation.
    model=cfg.scene.entities['robot'].spec_fn().compile()
    cfg.actions['joint_pos'].clip={model.joint(j).name:tuple(float(x) for x in model.jnt_range[j])
        for j in range(model.njnt) if model.jnt_limited[j] and model.joint(j).name.startswith(
            ('left_','right_','neck_','head_'))}
    cfg.scene.terrain.collisions=deepcopy(physics.scene.terrain.collisions)
    cfg.events['reset_base']=deepcopy(physics.events['reset_base'])
    cfg.events['reset_robot_joints']=deepcopy(physics.events['reset_robot_joints'])
    cfg.events.pop('base_com',None)
    cfg.curriculum.pop('com_range',None)
    cfg.curriculum.pop('head_pose_range',None)
    cfg.curriculum.pop('body_pose_range',None)
    cfg.curriculum.pop('head_pose_bias_weight',None)
    twist=cfg.commands['twist']
    twist=mdp.HD1910WalkVelocityCommandCfg(**vars(twist))
    cfg.commands['twist']=twist
    twist.ranges.lin_vel_x=(-.2,.3)
    twist.ranges.lin_vel_y=(-.15,.15)
    twist.ranges.ang_vel_z=(-1.,1.)
    twist.rel_turn_in_place_envs=.1
    twist.rel_forward_envs=0.
    twist.rel_world_envs=0.
    twist.heading_command=False
    twist.ranges.heading=None
    twist.rel_heading_envs=0.
    twist.init_velocity_prob=0.
    twist.resampling_time_range=(2.,4.)
    cfg.curriculum['standing_envs'].params['standing_stages']=[
        {'step':0,'rel_standing_envs':.2},{'step':250*24,'rel_standing_envs':.25}]
    # Explicit zero commands are taught by the inherited standing curriculum.
    for name in ('head_pose','body_pose'):
        cfg.commands[name].ranges=tuple((-.005,.005) for _ in cfg.commands[name].ranges)
    cfg.rewards['idle_home']=RewardTermCfg(func=mdp.hd1910_walk_idle_home,weight=2.,
                                         params={'nominal_height':STAND_HEIGHT})
    # Avoid inheriting the stationary-only motion/leg taxes of stand training.
    cfg.rewards['head_target_cost']=RewardTermCfg(func=mdp.hd1910_settle_head_target_cost,weight=-.03)
    cfg.rewards['official_target_cost']=RewardTermCfg(func=mdp.hd1910_walk_official_target_cost,weight=-.2)
    cfg.rewards['failure_cost']=RewardTermCfg(func=mdp.hd1910_walk_failure_cost,weight=-5.)
    # V1 parked just below the inherited 70-degree fall threshold and earned
    # tracking/air-time/head rewards with its body supported by the floor.
    cfg.terminations['fell_over'].params['limit_angle']=math.radians(45)
    cfg.terminations['low_trunk']=TerminationTermCfg(
        func=mdp.hd1910_walk_low_trunk,params={'nominal_height':STAND_HEIGHT})
    cfg.rewards['track_linear_velocity'].weight=5.
    cfg.rewards['track_linear_velocity'].params['std']=math.sqrt(.02)
    cfg.rewards['track_angular_velocity'].weight=6.
    cfg.rewards['track_angular_velocity'].params['std']=math.sqrt(.04)
    cfg.rewards['yaw_error_cost']=RewardTermCfg(func=mdp.hd1910_walk_yaw_error_cost,weight=-3.)
    cfg.rewards['yaw_bias_cost']=RewardTermCfg(func=mdp.hd1910_walk_yaw_bias_cost,weight=-3.,
                                            params={'tau':.5,'std':.2})
    cfg.rewards['idle_stop_cost']=RewardTermCfg(func=mdp.hd1910_walk_idle_stop_cost,weight=-1.)
    cfg.rewards['upright'].weight=4.
    cfg.rewards['air_time'].weight=.8
    cfg.rewards['pose'].weight=.15
    cfg.rewards['head_pose_tracking'].weight=.5
    cfg.rewards['head_pose_bias'].weight=.6
    # V7's heavy negative head tax incentivized short, low-trunk episodes.
    # Keep the mild DC cost and couple positive gait rewards to head quality.
    # V4 already learned full swings: preserve that incentive when warm-starting
    # the next batch instead of spending another 150 iterations relearning it.
    cfg.rewards['stride_quality']=RewardTermCfg(func=mdp.hd1910_walk_stride_reward,weight=4.,
        params={'cycle_time':.55,'target_air':.2,'target_height':.022})
    cfg.curriculum['stride_quality_weight']=CurriculumTermCfg(func=mdp.reward_weight,params={
        'reward_name':'stride_quality','weight_stages':[
            {'step':0,'weight':4.},{'step':250*24,'weight':5.}]})
    cfg.rewards['foot_swing_height'].weight=-.5
    cfg.rewards['foot_swing_height'].params['target_height']=.022
    cfg.rewards['foot_clearance'].params['target_height']=.022
    # A gait needs to be discovered before strong smoothing is introduced.
    cfg.curriculum['action_rate_weight'].params['weight_stages']=[
        {'step':0,'weight':-.03},{'step':1500*24,'weight':-.15},
        {'step':3000*24,'weight':-.3}]
    for name,term in cfg.rewards.items():
        # head_pose_bias self-negates: it is a cost despite its positive weight,
        # and lowering the trunk must not be a way to switch that cost off.
        if term.weight>0 and name!='head_pose_bias':
            term.params={'base_func':term.func,'base_params':deepcopy(term.params),
                         'nominal_height':STAND_HEIGHT}
            if name in ('track_linear_velocity','stride_quality'):
                term.params['head_quality']=True
            term.func=mdp.hd1910_walk_supported_reward
    cfg.episode_length_s=20.
    if play:
        for name in ('head_pose','body_pose'):
            cfg.commands[name].ranges=tuple((0.,0.) for _ in cfg.commands[name].ranges)
        cfg.events.pop('push_robot',None)
    return cfg


HD1910WalkRlCfg=replace(deepcopy(MicroduckRlCfg),experiment_name='microduck_hd1910_walk',
                       run_name='replica_m6_walk',logger='tensorboard',max_iterations=500)
# Replica foot kinematics have been checked against the reflection table.
# Mirror loss only: do not duplicate unmirrored privileged critic observations.
HD1910WalkRlCfg.algorithm.symmetry_cfg=deepcopy(SYMMETRY_CFG)
HD1910WalkRlCfg.algorithm.learning_rate=1e-4
HD1910WalkRlCfg.algorithm.schedule='fixed'

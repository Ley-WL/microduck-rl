"""Push recovery followed by a quiet, forward-looking HOME stance."""
from copy import deepcopy
from dataclasses import replace
from mjlab.managers import EventTermCfg, RewardTermCfg, CurriculumTermCfg
from .microduck_hd1910_recovery_env_cfg import make_hd1910_recovery_env_cfg, HD1910RecoveryRlCfg
from .microduck_hd1910_stand_env_cfg import STAND_HEIGHT
from . import mdp


def make_hd1910_settle_env_cfg(play=False):
    cfg=make_hd1910_recovery_env_cfg(play=play)
    cfg.events["reset_push_schedule"]=EventTermCfg(func=mdp.hd1910_reset_settle_schedule,mode="reset")
    if not play:
        cfg.events["push_robot"].params.update(
            stages=((0,.25),(250*24,.4),(1000*24,.6)),calm_interval=(4.,7.))
    cfg.rewards["standing_composite"]=RewardTermCfg(
        func=mdp.hd1910_settle_balance,weight=10.,params={"nominal_height":STAND_HEIGHT})
    cfg.rewards["leg_alignment_cost"].weight=-2.
    cfg.rewards["head_home_cost"]=RewardTermCfg(func=mdp.hd1910_settle_head_cost,weight=-2.)
    cfg.rewards["head_target_cost"]=RewardTermCfg(func=mdp.hd1910_settle_head_target_cost,weight=-.5)
    cfg.rewards["quiet_motion_cost"]=RewardTermCfg(func=mdp.hd1910_settle_motion_cost,weight=-.1)
    if not play:
        cfg.curriculum["settle_motion_weight"]=CurriculumTermCfg(
            func=mdp.reward_weight,params={"reward_name":"quiet_motion_cost","weight_stages":[
                {"step":0,"weight":-.1},{"step":200*24,"weight":-.3},
                {"step":500*24,"weight":-.8},{"step":1000*24,"weight":-2.}]})
    cfg.rewards["head_pose_tracking"].weight=2.
    cfg.rewards["track_angular_velocity"].weight=2.
    # Motion regularization is conditional on the external recovery window;
    # do not freeze outputs, add filtering or cap recovery action velocities.
    return cfg


HD1910SettleRlCfg=replace(deepcopy(HD1910RecoveryRlCfg),
    experiment_name="microduck_hd1910_settle",run_name="quiet_after_push",max_iterations=2000)

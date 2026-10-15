"""Ten-beat command-conditioned duck dance using current replica + HD1910."""
from copy import deepcopy
from dataclasses import replace
from mjlab.managers import RewardTermCfg
from . import mdp
from .microduck_hd1910_stand_env_cfg import (
    make_hd1910_stand_env_cfg, HD1910StandRlCfg, STAND_HEIGHT,
)


def make_hd1910_dance_env_cfg(play=False):
    cfg = make_hd1910_stand_env_cfg(play=play)
    for name, part, dim in (("head_pose", "head", 4), ("body_pose", "body", 6)):
        cfg.commands[name] = mdp.DuckDanceCommandCfg(
            part=part, ranges=((0., 0.),)*dim, resampling_time_range=(1000., 1000.),
        )
    cfg.commands["twist"].rel_standing_envs = 1.0
    # Absolute body command frame starts facing forward at the terrain origin.
    cfg.events["reset_base"].params["pose_range"] = {
        "z": (STAND_HEIGHT+.002, STAND_HEIGHT+.004),
        "roll": (-.025, .025), "pitch": (-.025, .025), "yaw": (-.025, .025),
    }
    # The pose-hold objective would directly oppose deliberate swaying/crouching.
    for name in ("standing_composite", "leg_alignment_cost", "head_pose_tracking",
                 "body_pose_tracking", "head_pose_bias", "track_linear_velocity",
                 "track_angular_velocity"):
        cfg.rewards.pop(name, None)
    cfg.rewards["dance_tracking"] = RewardTermCfg(
        func=mdp.duck_dance_tracking, weight=12., params={"nominal_height": STAND_HEIGHT},
    )
    cfg.rewards["upright"].weight = 1.
    cfg.rewards["action_rate_l2"].weight = -.025
    cfg.rewards["body_ang_vel"].weight = -.01
    cfg.rewards["angular_momentum"].weight = -.005
    cfg.episode_length_s = 16.0  # two complete phrases, including entry/exit holds
    return cfg


HD1910DanceRlCfg = replace(deepcopy(HD1910StandRlCfg),
    experiment_name="microduck_hd1910_dance10", run_name="duck_groove_v1",
    max_iterations=1000, save_interval=100)


def make_hd1910_dance_stable_env_cfg(play=False):
    """Separate refinement recipe; preserve v1 for matched baseline evaluation."""
    cfg = make_hd1910_dance_env_cfg(play=play)
    # feet_slip gates on norm(twist) > threshold. Dance always has zero twist.
    cfg.rewards['foot_slip'].params['command_threshold'] = -1.0
    cfg.rewards['foot_slip'].weight = -1.0
    cfg.rewards['dance_balance_cost'] = RewardTermCfg(func=mdp.duck_dance_balance_cost, weight=-2.)
    cfg.rewards['upright'].weight = 2.
    if not play:
        cfg.events['reset_base'].params['pose_range'].update(roll=(-.04,.04),pitch=(-.06,.06))
        cfg.events['push_robot'].params['velocity_range']={'x':(-.12,.12),'y':(-.12,.12)}
        cfg.events['push_robot'].interval_range_s=(3.,6.)
    return cfg


HD1910DanceStableRlCfg = replace(deepcopy(HD1910DanceRlCfg),
    experiment_name='microduck_hd1910_dance10_stable',run_name='stable_v2',max_iterations=400)
HD1910DanceStableRlCfg.algorithm.learning_rate = 3e-5
HD1910DanceStableRlCfg.algorithm.schedule = 'fixed'
HD1910DanceStableRlCfg.algorithm.entropy_coef = .001


def make_hd1910_dance_recovery_env_cfg(play=False):
    cfg = make_hd1910_dance_stable_env_cfg(play=play)
    if not play:
        cfg.events['reset_base'].params['pose_range'].update(roll=(-.08,.08),pitch=(-.10,.10))
        cfg.events['reset_robot_joints'].params['position_range']=(-.04,.04)
        cfg.events['push_robot'].params['velocity_range']={'x':(-.16,.16),'y':(-.16,.16)}
    return cfg


HD1910DanceRecoveryRlCfg = replace(deepcopy(HD1910DanceStableRlCfg),
    experiment_name='microduck_hd1910_dance10_recovery',run_name='recovery_v3',max_iterations=400)
HD1910DanceRecoveryRlCfg.algorithm.learning_rate=1e-5
HD1910DanceRecoveryRlCfg.algorithm.entropy_coef=.0005


EXPRESSIVE_GAINS = (2., 2., 2., 2.5, 1., 2., 2.5, 2., 1., 1.)


def make_hd1910_dance_expressive_env_cfg(play=False):
    cfg = make_hd1910_dance_recovery_env_cfg(play=play)
    for name in ('head_pose','body_pose'):
        cfg.commands[name].gains=EXPRESSIVE_GAINS
        cfg.commands[name].start_fraction=.15
        cfg.commands[name].ramp_steps=0 if play else 300*24
    # Stronger body tracking prevents satisfying the dance mostly with the head.
    cfg.rewards['dance_tracking'].params.update(z_std=.006,roll_std=.045)
    cfg.rewards['pose'].weight=.3  # leave the legs room for deliberate crouching
    return cfg


HD1910DanceExpressiveRlCfg = replace(deepcopy(HD1910DanceRecoveryRlCfg),
    experiment_name='microduck_hd1910_dance10_expressive',run_name='expressive_v4',max_iterations=800)
HD1910DanceExpressiveRlCfg.algorithm.learning_rate=3e-5


def make_hd1910_dance_resilient_env_cfg(play=False):
    """Keep the full expressive choreography while practicing repeated recovery."""
    cfg=make_hd1910_dance_expressive_env_cfg(play=play)
    for name in ('head_pose','body_pose'):
        cfg.commands[name].ramp_steps=0
    if not play:
        cfg.events['push_robot'].params['velocity_range']={'x':(-.18,.18),'y':(-.18,.18)}
        cfg.events['push_robot'].interval_range_s=(1.5,4.)
        cfg.episode_length_s=24.
    return cfg


HD1910DanceResilientRlCfg=replace(deepcopy(HD1910DanceExpressiveRlCfg),
    experiment_name='microduck_hd1910_dance10_resilient',run_name='resilient_v5',max_iterations=1000)
HD1910DanceResilientRlCfg.algorithm.learning_rate=1e-5

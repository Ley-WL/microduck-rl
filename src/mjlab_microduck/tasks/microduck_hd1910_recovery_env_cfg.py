"""Near-standing push recovery, including bursts; not lying-down standup."""
from copy import deepcopy
from dataclasses import replace
import math
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab_microduck.robot.ground_contact import add_body_ground_contacts
from .microduck_hd1910_stand_env_cfg import make_hd1910_stand_env_cfg, get_replica_spec, HD1910StandRlCfg, STAND_HEIGHT
from . import mdp

# Iterations use 24 rollout steps; warm start resets common_step_counter.
PUSH_STAGES = ((0, .15), (250 * 24, .25), (750 * 24, .40), (1500 * 24, .60))


def get_recovery_spec():
    spec = get_replica_spec()
    add_body_ground_contacts(spec)
    return spec


def make_hd1910_recovery_env_cfg(play=False):
    cfg = make_hd1910_stand_env_cfg(play=play)
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = get_recovery_spec
    # Preserve floor-only proxies and the original self-contact masks. The
    # inherited disable_other_geoms would silently erase the spec's masks.
    robot.collisions = (*(replace(c,disable_other_geoms=False) for c in robot.collisions), CollisionCfg(
        geom_names_expr=(r"body_ground_.*",), contype=4, conaffinity=0,
        condim=3, friction=(1., .005, .0001), solref=(.004, 1.),
        disable_other_geoms=False,
    ))
    cfg.scene.terrain.collisions = (CollisionCfg(
        geom_names_expr=(r"terrain",), contype=1, conaffinity=5, condim=3,
        friction=(1., .005, .0001), disable_other_geoms=False,
    ),)
    cfg.events["reset_push_schedule"] = EventTermCfg(func=mdp.hd1910_reset_push_schedule, mode="reset")
    if not play:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.hd1910_burst_push, mode="interval", interval_range_s=(.1,.1),
            params={"stages": PUSH_STAGES},
        )
        cfg.events["reset_base"].params["pose_range"].update(roll=(-.10,.10),pitch=(-.14,.14))
    # Keep HOME as the quiet-time target, but permit hips/knees/ankles to move
    # during recovery. Do not change action limits or the 61D observation.
    cfg.rewards["pose"].weight = .2
    cfg.rewards["track_linear_velocity"].weight = .5
    cfg.rewards["track_angular_velocity"].weight = .5
    cfg.rewards["head_pose_tracking"].weight = .5
    cfg.rewards["upright"].weight = 4.
    cfg.rewards["action_rate_l2"].weight = -.03
    cfg.rewards["foot_slip"].weight = -.03
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=mdp.hd1910_recovery_balance, weight=10.,
        params={"nominal_height":STAND_HEIGHT},
    )
    cfg.rewards["leg_alignment_cost"] = RewardTermCfg(
        func=mdp.hd1910_recovery_pose_cost, weight=-1.,
    )
    cfg.rewards["quiet_pose"] = RewardTermCfg(func=mdp.hd1910_quiet_pose, weight=2.)
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60)
    cfg.episode_length_s = 25.
    return cfg


HD1910RecoveryRlCfg = replace(deepcopy(HD1910StandRlCfg),
    experiment_name="microduck_hd1910_recovery",run_name="burst_recovery",max_iterations=3000)

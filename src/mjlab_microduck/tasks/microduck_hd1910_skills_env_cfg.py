"""Replica STL + HD1910 recovery and commanded sit/stand tasks.

The training mesh/actuator source is the local replica, never the donor asset.
Floor-only mesh contact permits head/body-assisted recovery without creating
overlapping visual-mesh self collisions. Preserve 61D actor / raw previous
actions and match the platform's 0.45-old action EMA.
"""

from copy import deepcopy
from pathlib import Path
import math

import mujoco
import numpy as np
from mjlab.managers import CurriculumTermCfg, EventTermCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_microduck.actuator.feetech_bam import FeetechBamActuatorCfg
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.robot.ground_contact import lowest_mesh_z
from . import mdp
from mjlab_microduck.robot.hardware_profile import ROBOT_DIR, HD1910_PARAMS as PARAMS, actuator_kwargs
from .microduck_standup_env_cfg import make_microduck_standup_env_cfg, MicroduckStandUpRlCfg
from .microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg, MicroduckSitStandRlCfg, SITTING_TARGET_OVERRIDES
from .xgoduck_runtime_actions import XgoDuckPositionActionCfg

# Measured with this geometry and 7.4V/kp5 BAM, 16 noisy starts held 3 seconds.
# Full evidence: artifacts/hd1910-skills-20261009/pose-hold.json.
STAND_Z = 0.11504956315901835
SIT_Z = 0.062396494608470326
POSE_NAMES = ("stand", "sit", "prone", "supine", "left_side", "right_side", "oblique", "inverted")
POSE_ANGLES = ((0, 0), (0, 0), (0, 90), (0, -90), (-90, 0), (90, 0), (70, 45), (180, 0))


def get_skills_spec():
    spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / "robot_walk.xml"))
    count = 0
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname not in ("sole_left", "sole_right") and geom.contype == 0 and geom.conaffinity == 0:
            # Terrain's default affinity includes bit 1; unlike bit 4 this
            # works when the terrain is assembled by mjlab after the robot.
            geom.name = f"body_ground_{count}"
            geom.contype = 1
            geom.conaffinity = 0
            geom.solref = np.array([.004, 1.])
            count += 1
    assert count > 0
    return spec


def pose_spawn_heights():
    model = get_skills_spec().compile()
    data = mujoco.MjData(model)
    import re
    home = np.array([next(angle for pattern, angle in HOME_FRAME.joint_pos.items() if re.fullmatch(pattern, model.joint(j).name)) for j in range(1, model.njnt)])
    heights = []
    for index, (roll, pitch) in enumerate(POSE_ANGLES):
        target = home.copy()
        if index == 1:
            for joint, value in SITTING_TARGET_OVERRIDES.items():
                target[joint] = value
        r, p = np.radians([roll, pitch]) / 2
        data.qpos[:7] = [0, 0, .12, math.cos(r)*math.cos(p), math.sin(r)*math.cos(p), math.cos(r)*math.sin(p), -math.sin(r)*math.sin(p)]
        data.qpos[7:] = target
        mujoco.mj_forward(model, data)
        heights.append(.12 - lowest_mesh_z(model, data))
    return tuple(heights)


SPAWN_HEIGHTS = pose_spawn_heights()


def _adapt(cfg, play):
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = get_skills_spec
    # Entity assembly disables geoms excluded by CollisionCfg. The inherited
    # '*_collision' selection silently removed all body_ground_* contacts.
    # One selection preserves body support and soles; affinity zero keeps
    # every robot geom from contacting another robot geom, while terrain's
    # affinity bit 1 still accepts them.
    robot.collisions = (CollisionCfg(
        geom_names_expr=(r'^(left|right)_foot_collision$', r'^body_ground_\d+$'),
        contype=1, conaffinity=0, condim=3,
        priority={r'^(left|right)_foot_collision$': 1},
        friction=(1., .005, .0001),
        solref={r'^body_ground_\d+$': (.004, 1.)},
    ),)
    actuator = actuator_kwargs()
    if play:
        actuator.update(vin_range=(7.4, 7.4), vin_drop_gain_range=(0., 0.))
    robot.articulation.actuators = (FeetechBamActuatorCfg(**actuator),)
    cfg.actions["joint_pos"] = XgoDuckPositionActionCfg(**vars(cfg.actions["joint_pos"]), alpha=.45)
    # Do not add XML target clipping: same model-output treatment as XgoDuck.
    cfg.actions["joint_pos"].clip = None
    cfg.events["foot_friction"].params["ranges"] = (.4, 1.5)
    # Head/body are zero in production for these skills. Retain all slots.
    for name in ("head_pose", "body_pose"):
        if name in cfg.commands:
            command = cfg.commands[name]
            command.ranges = tuple((0., 0.) for _ in command.ranges)
    for name in ("head_pose_range", "body_pose_range", "body_control_weight", "body_control_std", "body_pose_tracking_weight"):
        cfg.curriculum.pop(name, None)
    cfg.rewards.pop("body_pose_tracking", None)
    for name, amplitude in (("base_ang_vel", .06), ("projected_gravity", .03), ("joint_pos", .01)):
        noise = cfg.observations["actor"].terms[name].noise
        noise.n_min, noise.n_max = -amplitude, amplitude
    # All rewards read the replica's measured target heights.
    for reward in cfg.rewards.values():
        for name in ("stand_z", "target_height", "nominal_height"):
            if name in reward.params:
                reward.params[name] = STAND_Z
        if "sit_z" in reward.params:
            reward.params["sit_z"] = SIT_Z
    cfg.episode_length_s = 15.
    cfg.sim.nconmax = 256
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    # Random pushes are a later robustness experiment, not fallen-spawn coverage.
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("push_magnitude", None)
    return cfg


def make_hd1910_getup_env_cfg(play=False):
    cfg = _adapt(deepcopy(make_microduck_standup_env_cfg(play=play)), play)
    cfg.curriculum.pop("ground_state_mix", None)
    for name in ("height_stand_sharp_weight", "upright_sharp_weight", "standing_composite_weight"):
        cfg.curriculum.pop(name, None)
    cfg.events["set_ground_state"] = EventTermCfg(
        func=mdp.hd1910_reset_skill_pose, mode="reset",
        params={"pose_probabilities": (.25, .25, .12, .08, .1, .1, .05, .05),
                "pose_angles": POSE_ANGLES, "spawn_heights": SPAWN_HEIGHTS,
                "sit_overrides": SITTING_TARGET_OVERRIDES, "joint_noise": .08,
                "tilt_noise_deg": 12., "clearance": .015},
    )
    cfg.curriculum["recovery_pose_mix"] = CurriculumTermCfg(
        func=mdp.event_param_curriculum,
        params={"event_name": "set_ground_state", "param_stages": [
            {"step": 0, "params": {"pose_probabilities": (.25, .25, .12, .08, .1, .1, .05, .05)}},
            {"step": 1000*24, "params": {"pose_probabilities": (.1, .1, .16, .16, .15, .15, .09, .09)}},
            {"step": 2500*24, "params": {"pose_probabilities": (.025, .025, .16, .2, .18, .18, .115, .115)}},
        ]},
    )
    # No velocity/body pose commands in the getup model.
    for command in cfg.commands.values():
        if hasattr(command, "ranges") and not isinstance(command.ranges, tuple):
            command.ranges.lin_vel_x = command.ranges.lin_vel_y = command.ranges.ang_vel_z = (0., 0.)
    return cfg


def make_hd1910_sitstand_env_cfg(play=False):
    cfg = _adapt(deepcopy(make_microduck_sitstand_env_cfg(play=play)), play)
    command = cfg.commands["twist"]
    command.sit_z, command.stand_z = SIT_Z, STAND_Z
    command.resampling_time_range = (4., 7.)
    cfg.events["set_ground_state"] = EventTermCfg(
        func=mdp.hd1910_reset_skill_pose, mode="reset",
        params={"pose_probabilities": (.5, .5, 0., 0., 0., 0., 0., 0.),
                "pose_angles": POSE_ANGLES, "spawn_heights": SPAWN_HEIGHTS,
                "sit_overrides": SITTING_TARGET_OVERRIDES, "joint_noise": .06,
                "tilt_noise_deg": 5., "clearance": .005},
    )
    cfg.rewards["rise_bootstrap"].params["max_height"] = STAND_Z + .01
    return cfg


HD1910GetupRlCfg = deepcopy(MicroduckStandUpRlCfg)
HD1910GetupRlCfg.experiment_name = "hd1910_getup_v1"
HD1910GetupRlCfg.run_name = "replica_getup"
HD1910GetupRlCfg.max_iterations = 12000
HD1910GetupRlCfg.save_interval = 250
HD1910SitStandRlCfg = deepcopy(MicroduckSitStandRlCfg)
HD1910SitStandRlCfg.experiment_name = "hd1910_sitstand_v1"
HD1910SitStandRlCfg.run_name = "replica_sitstand"
HD1910SitStandRlCfg.max_iterations = 10000
HD1910SitStandRlCfg.save_interval = 250

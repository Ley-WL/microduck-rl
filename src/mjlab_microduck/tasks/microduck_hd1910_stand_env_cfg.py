"""Replica + supplied Feetech M6; balance in a HOME-aligned standing pose."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import re
import math
import mujoco
import numpy as np
from mjlab.managers import RewardTermCfg
from mjlab_microduck.actuator.feetech_bam import FeetechBamActuatorCfg
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg, MicroduckRlCfg
from . import mdp
from .xgoduck_runtime_actions import XgoDuckPositionActionCfg

ROBOT_DIR = Path(__file__).resolve().parents[1] / "robot" / "replica"
PARAMS = ROBOT_DIR.parent / "hd1910" / "1910_m6.json"

def get_replica_spec():
    return mujoco.MjSpec.from_file(str(ROBOT_DIR / "robot_walk.xml"))

def nominal_stand_height():
    """Measure HOME with both soles on z=0; do not reuse old model height."""
    model = get_replica_spec().compile()
    data = mujoco.MjData(model)
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        for pattern, angle in HOME_FRAME.joint_pos.items():
            if re.fullmatch(pattern, name):
                data.qpos[model.jnt_qposadr[j]] = angle
                break
    mujoco.mj_forward(model, data)
    bottom = []
    for name in ("left_foot_collision", "right_foot_collision"):
        g = model.geom(name).id
        z_axis = data.geom_xmat[g].reshape(3, 3)[2]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            bottom.append(data.geom_xpos[g, 2] + (model.mesh_vert[start:start+count] @ z_axis).min())
        else:
            assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            bottom.append(data.geom_xpos[g, 2] - np.abs(z_axis) @ model.geom_size[g])
    trunk = model.body("trunk_base").id
    return float(data.xpos[trunk, 2] - min(bottom))

STAND_HEIGHT = nominal_stand_height()

def make_hd1910_stand_env_cfg(play=False):
    cfg = deepcopy(make_microduck_velocity_env_cfg(play=play))
    cfg.actions["joint_pos"] = XgoDuckPositionActionCfg(**vars(cfg.actions["joint_pos"]), alpha=0.45)
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = get_replica_spec
    robot.articulation.actuators = (FeetechBamActuatorCfg(
        json_path=str(PARAMS), target_names_expr=(r"^(?!passive_).*",),
        kp_fw=5.0, vin_range=(7.4, 8.0), vin_drop_gain_range=(0.0, 0.2),
        vin_min=4.0, delay_min_lag=3, delay_max_lag=6,
    ),)
    # Keep all 61 observation slots; tiny nonzero samples keep command weights alive.
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.9 if not play else 1.0
    twist.rel_turn_in_place_envs = 0.0
    twist.ranges.lin_vel_x = (-0.005, 0.005)
    twist.ranges.lin_vel_y = (-0.005, 0.005)
    twist.ranges.ang_vel_z = (-0.005, 0.005)
    for command in ("head_pose", "body_pose"):
        c = cfg.commands[command]
        c.ranges = tuple((-0.005, 0.005) for _ in c.ranges)
    # XgoDuck-matched observation uncertainty; keep this robot's geometry/HOME.
    cfg.events["foot_friction"].params["ranges"] = (0.4, 1.5)
    for name, magnitude in (("base_ang_vel", .06), ("projected_gravity", .03), ("joint_pos", .01)):
        noise = cfg.observations["actor"].terms[name].noise
        noise.n_min, noise.n_max = -magnitude, magnitude
    # Stand-hold starts upright, with small deviations; no lying-down recovery.
    cfg.events["reset_base"].params = {
        "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03),
                       "z": (STAND_HEIGHT + 0.002, STAND_HEIGHT + 0.004),
                       "roll": (-0.06, 0.06), "pitch": (-0.08, 0.08), "yaw": (-math.pi, math.pi)},
        "velocity_range": {},
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.025, 0.025)
    cfg.events["push_robot"].params["velocity_range"] = {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}
    cfg.events["push_robot"].interval_range_s = (4.0, 7.0)
    # Drop inherited walking curricula and unused empty-base COM randomization.
    cfg.curriculum = {}
    cfg.events.pop("base_com", None)
    for name in ("air_time", "foot_clearance", "foot_swing_height"):
        cfg.rewards.pop(name, None)
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["upright"].weight = 3.0
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=mdp.hd1910_standing_composite, weight=8.0,
        params={"nominal_height": STAND_HEIGHT, "height_std": 0.018},
    )
    cfg.rewards["leg_alignment_cost"] = RewardTermCfg(func=mdp.hd1910_leg_pose_cost, weight=-2.0)
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(45)
    cfg.episode_length_s = 15.0
    if play:
        for command in ("head_pose", "body_pose"):
            cfg.commands[command].ranges = tuple((0.0, 0.0) for _ in cfg.commands[command].ranges)
        for name in list(cfg.events):
            if name.startswith("randomize_") or name in ("encoder_bias", "push_robot"):
                cfg.events.pop(name)
        cfg.events["reset_base"].params["pose_range"] = {"z": (STAND_HEIGHT + 0.002, STAND_HEIGHT + 0.002)}
        cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
        robot.articulation.actuators = (replace(robot.articulation.actuators[0], vin_range=(7.4, 7.4), vin_drop_gain_range=(0.0, 0.0)),)
    return cfg

HD1910StandRlCfg = replace(deepcopy(MicroduckRlCfg), experiment_name="microduck_hd1910_stand", run_name="replica_m6", logger="tensorboard", max_iterations=1500)

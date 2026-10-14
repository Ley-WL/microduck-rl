from pathlib import Path

import mujoco
from mjlab_microduck.actuator.feetech_bam import (
    FeetechBamActuatorCfg, FeetechBacklashActuatorCfg,
)
from .hardware_profile import ROBOT_DIR, actuator_kwargs
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


_ROBOT_DIR: Path = ROBOT_DIR

MICRODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
# Local body-contact variant; collision selection belongs to each task.
MICRODUCK_GROUNDCONTACT_XML: Path = _ROBOT_DIR / "robot_allcollisions.xml"
# Full local collision model; shares the current replica assembly.
MICRODUCK_ALLCOLLISIONS_XML: Path = _ROBOT_DIR / "robot_allcollisions.xml"
# 70mm / 15g ball prop for the BallKick task.
MICRODUCK_BALL_XML: Path = _ROBOT_DIR / "ball.xml"
# Roller-skate model: 14 actuated joints + passive wheel hinges (passive_*wheel).
MICRODUCK_GROUNDCONTACT_ROLLERS_XML: Path = _ROBOT_DIR / "robot_allcollisions_rollers.xml"
# Backlash models: every servo joint gets an unactuated passive_<joint>_backlash
# hinge in series (±1° play, 2° total). Exported via
# config_mjcf_{groundcontact,walk}_backlash.json (add_backlash.py post-processor).
MICRODUCK_GROUNDCONTACT_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_backlash.xml"
MICRODUCK_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_backlash.xml"
MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_rollers_backlash.xml"
MICRODUCK_ALLCOLLISIONS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_backlash.xml"

assert MICRODUCK_WALK_XML.exists(), f"XML not found: {MICRODUCK_WALK_XML}"
assert MICRODUCK_GROUNDCONTACT_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_XML}"
assert MICRODUCK_ALLCOLLISIONS_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_XML}"
assert MICRODUCK_BALL_XML.exists(), f"XML not found: {MICRODUCK_BALL_XML}"
assert MICRODUCK_GROUNDCONTACT_ROLLERS_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_ROLLERS_XML}"
assert MICRODUCK_GROUNDCONTACT_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_BACKLASH_XML}"
assert MICRODUCK_WALK_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_WALK_BACKLASH_XML}"
assert MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML}"
assert MICRODUCK_ALLCOLLISIONS_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_BACKLASH_XML}"


def get_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML))


def get_standup_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_XML))


def get_ground_pick_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_XML))


def get_walk_rollers_spec() -> mujoco.MjSpec:
    # NOTE: was loading robot_groundcontact.xml (no wheels) — the roller env
    # silently ran on the wheel-less standup model.
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_ROLLERS_XML))


SERVO_MESH_NAME = "xl330"
SERVO_GEOM_SUFFIX = "_servo_collision"


def name_servo_collision_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Name housing collision geoms from the current CAD identifiers.

    The onshape-to-robot export leaves collision geoms unnamed (only the two
    foot soles are named by the export config). The protective-fall tasks need a
    contact sensor on the servo housings specifically — a servo hitting the floor
    is the event that strips gears — so this names each housing collision geom
    ``<body>_<k>_servo_collision``. The ``_collision`` suffix puts them under
    FULL_COLLISION (condim 1) like the other named collision geoms.
    """
    counts: dict[str, int] = {}
    for geom in list(spec.geoms):
        if geom.meshname != SERVO_MESH_NAME:
            continue
        body = geom.parent.name
        k = counts.get(body, 0)
        counts[body] = k + 1
        name = f"{body}_{k}{SERVO_GEOM_SUFFIX}"
        if geom.contype == 0 and geom.conaffinity == 0:
            # The local export has visual housings only. Add a floor-contact
            # copy for the protective-fall sensor, without altering the mesh,
            # explicit body inertia or enabling overlapping self contacts.
            geom.parent.add_geom(
                name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=geom.meshname, pos=geom.pos, quat=geom.quat,
                contype=1, conaffinity=0, group=3, mass=0.,
            )
        else:
            geom.name = name
    return spec


def get_allcollisions_spec() -> mujoco.MjSpec:
    return name_servo_collision_geoms(mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML)))


def get_allcollisions_backlash_spec() -> mujoco.MjSpec:
    return name_servo_collision_geoms(mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_BACKLASH_XML)))


def get_ball_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_BALL_XML))


def get_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_BACKLASH_XML))


def get_walk_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_BACKLASH_XML))


def get_rollers_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_ROLLERS_BACKLASH_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # Lower body — STAND2 pose: trunk shifted ~5mm forward over the feet so
        # the CoM sits over the ankle axis (was ~5mm behind it at the old HOME,
        # which biased the robot backward and made the standup policy droop its
        # head forward as a counterweight). Leg pitch chain leaned forward:
        # hip_pitch 30°→26.24°, ankle 30°→25.95°, knee 0°→0.28°. Matches the
        # STAND keyframe in scene.xml / scene_walk.xml.
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.0873,
        r".*right_hip_roll.*": 0.0873,
        r".*left_hip_pitch.*": -0.4579,
        r".*right_hip_pitch.*": 0.4579,
        r".*left_knee.*": -0.0049,
        r".*right_knee.*": 0.0049,
        r".*left_ankle.*": 0.4530,
        r".*right_ankle.*": -0.4530,
        # Head
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# All task factories use HD1910, including legacy task IDs.
# "xl330" remains a CAD mesh identifier, never an actuator selection.
_BAM_ACTUATOR_KWARGS = actuator_kwargs()
actuators = FeetechBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)
backlash_actuators = FeetechBacklashActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# HOME frame for the backlash model. HOME_FRAME's unanchored patterns
# (e.g. r".*left_hip_roll.*") would also match passive_left_hip_roll_backlash
# and try to initialize it at -0.0873 rad — outside its ±1° range. Pattern
# matching is first-match-wins in declaration order, so the anchored backlash
# rule placed FIRST pins every backlash joint at 0 and the servo joints fall
# through to the normal HOME values.
BACKLASH_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={r".*_backlash$": 0.0, **HOME_FRAME.joint_pos},
    joint_vel={".*": 0.0},
)

MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# True full-collision model (every part collides, servo housings named — see
# name_servo_collision_geoms). For tasks where the robot falls onto arbitrary
# parts and the reward must know WHICH part hit: VelStand / protective fall.
MICRODUCK_ALLCOLLISIONS_ROBOT_CFG = EntityCfg(
    spec_fn=get_allcollisions_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_GROUND_PICK_ROBOT_CFG = EntityCfg(
    spec_fn=get_ground_pick_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Backlash robots: base model + ±1° serial backlash hinge per servo.
# Encoder reads through the backlash (BacklashEncoderBamActuator feedback +
# joint_pos/vel_rel_backlash observations — see tasks/backlash.py).
# Groundcontact variant → VelStand/StandUp backlash tasks (mirrors
# MICRODUCK_STANDUP_ROBOT_CFG); walk variant → Velocity backlash
# tasks (mirrors MICRODUCK_WALK_ROBOT_CFG, keeps backlash-vs-base comparisons
# unconfounded by the collision model).
MICRODUCK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_WALK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# All-collisions backlash robot → VelStand backlash tasks (mirrors
# MICRODUCK_ALLCOLLISIONS_ROBOT_CFG).
MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_allcollisions_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Roller-skate backlash robot: wheels stay free (passive_*wheel untouched by
# add_backlash.py). collisions=() mirrors MICRODUCK_WALK_ROLLERS_ROBOT_CFG —
# roller wheel collision geoms have no explicit names; XML defaults apply.
MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_rollers_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Free-floating, non-articulated ball prop for the BallKick task. Position is
# set each episode by the reset_ball_in_front_of_foot event; the init pos here
# only matters for the pristine pre-first-reset state.
MICRODUCK_BALL_CFG = EntityCfg(
    spec_fn=get_ball_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, 0.035)),
)

# Roller skate robot: the 4 passive wheel joints (passive_*wheel) have no XML
# actuators; the BAM cfg's target regex already excludes them, so the action
# space stays 14-dimensional. Uses the SAME canonical BAM actuator as every
# other variant (was a plain XmlActuatorCfg PD — an actuator-physics mismatch
# vs the rest of the family, and joint-friction DR was impossible).
MICRODUCK_WALK_ROLLERS_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_rollers_spec,
    init_state=HOME_FRAME,
    collisions=(),  # roller wheel collision geoms have no explicit names; XML defaults apply
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": MICRODUCK_WALK_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    viewer.launch(scene.compile())

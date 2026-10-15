"""Bounded stop-focused transfer from deployed ONNX, with a fresh critic."""
from dataclasses import asdict
from mjlab.managers import RewardTermCfg
from mjlab.utils.spec_config import CollisionCfg
from . import mdp
from . import walk_surface_transfer as base
import kissqy_walk_physics as physics
import mujoco


def robot_spec():
    spec = mujoco.MjSpec.from_file(str(base.ROBOT/'replica/robot_walk_backlash.xml'))
    contract = physics.apply(spec)
    physics.configure_and_validate(spec.compile(), contract)
    return spec


def make_cfg(surface=False, play=False):
    cfg = base.make_cfg(False, play)
    robot = cfg.scene.entities['robot']
    robot.spec_fn = robot_spec
    robot.collisions = (CollisionCfg(geom_names_expr=(r'.*_collision',),
        contype=1, conaffinity=1, condim=1,
        priority={r'^(left|right)_foot_collision$':1},
        friction=(1.,.005,.0001), solref=(.02,1.)),)
    # Feet require tangential contact; preserve the contract on compiled spec.
    robot.collisions = (
        CollisionCfg(geom_names_expr=(r'.*_servo_collision',),contype=1,conaffinity=1,condim=1),
        CollisionCfg(geom_names_expr=(r'^(left|right)_foot_collision$',),contype=1,conaffinity=1,condim=3,priority=1),
    )
    actuator = robot.articulation.actuators[0]
    actuator.kp_fw = 5.
    actuator.vin_range = (7.,8.)
    actuator.vin_drop_gain_range = (0.,.15)
    actuator.vin_min = 4.
    actuator.delay_min_lag = 3
    actuator.delay_max_lag = 6
    cfg.sim.mujoco.timestep = .005
    cfg.decimation = 4
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    cfg.sim.mujoco.disableflags = ()
    cfg.episode_length_s = 24.
    command = cfg.commands['twist']
    cfg.commands['twist'] = mdp.WalkStopCommandCfg(**{k:v for k,v in vars(command).items()
        if k in mdp.WalkStopCommandCfg.__dataclass_fields__})
    cfg.commands['twist'].heading_command = False
    cfg.commands['twist'].ranges.heading = None
    cfg.events['foot_friction'].params['ranges'] = (.7,1.3)
    cfg.rewards['stop_motion'] = RewardTermCfg(func=mdp.walk_stop_motion_cost,weight=-8.)
    cfg.rewards['stop_tilt'] = RewardTermCfg(func=mdp.walk_stop_tilt_cost,weight=-4.)
    # Existing locomotion/head tracking rewards stay: stopping is not rewarded
    # by refusing to walk, and no fixed head pose is forced into balance recovery.
    return cfg


def rl_cfg():
    cfg = base.rl_cfg()
    cfg.algorithm.learning_rate = 1e-5
    cfg.actor.distribution_cfg['init_std'] = .05
    cfg.max_iterations = 100
    cfg.save_interval = 25
    cfg.experiment_name = 'walk_stop_transfer'
    return cfg

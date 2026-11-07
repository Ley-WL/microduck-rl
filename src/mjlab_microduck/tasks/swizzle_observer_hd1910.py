"""HD1910 roller swizzle: retain published task, PPO and four passive wheels."""
from copy import deepcopy
from .microduck_velocity_swizzle_env_cfg import make_microduck_velocity_swizzle_env_cfg, MicroduckSwizzleRlCfg
from .getup_mechanical_contract import edit_mechanical_spec, LIMITS
from mjlab_microduck.robot.microduck_constants import get_walk_rollers_spec


def roller_spec():
    return edit_mechanical_spec(get_walk_rollers_spec())


def make_cfg(play=False):
    cfg = deepcopy(make_microduck_velocity_swizzle_env_cfg(play=play))
    cfg.scene.entities['robot'].spec_fn = roller_spec
    cfg.actions['joint_pos'].clip = dict(LIMITS)
    return cfg


def rl_cfg():
    cfg = deepcopy(MicroduckSwizzleRlCfg)
    cfg.logger = 'tensorboard'
    cfg.upload_model = False
    cfg.experiment_name = 'hd1910_swizzle_observer'
    return cfg

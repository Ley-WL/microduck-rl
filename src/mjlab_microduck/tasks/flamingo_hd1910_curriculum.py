"""HD1910 transfer using the published FlamingoCycleHard curriculum and PPO."""
from copy import deepcopy

from .flamingo_hd1910 import spec
from .getup_mechanical_contract import LIMITS
from .microduck_flamingo_cycle_env_cfg import (
    MicroduckFlamingoCycleRlCfg,
    make_microduck_flamingo_cycle_env_cfg,
)


def make_cfg():
    cfg = deepcopy(make_microduck_flamingo_cycle_env_cfg(hard=True))
    cfg.scene.entities['robot'].spec_fn = spec
    cfg.actions['joint_pos'].clip = dict(LIMITS)
    cfg.sim.nconmax = 200
    cfg.sim.contact_sensor_maxmatch = 256
    return cfg


def rl_cfg():
    cfg = deepcopy(MicroduckFlamingoCycleRlCfg)
    cfg.logger = 'tensorboard'
    cfg.upload_model = False
    cfg.experiment_name = 'hd1910_flamingo_published_curriculum'
    return cfg

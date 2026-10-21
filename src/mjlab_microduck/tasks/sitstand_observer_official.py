"""Public SitStand recipe, with only HD1910 geometry and Observer limits."""
from copy import deepcopy
from .getup_mechanical_contract import edit_mechanical_spec, LIMITS
from mjlab_microduck.robot.microduck_constants import get_standup_spec
from .microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg, MicroduckSitStandRlCfg


def spec():
    return edit_mechanical_spec(get_standup_spec())


def make_cfg(play=False):
    cfg = deepcopy(make_microduck_sitstand_env_cfg(play=play))
    cfg.scene.entities['robot'].spec_fn = spec
    cfg.actions['joint_pos'].clip = dict(LIMITS)
    cfg.sim.nconmax = 200
    cfg.sim.contact_sensor_maxmatch = 256
    return cfg


def rl_cfg():
    cfg = deepcopy(MicroduckSitStandRlCfg)
    cfg.logger = 'tensorboard'
    cfg.upload_model = False
    cfg.experiment_name = 'hd1910_sitstand_observer_official'
    return cfg

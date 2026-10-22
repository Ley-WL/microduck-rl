"""Public GroundPick/Roulade recipes with HD1910 and Observer joint limits."""
from copy import deepcopy
from .getup_mechanical_contract import edit_mechanical_spec, LIMITS
from mjlab_microduck.robot.microduck_constants import get_ground_pick_spec, get_standup_spec
from .microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg, MicroduckGroundPickRlCfg
from .microduck_roulade_env_cfg import make_microduck_roulade_env_cfg, MicroduckRouladeRlCfg

FACTORIES = {'pick': make_microduck_ground_pick_env_cfg, 'roll': make_microduck_roulade_env_cfg}
RUNNERS = {'pick': MicroduckGroundPickRlCfg, 'roll': MicroduckRouladeRlCfg}


def pick_spec():
    return edit_mechanical_spec(get_ground_pick_spec())


def roll_spec():
    return edit_mechanical_spec(get_standup_spec())


def make_cfg(task, play=False):
    cfg = deepcopy(FACTORIES[task](play=play))
    cfg.scene.entities['robot'].spec_fn = pick_spec if task == 'pick' else roll_spec
    cfg.actions['joint_pos'].clip = dict(LIMITS)
    cfg.sim.nconmax = 200
    cfg.sim.contact_sensor_maxmatch = 256
    return cfg


def rl_cfg(task):
    cfg = deepcopy(RUNNERS[task])
    cfg.logger = 'tensorboard'
    cfg.upload_model = False
    cfg.experiment_name = f'hd1910_{task}_observer_official'
    return cfg

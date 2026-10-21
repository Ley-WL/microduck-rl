"""Official cycle rewards with the current HD1910 hardware and target limits."""
from copy import deepcopy
from .microduck_flamingo_cycle_env_cfg import make_microduck_flamingo_cycle_env_cfg, MicroduckFlamingoCycleRlCfg
from .getup_mechanical_contract import edit_mechanical_spec, LIMITS
from mjlab_microduck.robot.microduck_constants import get_standup_spec

def spec():
    return edit_mechanical_spec(get_standup_spec())

def make_cfg(play=False):
    cfg=deepcopy(make_microduck_flamingo_cycle_env_cfg(play=play,hard=True))
    cfg.scene.entities['robot'].spec_fn=spec
    cfg.actions['joint_pos'].clip=dict(LIMITS)
    cfg.sim.nconmax=200
    cfg.sim.contact_sensor_maxmatch=256
    # Imported actor already learned mature curricula; do not restart discovery.
    for term in cfg.curriculum.values():
        for key,value in term.params.items():
            if key.endswith('_stages') and isinstance(value,list) and value:
                term.params[key]=[{**value[-1], 'step':0}]
    return cfg

def rl_cfg():
    cfg=deepcopy(MicroduckFlamingoCycleRlCfg)
    cfg.algorithm.learning_rate=1e-5
    cfg.algorithm.schedule='fixed'
    cfg.actor.distribution_cfg['init_std']=.08
    cfg.logger='tensorboard'
    cfg.upload_model=False
    cfg.experiment_name='hd1910_flamingo_cycle_transfer'
    cfg.save_interval=50
    return cfg

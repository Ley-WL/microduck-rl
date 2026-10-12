"""Independent transfer candidate; preserve old tasks and deployment v3."""
from copy import deepcopy
from mjlab.managers import CurriculumTermCfg,EventTermCfg,RewardTermCfg
from . import mdp
from .microduck_hd1910_sit_contact_env_cfg import make_hd1910_sit_contact_env_cfg,HD1910SitContactRlCfg
from .sit_transfer_ppo import SitTransferPpoCfg


def make_hd1910_sit_transfer_env_cfg(play=False):
    cfg=make_hd1910_sit_contact_env_cfg(play=play)
    cfg.rewards['action_rate_l2'].weight=-.1
    cfg.curriculum['action_rate_weight'].params['weight_stages']=[
        {'step':0,'weight':-.1},{'step':250*24,'weight':-.15},{'step':500*24,'weight':-.2}]
    cfg.rewards['target_acceleration']=RewardTermCfg(func=mdp.hd1910_target_acceleration_cost,weight=-.05)
    if not play:
        cfg.events['transfer_pitch_kick']=EventTermCfg(func=mdp.hd1910_transition_pitch_kick,mode='interval',
            interval_range_s=(.12,.20),params={'magnitude':(.7,1.6),'probability':.25,'window_s':.8})
        cfg.events['transfer_motor_gains']=EventTermCfg(func=mdp.randomize_delayed_actuator_gains,mode='reset',
            params={'kp_range':(.8,1.2),'kd_range':(.9,1.1)})
    return cfg


HD1910SitTransferRlCfg=deepcopy(HD1910SitContactRlCfg)
old=vars(HD1910SitTransferRlCfg.algorithm)
HD1910SitTransferRlCfg.algorithm=SitTransferPpoCfg(**{k:v for k,v in old.items() if k!='class_name'})
HD1910SitTransferRlCfg.algorithm.learning_rate=1e-5
HD1910SitTransferRlCfg.algorithm.schedule='fixed'
HD1910SitTransferRlCfg.algorithm.entropy_coef=0.
HD1910SitTransferRlCfg.algorithm.num_learning_epochs=2
HD1910SitTransferRlCfg.algorithm.num_mini_batches=4
HD1910SitTransferRlCfg.algorithm.clip_param=.1
HD1910SitTransferRlCfg.experiment_name='hd1910_sitstand_transfer_v1'
HD1910SitTransferRlCfg.run_name='retained-smooth1000'
HD1910SitTransferRlCfg.max_iterations=1000
HD1910SitTransferRlCfg.save_interval=100

"""Second independent transfer run: optimize normalized knee process cost."""
from copy import deepcopy
from mjlab.managers import EventTermCfg,RewardTermCfg
from .microduck_hd1910_sit_transfer_env_cfg import make_hd1910_sit_transfer_env_cfg,HD1910SitTransferRlCfg
from . import sit_process_cost as process


def make_hd1910_sit_process_env_cfg(play=False):
    cfg=make_hd1910_sit_transfer_env_cfg(play)
    cfg.rewards.pop('target_acceleration')
    cfg.rewards['knee_process']=RewardTermCfg(func=process.knee_process_cost,weight=-.15,
                                            params={'scale_rad':.05})
    if not play:
        cfg.events['process_reset']=EventTermCfg(func=process.reset_transition_state,mode='reset')
        cfg.events['transfer_pitch_kick'].func=process.process_pitch_kick
        cfg.events['transfer_pitch_kick'].params['probability']=1.
        cfg.events['transfer_motor_gains'].func=process.randomize_process_gains
    return cfg


HD1910SitProcessRlCfg=deepcopy(HD1910SitTransferRlCfg)
HD1910SitProcessRlCfg.experiment_name='hd1910_sitstand_process_v2'
HD1910SitProcessRlCfg.algorithm.temporal_coef=3.
HD1910SitProcessRlCfg.algorithm.onpolicy_retention_coef=.2
HD1910SitProcessRlCfg.run_name='process1000'

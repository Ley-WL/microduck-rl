"""Pre-contact forward-pitch cost for commanded rise; paired diagnostic task."""
from copy import deepcopy
import math
import torch
from mjlab.managers import RewardTermCfg
from .microduck_hd1910_sit_head_env_cfg import make_sit_head_env_cfg,SitHeadRlCfg

def forward_pitch_cost(gravity,stand):
    # R_y(+pitch) makes projected world-down x positive. Penalize forward
    # inclination before head contact, without prescribing a joint trajectory.
    threshold=math.sin(math.radians(15.))
    return stand*((gravity[:,0]-threshold).clamp_min(0.)/threshold).square().clamp(max=4.)

def standing_forward_pitch_cost(env):
    gravity=env.scene['robot'].data.projected_gravity_b
    stand=1.-env.command_manager.get_command('twist')[:,0].clamp(0.,1.)
    return forward_pitch_cost(gravity,stand)

def make_sit_pitch_env_cfg(play=False,penalty=True):
    cfg=make_sit_head_env_cfg(play=play,penalty=True)
    cfg.rewards['standing_forward_pitch']=RewardTermCfg(func=standing_forward_pitch_cost,weight=-2. if penalty else 0.)
    return cfg

SitPitchRlCfg=deepcopy(SitHeadRlCfg)
SitPitchRlCfg.experiment_name='hd1910_sit_pitch_guard'
SitPitchFixedRlCfg=deepcopy(SitPitchRlCfg)
SitPitchFixedRlCfg.experiment_name='hd1910_sit_pitch_fixedlr'
SitPitchFixedRlCfg.algorithm.schedule='fixed'

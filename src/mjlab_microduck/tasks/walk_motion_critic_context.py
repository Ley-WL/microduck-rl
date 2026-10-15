"""Expose reward memory and scheduled commands to the training critic only."""
import math
import torch
from mjlab.managers import ObservationTermCfg
from . import walk_mean_motion_tracking as base


def reward_context(env):
    manager=getattr(env,'reward_manager',None)
    # ObservationManager determines shapes before RewardManager is constructed.
    if manager is None:
        errors=torch.zeros((env.num_envs,6),device=env.device)
    else:
        errors=torch.cat([manager.get_term_cfg(name).func.error for name in
            ('track_linear_velocity','track_angular_velocity')],dim=-1)
    term=env.command_manager.get_term('twist')
    phase=term.stop_time*(2*math.pi/30.)
    plan=torch.stack((phase.sin(),phase.cos(),term.stop_speed,term.stop_yaw),dim=-1)
    return torch.cat((errors,plan),dim=-1)


def make_cfg(surface=False,play=False):
    cfg=base.make_cfg(surface,play)
    cfg.observations['critic'].terms['reward_context']=ObservationTermCfg(func=reward_context)
    return cfg


def rl_cfg():
    cfg=base.rl_cfg();cfg.experiment_name='walk_motion_critic_context'
    return cfg

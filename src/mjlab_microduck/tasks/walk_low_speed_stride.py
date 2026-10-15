"""Experimental command coverage and minimum useful stride; not an accepted policy."""
from dataclasses import dataclass
import torch
from . import walk_sustained_transfer as base

SPEEDS=(-.4,-.3,-.2,-.1,-.05,0.,.05,.1,.2,.3,.4)

class CoveredCommand(base.SustainedCommand):
    def reset(self,env_ids=None):
        result=super().reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        speed=torch.tensor(SPEEDS,device=self.device)
        self.stop_speed[ids]=speed[torch.randint(len(speed),(self.stop_time[ids].numel(),),device=self.device)]
        return result

@dataclass(kw_only=True)
class CoveredCommandCfg(base.SustainedCommandCfg):
    def build(self,env):return CoveredCommand(self,env)

def make_cfg(surface=False,play=False,larger_steps=True):
    cfg=base.make_cfg(surface,play)
    cfg.commands['twist']=CoveredCommandCfg(**vars(cfg.commands['twist']))
    if larger_steps:
        # At 0.05m/s, 2.75cm no longer saturates the distance component.
        # Credit a completed slow swing without the old 0.3s ceiling.
        cfg.rewards['long_stride'].params.update(min_stride=.06,max_air_credit=1.)
    return cfg

def rl_cfg():
    cfg=base.rl_cfg();cfg.experiment_name='walk_low_speed_stride'
    return cfg

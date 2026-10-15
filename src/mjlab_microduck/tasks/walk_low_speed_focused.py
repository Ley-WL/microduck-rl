"""Low-speed stride experiment from a partially starting student; not accepted."""
from dataclasses import dataclass
import torch
from . import walk_low_speed_stride as base


class FocusedCommand(base.CoveredCommand):
    def reset(self,env_ids=None):
        result=super().reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        values=self.stop_speed[ids]
        selected=torch.rand(values.shape,device=self.device)<.5
        low=torch.where(torch.rand(values.shape,device=self.device)<.5,-.05,.05)
        self.stop_speed[ids]=torch.where(selected,low,values)
        return result


@dataclass(kw_only=True)
class FocusedCommandCfg(base.CoveredCommandCfg):
    def build(self,env):return FocusedCommand(self,env)


def make_cfg(surface=False,play=False,stronger_stride=True):
    cfg=base.make_cfg(surface,play)
    cfg.commands['twist']=FocusedCommandCfg(**vars(cfg.commands['twist']))
    # Both arms retain six-centimetre distance credit and slow-swing air credit.
    # Only this weight differs; no reward for idle, repeated-foot or hopping.
    cfg.rewards['long_stride'].weight=6. if stronger_stride else 1.
    return cfg


def rl_cfg():
    cfg=base.rl_cfg();cfg.experiment_name='walk_low_speed_focused'
    return cfg

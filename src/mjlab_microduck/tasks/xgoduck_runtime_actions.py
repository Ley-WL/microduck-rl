"""Model-specific action EMA shared with the XgoDuck deployment profile.

Raw manager actions remain the observation/history; only position targets are filtered.
"""
from dataclasses import dataclass
import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class XgoDuckPositionActionCfg(JointPositionActionCfg):
    alpha: float = 0.45

    def build(self, env):
        return XgoDuckPositionAction(self, env)


class XgoDuckPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not 0 <= cfg.alpha < 1:
            raise ValueError("action alpha must be in [0,1)")
        self._filtered_actions = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions):
        self._filtered_actions.mul_(self.cfg.alpha).add_(actions, alpha=1-self.cfg.alpha)
        super().process_actions(self._filtered_actions)
        self._raw_actions.copy_(actions)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._filtered_actions[slice(None) if env_ids is None else env_ids] = 0

    @property
    def position_target(self):
        """Absolute target after EMA, HOME and optional target bounds."""
        return self._processed_actions

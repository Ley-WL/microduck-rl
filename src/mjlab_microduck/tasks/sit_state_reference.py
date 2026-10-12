"""Diagnostic reward reference determined by observable joint posture.

The deployed command stays binary and the actor remains 61D. This changes
training rewards only; it does not insert a hand-made actuator trajectory.
Compare against WeakProcess before considering any new weights for deployment.
"""
import torch
from dataclasses import dataclass
from . import mdp
from .microduck_hd1910_sit_contact_env_cfg import SIT_OVERRIDES


def posture_reference(position, home, seated, flag, lead=.1):
    """Project leg posture onto HOME->SIT, then reward a bounded lead.

    Unlike an elapsed-time ramp, identical posture and command have identical
    targets even after a disturbance. The lead is an experimental reward
    parameter, not an enforced physical speed or a hardware setting.
    """
    direction = seated - home
    progress = ((position-home)*direction).sum(-1)/direction.square().sum(-1).clamp_min(1e-8)
    progress = progress.clamp(0., 1.)
    return progress + (flag-progress).clamp(-lead, lead)


class StateReferenceCommand(mdp.SitStandCommand):
    @property
    def alpha(self):
        asset = self.robot
        position = mdp._servo_joint_pos(self._env_ref, asset)
        home = mdp._servo_default_joint_pos(self._env_ref, asset)
        seated = home.clone()
        # The v3 contract's fixed 14-servo layout; helpers handle passive joints.
        for index, value in SIT_OVERRIDES.items():
            seated[:, index] = value
        ids = [2, 3, 11, 12]
        return posture_reference(position[:, ids], home[:, ids], seated[:, ids], self.command[:, 0])


@dataclass(kw_only=True)
class StateReferenceCommandCfg(mdp.SitStandCommandCfg):
    class_type: type = StateReferenceCommand

    def build(self, env):
        return StateReferenceCommand(self, env)

"""Diagnostic gait-period averaging of tracking errors; no runtime input changes."""
import torch

from . import walk_mean_motion_tracking as base
from .walk_sustained_transfer import SustainedBias


def tracking_tau(command):
    # A 10cm progress cycle lasts 0.1 / |vx| seconds. Preserve the old window
    # for rest/pure yaw and faster motion; never divide by a near-zero command.
    speed = command[:, :2].norm(dim=-1)
    tau = (.1 / speed.clamp_min(.025)).clamp(.7, 4.)
    return torch.where(speed > .001, tau, .7)[:, None]


class CycleTracking(base.MeanMotionTracking):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reference = base.MeanMotionTracking(cfg, env)

    def reset(self, env_ids):
        super().reset(env_ids)
        self.reference.reset(env_ids)

    def __call__(self, env, original, std, command_name, angular=False, tau=.7):
        reference = self.reference(env, original, std, command_name, angular, tau)
        score = super().__call__(env, original, std, command_name, angular,
                                 tracking_tau(env.command_manager.get_command(command_name)))
        self.legacy_reward = reference
        return score


class CycleBias(SustainedBias):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reference = SustainedBias(cfg, env)
        self.legacy_reward = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids):
        super().reset(env_ids)
        self.reference.reset(env_ids)
        self.legacy_reward[env_ids] = 0.

    def __call__(self, env):
        self.legacy_reward = self.reference(env)
        robot = env.scene['robot']
        actual = torch.cat((robot.data.root_link_lin_vel_b[:, :2],
                            robot.data.root_link_ang_vel_b[:, 2:3]), -1)
        command = env.command_manager.get_command('twist')
        # Read-only diagnostic snapshot at reward time, before env.step's
        # forward refresh and command update. Never fed to the actor/critic.
        self.audit_inputs = torch.cat((actual, command), -1)
        self.error.lerp_(actual - command, env.step_dt / (tracking_tau(command) + env.step_dt))
        scale = torch.tensor([.1, .1, .4], device=env.device)
        return (self.error / scale).square().sum(-1).clamp(max=25.)


def make_cfg(surface=False, play=False):
    cfg = base.make_cfg(surface, play)
    for name in ('track_linear_velocity', 'track_angular_velocity'):
        cfg.rewards[name].func = CycleTracking
    cfg.rewards['tracking_bias'].func = CycleBias
    return cfg


def rl_cfg():
    cfg = base.rl_cfg()
    cfg.experiment_name = 'walk_cycle_mean_tracking'
    return cfg

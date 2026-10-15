"""Refine the large-step student using sustained tracking and stop penalties."""
from dataclasses import dataclass
import torch
from . import walk_tracking_transfer as base


class SustainedCommand(base.TrackingCommand):
    def compute(self, dt):
        self.stop_time += dt
        t = torch.remainder(self.stop_time, 30.)
        moving = ((t >= 1.) & (t < 11.)) | ((t >= 16.) & (t < 26.))
        direction = torch.where(t < 16., 1., -1.)
        self.vel_command_b[:] = torch.stack((self.stop_speed*direction*moving,
            torch.zeros_like(t), self.stop_yaw*direction*moving), -1)
        self.is_standing_env[:] = ~moving | (self.vel_command_b.norm(dim=-1) < .001)
        self.is_heading_env[:] = False
        self.is_world_env[:] = False


@dataclass(kw_only=True)
class SustainedCommandCfg(base.TrackingCommandCfg):
    def build(self, env):
        return SustainedCommand(self, env)


class SustainedBias(base.TrackingBias):
    def __call__(self, env):
        robot=env.scene['robot']
        actual=torch.cat((robot.data.root_link_lin_vel_b[:,:2],
            robot.data.root_link_ang_vel_b[:,2:3]),-1)
        self.error.lerp_(actual-env.command_manager.get_command('twist'), env.step_dt/(.7+env.step_dt))
        scale=torch.tensor([.1,.1,.4],device=env.device)
        return (self.error/scale).square().sum(-1).clamp(max=25.)


def make_cfg(surface=False, play=False):
    cfg=base.make_cfg(surface,play)
    cfg.commands['twist']=SustainedCommandCfg(**vars(cfg.commands['twist']))
    cfg.episode_length_s=32.
    cfg.rewards['tracking_bias'].func=SustainedBias
    cfg.rewards['tracking_bias'].weight=-4.
    cfg.rewards['track_angular_velocity'].weight=1.
    cfg.rewards['stop_motion'].weight=-8.
    # Preserve the existing complete alternating-step objective and actuator PD.
    return cfg


def rl_cfg():
    cfg=base.rl_cfg()
    cfg.algorithm.learning_rate=1e-5
    cfg.experiment_name='walk_sustained_transfer'
    return cfg

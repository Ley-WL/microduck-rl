"""Experimental low-speed tracking over gait oscillation; runtime unchanged."""
import torch
from . import walk_low_speed_focused_control as base


class MeanTracking:
    def __init__(self,cfg,env):
        self.error=torch.zeros((env.num_envs,3),device=env.device)
        self.legacy_reward=torch.zeros(env.num_envs,device=env.device)

    def reset(self,env_ids):
        self.error[env_ids]=0.
        self.legacy_reward[env_ids]=0.

    def __call__(self,env,original,std,command_name,angular=False,tau=.7):
        command=env.command_manager.get_command(command_name)
        robot=env.scene['robot']
        actual=robot.data.root_link_ang_vel_b if angular else robot.data.root_link_lin_vel_b
        desired=torch.zeros_like(actual)
        if angular:desired[:,2]=command[:,2]
        else:desired[:,:2]=command[:,:2]
        self.error.lerp_(actual-desired,env.step_dt/(tau+env.step_dt))
        self.legacy_reward=original(env,std=std,command_name=command_name)
        averaged=torch.exp(-self.error.square().sum(-1)/std**2)
        speed=command[:,:2].norm(dim=-1)
        # Zero commands/pure yaw and >=0.1m/s use the unchanged original term.
        return torch.where((speed>.03)&(speed<.1),averaged,self.legacy_reward)


def make_cfg(surface=False,play=False):
    cfg=base.make_cfg(surface,play)
    for name,angular in [('track_linear_velocity',False),('track_angular_velocity',True)]:
        term=cfg.rewards[name]
        term.params.update(original=term.func,angular=angular)
        term.func=MeanTracking
    return cfg


def rl_cfg():
    cfg=base.rl_cfg();cfg.experiment_name='walk_low_speed_mean_tracking'
    return cfg

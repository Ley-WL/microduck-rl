"""Small, anchored speed/yaw/long-stride transfer; never changes runtime PD."""
from dataclasses import dataclass
import torch
from mjlab.managers import RewardTermCfg
from . import mdp
from . import walk_stop_transfer as base


class TrackingCommand(mdp.WalkStopCommand):
    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        n = self.stop_time[ids].numel()
        # Equal coverage of forward/backward, straight, turning, and pure yaw.
        speed = torch.tensor([-.2, -.1, 0., .1, .2], device=self.device)
        yaw = torch.tensor([-.6, 0., .6], device=self.device)
        self.stop_speed[ids] = speed[torch.randint(5, (n,), device=self.device)]
        self.stop_yaw[ids] = yaw[torch.randint(3, (n,), device=self.device)]
        return result


@dataclass(kw_only=True)
class TrackingCommandCfg(mdp.WalkStopCommandCfg):
    def build(self, env):
        return TrackingCommand(self, env)


class TrackingBias:
    """Average tracking error over gait oscillation, including lateral slip."""
    def __init__(self, cfg, env):
        self.error = torch.zeros((env.num_envs, 3), device=env.device)

    def reset(self, env_ids):
        self.error[env_ids] = 0.

    def __call__(self, env):
        robot = env.scene['robot']
        actual = torch.cat((robot.data.root_link_lin_vel_b[:, :2],
                            robot.data.root_link_ang_vel_b[:, 2:3]), dim=-1)
        command = env.command_manager.get_command('twist')
        self.error.lerp_(actual - command, env.step_dt / (.35 + env.step_dt))
        scale = torch.tensor([.1, .1, .4], device=env.device)
        return (self.error / scale).square().sum(-1).clamp(max=25.)


def make_cfg(surface=False, play=False):
    cfg = base.make_cfg(False, play)
    cfg.commands['twist'] = TrackingCommandCfg(**vars(cfg.commands['twist']))
    cfg.rewards['track_linear_velocity'].weight = 5.
    cfg.rewards['track_linear_velocity'].params['std'] = .12
    cfg.rewards['track_angular_velocity'].weight = 4.
    cfg.rewards['track_angular_velocity'].params['std'] = .35
    cfg.rewards['tracking_bias'] = RewardTermCfg(func=TrackingBias, weight=-.5)
    cfg.rewards['stop_motion'].weight = -2.
    # Reward complete alternating swings, weighted by air time, not step count.
    # At .2m/s distance credit saturates at .11m; larger is preferred up to
    # that bound, while contact/clearance/sideways error still constrain it.
    cfg.rewards['long_stride'] = RewardTermCfg(func=mdp.hd1910_walk_stride_reward,
        weight=1., params=dict(cycle_time=.55, target_air=.18, target_height=.018,
            require_alternation=True, complete_swing=True, component_shaping=True))
    cfg.rewards['failure_cost'] = RewardTermCfg(func=mdp.hd1910_walk_failure_cost, weight=-5.)
    cfg.terminations['fell_over'].params['limit_angle'] = .7853981633974483
    return cfg


def rl_cfg():
    cfg = base.rl_cfg()
    cfg.algorithm.learning_rate = 3e-6
    cfg.max_iterations = 50
    cfg.save_interval = 10
    cfg.experiment_name = 'walk_tracking_stride_transfer'
    return cfg

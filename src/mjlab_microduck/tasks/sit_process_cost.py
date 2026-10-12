"""Normalized final-target costs and reset-safe transition perturbation.

Diagnostic training scales are not physical servo rate limits. Runtime remains
the same ONNX/EMA/HOME contract. Store selected knee columns by action names.
"""
import torch
from . import mdp


def knee_process_cost(env, scale_rad=.05):
    term=env.action_manager.get_term('joint_pos')
    names=term._target_names
    columns=[i for i,name in enumerate(names) if name.endswith('knee')]
    assert len(columns)==2
    target=term.position_target[:,columns]
    if not hasattr(env,'_process_knee_last'):
        env._process_knee_last=target.detach().clone()
        env._process_knee_delta=torch.zeros_like(target)
    delta=target-env._process_knee_last
    acceleration=delta-env._process_knee_delta
    cost=(delta/scale_rad).square().mean(-1)+.5*(acceleration/scale_rad).square().mean(-1)
    reset=env.episode_length_buf<=1
    env._process_knee_last=target.detach().clone()
    env._process_knee_delta=torch.where(reset[:,None],torch.zeros_like(delta),delta).detach().clone()
    return torch.where(reset,torch.zeros_like(cost),cost)


def reset_transition_state(env,env_ids):
    if hasattr(env,'_sit_transfer_command'):
        env._sit_transfer_command[env_ids]=env.command_manager.get_command('twist')[env_ids,0]
        env._sit_transfer_edge_tick[env_ids]=-100000
    # Reset events may precede command-manager reset; the first interval
    # callback explicitly synchronizes with the new sampled reset command.
    if not hasattr(env,'_process_transition_synced'):
        env._process_transition_synced=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    env._process_transition_synced[env_ids]=False


def process_pitch_kick(env,env_ids,magnitude=(.7,1.6),probability=1.,window_s=.8):
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    if not hasattr(env,'_sit_transfer_command'):
        env._sit_transfer_command=env.command_manager.get_command('twist')[:,0].clone()
        env._sit_transfer_edge_tick=env.episode_length_buf.clone()-100000
        env._sit_transfer_kick_count=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    unsynced=env_ids[~env._process_transition_synced[env_ids]]
    env._sit_transfer_command[unsynced]=env.command_manager.get_command('twist')[unsynced,0]
    env._sit_transfer_edge_tick[unsynced]=-100000
    env._process_transition_synced[unsynced]=True
    mdp.hd1910_transition_pitch_kick(env,env_ids,magnitude,probability,window_s)


def randomize_process_gains(env,env_ids,kp_range=(.8,1.2),kd_range=(.9,1.1)):
    """Sample one factor per actuator/world, without averaging away its range.

    kd is the BAM electrical velocity factor, not identified firmware D.
    """
    from bam.mjlab import BamActuator
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    for actuator in env.scene['robot'].actuators:
        if not isinstance(actuator,BamActuator):continue
        shape=(len(env_ids),1)
        kp=torch.empty(shape,device=env.device).uniform_(*kp_range)
        kd=torch.empty(shape,device=env.device).uniform_(*kd_range)
        actuator.reset_gains(env_ids)
        actuator.set_gains(env_ids,kp_scale=kp,kd_scale=kd)

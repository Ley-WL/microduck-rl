"""Endpoint reward eligibility aligned with the diagnostic upright criterion.

Do not gate transitional leaning, directional L1 costs, or upright bootstrap.
These are training reward choices, not hardware protection thresholds.
"""
import math
import torch
from . import mdp


def arrival_gate_values(quat,flag,blend,full_degrees=12.,zero_degrees=20.):
    assert 0<=full_degrees<zero_degrees<=180
    cos_tilt=1.-2.*(quat[:,1].square()+quat[:,2].square())
    full,zero=math.cos(math.radians(full_degrees)),math.cos(math.radians(zero_degrees))
    t=((cos_tilt-zero)/(full-zero)).clamp(0.,1.)
    gate=t.square()*(3.-2.*t)
    # Existing rewards guide the moving setpoint without requiring uprightness.
    return torch.where((flag-blend).abs()<.02,gate,torch.ones_like(gate))


def arrival_gate(env,command_name='twist'):
    term=env.command_manager.get_term(command_name)
    return arrival_gate_values(env.scene['robot'].data.root_link_quat_w,
                               term.command[:,0],term.alpha)


def arrival_pose_match(env,**params):
    return mdp.posture_pose_group_match(env,**params)*arrival_gate(env,params['command_name'])


def arrival_height_match(env,**params):
    return mdp.posture_height_gaussian(env,**params)*arrival_gate(env,params['command_name'])


def arrival_stillness(env,**params):
    return mdp.posture_stillness(env,**params)*arrival_gate(env,params['command_name'])
